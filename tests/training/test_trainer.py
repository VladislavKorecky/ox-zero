"""Tests for the trainer: the AdamW step and one generation of steps.

The trainer is where self-play data turns into better weights: sample a
mini-batch from the replay buffer, apply a random symmetry per example,
compute the AlphaZero loss `(z - v)² - π^T log p` and take one AdamW step.
These tests pin the wiring around that loop: the optimiser actually learns
(a fixed batch is memorised), the network object shared with the
`NetworkEvaluator` flips between train and eval mode safely, the optimiser is
built from the config and resumes from a checkpoint, a generation runs the
configured number of steps and reports their mean losses, and augmentation
is applied to every sampled batch.

Plan: docs/plans/archive/04-training-pipeline.md, step 3. Tiny network, CPU only.
"""

import copy
import math
from typing import Any

import numpy as np
import pytest
import torch

import ox_zero.training.trainer as trainer_module
from ox_zero.engine.encoding import encode, legal_mask
from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.network_evaluator import NetworkEvaluator
from ox_zero.game.rules import State, apply_move, initial_state, is_terminal, legal_moves
from ox_zero.training.checkpoint import load_checkpoint, save_checkpoint
from ox_zero.training.replay import Batch, ReplayBuffer, augment
from ox_zero.training.selfplay import Examples
from ox_zero.training.trainer import StepLosses, TrainConfig, Trainer

TINY = NetworkConfig(blocks=1, filters=4, value_hidden=8)
CPU = torch.device("cpu")


def random_states(size: int, count: int, seed: int) -> list[State]:
    """`count` distinct live positions reached by random play."""
    rng = np.random.default_rng(seed)
    states: list[State] = []
    while len(states) < count:
        state = initial_state(size)
        for _ in range(int(rng.integers(0, size + 1))):
            if is_terminal(state):
                break
            moves = legal_moves(state)
            state = apply_move(state, moves[int(rng.integers(len(moves)))])
        if not is_terminal(state) and state not in states:
            states.append(state)
    return states


def make_examples(size: int, count: int, seed: int) -> Examples:
    """Real planes, a one-hot `π` on a random legal cell, and a random `z` in {-1, 0, 1}.

    One-hot targets make the policy cross-entropy's floor 0 (for a soft `π`
    the floor would be its entropy `H(π)`), so "memorised" has a clear meaning.
    """
    rng = np.random.default_rng(seed)
    states = random_states(size, count, seed)
    planes = np.stack([encode(s) for s in states]).astype(np.uint8)
    pi = np.zeros((count, size * size), dtype=np.float32)
    for i, state in enumerate(states):
        legal = np.flatnonzero(legal_mask(state))
        pi[i, int(rng.choice(legal))] = 1.0
    z = rng.choice([-1.0, 0.0, 1.0], size=count).astype(np.float32)
    return Examples(planes=planes, pi=pi, z=z)


def as_batch(examples: Examples) -> Batch:
    return Batch(planes=examples.planes.astype(np.float32), pi=examples.pi, z=examples.z)


def tiny_network(size: int = 4, seed: int = 0) -> Network:
    torch.manual_seed(seed)
    return Network(size, TINY)


def assert_nested_equal(a: Any, b: Any) -> None:
    """Deep equality for optimiser state dicts (tensors compared with `torch.equal`)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def is_finite(losses: StepLosses) -> bool:
    return all(math.isfinite(x) for x in (losses.total, losses.policy, losses.value))


# --- 1. A fixed batch is memorised -------------------------------------------


def test_fixed_batch_is_memorised() -> None:
    # The network's own "memorise one batch" test, now through the trainer:
    # what is under test is the optimiser wiring (zero_grad, backward, step on
    # the right parameters). lr 1e-2 and no weight decay because the defaults
    # are tuned for a real run, not for overfitting 8 examples in seconds.
    batch = as_batch(make_examples(4, 8, seed=1))
    trainer = Trainer(tiny_network(), TrainConfig(learning_rate=1e-2, weight_decay=0.0), CPU)

    history = [trainer.step(batch) for _ in range(200)]

    assert all(is_finite(losses) for losses in history)
    first, last = history[0], history[-1]
    assert last.total < 0.25 * first.total
    assert last.value < 0.05
    # The three numbers are one loss: total = policy + value.
    assert last.total == pytest.approx(last.policy + last.value, rel=1e-5)


# --- 2. Mode switching on the shared network ----------------------------------


def test_mode_switching_with_a_shared_evaluator() -> None:
    # One Network object serves both self-play (through the evaluator) and
    # training. BatchNorm normalises with batch statistics in train mode and
    # with running statistics in eval mode, so each user sets its own mode on
    # every call rather than trusting what the other left behind.
    network = tiny_network()
    trainer = Trainer(network, TrainConfig(), CPU)
    batch = as_batch(make_examples(4, 8, seed=2))

    trainer.step(batch)
    assert network.training

    evaluator = NetworkEvaluator(network, CPU)
    policies, values = evaluator.evaluate(random_states(4, 3, seed=3))
    assert policies.shape == (3, 16) and values.shape == (3,)
    assert np.isfinite(policies).all() and np.isfinite(values).all()
    assert not network.training

    assert is_finite(trainer.step(batch))
    assert network.training


# --- 3. Optimiser construction and resume -------------------------------------


def test_optimizer_is_adamw_from_the_config() -> None:
    config = TrainConfig(learning_rate=3e-3, weight_decay=2e-4)
    network = tiny_network()
    trainer = Trainer(network, config, CPU)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    for group in trainer.optimizer.param_groups:
        assert group["lr"] == config.learning_rate
        assert group["weight_decay"] == config.weight_decay
    # No parameter groups: every parameter, BatchNorm's included, is decayed.
    optimised = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert optimised == {id(p) for p in network.parameters()}


def test_optimizer_state_resumes_from_a_checkpoint(tmp_path) -> None:
    config = TrainConfig()
    network = tiny_network()
    trainer = Trainer(network, config, CPU)
    trainer.step(as_batch(make_examples(4, 8, seed=4)))

    path = tmp_path / "gen_001.pt"
    save_checkpoint(path, network, generation=1, optimizer=trainer.optimizer)
    checkpoint = load_checkpoint(path)

    resumed = Trainer(checkpoint.network, config, CPU, optimizer_state=checkpoint.optimizer_state)
    assert_nested_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())


# --- 4. A generation of steps -------------------------------------------------


def test_train_generation_runs_the_configured_steps_and_returns_the_mean() -> None:
    config = TrainConfig(batch_size=8, steps_per_generation=5)
    buffer = ReplayBuffer(size=4, generations=2)
    buffer.add(make_examples(4, 20, seed=5), generation=1)

    network = tiny_network()
    twin = copy.deepcopy(network)
    trainer = Trainer(network, config, CPU)
    mean = trainer.train_generation(buffer, np.random.default_rng(9))

    # Adam keeps a per-parameter step counter; it counts optimiser steps.
    parameter = next(network.parameters())
    assert int(trainer.optimizer.state[parameter]["step"]) == config.steps_per_generation

    # Replay the same rng by hand on an identical twin: sample -> augment -> step.
    replay_trainer = Trainer(twin, config, CPU)
    rng = np.random.default_rng(9)
    collected = [
        replay_trainer.step(augment(buffer.sample(config.batch_size, rng), rng))
        for _ in range(config.steps_per_generation)
    ]
    assert mean.total == pytest.approx(np.mean([s.total for s in collected]), rel=1e-6)
    assert mean.policy == pytest.approx(np.mean([s.policy for s in collected]), rel=1e-6)
    assert mean.value == pytest.approx(np.mean([s.value for s in collected]), rel=1e-6)


# --- 5. Augmentation is applied -----------------------------------------------


def test_augmentation_is_applied_to_every_sampled_batch(monkeypatch) -> None:
    config = TrainConfig(batch_size=8, steps_per_generation=3)
    buffer = ReplayBuffer(size=4, generations=2)
    buffer.add(make_examples(4, 20, seed=6), generation=1)

    # Record what the buffer hands out, to check augment gets exactly that.
    sampled: list[Batch] = []
    original_sample = buffer.sample

    def recording_sample(batch_size: int, rng: np.random.Generator) -> Batch:
        batch = original_sample(batch_size, rng)
        sampled.append(batch)
        return batch

    monkeypatch.setattr(buffer, "sample", recording_sample)

    calls: list[tuple[Batch, np.random.Generator]] = []

    def spy(batch: Batch, rng: np.random.Generator) -> Batch:
        calls.append((batch, rng))
        return augment(batch, rng)

    monkeypatch.setattr(trainer_module, "augment", spy)

    rng = np.random.default_rng(10)
    Trainer(tiny_network(), config, CPU).train_generation(buffer, rng)

    assert len(calls) == config.steps_per_generation
    assert len(sampled) == config.steps_per_generation
    for (batch, call_rng), drawn in zip(calls, sampled):
        assert batch is drawn
        assert call_rng is rng


@pytest.mark.parametrize(
    "bad",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"steps_per_generation": 0},
        {"learning_rate": 0.0},
        {"learning_rate": -1e-3},
        {"weight_decay": -1e-4},
        {"buffer_generations": 0},
    ],
)
def test_train_config_rejects_invalid_values(bad: dict) -> None:
    # Fail at construction, not later: zero steps would log NaN mean losses,
    # a non-positive batch size would fail deep inside torch.
    with pytest.raises(ValueError):
        TrainConfig(**bad)


def test_train_config_accepts_boundary_values() -> None:
    TrainConfig(batch_size=1, steps_per_generation=1, weight_decay=0.0, buffer_generations=1)
