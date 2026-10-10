"""The trainer: AdamW steps on the AlphaZero loss, one generation at a time.

Design: docs/design/network.md ("Loss", "Optimiser", "Symmetries") and
docs/design/training.md ("The generation loop").
Plan: docs/plans/04-training-pipeline.md, step 3.

What one generation of training is
----------------------------------
After self-play has added its games to the replay buffer, the trainer runs
`steps_per_generation` iterations of

    sample (uniform over the buffer) -> augment (a random symmetry per example)
    -> forward -> alphazero_loss -> backward -> AdamW step

and reports the mean losses over those steps. Each step is ordinary
mini-batch gradient descent on `l = (z - v)² - π^T log p`: the value head is
pulled towards the game results `z`, the policy head towards the search's
visit distributions `π` (the search is stronger than the raw network, so this
"distils" search into the network; that is the policy-improvement half of
AlphaZero).

Weight decay: AdamW instead of the paper's L2 term
--------------------------------------------------
The paper's loss carries `+ c·‖θ‖²`. With plain SGD, adding that term to the
loss is exactly the same as shrinking every weight by `lr·2c` each step
("weight decay"). With Adam it is not: Adam divides each parameter's gradient
by a running estimate of its magnitude, so an L2 gradient `2cθ` would be
rescaled per parameter and parameters with large gradients would barely be
decayed. AdamW (Loshchilov & Hutter, 2019) *decouples* the decay: the Adam
update uses only the loss gradient, and the weights are then shrunk by
`lr · weight_decay · θ` directly. That keeps the paper's intent (a uniform
pull of every weight towards 0, a regulariser against overfitting the
buffer) without being numerically identical to it. This is why
`alphazero_loss` has no regularisation term: adding one there as well would
count it twice (see its docstring).

Every parameter is decayed, BatchNorm's `γ`/`β` and the linear biases
included: the paper's `‖θ‖²` is over all of θ. Excluding norms and biases is
a common refinement, kept on the upgrades list (plan 04, "AdamW decays every
parameter").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ox_zero.engine.network import Network, alphazero_loss
from ox_zero.training.replay import Batch, ReplayBuffer, augment, legal_from_planes


@dataclass(frozen=True)
class TrainConfig:
    """How the network is trained each generation.

    Attributes:
        batch_size: Examples per mini-batch.
        steps_per_generation: Optimiser steps after each round of self-play.
        learning_rate: AdamW's step size.
        weight_decay: AdamW's decoupled decay, the role of the paper's `c`.
        buffer_generations: `K`, how many generations the replay buffer keeps.
            Read by `run.py` when it builds the buffer, not by the trainer;
            it lives here because it is a training-data setting.
    """

    batch_size: int = 256
    steps_per_generation: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    buffer_generations: int = 10


@dataclass(frozen=True)
class StepLosses:
    """The three losses of one step (or their means over a generation), as floats.

    `total = policy + value`. Batch means, not sums: a mean keeps the loss
    scale (and hence the gradient scale, and the meaning of a learning rate)
    independent of the batch size, so `batch_size` can change without
    retuning `learning_rate`, and the logged numbers are comparable across
    runs.
    """

    total: float
    policy: float
    value: float


class Trainer:
    """Owns the AdamW optimiser over a `Network` and takes training steps on it.

    The network object is shared with the `NetworkEvaluator` used for
    self-play (plan 04, "Device"), so both see the newest weights with no
    copying. They disagree about the mode, which is why each sets it on every
    call (see `step`).

    Args:
        network: The network to train; moved to `device`.
        config: Batch size, step count, learning rate and weight decay.
        device: Where the network and every batch live.
        optimizer_state: `optimizer.state_dict()` from a checkpoint, to resume
            Adam's moment estimates and step counts instead of starting cold.
    """

    optimizer: torch.optim.AdamW

    def __init__(
        self,
        network: Network,
        config: TrainConfig,
        device: torch.device,
        optimizer_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.network = network.to(device)
        self.config = config
        self.device = device
        # One parameter group over every parameter (see the module docstring).
        # The network must be on `device` before this line: the optimiser's
        # state tensors are created next to the parameters they belong to.
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if optimizer_state is not None:
            # `load_state_dict` moves the saved moments (written on CPU by
            # `save_checkpoint`) onto each parameter's device. It also restores
            # the saved `lr` and `weight_decay`; run.py refuses a resume with a
            # changed TrainConfig, so those equal the config's anyway.
            self.optimizer.load_state_dict(optimizer_state)

    def step(self, batch: Batch) -> StepLosses:
        """One AdamW step on `batch`; returns its losses."""
        # NumPy -> torch. `from_numpy` shares the array's memory (no copy) and
        # `.to(device)` copies it once to the training device.
        planes = torch.from_numpy(batch.planes).to(self.device)
        pi = torch.from_numpy(batch.pi).to(self.device)
        z = torch.from_numpy(batch.z).to(self.device)
        # The legal mask is derived from the planes rather than stored with
        # each example: a cell is legal exactly when it is empty (planes 0 and
        # 1 both zero), and recorded positions are never terminal. Deriving it
        # costs one comparison, saves storing S² booleans per example, and
        # cannot go out of sync with the planes after augmentation, because it
        # is computed from the already-transformed planes.
        legal = torch.from_numpy(legal_from_planes(batch.planes)).to(self.device)

        # Train mode on *every* step, not once: the evaluator sharing this
        # network sets eval mode on each of its calls. In train mode BatchNorm
        # normalises with the current batch's mean and variance (and updates
        # its running averages); in eval mode it uses those frozen running
        # averages. Training in eval mode would freeze the statistics and
        # mis-scale the gradients; evaluating in train mode would make a
        # position's value depend on its batch mates.
        self.network.train()
        logits, values = self.network(planes)
        total, policy_loss, value_loss = alphazero_loss(logits, values, pi, z, legal)

        # Standard PyTorch update. Gradients *accumulate* into `.grad` across
        # backward calls, so clear them first; `backward` fills `.grad` with
        # ∂total/∂θ by backprop; `step` applies the Adam update and then the
        # decoupled weight decay.
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        # `.item()` copies a scalar to the host (a device sync on a GPU);
        # three per step is negligible at this scale.
        return StepLosses(
            total=total.item(), policy=policy_loss.item(), value=value_loss.item()
        )

    def train_generation(self, buffer: ReplayBuffer, rng: np.random.Generator) -> StepLosses:
        """`steps_per_generation` steps of sample -> augment -> step; the mean losses.

        The run's one NumPy generator drives both the sampling and the
        symmetries, so a seeded run is reproducible.
        """
        history: list[StepLosses] = []
        for _ in range(self.config.steps_per_generation):
            batch = buffer.sample(self.config.batch_size, rng)
            # Called through the module-level name so tests can substitute it.
            history.append(self.step(augment(batch, rng)))
        return StepLosses(
            total=float(np.mean([s.total for s in history])),
            policy=float(np.mean([s.policy for s in history])),
            value=float(np.mean([s.value for s in history])),
        )
