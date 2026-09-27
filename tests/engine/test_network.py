"""Tests for the policy/value network and the AlphaZero loss.

Design: docs/design/network.md. The network is `f_θ(s) = (p, v)`: a residual
convolutional tower with a policy head (one logit per cell) and a value head
(one scalar in `[-1, 1]` after tanh). The loss is

    L = (z - v)²  -  π^T log p      (+ c·||θ||², applied by AdamW, not here)

Everything runs on CPU with a tiny network unless the test is about defaults.
"""

import math

import numpy as np
import pytest
import torch

from ox_zero.engine.encoding import encode_batch, legal_mask
from ox_zero.engine.network import Network, NetworkConfig, alphazero_loss
from ox_zero.game import apply_move, initial_state, is_terminal, legal_moves

TINY = NetworkConfig(blocks=2, filters=8, value_hidden=16)


def _random_states(size, count, seed, max_moves=None):
    """Live positions reached by random play, all distinct."""
    rng = np.random.default_rng(seed)
    max_moves = max_moves if max_moves is not None else size
    states = []
    while len(states) < count:
        state = initial_state(size)
        for _ in range(int(rng.integers(0, max_moves + 1))):
            if is_terminal(state):
                break
            moves = legal_moves(state)
            state = apply_move(state, moves[int(rng.integers(len(moves)))])
        if not is_terminal(state) and state not in states:
            states.append(state)
    return states


# --- Shapes -----------------------------------------------------------------------


@pytest.mark.parametrize("size", [3, 6, 12])
def test_output_shapes(size):
    torch.manual_seed(0)
    net = Network(size, TINY).eval()
    planes = torch.from_numpy(encode_batch(_random_states(size, 5, seed=size)))
    with torch.inference_mode():
        logits, values = net(planes)
    assert logits.shape == (5, size * size) and logits.dtype == torch.float32
    assert values.shape == (5,) and values.dtype == torch.float32
    assert torch.all(values.abs() <= 1.0)


def test_default_config():
    # 4 blocks x 64 filters: the starting point for 6x6 (network.md). On 12x12
    # the stem plus 4 blocks is 9 stacked 3x3 convolutions, each growing the
    # receptive field by 2 cells: 1 + 2·9 = 19, so every output cell sees a
    # 19x19 window, more than the whole 12x12 board.
    config = NetworkConfig()
    assert (config.blocks, config.filters, config.value_hidden) == (4, 64, 256)
    net = Network(12)
    logits, values = net.eval()(torch.zeros(2, 3, 12, 12))
    assert logits.shape == (2, 144) and values.shape == (2,)


def test_wrong_board_size_is_rejected():
    net = Network(6, TINY)
    with pytest.raises(ValueError):
        net(torch.zeros(1, 3, 4, 4))


def test_config_is_recorded():
    # A checkpoint (plan 04) rebuilds the architecture from these two fields.
    net = Network(6, TINY)
    assert net.size == 6
    assert net.config == TINY


# --- BatchNorm modes --------------------------------------------------------------


def test_eval_mode_is_deterministic_and_batch_independent():
    # In eval mode BatchNorm normalises with the running statistics it
    # accumulated during training, fixed numbers, so a position's output does
    # not depend on what else is in the batch.
    torch.manual_seed(0)
    net = Network(6, TINY).eval()
    planes = torch.from_numpy(encode_batch(_random_states(6, 4, seed=1)))
    with torch.inference_mode():
        first = net(planes)
        second = net(planes)
        alone = net(planes[:1])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(alone[0][0], first[0][0])
    torch.testing.assert_close(alone[1][0], first[1][0])


def test_train_mode_normalises_with_batch_statistics():
    # In train mode BatchNorm normalises each channel with the mean and
    # variance of the *current batch*, so the same position gives different
    # outputs in different batches. That is right for training and wrong for
    # search, which is why the evaluator always switches to eval mode.
    # (A batch of one does not raise here: the statistics are taken over the
    # batch *and* the S x S cells, so there are still many values per channel.)
    torch.manual_seed(0)
    net = Network(6, TINY).train()
    planes = torch.from_numpy(encode_batch(_random_states(6, 4, seed=1)))
    with torch.no_grad():
        in_batch = net(planes)[1][0]
        alone = net(planes[:1])[1][0]
    assert not torch.allclose(in_batch, alone)


# --- Loss -------------------------------------------------------------------------


def test_loss_by_hand():
    # 2x2 board, one example. Cell 3 is illegal and has the *largest* logit:
    # if it leaked into the softmax the policy loss would be much larger.
    logits = torch.tensor([[2.0, 0.5, -1.0, 5.0]])
    legal = torch.tensor([[True, True, True, False]])
    pi = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    values = torch.tensor([0.5])
    z = torch.tensor([1.0])

    total, policy_loss, value_loss = alphazero_loss(logits, values, pi, z, legal)

    # Softmax over the three legal logits only: -log(e^2 / (e^2 + e^0.5 + e^-1)).
    expected_policy = -(2.0 - math.log(math.exp(2.0) + math.exp(0.5) + math.exp(-1.0)))
    assert value_loss.item() == pytest.approx(0.25)  # (1 - 0.5)²
    assert policy_loss.item() == pytest.approx(expected_policy, rel=1e-6)
    assert total.item() == pytest.approx(0.25 + expected_policy, rel=1e-6)
    assert total.shape == policy_loss.shape == value_loss.shape == ()


def test_loss_has_no_nan_from_masked_cells():
    # The masked log-probabilities are -inf and π is 0 there. A naive
    # (π · log p).sum() computes 0 · -inf = NaN, which would poison every
    # weight on the next step. The gradient must be finite too.
    logits = torch.randn(3, 9, requires_grad=True)
    legal = torch.ones(3, 9, dtype=torch.bool)
    legal[:, :4] = False
    pi = torch.zeros(3, 9)
    pi[:, 4:] = 0.2
    values = torch.zeros(3, requires_grad=True)

    total, policy_loss, _ = alphazero_loss(logits, values, pi, torch.zeros(3), legal)
    total.backward()

    assert torch.isfinite(total) and torch.isfinite(policy_loss)
    assert torch.all(torch.isfinite(logits.grad))
    assert torch.all(logits.grad[:, :4] == 0.0)  # illegal logits get no gradient


def test_network_memorises_one_batch():
    # The standard "can it overfit a single batch" check: if the network, the
    # loss and the optimiser are wired correctly, 16 fixed examples are easy
    # to memorise. One-hot targets make the cross-entropy's floor 0; with a
    # soft target π the floor would be the entropy of π, H(π) = -Σ π log π,
    # because cross-entropy = H(π) + KL(π || p) and only the KL part can
    # reach 0.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    states = _random_states(6, 16, seed=2)
    planes = torch.from_numpy(encode_batch(states))
    legal = torch.from_numpy(np.stack([legal_mask(s) for s in states]))
    pi = torch.zeros(16, 36)
    for i, state in enumerate(states):
        row, col = legal_moves(state)[int(rng.integers(len(legal_moves(state))))]
        pi[i, row * 6 + col] = 1.0
    z = torch.from_numpy(rng.choice([-1.0, 0.0, 1.0], size=16).astype(np.float32))

    net = Network(6, TINY).train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(300):
        logits, values = net(planes)
        total, policy_loss, value_loss = alphazero_loss(logits, values, pi, z, legal)
        optimiser.zero_grad()
        total.backward()
        optimiser.step()
        if policy_loss.item() < 0.05 and value_loss.item() < 0.01:
            break

    assert policy_loss.item() < 0.05
    assert value_loss.item() < 0.01
