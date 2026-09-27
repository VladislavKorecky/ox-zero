"""Tests for `NetworkEvaluator` (the network behind the evaluator seam) and `select_device`.

CPU only, tiny network, random weights: these tests check the plumbing
(encoding, masking, softmax, eval mode, dtypes), not move quality.
"""

import numpy as np
import pytest
import torch

from ox_zero.engine.encoding import encode_batch, legal_mask
from ox_zero.engine.evaluator import Evaluator
from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.network_evaluator import NetworkEvaluator, select_device
from ox_zero.engine.search import analyse
from ox_zero.game import legal_moves, play

TINY = NetworkConfig(blocks=2, filters=8, value_hidden=16)
CPU = torch.device("cpu")


def _states():
    # Different numbers of marks, so different legal cells per row.
    return [
        play([(0, 0), (1, 2)], size=4),
        play([(3, 3)], size=4),
        play([(0, 1), (2, 2), (3, 0), (1, 3), (0, 3)], size=4),
    ]


@pytest.fixture
def net():
    torch.manual_seed(0)
    return Network(4, TINY)


def test_protocol_shapes_and_masking(net):
    evaluator = NetworkEvaluator(net, CPU)
    assert isinstance(evaluator, Evaluator)
    states = _states()
    policies, values = evaluator.evaluate(states)

    assert policies.shape == (3, 16) and policies.dtype == np.float32
    assert values.shape == (3,) and values.dtype == np.float32
    assert np.all(np.abs(values) <= 1.0)
    for policy, state in zip(policies, states):
        mask = legal_mask(state)
        assert np.all(policy[~mask] == 0.0)
        assert np.all(policy[mask] > 0.0)
        assert abs(policy[mask].sum() - 1.0) < 1e-5


def test_agrees_with_the_raw_network(net):
    state = _states()[0]
    policies, values = NetworkEvaluator(net, CPU).evaluate([state])

    net.eval()
    with torch.inference_mode():
        logits, value = net(torch.from_numpy(encode_batch([state])))
    mask = torch.from_numpy(legal_mask(state))
    expected = torch.softmax(logits[0].masked_fill(~mask, float("-inf")), dim=-1)

    np.testing.assert_allclose(policies[0], expected.numpy(), rtol=1e-6, atol=1e-7)
    assert values[0] == pytest.approx(value[0].item(), abs=1e-7)


def test_evaluation_is_batch_independent(net):
    # Eval-mode BatchNorm uses fixed running statistics, so a position's
    # answer cannot depend on its batch mates. In train mode it would.
    net.train()  # the evaluator must switch to eval mode itself
    evaluator = NetworkEvaluator(net, CPU)
    a, b, _ = _states()
    pair_policies, pair_values = evaluator.evaluate([a, b])
    alone_policies, alone_values = evaluator.evaluate([a])
    np.testing.assert_allclose(pair_policies[0], alone_policies[0], rtol=1e-5, atol=1e-7)
    assert pair_values[0] == pytest.approx(alone_values[0], abs=1e-6)


def test_board_size_mismatch_is_rejected():
    evaluator = NetworkEvaluator(Network(6, TINY), CPU)
    with pytest.raises(ValueError):
        evaluator.evaluate([play([(0, 0)], size=4)])


# --- Device selection -------------------------------------------------------------


def test_select_device_honours_an_explicit_choice():
    assert select_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize(
    "cuda, mps, expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_select_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device().type == expected
    assert select_device("auto").type == expected


def test_select_device_without_patching_returns_a_known_type():
    assert select_device().type in {"cuda", "mps", "cpu"}


# --- End to end -------------------------------------------------------------------


def test_analysis_runs_on_a_network(net):
    state = _states()[0]
    snapshots = list(analyse(state, NetworkEvaluator(net, CPU), max_simulations=50))
    last = snapshots[-1]
    assert last.simulations == 50
    assert list(last.visits) == legal_moves(state)
    assert sum(last.visits.values()) == 50 + len(legal_moves(state))
    assert -1.0 <= last.value <= 1.0
    assert last.best in last.visits
