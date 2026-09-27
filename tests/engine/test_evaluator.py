"""Tests for the torch-free evaluators: `UniformEvaluator` and `TableEvaluator`.

An evaluator answers the search's one question about a leaf: "what are the
move priors `p` and the value `v` here?" (docs/design/engineering.md, "The
evaluator seam"). These two answer it without a network, so the search can be
tested exactly and without torch.
"""

import subprocess
import sys

import numpy as np

from ox_zero.engine.evaluator import Evaluator, TableEvaluator, UniformEvaluator
from ox_zero.game import legal_moves, play


def test_uniform_evaluator_spreads_the_prior_over_legal_moves():
    # 3 marks vs 5 marks on 4x4: 13 and 11 legal moves.
    states = [
        play([(0, 0), (0, 3), (3, 0)], size=4),
        play([(0, 0), (0, 3), (3, 0), (1, 2), (3, 3)], size=4),
    ]
    policies, values = UniformEvaluator().evaluate(states)

    assert policies.shape == (2, 16) and policies.dtype == np.float32
    assert values.shape == (2,) and values.dtype == np.float32
    np.testing.assert_array_equal(values, 0.0)

    for policy, state in zip(policies, states):
        legal = {r * 4 + c for r, c in legal_moves(state)}
        for index in range(16):
            if index in legal:
                assert policy[index] == np.float32(1 / len(legal))
            else:
                assert policy[index] == 0.0
        assert abs(policy.sum() - 1.0) < 1e-6


def test_table_evaluator_returns_scripted_answers_and_falls_back():
    known = play([(0, 0)], size=3)
    unknown = play([(1, 1)], size=3)
    scripted = np.zeros(9, dtype=np.float32)
    scripted[4] = 0.75
    scripted[8] = 0.25
    evaluator = TableEvaluator({known: (scripted, -0.5)})

    policies, values = evaluator.evaluate([known, unknown, known])

    assert policies.shape == (3, 9) and policies.dtype == np.float32
    assert values.shape == (3,) and values.dtype == np.float32
    np.testing.assert_array_equal(policies[0], scripted)
    np.testing.assert_array_equal(policies[2], scripted)
    assert values[0] == np.float32(-0.5) and values[2] == np.float32(-0.5)

    # Fallback defaults to uniform: 8 legal cells, value 0.
    expected_uniform, _ = UniformEvaluator().evaluate([unknown])
    np.testing.assert_array_equal(policies[1], expected_uniform[0])
    assert values[1] == 0.0


def test_table_evaluator_uses_the_given_fallback():
    state = play([(0, 0)], size=3)
    inner = TableEvaluator({state: (np.full(9, 1 / 9, dtype=np.float32), 0.25)})
    outer = TableEvaluator({}, fallback=inner)
    _, values = outer.evaluate([state])
    assert values[0] == np.float32(0.25)


def test_simple_evaluators_satisfy_the_protocol():
    # Structural typing: anything with a matching `evaluate` is an Evaluator.
    assert isinstance(UniformEvaluator(), Evaluator)
    assert isinstance(TableEvaluator({}), Evaluator)


def test_evaluator_module_does_not_import_torch():
    # The search depends only on this module; it must stay torch-free so that
    # tree tests (and anything else that just searches) never pay for torch.
    # A fresh interpreter is needed because this test process may already
    # have torch loaded by other tests.
    code = "import sys, ox_zero.engine.evaluator; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
