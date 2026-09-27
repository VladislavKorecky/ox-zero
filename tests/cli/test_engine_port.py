"""Tests for the CLI's engine port: the `Analysis` value the CLI displays and
the `analyze` helper that runs a fixed-budget search through any engine.
"""

import math

import pytest

from ox_zero.cli.engine_port import Analysis, analyze
from ox_zero.cli.placeholder import PlaceholderEngine
from ox_zero.game import initial_state


# --- Analysis ----------------------------------------------------------------


def test_top_ranks_by_score_then_board_order():
    analysis = Analysis(
        value=0.9,
        scores={(0, 0): 0.2, (0, 1): 0.9, (1, 0): 0.5, (1, 1): 0.5},
        simulations=10,
    )
    assert analysis.top(3) == [((0, 1), 0.9), ((1, 0), 0.5), ((1, 1), 0.5)]


def test_top_with_n_larger_than_move_count_returns_all():
    analysis = Analysis(value=0.5, scores={(0, 0): 0.5}, simulations=1)
    assert analysis.top(10) == [((0, 0), 0.5)]


def test_best_is_the_top_move():
    analysis = Analysis(value=0.9, scores={(0, 0): 0.2, (0, 1): 0.9}, simulations=10)
    assert analysis.best == ((0, 1), 0.9)



# --- analyze -----------------------------------------------------------------


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        analyze(PlaceholderEngine(seed=0, rate=math.inf), initial_state(), simulations=0)
