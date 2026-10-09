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
        chosen=(0, 1),
    )
    assert analysis.top(3) == [((0, 1), 0.9), ((1, 0), 0.5), ((1, 1), 0.5)]


def test_top_with_n_larger_than_move_count_returns_all():
    analysis = Analysis(value=0.5, scores={(0, 0): 0.5}, simulations=1, chosen=(0, 0))
    assert analysis.top(10) == [((0, 0), 0.5)]


def test_best_is_the_chosen_move_not_the_top_score():
    # The real engine plays its most visited move, which need not be the move
    # with the highest score (Q): a move visited a handful of times can carry
    # a lucky high average. `best` must follow the engine's choice, while
    # `top` keeps ranking the displayed numbers.
    analysis = Analysis(
        value=0.9,
        scores={(0, 0): 0.2, (0, 1): 0.9, (1, 0): 0.6},
        simulations=10,
        chosen=(1, 0),
    )
    assert analysis.best == ((1, 0), 0.6)
    assert analysis.top(1) == [((0, 1), 0.9)]


def test_chosen_is_required():
    # No default: an adapter that forgot to set it must fail loudly instead
    # of silently falling back to the top score.
    with pytest.raises(TypeError):
        Analysis(value=0.5, scores={(0, 0): 0.5}, simulations=1)  # type: ignore[call-arg]



def test_chosen_must_be_a_scored_move():
    # Caught where the Analysis is built, not later as a KeyError in `best`.
    with pytest.raises(ValueError, match="9, 9"):
        Analysis(value=0.5, scores={(0, 0): 0.5}, simulations=1, chosen=(9, 9))


# --- analyze -----------------------------------------------------------------


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        analyze(PlaceholderEngine(seed=0, rate=math.inf), initial_state(), simulations=0)
