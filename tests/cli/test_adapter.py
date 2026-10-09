"""Tests for `SearchEngine`, the adapter that connects the AlphaZero search
(`ox_zero.engine.search.analyse`) to the CLI's engine port.

Everything here runs on `UniformEvaluator`: no torch, no model. That is also
the CLI's no-model mode, so the tactics tests double as evidence that the
CLI is a real engine before any training.
"""

import itertools

import pytest

from ox_zero.cli.adapter import SearchEngine
from ox_zero.cli.engine_port import analyze
from ox_zero.engine.evaluator import UniformEvaluator
from ox_zero.engine.search import ANALYSIS, analyse
from ox_zero.game import from_board_string, legal_moves, play

# X at (0,0), O at (0,1): X to move completes X O X at (0,2).
WIN_IN_ONE = play([(0, 0), (0, 1)], size=3)


def uniform_engine(**kwargs) -> SearchEngine:
    return SearchEngine(UniformEvaluator(), **kwargs)


# --- The score mapping ---------------------------------------------------------


def test_first_snapshot_maps_the_engine_snapshot():
    # The search speaks values in [-1, 1] (loss .. win, side to move); the
    # CLI shows probabilities in [0, 1]. The mapping is the affine
    # (x + 1) / 2, applied to the root value and to every move's Q.
    snapshot = next(analyse(WIN_IN_ONE, UniformEvaluator(), ANALYSIS))
    analysis = next(uniform_engine().search(WIN_IN_ONE))

    assert analysis.simulations == 0
    assert analysis.value == pytest.approx((snapshot.value + 1) / 2)
    assert list(analysis.scores) == legal_moves(WIN_IN_ONE)
    for move in legal_moves(WIN_IN_ONE):
        assert analysis.scores[move] == pytest.approx((snapshot.q[move] + 1) / 2)
    assert analysis.chosen == snapshot.best
    # Root expansion evaluates every child up front; the winning child is a
    # finished game with an exact value, so its Q is already 1 here.
    assert analysis.scores[(0, 2)] == 1.0


# --- The port contract ---------------------------------------------------------


def test_capped_search_counts_up_to_the_cap_and_ends():
    snapshots = list(uniform_engine().search(WIN_IN_ONE, max_simulations=5))
    assert [s.simulations for s in snapshots] == [0, 1, 2, 3, 4, 5]
    assert analyze(uniform_engine(), WIN_IN_ONE, 5).simulations == 5


def test_uncapped_search_keeps_going_and_can_be_dropped():
    position = play([(1, 1)], size=4)
    search = uniform_engine().search(position)
    snapshots = list(itertools.islice(search, 50))
    assert len(snapshots) == 50
    # Cancelling is just dropping the iterator (that is what the sandbox does).
    del search


def test_scores_cover_exactly_the_legal_moves_in_board_order():
    position = play([(1, 1), (0, 0)], size=4)
    for analysis in uniform_engine().search(position, max_simulations=20):
        assert list(analysis.scores) == legal_moves(position)


def test_searching_a_finished_game_is_an_error_without_iterating():
    finished = play([(0, 0), (0, 1), (0, 2)], size=3)
    with pytest.raises(ValueError):
        uniform_engine().search(finished)


# --- Tactics with no model -----------------------------------------------------


def test_finds_a_win_in_one():
    analysis = analyze(uniform_engine(), WIN_IN_ONE, 100)
    assert analysis.best[0] == (0, 2)
    assert analysis.value > 0.9


def test_avoids_a_loss_in_one():
    # O to move; every move except (1,3) hands X an immediate alternating line.
    position = from_board_string("____XXO_________", size=4)
    assert analyze(uniform_engine(), position, 200).best[0] == (1, 3)


# --- Board size ----------------------------------------------------------------


def test_size_mismatch_is_an_error_naming_both_sizes():
    with pytest.raises(ValueError, match=r"6x6.*4x4"):
        uniform_engine(size=6).search(play([], size=4))


def test_without_a_size_any_board_is_accepted():
    for size in (3, 4, 6):
        next(uniform_engine(size=None).search(play([], size=size)))


# --- Determinism ---------------------------------------------------------------


def test_the_seed_does_not_change_the_analysis():
    # Analysis mode has no randomness at all: no Dirichlet noise at the root,
    # the move is the most visited (not sampled), and ties break in board
    # order. So the seed has nothing to act on, and two seeds must agree.
    # This is intended: don't "fix" the seed into doing something here.
    position = play([(1, 1), (2, 2)], size=4)
    a = list(uniform_engine(seed=1).search(position, max_simulations=30))
    b = list(uniform_engine(seed=2).search(position, max_simulations=30))
    assert a == b
