"""Tests for the engine interface (`Analysis`, `analyze`, `load_engine`) and the
placeholder `DummyEngine`.

The dummy's numbers are meaningless by design. What is tested is the contract
the CLI relies on, which the real MCTS engine will have to honour too.
"""

import itertools
import math

import pytest

from ox_zero.engine import Analysis, DummyEngine, analyze, load_engine
from ox_zero.game import initial_state, legal_moves, play

POSITION = play([(5, 5), (6, 6), (5, 6)])


def fast_engine(seed: int = 0) -> DummyEngine:
    # No fake throughput limit, so tests don't sleep.
    return DummyEngine(seed=seed, rate=math.inf)


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


# --- DummyEngine: the Engine contract ----------------------------------------


def test_scores_every_legal_move_with_a_probability():
    analysis = analyze(fast_engine(), POSITION, simulations=200)
    assert list(analysis.scores) == legal_moves(POSITION)
    assert all(0.0 <= score <= 1.0 for score in analysis.scores.values())


def test_value_is_the_best_move_score():
    # The side to move is assumed to pick its best move, so the position is
    # worth what its best move is worth.
    analysis = analyze(fast_engine(), POSITION, simulations=200)
    assert analysis.value == max(analysis.scores.values())


def test_analyze_stops_exactly_at_the_budget():
    assert analyze(fast_engine(), POSITION, simulations=333).simulations == 333


def test_search_yields_snapshots_with_increasing_simulations():
    snapshots = list(itertools.islice(fast_engine().search(POSITION), 20))
    counts = [snapshot.simulations for snapshot in snapshots]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_uncapped_search_keeps_going():
    snapshots = list(itertools.islice(fast_engine().search(POSITION), 500))
    assert len(snapshots) == 500


def test_capped_search_ends_at_the_cap():
    snapshots = list(fast_engine().search(POSITION, max_simulations=1000))
    assert snapshots[-1].simulations == 1000


def test_same_seed_gives_same_analysis():
    a = analyze(fast_engine(seed=7), POSITION, simulations=500)
    b = analyze(fast_engine(seed=7), POSITION, simulations=500)
    assert a == b


def test_different_seeds_give_different_analyses():
    a = analyze(fast_engine(seed=1), POSITION, simulations=500)
    b = analyze(fast_engine(seed=2), POSITION, simulations=500)
    assert a.scores != b.scores


def test_scores_settle_as_the_search_deepens():
    # The live views rely on numbers that move a lot early and settle later,
    # like a real search whose estimates converge.
    snapshots = list(fast_engine().search(POSITION, max_simulations=20_000))

    def mean_change(a: Analysis, b: Analysis) -> float:
        return sum(abs(a.scores[m] - b.scores[m]) for m in a.scores) / len(a.scores)

    early = mean_change(snapshots[0], snapshots[1])
    late = mean_change(snapshots[-2], snapshots[-1])
    assert late < early / 3


def test_searching_a_finished_game_is_an_error():
    finished = play([(5, 5), (6, 6), (5, 7), (5, 6)])
    with pytest.raises(ValueError):
        next(fast_engine().search(finished))


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        analyze(fast_engine(), initial_state(), simulations=0)


def test_rate_limits_throughput():
    # 1000 simulations at 20k/s must take at least ~50 ms.
    import time

    start = time.perf_counter()
    analyze(DummyEngine(seed=0, rate=20_000), POSITION, simulations=1000)
    assert time.perf_counter() - start >= 0.04


# --- load_engine -------------------------------------------------------------


def test_without_a_model_the_dummy_is_used():
    assert isinstance(load_engine(None, seed=0), DummyEngine)


def test_missing_model_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_engine(tmp_path / "nope.pt", seed=0)


def test_existing_model_path_still_loads_the_dummy_for_now(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"")
    assert isinstance(load_engine(checkpoint, seed=0), DummyEngine)
