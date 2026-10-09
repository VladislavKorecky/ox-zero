"""Tests for `PlaceholderEngine`, the stand-in engine the CLI runs on until a
real one exists. Its numbers are meaningless by design; what is tested is the
engine-port contract the CLI relies on.
"""

import itertools
import math

import pytest

from ox_zero.cli.engine_port import Analysis, analyze
from ox_zero.cli.placeholder import PlaceholderEngine
from ox_zero.game import legal_moves, play

POSITION = play([(5, 5), (6, 6), (5, 6)])


def fast_engine(seed: int = 0) -> PlaceholderEngine:
    # No fake throughput limit, so tests don't sleep.
    return PlaceholderEngine(seed=seed, rate=math.inf)


# --- The engine-port contract ----------------------------------------


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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("moves", [[], [(5, 5), (6, 6), (5, 6)], [(0, 0), (11, 11)]])
def test_chosen_is_the_top_scoring_move(seed, moves):
    # The placeholder has no visit counts, so its "choice" is its top score.
    for snapshot in fast_engine(seed).search(play(moves), max_simulations=300):
        assert snapshot.chosen == snapshot.top(1)[0][0]


def test_searching_a_finished_game_is_an_error():
    finished = play([(5, 5), (6, 6), (5, 7), (5, 6)])
    with pytest.raises(ValueError):
        next(fast_engine().search(finished))


def test_rate_limits_throughput():
    # 1000 simulations at 20k/s must take at least ~50 ms.
    import time

    start = time.perf_counter()
    analyze(PlaceholderEngine(seed=0, rate=20_000), POSITION, simulations=1000)
    assert time.perf_counter() - start >= 0.04
