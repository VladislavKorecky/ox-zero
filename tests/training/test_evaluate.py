"""Tests for checkpoint tournaments and the Elo fit (plan 04, step 4).

Search-only: every evaluator here is a plain Python object (uniform, a
scripted prior, or an oracle built from the exact solver), so this file never
imports torch, like `evaluate.py` itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

import ox_zero.training.evaluate as evaluate_module
from ox_zero.engine.evaluator import Evaluator, UniformEvaluator
from ox_zero.game import Solver
from ox_zero.game.rules import Mark, State, initial_state, legal_moves
from ox_zero.training.evaluate import (
    EVALUATION,
    EvalConfig,
    MatchGame,
    MatchResult,
    elo_difference,
    elo_ratings,
    opponents_for,
    play_match,
)
from ox_zero.training.selfplay import lockstep_step

# --- Helpers ---------------------------------------------------------------------


class SpyEvaluator:
    """Wraps an evaluator and records every batch of states it is asked for."""

    def __init__(self, inner: Evaluator | None = None) -> None:
        self.inner = inner if inner is not None else UniformEvaluator()
        self.batches: list[list[State]] = []

    def evaluate(self, states: Sequence[State]):
        self.batches.append(list(states))
        return self.inner.evaluate(states)


class FirstCellEvaluator:
    """All prior on the first legal cell in board order, value 0, for every state.

    Its priors are unmistakable (one child at 1, the rest at 0), which is what
    the leak test needs to tell its tree apart from a uniform one.
    """

    def evaluate(self, states: Sequence[State]):
        size = states[0].size
        policies = np.zeros((len(states), size * size), dtype=np.float32)
        for row, state in enumerate(states):
            r, c = legal_moves(state)[0]
            policies[row, r * size + c] = 1.0
        return policies, np.zeros(len(states), dtype=np.float32)


class OracleEvaluator:
    """A perfect evaluator from the exact solver.

    Uniform prior over the solver's `best_moves`, value from `value`. Not a
    `TableEvaluator`: the table would have to list every 4x4 position up front,
    whereas this answers any position on demand from the solver's cache.
    """

    def __init__(self, solver: Solver) -> None:
        self.solver = solver

    def evaluate(self, states: Sequence[State]):
        size = states[0].size
        policies = np.zeros((len(states), size * size), dtype=np.float32)
        values = np.zeros(len(states), dtype=np.float32)
        for row, state in enumerate(states):
            best = self.solver.best_moves(state)
            for r, c in best:
                policies[row, r * size + c] = 1.0 / len(best)
            values[row] = self.solver.value(state)
        return policies, values


@pytest.fixture
def recorded_games(monkeypatch: pytest.MonkeyPatch) -> list[MatchGame]:
    """Every `MatchGame` that `play_match` builds, in construction order.

    `MatchResult` only carries totals; the tests that need per-game facts
    (which opening, which colours, which trees) swap in a subclass that
    remembers its instances and its starting position.
    """
    games: list[MatchGame] = []

    class RecordingMatchGame(MatchGame):
        def __init__(self, state: State, *args, **kwargs) -> None:
            super().__init__(state, *args, **kwargs)
            self.start = state
            games.append(self)

    monkeypatch.setattr(evaluate_module, "MatchGame", RecordingMatchGame)
    return games


def first_move(state: State) -> tuple[int, int]:
    """The single occupied cell of a position after one move."""
    (index,) = [i for i, mark in enumerate(state.board) if mark is not None]
    return divmod(index, state.size)


# --- 1. elo_difference -------------------------------------------------------------


def test_elo_difference_known_values() -> None:
    # The logistic Elo curve: a 75% score is 191 Elo, by symmetry 25% is -191.
    assert elo_difference(0.5) == pytest.approx(0.0, abs=1e-9)
    assert elo_difference(0.75) == pytest.approx(191, abs=1)
    assert elo_difference(0.25) == pytest.approx(-191, abs=1)


def test_elo_difference_is_infinite_at_the_extremes() -> None:
    # A perfect score has no finite Elo difference. It must come back as
    # math.inf rather than crash with ZeroDivisionError or log10(0).
    assert elo_difference(1.0) == math.inf
    assert elo_difference(0.0) == -math.inf


# --- 2. elo_ratings on one pairing -------------------------------------------------


def test_elo_ratings_single_pairing_includes_the_virtual_draw() -> None:
    ratings = elo_ratings([(1, 0, 30, 0, 10)])
    # The anchor is exactly 0, by definition.
    assert ratings[0] == 0.0
    # One virtual draw: 30 + 0.5 points out of 41 games, not 30 out of 40.
    assert ratings[1] == pytest.approx(elo_difference(30.5 / 41), abs=2)


# --- 3. Chains, sweeps, the anchor and the ladder ----------------------------------


def test_elo_ratings_chain() -> None:
    ratings = elo_ratings([(1, 0, 30, 0, 10), (2, 1, 30, 0, 10)])
    assert ratings[0] == 0.0
    # 75% with the virtual draw is 30.5/41, about 185 Elo per link. Without the
    # virtual draw it would be elo_difference(0.75) = 191 per link.
    assert ratings[1] == pytest.approx(185, abs=2)
    assert ratings[2] == pytest.approx(370, abs=2)
    assert elo_difference(0.75) == pytest.approx(191, abs=1)


def test_elo_ratings_sweep_is_finite() -> None:
    # 40-0 alone would be +inf; the virtual draw (40.5/41) keeps it finite.
    ratings = elo_ratings([(1, 0, 40, 0, 0)])
    assert math.isfinite(ratings[1])
    assert 0 < ratings[1] < 1000


def test_elo_ratings_players_are_generations_and_anchor_must_exist() -> None:
    ratings = elo_ratings([(5, 3, 20, 0, 20)], anchor=3)
    assert set(ratings) == {3, 5}
    assert ratings[3] == 0.0
    with pytest.raises(ValueError):
        elo_ratings([(1, 0, 20, 0, 20)], anchor=7)


def test_elo_ratings_rejects_players_disconnected_from_the_anchor() -> None:
    # 6 and 5 only ever played each other: their difference is measured, but
    # nothing ties either to the anchor, so their absolute ratings would be
    # arbitrary. That must be an error, naming them, not a silent number.
    with pytest.raises(ValueError, match=r"5.*6"):
        elo_ratings([(1, 0, 20, 0, 20), (6, 5, 20, 0, 20)])


def test_ladder_match_corrects_a_drifting_chain() -> None:
    # Why the ladder exists: with only neighbour matches, a late generation's
    # rating is the sum of many noisy links. Here every link is 55% (22 of 40,
    # 22.5/41 with the virtual draw, about 34 Elo), so the chain alone puts
    # player 8 at about 8 * 34 = 272.
    chain = [(g, g - 1, 22, 0, 18) for g in range(1, 9)]
    assert elo_ratings(chain)[8] == pytest.approx(272, abs=2)
    # One direct long-range measurement, 8 over 0 by 30-0-10 (185 on its own),
    # pulls player 8 to a compromise between the two estimates.
    ratings = elo_ratings([*chain, (8, 0, 30, 0, 10)])
    assert ratings[8] == pytest.approx(198, abs=2)


# --- 4. Opponent selection ---------------------------------------------------------


def test_opponents_for() -> None:
    # Nearest `opponents` previous generations, plus the ladder opponent
    # g - ladder: one long-range match per generation that pins the chain of
    # neighbour matches to the anchor and stops the ratings drifting.
    config = EvalConfig()
    assert opponents_for(2, config) == [1, 0]
    assert opponents_for(5, config) == [4, 3, 2]  # 5 - 8 < 0: no ladder yet
    assert opponents_for(10, config) == [9, 8, 7, 2]
    # The ladder is already a nearest opponent: no duplicate.
    assert opponents_for(3, EvalConfig(ladder=2)) == [2, 1, 0]
    # Ladder off: None or 0 (0 would make a generation its own opponent).
    assert opponents_for(5, EvalConfig(ladder=None)) == [4, 3, 2]
    assert opponents_for(5, EvalConfig(ladder=0)) == [4, 3, 2]
    assert opponents_for(10, EvalConfig(ladder=None)) == [9, 8, 7]


# --- 5. Paired openings ------------------------------------------------------------


def test_paired_openings(recorded_games: list[MatchGame]) -> None:
    rng = np.random.default_rng(0)
    result = play_match(
        UniformEvaluator(), UniformEvaluator(), size=4, games_per_colour=3, simulations=4, rng=rng
    )
    assert isinstance(result, MatchResult)
    assert result.games == 6
    assert result.wins + result.draws + result.losses == 6
    assert len(result.openings) == 3
    assert len(set(result.openings)) == 3

    assert len(recorded_games) == 6
    for i in range(3):
        first, second = recorded_games[i], recorded_games[i + 3]
        # Both games of a pair start from the same opening move...
        assert first_move(first.start) == result.openings[i]
        assert first_move(second.start) == result.openings[i]
        # ...with the colours swapped.
        assert first.a_plays is not second.a_plays


def test_openings_are_cycled_when_the_board_runs_out() -> None:
    rng = np.random.default_rng(1)
    result = play_match(
        UniformEvaluator(), UniformEvaluator(), size=3, games_per_colour=10, simulations=2, rng=rng
    )
    assert result.games == 20
    assert len(result.openings) == 10
    # 3x3 has only 9 cells: all 9 are used once, then the draw is cycled.
    assert len(set(result.openings[:9])) == 9
    assert result.openings[9] == result.openings[0]


# --- 6. Strength shows -------------------------------------------------------------


def test_oracle_wins_every_game_as_o(solver: Solver, recorded_games: list[MatchGame]) -> None:
    oracle = OracleEvaluator(solver)
    rng = np.random.default_rng(2)
    result = play_match(oracle, UniformEvaluator(), size=4, games_per_colour=4, simulations=16, rng=rng)
    assert result.games == 8

    # 4x4 is an O win with perfect play and every first move loses, so the
    # random opening (X's move) cannot save X: as O the oracle must win every
    # game. The X games are not asserted: there the oracle starts from a lost
    # position and its result depends on whether the uniform player blunders.
    as_o = [game for game in recorded_games if game.a_plays is Mark.O]
    assert len(as_o) == 4
    for game in as_o:
        assert game.done
        assert game.final.winner is Mark.O
    assert result.wins >= 4
    # The colour split, from the oracle's (a's) perspective: every O game won.
    assert result.as_o == (4, 0, 0)


def test_colour_split_sums_to_the_totals() -> None:
    result = play_match(
        UniformEvaluator(), FirstCellEvaluator(), size=3, games_per_colour=3,
        simulations=4, rng=np.random.default_rng(5),
    )
    assert sum(result.as_x) == sum(result.as_o) == 3
    totals = tuple(x + o for x, o in zip(result.as_x, result.as_o))
    assert totals == (result.wins, result.draws, result.losses)


def test_match_result_rejects_inconsistent_totals() -> None:
    MatchResult(wins=1, draws=1, losses=0, openings=((0, 0),), as_x=(1, 0, 0), as_o=(0, 1, 0))
    with pytest.raises(ValueError):
        MatchResult(wins=2, draws=0, losses=0, openings=((0, 0),), as_x=(1, 0, 0), as_o=(0, 1, 0))


# --- 7. Two evaluators, two trees, two batches -------------------------------------


def test_each_evaluator_only_sees_its_own_games(
    monkeypatch: pytest.MonkeyPatch, recorded_games: list[MatchGame]
) -> None:
    a, b = SpyEvaluator(), SpyEvaluator()
    calls: list[tuple[list, object]] = []
    real_step = evaluate_module.lockstep_step

    def checking_step(trees, evaluator):
        # Every tree stepped with an evaluator must be the mover's tree of a
        # game where that evaluator is to move.
        for tree in trees:
            owners = [g for g in recorded_games if any(t is tree for t in g.trees.values())]
            assert len(owners) == 1
            game = owners[0]
            assert game.tree is tree
            assert game.mover is evaluator
        a_before, b_before = len(a.batches), len(b.batches)
        steps = real_step(trees, evaluator)
        # Only the evaluator being stepped was called.
        if evaluator is a:
            assert len(b.batches) == b_before
        else:
            assert len(a.batches) == a_before
        calls.append((list(trees), evaluator))
        return steps

    monkeypatch.setattr(evaluate_module, "lockstep_step", checking_step)
    play_match(a, b, size=4, games_per_colour=2, simulations=4, rng=np.random.default_rng(3))
    assert any(e is a for _, e in calls)
    assert any(e is b for _, e in calls)
    assert a.batches and b.batches


def test_no_opinions_leak_between_trees() -> None:
    a, b = FirstCellEvaluator(), UniformEvaluator()
    game = MatchGame(
        initial_state(3), a, b, a_plays=Mark.O, simulations=1, search=EVALUATION,
        rng=np.random.default_rng(4),
    )

    def search_once() -> None:
        # Root setup (not a simulation), then one simulation.
        mover = game.mover
        while game.tree.simulations < 1:
            lockstep_step([game.tree], mover)

    # (1) b plays X and moves first: set up, search once, move.
    assert game.mover is b
    assert game.trees[Mark.O] is None  # a's tree does not exist yet
    search_once()
    game.play_if_ready()
    assert not game.done

    # (2) a's tree is created from the new position; a sets up, searches once.
    assert game.mover is a
    search_once()
    # Check now, before a moves.
    a_root = game.trees[Mark.O].root
    a_priors = sorted(child.prior for child in a_root.children.values())
    assert a_priors[-1] == pytest.approx(1.0)
    assert all(p == 0 for p in a_priors[:-1])
    # b's root is the child b chose, expanded by b's own simulation: uniform.
    # A shared tree would have given a a root expanded with b's uniform priors.
    b_root = game.trees[Mark.X].root
    assert b_root.expanded
    b_priors = [child.prior for child in b_root.children.values()]
    assert b_priors == pytest.approx([1 / len(b_priors)] * len(b_priors))
    assert game.trees[Mark.X] is not game.trees[Mark.O]

    # (3) a moves, both trees play it; b sets up its new root, searches once.
    game.play_if_ready()
    assert not game.done
    assert game.mover is b
    search_once()
    b_root = game.trees[Mark.X].root
    b_priors = [child.prior for child in b_root.children.values()]
    assert b_priors == pytest.approx([1 / len(b_priors)] * len(b_priors))


# --- 8. Determinism ----------------------------------------------------------------


def test_same_seed_same_result() -> None:
    def run(seed: int) -> MatchResult:
        return play_match(
            UniformEvaluator(), UniformEvaluator(), size=4, games_per_colour=3, simulations=8,
            rng=np.random.default_rng(seed),
        )

    assert run(5) == run(5)


# --- 9. Self-match -----------------------------------------------------------------


def test_self_match_uses_two_trees_per_game(recorded_games: list[MatchGame]) -> None:
    e = UniformEvaluator()
    result = play_match(e, e, size=4, games_per_colour=2, simulations=4, rng=np.random.default_rng(6))
    assert result.games == 4
    for game in recorded_games:
        assert game.done
        # Trees are keyed by side, never by evaluator identity, so the same
        # object on both sides still gets one tree per side.
        x_tree, o_tree = game.trees[Mark.X], game.trees[Mark.O]
        if x_tree is not None and o_tree is not None:
            assert x_tree is not o_tree
    assert any(
        g.trees[Mark.X] is not None and g.trees[Mark.O] is not None for g in recorded_games
    )
