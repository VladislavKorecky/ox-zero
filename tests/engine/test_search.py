"""Tests for the search driver: `SearchConfig`, `SearchTree`, `analyse`.

`SearchTree` is the stepped, two-phase search that self-play drives
(`select` a leaf, evaluate it elsewhere, `expand_and_backup`), and `analyse`
is the snapshot generator the CLI will consume. See docs/design/search.md
("Modes") and docs/plans/02-engine-search-network.md ("Interfaces").
"""

import itertools
import random
import subprocess
import sys

import numpy as np
import pytest

from ox_zero.engine.evaluator import TableEvaluator, UniformEvaluator
from ox_zero.engine.mcts import most_visited
from ox_zero.engine.search import (
    ANALYSIS,
    SELF_PLAY,
    SearchConfig,
    SearchTree,
    Snapshot,
    analyse,
)
from ox_zero.game import (
    apply_move,
    from_board_string,
    initial_state,
    is_terminal,
    legal_moves,
    play,
    to_board_string,
)

WIN_IN_ONE = [(0, 0), (0, 1)]  # 3x3, X to move: (0,2) completes X O X


class CountingEvaluator:
    """Wraps an evaluator and counts how many states it was asked about."""

    def __init__(self, inner=None):
        self.inner = inner or UniformEvaluator()
        self.states = 0

    def evaluate(self, states):
        self.states += len(states)
        return self.inner.evaluate(states)


def _evaluate_one(evaluator, state):
    policies, values = evaluator.evaluate([state])
    return policies[0], float(values[0])


# --- Config -----------------------------------------------------------------------


def test_config_resolves_alpha_and_cutoff_per_size():
    config = SearchConfig()
    assert config.alpha(6) == 11 / 36
    assert config.alpha(12) == 11 / 144
    assert (config.cutoff(4), config.cutoff(6), config.cutoff(8), config.cutoff(12)) == (2, 3, 4, 7)
    assert config.cutoff(5) == 2  # provisional fallback: round(5 / 2)
    assert config.cutoff(3) == 2


def test_explicit_config_values_override():
    config = SearchConfig(dirichlet_alpha=0.5, temperature_cutoff=9)
    assert config.alpha(6) == 0.5
    assert config.cutoff(12) == 9


def test_mode_presets():
    assert ANALYSIS.expand_root and not ANALYSIS.root_noise
    assert SELF_PLAY.root_noise and not SELF_PLAY.expand_root
    assert (SearchConfig().c_base, SearchConfig().c_init) == (19652.0, 1.25)
    assert SearchConfig().dirichlet_epsilon == 0.25


# --- Two-phase protocol -----------------------------------------------------------


def test_first_select_returns_the_root_for_setup():
    state = play([(0, 0)], size=3)
    tree = SearchTree(state, SearchConfig())
    assert tree.select() == state

    policy, value = _evaluate_one(UniformEvaluator(), state)
    tree.expand_and_backup(policy, value)

    # Expanding the root is setup, not a simulation: nothing counted, no visits.
    assert tree.simulations == 0
    assert tree.root.expanded
    assert tree.root.visit_count == 0
    assert all(child.visit_count == 0 for child in tree.root.children.values())

    leaf = tree.select()
    assert leaf in {apply_move(state, move) for move in legal_moves(state)}


def test_select_twice_without_backup_raises():
    tree = SearchTree(play([(0, 0)], size=3), SearchConfig())
    tree.select()
    with pytest.raises(RuntimeError):
        tree.select()


def test_expand_and_backup_without_a_pending_leaf_raises():
    tree = SearchTree(play([(0, 0)], size=3), SearchConfig())
    with pytest.raises(RuntimeError):
        tree.expand_and_backup(np.full(9, 1 / 9, dtype=np.float32), 0.0)


def test_a_finished_game_cannot_be_searched():
    with pytest.raises(ValueError):
        SearchTree(play([(0, 0), (0, 1), (0, 2)], size=3), SearchConfig())


def test_simulation_counts_after_each_backup():
    tree = SearchTree(play([(0, 0)], size=3), SearchConfig())
    tree.simulate(UniformEvaluator(), 7)
    assert tree.simulations == 7
    # Root N(s) = Σ N(root, a) exactly: setup added no visit.
    assert tree.root.visit_count == 7
    assert sum(c.visit_count for c in tree.root.children.values()) == 7


# --- Terminal leaves --------------------------------------------------------------


def test_terminal_leaves_are_scored_without_the_evaluator():
    state = play(WIN_IN_ONE, size=3)
    evaluator = CountingEvaluator()
    tree = SearchTree(state, SearchConfig())
    tree.simulate(evaluator, 30)

    assert tree.simulations == 30
    winning = tree.root.children[(0, 2)]
    # Every visit to the winning child hit a terminal position and was backed
    # up without asking the evaluator. The evaluator saw the root once (setup)
    # plus one state per non-terminal simulation.
    assert evaluator.states == 1 + (30 - winning.visit_count)
    assert winning.visit_count > 0
    # Each of those visits backed up -terminal_value = -(-1) = +1 for X.
    assert winning.q == 1.0


def test_select_returns_none_on_a_terminal_leaf():
    state = play(WIN_IN_ONE, size=3)
    tree = SearchTree(state, SearchConfig())
    evaluator = UniformEvaluator()
    nones = 0
    for _ in range(40):
        leaf = tree.select()
        if leaf is None:
            nones += 1
            continue
        tree.expand_and_backup(*_evaluate_one(evaluator, leaf))
    # The winning child is terminal and is never expanded, so each of its
    # visits is exactly one `None` from select().
    assert nones == tree.root.children[(0, 2)].visit_count
    assert not tree.root.children[(0, 2)].expanded


# --- Root noise -------------------------------------------------------------------


def _uniform_priors(state):
    return 1 / len(legal_moves(state))


@pytest.mark.parametrize("config", [SELF_PLAY, ANALYSIS], ids=["self-play", "analysis"])
def test_root_noise_only_at_the_root_and_only_when_configured(config):
    state = play([(0, 0), (1, 2)], size=4)
    tree = SearchTree(state, config, rng=np.random.default_rng(3))
    tree.simulate(UniformEvaluator(), 30)

    root_priors = [child.prior for child in tree.root.children.values()]
    uniform = _uniform_priors(state)
    if config.root_noise:
        assert all(p != pytest.approx(uniform, abs=1e-9) for p in root_priors)
    else:
        assert root_priors == pytest.approx([uniform] * len(root_priors))

    expanded = [c for c in tree.root.children.values() if c.expanded]
    assert expanded, "30 simulations must expand some root child"
    for child in expanded:
        expected = _uniform_priors(child.state)
        assert [g.prior for g in child.children.values()] == pytest.approx(
            [expected] * len(child.children)
        )


# --- Move choice and policy target ------------------------------------------------


def _self_play_tree(seed=0, simulations=50):
    state = play([(0, 0), (1, 2)], size=4)
    tree = SearchTree(state, SELF_PLAY, rng=np.random.default_rng(seed))
    tree.simulate(UniformEvaluator(), simulations)
    return tree


def test_policy_target_is_a_distribution_over_legal_moves():
    tree = _self_play_tree()
    pi = tree.policy_target()
    assert pi.shape == (16,) and pi.dtype == np.float32
    assert pi.sum() == pytest.approx(1.0, abs=1e-6)
    assert pi[0] == 0.0 and pi[1 * 4 + 2] == 0.0  # the two occupied cells


def test_choose_move_samples_before_the_cutoff_and_is_greedy_from_it():
    tree = _self_play_tree()
    cutoff = SELF_PLAY.cutoff(4)
    for _ in range(20):
        move = tree.choose_move(move_index=0)
        assert tree.root.children[move].visit_count > 0
    assert tree.choose_move(move_index=cutoff) == most_visited(tree.root)
    assert tree.choose_move(move_index=cutoff + 5) == most_visited(tree.root)


# --- Tree reuse -------------------------------------------------------------------


def test_play_reuses_the_subtree_and_resets_the_budget():
    tree = _self_play_tree(simulations=100)
    move = most_visited(tree.root)
    child = tree.root.children[move]
    assert child.expanded
    before = {m: (g.visit_count, g.value_sum) for m, g in child.children.items()}
    priors_before = [g.prior for g in child.children.values()]

    tree.play(move)

    assert tree.root is child
    assert tree.simulations == 0
    after = {m: (g.visit_count, g.value_sum) for m, g in tree.root.children.items()}
    assert after == before
    # Self-play re-noises the new root's priors (they were the evaluator's).
    priors_after = [g.prior for g in tree.root.children.values()]
    assert priors_after != pytest.approx(priors_before, abs=1e-9)
    assert sum(priors_after) == pytest.approx(1.0)

    tree.simulate(UniformEvaluator(), 10)
    assert tree.simulations == 10


def test_play_on_an_unexpanded_child_sets_up_the_new_root_on_next_select():
    state = play([(0, 0)], size=3)
    tree = SearchTree(state, SELF_PLAY, rng=np.random.default_rng(0))
    tree.simulate(UniformEvaluator(), 1)  # only one root child gets expanded
    unexpanded = next(m for m, c in tree.root.children.items() if not c.expanded)

    tree.play(unexpanded)

    assert tree.select() == apply_move(state, unexpanded)


def test_play_with_a_pending_leaf_raises():
    tree = _self_play_tree()
    tree.select()
    with pytest.raises(RuntimeError):
        tree.play(most_visited(tree.root))


# --- Determinism ------------------------------------------------------------------


def test_same_seed_same_search():
    trees = [_self_play_tree(seed=7, simulations=100) for _ in range(2)]
    visits = [[c.visit_count for c in t.root.children.values()] for t in trees]
    assert visits[0] == visits[1]


def test_search_modules_do_not_import_torch():
    # The search must stay torch-free (docs/design/engineering.md, "The
    # evaluator seam"). A fresh interpreter, because this test process may
    # already have torch loaded by the network tests.
    code = "import sys, ox_zero.engine.search; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_initial_state_root_has_every_cell_as_a_child():
    tree = SearchTree(initial_state(3), ANALYSIS)
    tree.simulate(UniformEvaluator(), 0)
    assert list(tree.root.children) == [(r, c) for r in range(3) for c in range(3)]


# === The analysis generator ========================================================


def test_analyse_refuses_a_finished_game_eagerly():
    finished = play([(0, 0), (0, 1), (0, 2)], size=3)
    # No next(): the error must come from the call itself.
    with pytest.raises(ValueError):
        analyse(finished, UniformEvaluator())


def test_analyse_yields_setup_then_one_snapshot_per_simulation():
    state = play([(0, 0), (1, 2)], size=4)
    legal = legal_moves(state)
    snapshots = list(analyse(state, UniformEvaluator(), max_simulations=5))

    assert [s.simulations for s in snapshots] == [0, 1, 2, 3, 4, 5]
    for snapshot in snapshots:
        assert isinstance(snapshot, Snapshot)
        assert list(snapshot.q) == legal
        assert list(snapshot.visits) == legal
        # Root expansion gave every legal move one visit, on top of the
        # counted simulations.
        assert sum(snapshot.visits.values()) == snapshot.simulations + len(legal)
        top = max(snapshot.visits.values())
        assert snapshot.best == next(m for m in legal if snapshot.visits[m] == top)
        assert -1.0 <= snapshot.value <= 1.0


def test_analyse_without_a_cap_runs_until_dropped():
    state = play([(0, 0)], size=3)
    snapshots = list(itertools.islice(analyse(state, UniformEvaluator()), 20))
    assert [s.simulations for s in snapshots] == list(range(20))
    # Dropping the iterator is the cancellation: nothing to close.


def test_first_snapshot_already_scores_every_move():
    # X to move on 3x3 after X(0,0) O(0,1). (0,2) wins on the spot; the
    # position after (1,1) is scripted at +0.5 for O (the side to move there),
    # so from X's view that move is worth -0.5. Everything else is uniform, 0.
    state = play(WIN_IN_ONE, size=3)
    after_centre = apply_move(state, (1, 1))
    evaluator = TableEvaluator({after_centre: (np.full(9, 1 / 6, dtype=np.float32), 0.5)})

    first = next(analyse(state, evaluator))

    assert first.simulations == 0
    assert first.q[(0, 2)] == 1.0
    assert first.q[(1, 1)] == pytest.approx(-0.5)
    others = [m for m in legal_moves(state) if m not in {(0, 2), (1, 1)}]
    assert all(first.q[m] == 0.0 for m in others)
    assert all(first.visits[m] == 1 for m in legal_moves(state))


# === Solver fixtures ===============================================================
#
# The exact solver is the only ground truth the search has. With a uniform
# evaluator (no knowledge at all) the search must still find short tactics
# purely from terminal positions, and pick the unique optimal move in deeper
# forced wins. Budgets were checked against a scratch PUCT following
# search.md; see docs/plans/02-engine-search-network.md, step 5.


def _final(state, simulations):
    *_, last = analyse(state, UniformEvaluator(), max_simulations=simulations)
    return last


def test_win_in_one(solver):
    state = play(WIN_IN_ONE, size=3)
    assert solver.best_moves(state) == [(0, 2)]
    last = _final(state, 100)
    assert last.best == (0, 2)
    assert last.value > 0.85


def test_forced_loss_in_two(solver):
    # O to move; both remaining moves let X complete a line next turn. The
    # value-sign test: every line ends in a loss, and averaging must say so.
    state = play([(0, 0), (1, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)], size=3)
    assert solver.value(state) == -1
    last = _final(state, 200)
    assert last.value < -0.9


def test_loss_in_one_avoided(solver):
    # O to move with 13 legal moves; every move except (1,3) hands X an
    # immediate alternating line.
    state = from_board_string("____XXO_________", size=4)
    assert len(legal_moves(state)) == 13
    last = _final(state, 200)
    assert last.best == (1, 3)


DEEP_FORCED_WINS = [
    ([(1, 3), (2, 0), (3, 0)], (1, 0)),
    ([(0, 3), (1, 3), (2, 0)], (2, 3)),
    ([(1, 1), (2, 2), (0, 0)], (3, 3)),
    ([(0, 0), (0, 1), (2, 3)], (0, 2)),
]


@pytest.mark.parametrize("moves, optimal", DEEP_FORCED_WINS)
def test_deep_forced_win_picks_the_unique_optimal_move(solver, moves, optimal):
    # O to move on 4x4, a forced win 9 to 11 plies deep, exactly one optimal
    # move out of 13. First make sure the rules still say so, so that a rules
    # change cannot silently turn this into a vacuous test.
    state = play(moves, size=4)
    assert solver.value(state) == 1
    assert solver.best_moves(state) == [optimal]

    last = _final(state, 1000)
    assert last.best == optimal
    # Deliberately NOT asserted: the value sign. On wins this deep the root
    # value stays near 0, because plain averaging never *proves* a win: a
    # proven-won subtree still averages in the draws that unvisited children
    # (Q = 0) and noisy inner nodes contribute. Propagating proofs is
    # MCTS-Solver, a deferred upgrade (docs/design/upgrades.md); this test is
    # the baseline it will be measured against.


def test_empty_three_by_three_avoids_the_centre(solver):
    # Eight first moves draw; the centre loses (docs/design/open-questions.md).
    state = initial_state(3)
    assert [m for m in legal_moves(state) if m not in solver.best_moves(state)] == [(1, 1)]
    last = _final(state, 2000)
    assert last.best != (1, 1)
    assert last.visits[(1, 1)] == min(last.visits.values())


def _sample_positions(solver, count, seed):
    """Reachable, live 3x3 positions whose optimal moves are a strict subset
    of at least 3 legal moves, so "best is optimal" is a real claim."""
    rng = random.Random(seed)
    found = {}
    while len(found) < count:
        state = initial_state(3)
        for _ in range(rng.randrange(0, 7)):
            if is_terminal(state):
                break
            state = apply_move(state, rng.choice(legal_moves(state)))
        if is_terminal(state) or state in found:
            continue
        legal = legal_moves(state)
        best = solver.best_moves(state)
        if len(legal) >= 3 and len(best) < len(legal):
            found[state] = best
    return list(found.items())


def test_random_three_by_three_positions_agree_with_the_solver(solver):
    failures = []
    for state, best in _sample_positions(solver, 30, seed=0):
        last = _final(state, 500)
        if last.best not in best:
            failures.append((to_board_string(state), last.best, best))
    assert failures == []
