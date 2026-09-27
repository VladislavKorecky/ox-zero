"""Tests for the search driver: `SearchConfig`, `SearchTree`, `analyse`.

`SearchTree` is the stepped, two-phase search that self-play drives
(`select` a leaf, evaluate it elsewhere, `expand_and_backup`), and `analyse`
is the snapshot generator the CLI will consume. See docs/design/search.md
("Modes") and docs/plans/02-engine-search-network.md ("Interfaces").
"""

import subprocess
import sys

import numpy as np
import pytest

from ox_zero.engine.evaluator import UniformEvaluator
from ox_zero.engine.mcts import most_visited
from ox_zero.engine.search import ANALYSIS, SELF_PLAY, SearchConfig, SearchTree
from ox_zero.game import apply_move, initial_state, legal_moves, play

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
