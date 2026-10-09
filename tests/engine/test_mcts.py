"""Tests for the tree primitives: PUCT, expansion, backup, noise, move choice.

These work on hand-built trees, with `expand` called directly with a scripted
policy, so every number can be derived by hand. The full search loop is
tested in test_search.py.

Formulas under test (docs/design/search.md):

    score(s,a) = Q(s,a) + U(s,a)
    U(s,a)     = C(s) · P(s,a) · √N(s) / (1 + N(s,a))
    C(s)       = log((1 + N(s) + c_base) / c_base) + c_init
"""

import math
import tracemalloc

import numpy as np
import pytest

from ox_zero.engine import mcts
from ox_zero.engine.evaluator import UniformEvaluator
from ox_zero.engine.mcts import (
    Node,
    add_dirichlet_noise,
    backup,
    expand,
    most_visited,
    puct_score,
    reuse_subtree,
    root_value,
    sample_move,
    select_child,
    select_leaf,
    terminal_value,
    visit_distribution,
)
from ox_zero.game import apply_move, initial_state, is_terminal, play

C_BASE = 19652.0
C_INIT = 1.25


def _policy(size: int, weights: dict[tuple[int, int], float]) -> np.ndarray:
    policy = np.zeros(size * size, dtype=np.float32)
    for (row, col), weight in weights.items():
        policy[row * size + col] = weight
    return policy


def _two_by_two_root(priors=(0.7, 0.2, 0.06, 0.04)) -> Node:
    # No line of three fits on 2x2, so nothing here can ever be won or lost:
    # every value is 0 and only the priors drive selection.
    root = Node(initial_state(2), prior=1.0)
    expand(root, np.array(priors, dtype=np.float32))
    return root


# --- PUCT score -------------------------------------------------------------------


def test_puct_score_by_hand():
    child = Node(initial_state(3), prior=0.5, visit_count=10, value_sum=2.0)  # Q = 0.2
    c = math.log((1 + 100 + 19652) / 19652) + 1.25
    u = c * 0.5 * math.sqrt(100) / (1 + 10)
    assert puct_score(100, child, C_BASE, C_INIT) == pytest.approx(0.2 + u, abs=1e-9)


def test_unvisited_child_has_zero_q():
    # Q = 0 for an unvisited child: "assume a draw" in [-1, 1] value space.
    child = Node(initial_state(3), prior=0.5)
    assert child.q == 0.0
    c = math.log((1 + 4 + 19652) / 19652) + 1.25
    assert puct_score(4, child, C_BASE, C_INIT) == pytest.approx(c * 0.5 * 2 / 1, abs=1e-12)


def test_every_child_scores_zero_before_the_parent_is_visited():
    # √N(s) = 0 kills the exploration term, and unvisited Q is 0, so the very
    # first selection is decided by board order alone.
    root = _two_by_two_root()
    assert all(puct_score(0, child, C_BASE, C_INIT) == 0.0 for child in root.children.values())
    move, _ = select_child(root, C_BASE, C_INIT)
    assert move == (0, 0)


# --- Selection --------------------------------------------------------------------


def test_selection_follows_priors_when_values_are_equal():
    # Children a, b, c, d = (0,0), (0,1), (1,0), (1,1) with priors 0.7, 0.2,
    # 0.06, 0.04. Q is 0 everywhere and C(s)·√N(s) is common to all children,
    # so each selection maximises P / (1 + N). With N(s) the number of
    # selections so far:
    #
    #   sel  N(s)  a: .7/(1+Na)  b: .2/(1+Nb)  pick
    #    1    0    (all scores 0: board order)  a
    #    2    1      .350          .200         a
    #    3    2      .233          .200         a
    #    4    3      .175          .200         b
    #    5    4      .175          .100         a
    #    6    5      .140          .100         a
    #    7    6      .117          .100         a
    #    8    7      .100          .100         tie (exact in real arithmetic)
    #    9    8    after a: .0875 vs .100 -> b;  after b: .100 vs .067 -> a
    #   10    9    either way the counts meet at a = 8, b = 2:
    #                after a,b: .0875 vs .067 -> a;  after b,a: .0875 vs .067 -> a
    #
    # c's best score, .06, never beats the leader, so c and d stay unvisited.
    # The tie at selection 8 is decided by float rounding, so only the
    # sequence before it and the final counts are asserted.
    root = _two_by_two_root()
    names = {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}
    sequence = []
    for _ in range(10):
        move, child = select_child(root, C_BASE, C_INIT)
        sequence.append(names[move])
        backup([root, child], 0.0)

    assert sequence[:7] == ["a", "a", "a", "b", "a", "a", "a"]
    counts = [root.children[m].visit_count for m in names]
    assert counts == [8, 2, 0, 0]
    assert root.visit_count == 10


def test_selection_prefers_value_over_prior():
    # Same tree, but b (prior 0.2, under a third of a's 0.7) turns out to be
    # good: every visit to b backs up Q = +0.8 for the player choosing it.
    # b is first visited at selection 4 (see the table above); from then on
    # its Q dominates the shrinking exploration terms and it takes almost
    # every selection. (Child c, prior 0.06, would not do: it is first
    # visited only at selection 15, too late to overtake a within 20.)
    root = _two_by_two_root()
    for _ in range(20):
        move, child = select_child(root, C_BASE, C_INIT)
        # `backup` negates the leaf value for the leaf's own edge (see
        # test_backup_signs), so a leaf value of -0.8 lands as W += +0.8.
        backup([root, child], -0.8 if move == (0, 1) else 0.0)

    visits = {move: child.visit_count for move, child in root.children.items()}
    assert max(visits, key=visits.get) == (0, 1)
    assert root.children[(0, 1)].q == pytest.approx(0.8)


def test_select_leaf_descends_to_an_unexpanded_node():
    root = _two_by_two_root()
    path = select_leaf(root, C_BASE, C_INIT)
    assert path[0] is root
    assert len(path) == 2
    assert not path[-1].expanded


def test_select_leaf_stops_at_a_terminal_node():
    # X to move on 3x3 with X _ O ... : root expanded with all mass on the
    # winning move (0,2) after X(0,0) O(0,1). The child is terminal.
    state = play([(0, 0), (0, 1)], size=3)
    root = Node(state, prior=1.0)
    expand(root, _policy(3, {(0, 2): 1.0}))
    backup([root], 0.0)  # give the root one visit so √N(s) > 0
    path = select_leaf(root, C_BASE, C_INIT)
    assert path[-1].state == apply_move(state, (0, 2))
    assert is_terminal(path[-1].state)
    assert not path[-1].expanded


# --- Backup -----------------------------------------------------------------------


def test_backup_signs():
    # root --a--> child --b--> leaf. The leaf is evaluated at +1: good for the
    # side to move *at the leaf*. The edge into the leaf (move b) was played
    # by the other player, the leaf's opponent, so from b's player's view the
    # result is -1: W(leaf) = -1. One level up, move a was played by the
    # leaf's side to move again, so W(child) = +1; then W(root) = -1.
    # Players alternate, so the sign flips at every step.
    root = Node(initial_state(3), prior=1.0)
    child = Node(apply_move(root.state, (0, 0)), prior=1.0)
    leaf = Node(apply_move(child.state, (1, 1)), prior=1.0)

    backup([root, child, leaf], 1.0)

    assert (leaf.value_sum, child.value_sum, root.value_sum) == (-1.0, 1.0, -1.0)
    assert leaf.visit_count == child.visit_count == root.visit_count == 1


# --- Terminal values --------------------------------------------------------------


def test_terminal_value_after_a_win_is_a_loss_for_the_side_to_move():
    assert terminal_value(play([(0, 0), (0, 1), (0, 2)], size=3)) == -1.0


def test_terminal_value_of_a_full_board_is_a_draw():
    assert terminal_value(play([(0, 0), (0, 1), (1, 0), (1, 1)], size=2)) == 0.0


def test_terminal_value_rejects_a_live_position():
    with pytest.raises(ValueError):
        terminal_value(initial_state(3))


# --- Expansion --------------------------------------------------------------------


def test_expand_creates_legal_children_with_renormalised_priors():
    # X(0,0) O(0,2) X(2,0) O(2,2): no line, 5 empty cells.
    state = play([(0, 0), (0, 2), (2, 0), (2, 2)], size=3)
    legal = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    # Half the mass sits on the occupied cell (0,0); it must be discarded and
    # the legal half rescaled to sum to 1.
    weights = {(0, 0): 0.5, (0, 1): 0.1, (1, 0): 0.05, (1, 1): 0.15, (1, 2): 0.1, (2, 1): 0.1}
    node = Node(state, prior=1.0)
    expand(node, _policy(3, weights))

    assert node.expanded
    assert list(node.children) == legal
    priors = [child.prior for child in node.children.values()]
    assert priors == pytest.approx([0.2, 0.1, 0.3, 0.2, 0.2])
    assert sum(priors) == pytest.approx(1.0)
    for move, child in node.children.items():
        assert child.state == apply_move(state, move)
        assert child.visit_count == 0 and child.value_sum == 0.0 and not child.expanded


def test_expanding_a_terminal_node_raises():
    node = Node(play([(0, 0), (0, 1), (0, 2)], size=3), prior=1.0)
    with pytest.raises(ValueError):
        expand(node, np.full(9, 1 / 9, dtype=np.float32))


# --- Lazy child states --------------------------------------------------------------
# An expansion creates a child per legal move (141 on a fresh 12x12 opening),
# but a simulation visits only one of them. Each `State` holds a full board,
# so building them all up front made boards ~88% of the tree's memory. A
# child builds its board on first access instead.


def uniform(state) -> np.ndarray:
    # The production path: a legal-only policy, as evaluators produce.
    return UniformEvaluator().evaluate([state])[0][0]


def test_expansion_builds_no_child_boards(monkeypatch):
    calls = []
    real_apply_move = mcts.apply_move
    monkeypatch.setattr(mcts, "apply_move", lambda s, m: calls.append(m) or real_apply_move(s, m))

    node = Node(play([(5, 5)]), prior=1.0)
    expand(node, uniform(node.state))
    assert calls == []

    child = node.children[(0, 0)]
    first = child.state
    assert calls == [(0, 0)]
    assert child.state is first  # cached: built once
    assert calls == [(0, 0)]


def test_a_lazy_child_state_is_the_parent_plus_its_move():
    node = Node(play([(1, 1)], size=3), prior=1.0)
    expand(node, uniform(node.state))
    for move, child in node.children.items():
        assert child.state == apply_move(node.state, move)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="nothing"),
        pytest.param({"parent_state": "P"}, id="parent without move"),
        pytest.param({"move": (0, 0)}, id="move without parent"),
    ],
)
def test_a_lazy_node_needs_both_a_parent_state_and_a_move(kwargs):
    if kwargs.get("parent_state") == "P":
        kwargs["parent_state"] = initial_state(3)
    with pytest.raises(TypeError):
        Node(None, prior=1.0, **kwargs)


def test_a_node_takes_a_state_or_lazy_arguments_not_both():
    state = initial_state(3)
    with pytest.raises(TypeError):
        Node(state, prior=1.0, parent_state=state, move=(0, 0))


def test_an_expansion_costs_far_less_than_its_child_boards():
    # Measured before the change: about 203 KB for 141 children, 178 KB of
    # which were the boards. The bound leaves room for the nodes themselves.
    state = play([(5, 5), (6, 6), (5, 6)])
    policy = uniform(state)
    nodes = [Node(state, prior=1.0) for _ in range(20)]
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for node in nodes:
            expand(node, policy)
        per_expansion = (tracemalloc.get_traced_memory()[0] - before) / len(nodes)
    finally:
        tracemalloc.stop()
    assert per_expansion < 50 * 1024


# --- Root value -------------------------------------------------------------------


def test_root_value_averages_over_children():
    root = _two_by_two_root()
    a, b = root.children[(0, 0)], root.children[(0, 1)]
    backup([root, a], -0.5)  # W_a = +0.5
    backup([root, b], 0.3)  # W_b = -0.3
    backup([root, b], 0.1)  # W_b = -0.4

    expected = (a.value_sum + b.value_sum) / (a.visit_count + b.visit_count)
    assert root_value(root) == pytest.approx(expected)
    assert root_value(root) == pytest.approx(0.1 / 3)
    # The sign trap: the root's own W was accumulated for the player who
    # moved *into* the root, i.e. the opponent of the side to move, so its
    # mean is the negation of the value we want.
    assert root_value(root) == pytest.approx(-root.q)


def test_root_value_without_visits_is_zero():
    assert root_value(_two_by_two_root()) == 0.0
    assert root_value(Node(initial_state(2), prior=1.0)) == 0.0


# --- Dirichlet noise --------------------------------------------------------------


def _three_level_tree() -> Node:
    root = Node(initial_state(3), prior=1.0)
    expand(root, np.full(9, 1 / 9, dtype=np.float32))
    expand(root.children[(0, 0)], np.full(9, 1 / 8, dtype=np.float32))
    return root


def test_dirichlet_noise_changes_root_priors_only():
    root = _three_level_tree()
    grandchildren_before = [c.prior for c in root.children[(0, 0)].children.values()]

    add_dirichlet_noise(root, np.random.default_rng(0), epsilon=0.25, alpha=0.3)

    priors = [child.prior for child in root.children.values()]
    assert sum(priors) == pytest.approx(1.0)
    assert all(p != pytest.approx(1 / 9, abs=1e-9) for p in priors)
    grandchildren_after = [c.prior for c in root.children[(0, 0)].children.values()]
    assert grandchildren_after == grandchildren_before


def test_dirichlet_noise_with_zero_epsilon_changes_nothing():
    root = _three_level_tree()
    add_dirichlet_noise(root, np.random.default_rng(0), epsilon=0.0, alpha=0.3)
    assert [c.prior for c in root.children.values()] == pytest.approx([1 / 9] * 9)


def test_dirichlet_noise_is_reproducible_from_the_seed():
    first, second = _three_level_tree(), _three_level_tree()
    add_dirichlet_noise(first, np.random.default_rng(0), epsilon=0.25, alpha=0.3)
    add_dirichlet_noise(second, np.random.default_rng(0), epsilon=0.25, alpha=0.3)
    assert [c.prior for c in first.children.values()] == [
        c.prior for c in second.children.values()
    ]


# --- Visit distribution and move choice -------------------------------------------


def _visited_root() -> Node:
    # 3x3 after X(0,0): 8 children. Give three of them 6, 3, 1 visits.
    root = Node(play([(0, 0)], size=3), prior=1.0)
    expand(root, np.full(9, 1 / 8, dtype=np.float32))
    for move, visits in {(0, 1): 6, (1, 1): 3, (2, 2): 1}.items():
        root.children[move].visit_count = visits
    return root


def test_visit_distribution_is_the_normalised_visit_counts():
    pi = visit_distribution(_visited_root(), 3)
    assert pi.shape == (9,) and pi.dtype == np.float32
    expected = _policy(3, {(0, 1): 0.6, (1, 1): 0.3, (2, 2): 0.1})
    np.testing.assert_allclose(pi, expected, atol=1e-7)
    assert pi[0] == 0.0  # the occupied cell


def test_most_visited_breaks_ties_in_board_order():
    root = _visited_root()
    assert most_visited(root) == (0, 1)
    root.children[(2, 2)].visit_count = 6  # now tied with (0,1), which comes first
    assert most_visited(root) == (0, 1)


def test_sample_move_follows_the_visit_fractions():
    # Frequencies over 1000 seeded draws land within 0.04 of 0.6 / 0.3 / 0.1
    # (about three standard errors), and a move with no visits is never drawn.
    # The frequency check covers both properties the design cares about.
    root = _visited_root()
    rng = np.random.default_rng(0)
    draws = [sample_move(root, rng) for _ in range(1000)]
    counts = {move: draws.count(move) / 1000 for move in set(draws)}
    assert set(counts) == {(0, 1), (1, 1), (2, 2)}
    assert counts[(0, 1)] == pytest.approx(0.6, abs=0.04)
    assert counts[(1, 1)] == pytest.approx(0.3, abs=0.04)
    assert counts[(2, 2)] == pytest.approx(0.1, abs=0.04)


# --- Subtree reuse ----------------------------------------------------------------


def test_reuse_subtree_keeps_the_child_and_its_statistics():
    root = _three_level_tree()
    child = root.children[(0, 0)]
    backup([root, child, child.children[(1, 1)]], 0.5)
    grandchildren = dict(child.children)

    new_root = reuse_subtree(root, (0, 0))

    assert new_root is child
    assert new_root.visit_count == 1 and new_root.value_sum == 0.5
    assert new_root.children == grandchildren
    # Nodes hold no parent pointers, so nothing reachable from the new root
    # leads back to the old one.
    stack, seen = [new_root], []
    while stack:
        node = stack.pop()
        seen.append(node)
        stack.extend((node.children or {}).values())
    assert all(node is not root for node in seen)


def test_reuse_subtree_rejects_an_unknown_move():
    root = _three_level_tree()
    with pytest.raises(ValueError):
        reuse_subtree(root.children[(0, 0)].children[(1, 1)], (2, 2))  # not expanded
