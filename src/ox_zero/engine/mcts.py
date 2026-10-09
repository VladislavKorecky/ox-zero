"""The search tree: nodes, PUCT selection, expansion, backup, and move choice.

Design: docs/design/search.md (every rule and constant) and
docs/design/engineering.md, "Tree representation". Reference: Silver et al.
2018, "A general reinforcement learning algorithm that masters chess, shogi,
and Go through self-play", and its published pseudocode.

What a simulation does
----------------------
Monte Carlo tree search (MCTS) builds a tree of positions below the one being
analysed, one *simulation* at a time:

1. **Select.** Walk down from the root, at each node taking the child with
   the highest PUCT score (below), until reaching a node that has not been
   expanded yet or a finished game.
2. **Expand and evaluate.** Ask the network for `(p, v)` at that leaf. `p`
   becomes the children's priors; `v` is the leaf's value. A finished game
   is not sent to the network: its value is known exactly.
3. **Back up.** Add `v` to every edge on the path back to the root, flipping
   its sign at every step, because the players alternate.

After many simulations the root's visit counts say which moves the search
found worth its time; that is the search's answer (see `most_visited`).

This module holds those steps as plain functions over `Node`s. Driving them
(the two-phase select / evaluate / backup loop, noise, snapshots) is
`search.py`'s job. Nothing here imports torch; priors arrive as NumPy arrays
through the evaluator seam (`evaluator.py`).

Node statistics
---------------
The paper stores `N, W, Q, P` per *edge* `(s, a)`. Our tree has exactly one
node per edge (a node is "the position after playing `a` at `s`"), so each
node stores the statistics of the edge that leads into it:

    N(s,a) = child.visit_count
    W(s,a) = child.value_sum      from the view of the player who played `a`
    Q(s,a) = child.q = W / N      0 when N = 0
    P(s,a) = child.prior

`N(s)`, the parent's visit count in the PUCT formula, is the parent node's
own `visit_count`. Backup increments it on every node of the path, exactly as
the pseudocode does. At the root that equals `Σ_a N(s,a)` (expanding the root
is setup and adds no visit). At an expanded inner node it is one more than
`Σ_a N(s,a)`: the extra visit is the simulation that expanded it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ox_zero.game.rules import Cell, State, apply_move, is_terminal, legal_moves


class Node:
    """One position in the search tree, plus the statistics of the edge into it.

    `children` is `None` until the node is expanded; terminal nodes are never
    expanded and keep `None`. Once expanded it maps every legal move, in board
    order, to its child.

    Lazy state: an expansion creates a child for every legal move (141 on an
    early 12x12 board), but a simulation then visits only one of them, and
    most children are never visited at all. A `State` holds a full copy of the
    board, so building every child's state up front made boards about 88% of
    the tree's memory. A child made by `expand` therefore stores only a
    reference to its parent's (shared, not copied) state and its own move, and
    builds its state the first time `state` is read, then caches it.

    The parent reference is kept after that, deliberately: the parent node
    holds the same `State` object anyway, so it costs nothing extra (after
    `reuse_subtree`, at most one old board stays alive), and never clearing it
    means two threads reading `state` at once cannot see a half-updated node.

    Nodes compare by identity, as tree nodes should: two nodes with equal
    statistics are still different places in the tree.

    `__slots__` instead of a per-instance `__dict__`: a few hundred bytes
    saved per node, which matters at hundreds of thousands of nodes.
    """

    __slots__ = ("_state", "_parent_state", "_move", "prior", "visit_count", "value_sum", "children")

    def __init__(
        self,
        state: State | None,
        prior: float,
        visit_count: int = 0,
        value_sum: float = 0.0,
        children: dict[Cell, Node] | None = None,
        *,
        parent_state: State | None = None,
        move: Cell | None = None,
    ) -> None:
        if state is None and (parent_state is None or move is None):
            raise TypeError("a Node needs a state, or a parent_state and a move to build it from")
        self._state = state
        self._parent_state = parent_state
        self._move = move
        self.prior = prior  # P(s,a) of the edge into this node
        self.visit_count = visit_count  # N(s,a)
        self.value_sum = value_sum  # W(s,a), for the player who played a
        self.children = children

    @classmethod
    def lazy(cls, parent_state: State, move: Cell, prior: float) -> Node:
        """A child whose state, `apply_move(parent_state, move)`, is built on first use."""
        return cls(None, prior, parent_state=parent_state, move=move)

    @property
    def state(self) -> State:
        """The position this node represents (built and cached on first read)."""
        if self._state is None:
            # A lazy node always has both (enforced in __init__), and neither
            # is ever cleared, so this is safe to run twice concurrently: both
            # threads build equal states and either assignment is correct.
            self._state = apply_move(self._parent_state, self._move)  # type: ignore[arg-type]
        return self._state

    def __repr__(self) -> str:
        move = "root" if self._move is None else f"move={self._move}"
        built = "built" if self._state is not None else "lazy"
        return (
            f"Node({move}, state {built}, prior={self.prior:.4f}, N={self.visit_count},"
            f" W={self.value_sum:.4f}, expanded={self.expanded})"
        )

    @property
    def q(self) -> float:
        """`Q(s,a) = W / N`, the mean backed-up value; `0.0` when unvisited.

        0 for an unvisited child is the pseudocode's choice. In `[-1, 1]`
        value space it means "assume a draw": neither optimistic nor
        pessimistic, so the prior term alone decides whether it is tried.
        """
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def expanded(self) -> bool:
        return self.children is not None


def terminal_value(state: State) -> float:
    """The exact value of a finished game, for the side to move.

    `-1.0` if someone won: the game ends on the move that completes a line,
    and that move was made by the *previous* player, so the side to move has
    lost. `0.0` for a full board with no line. `+1.0` is impossible here.
    """
    if not is_terminal(state):
        raise ValueError("terminal_value needs a finished game")
    return -1.0 if state.winner is not None else 0.0


def puct_score(parent_visits: int, child: Node, c_base: float, c_init: float) -> float:
    """PUCT: `Q(s,a) + U(s,a)`, the value of trying `a` at `s` next.

    PUCT is "predictor + upper confidence bound applied to trees". The score
    is the move's average result so far (`Q`, exploitation) plus a bonus
    (`U`, exploration) that is large for moves the network likes and shrinks
    as a move is visited:

        U(s,a) = C(s) · P(s,a) · √N(s) / (1 + N(s,a))
        C(s)   = log((1 + N(s) + c_base) / c_base) + c_init

    - `P(s,a)` is the prior: the network's opinion before any search.
    - `√N(s) / (1 + N(s,a))`: as the parent gets visited, the bonus of a
      neglected child grows (like `√N`); every visit to the child itself
      divides it down. So every move is eventually retried, but moves with a
      high prior or a high `Q` get most of the budget.
    - `C(s)` is the 2018 paper's slowly growing exploration weight: about
      `c_init = 1.25` for small `N(s)`, rising logarithmically once `N(s)` is
      comparable to `c_base = 19652`, i.e. only in very long searches. (The
      2017 AlphaGo Zero paper used a fixed `c_puct`.)
    """
    c = math.log((1 + parent_visits + c_base) / c_base) + c_init
    u = c * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    return child.q + u


def select_child(node: Node, c_base: float, c_init: float) -> tuple[Cell, Node]:
    """The child with the highest PUCT score. Ties go to the first in board order.

    `children` is filled from `legal_moves`, which is in board order, and the
    strict `>` below keeps the first maximum, so hand-computed tests are
    deterministic.
    """
    assert node.children, "select_child needs an expanded, non-terminal node"
    parent_visits = node.visit_count
    best_move, best_child, best_score = None, None, -math.inf
    for move, child in node.children.items():
        score = puct_score(parent_visits, child, c_base, c_init)
        if score > best_score:
            best_move, best_child, best_score = move, child, score
    return best_move, best_child


def select_leaf(root: Node, c_base: float, c_init: float) -> list[Node]:
    """Walk down by PUCT from `root`; return the path to the first unexpanded node.

    The last node is either unexpanded and in need of an evaluation, or a
    finished game (terminal nodes are never expanded).
    """
    path = [root]
    node = root
    while node.expanded:
        _, node = select_child(node, c_base, c_init)
        path.append(node)
    return path


def expand(node: Node, policy: np.ndarray) -> None:
    """Create a child for every legal move, with priors renormalised over them.

        P(s,a) = p_a / Σ_{legal b} p_b

    The evaluator already zeroes illegal cells, so the renormalisation is
    usually a no-op; it makes `expand` safe for any policy (a scripted one in
    a test, or a network policy after float rounding).
    """
    if is_terminal(node.state):
        raise ValueError("cannot expand a finished game; its value is exact")
    if node.expanded:
        raise ValueError("node is already expanded")
    size = node.state.size
    moves = legal_moves(node.state)
    weights = [float(policy[row * size + col]) for row, col in moves]
    total = sum(weights)
    if not total > 0.0:
        raise ValueError("policy puts no mass on any legal move")
    node.children = {
        move: Node.lazy(node.state, move, prior=weight / total)
        for move, weight in zip(moves, weights)
    }


def backup(path: Sequence[Node], leaf_value: float) -> None:
    """Propagate a leaf's value up the path, flipping its sign at every step.

    `leaf_value` is from the view of the side to move *at the leaf*. Each
    node's `value_sum` is from the view of the player who made the move into
    that node, which for the leaf is the leaf's opponent. So the value is
    negated *before* the leaf's own update, and again at each step up,
    because the players alternate:

        for each node, from leaf to root:
            v = -v;  N += 1;  W += v

    The edge into the leaf receives `-v_leaf`, its parent `+v_leaf`, and so
    on. Every node on the path, the root included, gets one more visit; that
    is what `N(s)` in PUCT counts (see the module docstring).
    """
    value = leaf_value
    for node in reversed(path):
        value = -value
        node.visit_count += 1
        node.value_sum += value


def root_value(root: Node) -> float:
    """The search's value of the root for the side to move: `Σ W / Σ N` over its children.

    Not `root.q`: the root's own `value_sum` was accumulated for the player
    who moved *into* the root, the opponent of the side to move, so it has
    the wrong sign. The children's `W(root, a)` are exactly from the side to
    move's view (they played `a`). `0.0` before any visit.
    """
    if not root.children:
        return 0.0
    visits = sum(child.visit_count for child in root.children.values())
    if visits == 0:
        return 0.0
    return sum(child.value_sum for child in root.children.values()) / visits


def add_dirichlet_noise(
    root: Node, rng: np.random.Generator, epsilon: float, alpha: float
) -> None:
    """Mix Dirichlet noise into the root's priors (self-play exploration).

        P'(s,a) = (1 - ε) · P(s,a) + ε · η_a,     η ~ Dir(α, ..., α)

    Why: in self-play the network is both the student and the teacher. A move
    it gives a near-zero prior would almost never be searched, so the search
    would never discover that it is good, and the network would never learn
    it. Noise at the root forces every move to get a chance.

    A Dirichlet sample `η` is a random probability vector. Its parameter `α`
    sets how spiky it is: small `α` puts most of the mass on a few moves
    (bold, targeted exploration), large `α` spreads it evenly. The paper
    scales `α` with the branching factor; we use `α = 11 / S²` (see
    `SearchConfig`). Only the root is noised: deeper nodes keep the
    network's priors so the search below the root stays sharp.
    """
    assert root.children, "noise needs an expanded, non-terminal root"
    children = list(root.children.values())
    noise = rng.dirichlet([alpha] * len(children))
    for child, eta in zip(children, noise):
        child.prior = (1 - epsilon) * child.prior + epsilon * float(eta)


def visit_distribution(root: Node, size: int) -> np.ndarray:
    """The policy target `π`: `N(root, a) / Σ_b N(root, b)`, `float32 [S²]`.

    This is the visit distribution at temperature `τ = 1`. It is always the
    training target, whatever temperature chose the move actually played:
    the target should describe what the search *found*, and `τ → 0` would
    collapse it to a one-hot vector that throws away the information about
    the second-best moves.
    """
    pi = np.zeros(size * size, dtype=np.float32)
    if not root.children:
        raise ValueError("the root has no children")
    total = sum(child.visit_count for child in root.children.values())
    if total == 0:
        raise ValueError("the root has no visits yet")
    for (row, col), child in root.children.items():
        pi[row * size + col] = child.visit_count / total
    return pi


def most_visited(root: Node) -> Cell:
    """The search's best move: the most visited child. Ties in board order.

    Not the highest `Q`: a child visited three times can have a wildly wrong
    `Q`, while the visit count already integrates both the value and the
    network's confidence (PUCT only keeps visiting a move whose `Q` holds up).
    """
    assert root.children, "most_visited needs an expanded root"
    # `max` returns the first maximal element, and children are in board order.
    return max(root.children.items(), key=lambda item: item[1].visit_count)[0]


def sample_move(root: Node, rng: np.random.Generator) -> Cell:
    """A move drawn with probability proportional to its visits (`τ = 1`).

    Used in the opening of self-play games so that the same network does not
    play the same game over and over. A move with no visits is never drawn.
    """
    assert root.children, "sample_move needs an expanded root"
    moves = list(root.children)
    visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float64)
    index = rng.choice(len(moves), p=visits / visits.sum())
    return moves[int(index)]


def reuse_subtree(root: Node, move: Cell) -> Node:
    """After `move` is played, its child becomes the new root.

    Everything the search learned below that child (visit counts, values,
    expanded grandchildren) are real simulations and are kept; the rest of
    the tree is dropped. Nodes hold no parent pointers, so once the caller
    lets go of the old root, the siblings are garbage-collected.
    """
    if not root.children or move not in root.children:
        raise ValueError(f"{move} is not a child of this node")
    return root.children[move]
