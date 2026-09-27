"""The search loop: `SearchConfig`, the stepped `SearchTree`, and `analyse`.

Design: docs/design/search.md ("One simulation", "Modes", "Constants") and
docs/design/cli-integration.md (what the analysis generator yields).

One search, two modes
---------------------
Self-play and the CLI run the same PUCT search with different settings:

| | Self-play (`SELF_PLAY`) | Analysis (`ANALYSIS`) |
|---|---|---|
| Root noise | Dirichlet, re-applied after every move | none |
| Root children | expanded when PUCT first picks them | all evaluated up front |
| Driver | many trees in lockstep, one batched `evaluate` per step | `analyse`, one tree, batch size 1 |

Why a two-phase API
-------------------
The expensive step of a simulation is the network call, and a network is far
more efficient on one batch of 256 positions than on 256 batches of one. So
self-play (plan 04) runs many games at once, in lockstep:

    leaves = [tree.select() for tree in trees]        # phase 1: walk down
    p, v   = evaluator.evaluate([l for l in leaves if l is not None])
    for each tree with a leaf: tree.expand_and_backup(p[i], v[i])   # phase 2

`SearchTree` therefore splits a simulation at the network call. Each tree
has at most one pending leaf at a time (no virtual loss: see
docs/design/upgrades.md). A leaf that is a finished game needs no network: its
value is exact, so `select` backs it up on the spot and returns `None`.

The root
--------
The paper's pseudocode evaluates (expands) the root *before* the simulation
loop, then adds exploration noise to it. We follow it: the first `select` on
a fresh tree returns the root's own state, and the matching
`expand_and_backup` expands the root, applies noise if configured, and backs
nothing up. That setup is not a simulation and is not counted (plan 02,
"Decisions"); `simulations` counts only full simulations since the last
`play`.

In analysis mode `expand_root_children` then evaluates every root child in
one batch and gives each one visit, so every legal move has a real `Q` from
the first snapshot on (docs/design/cli-integration.md). Those visits are also
setup and are not counted either.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from ox_zero.engine.evaluator import Evaluator
from ox_zero.engine.mcts import (
    Node,
    add_dirichlet_noise,
    backup,
    expand,
    most_visited,
    reuse_subtree,
    root_value,
    sample_move,
    select_leaf,
    terminal_value,
    visit_distribution,
)
from ox_zero.game.rules import Cell, State, is_terminal

# Temperature cutoff per board size: a quarter of the mean game length
# measured under random play (docs/design/search.md, "Move selection").
_CUTOFF_TABLE: dict[int, int] = {4: 2, 6: 3, 8: 4, 12: 7}


@dataclass(frozen=True)
class SearchConfig:
    """Every constant of one search. Frozen: a config is a value, not a knob.

    Attributes:
        c_base, c_init: PUCT exploration constants, `C(s) = log((1 + N(s) +
            c_base) / c_base) + c_init` (AlphaZero 2018 pseudocode).
        dirichlet_epsilon: Weight `ε` of the root noise (0.25, as the paper).
        dirichlet_alpha: Concentration `α` of the root noise. `None` means
            `11 / S²` (see `alpha`).
        temperature_cutoff: Self-play moves sampled with `τ = 1` before
            switching to the most visited move. `None` means per size (see
            `cutoff`).
        root_noise: Mix Dirichlet noise into the root priors. Self-play only.
        expand_root: Evaluate every root child up front. Analysis only.
    """

    c_base: float = 19652.0
    c_init: float = 1.25
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float | None = None
    temperature_cutoff: int | None = None
    root_noise: bool = False
    expand_root: bool = False

    def alpha(self, size: int) -> float:
        """Dirichlet `α` for an `S x S` board: `11 / S²` unless set explicitly.

        The rule of thumb behind the paper's values (0.03 for Go, 0.3 for
        chess) is `α ≈ 10 / (average legal moves)`. Measured on OXOX, games
        end with most of the board empty, so the average is about `0.9 · S²`
        and `10 / (0.9 · S²) ≈ 11 / S²` (docs/design/open-questions.md).
        """
        return self.dirichlet_alpha if self.dirichlet_alpha is not None else 11 / (size * size)

    def cutoff(self, size: int) -> int:
        """Number of opening moves played with `τ = 1` on an `S x S` board.

        From the measured table (2, 3, 4, 7 moves on 4x4, 6x6, 8x8, 12x12).
        Other sizes fall back to `max(1, round(S / 2))`. PROVISIONAL: nothing
        was measured for those sizes; the fallback only exists so small test
        boards (3x3, 5x5) have a value. It gives 2, 3, 4, 6 on the tabulated
        sizes, close to the table.
        """
        if self.temperature_cutoff is not None:
            return self.temperature_cutoff
        return _CUTOFF_TABLE.get(size, max(1, round(size / 2)))


ANALYSIS = SearchConfig(expand_root=True)
SELF_PLAY = SearchConfig(root_noise=True)


class SearchTree:
    """One search tree, stepped one simulation at a time.

    Attributes:
        root: The current root node. Replaced by `play`.
        simulations: Full simulations completed since construction or the last
            `play`. Root setup (expansion, and in analysis the up-front child
            evaluations) is not counted.
    """

    def __init__(
        self, state: State, config: SearchConfig, rng: np.random.Generator | None = None
    ) -> None:
        if is_terminal(state):
            raise ValueError("the game is over; there is nothing to search")
        self.root = Node(state, prior=1.0)
        self.config = config
        self.simulations = 0
        # One generator per tree, used for the Dirichlet noise and for move
        # sampling. Passing a seeded one makes the whole search reproducible
        # (docs/design/engineering.md, "Devices and determinism").
        self.rng = rng if rng is not None else np.random.default_rng()
        # The path from the root to the leaf returned by the last `select`,
        # waiting for its evaluation. `None` when nothing is pending.
        self._pending: list[Node] | None = None

    # --- Phase 1 -----------------------------------------------------------------

    def select(self) -> State | None:
        """Walk down by PUCT and return the leaf state that needs an evaluation.

        On a fresh tree (or after `play` onto an unexpanded child) this is the
        root itself. If the walk ends on a finished game, its exact value is
        backed up here, the simulation is counted, and `None` is returned:
        there is nothing for the network to do.

        Raises:
            RuntimeError: A leaf from the previous `select` is still pending.
            ValueError: The root is a finished game.
        """
        if self._pending is not None:
            raise RuntimeError("select() called while a leaf is still pending")
        if is_terminal(self.root.state):
            raise ValueError("the game is over; there is nothing to search")

        if not self.root.expanded:
            self._pending = [self.root]
            return self.root.state

        path = select_leaf(self.root, self.config.c_base, self.config.c_init)
        leaf = path[-1]
        if is_terminal(leaf.state):
            # A finished game: no network needed, the value is exact. It
            # enters the average like any other backup (no proof propagation;
            # MCTS-Solver is a deferred upgrade, docs/design/upgrades.md).
            backup(path, terminal_value(leaf.state))
            self.simulations += 1
            return None
        self._pending = path
        return leaf.state

    # --- Phase 2 -----------------------------------------------------------------

    def expand_and_backup(self, policy: np.ndarray, value: float) -> None:
        """Expand the pending leaf with `policy` and back `value` up its path.

        `policy` is the evaluator's prior for the leaf (`[S²]`), `value` its
        value for the leaf's side to move. If the pending leaf is the root,
        this is setup: the root is expanded, noise is applied if configured,
        nothing is backed up and nothing is counted.

        Raises:
            RuntimeError: No leaf is pending.
        """
        if self._pending is None:
            raise RuntimeError("expand_and_backup() called without a pending leaf")
        path, self._pending = self._pending, None
        leaf = path[-1]
        expand(leaf, policy)

        if leaf is self.root:
            # Root setup, as in the pseudocode: evaluate the root, then add
            # exploration noise, *then* start simulating. The root's value
            # is not backed up: there is no edge above the root to credit.
            if self.config.root_noise:
                self._add_noise()
            return

        backup(path, value)
        self.simulations += 1

    def expand_root_children(self, evaluator: Evaluator) -> None:
        """Analysis setup: evaluate every root child in one batch; one visit each.

        This is exactly what one simulation per child would do (expand the
        child with its own policy, back its value up through the root, so
        `W(root, a) = -v_child`), but batched into a single `evaluate` call
        and not counted as simulations. Finished-game children get their
        exact value and stay unexpanded. Afterwards every legal move has a
        real `Q` from its own evaluation, which is what the CLI shows as that
        move's score (docs/design/cli-integration.md).

        Raises:
            RuntimeError: The root is not expanded yet, or a leaf is pending,
                or root children were already visited.
        """
        if not self.root.expanded:
            raise RuntimeError("expand the root before its children")
        if self._pending is not None:
            raise RuntimeError("a leaf is still pending")
        children = list(self.root.children.values())
        if any(child.visit_count for child in children):
            raise RuntimeError("root children have already been visited")

        live = [child for child in children if not is_terminal(child.state)]
        if live:
            policies, values = evaluator.evaluate([child.state for child in live])
            for child, policy, value in zip(live, policies, values):
                expand(child, policy)
                backup([self.root, child], float(value))
        for child in children:
            if is_terminal(child.state):
                backup([self.root, child], terminal_value(child.state))

    # --- Convenience -------------------------------------------------------------

    def simulate(self, evaluator: Evaluator, n: int = 1) -> None:
        """Run `n` full simulations, one evaluation at a time (batch size 1).

        Does the root setup first if it has not happened yet, including the
        up-front child evaluations when `config.expand_root` is set, so
        `simulate(evaluator, 0)` is "just set up".
        """
        if not self.root.expanded:
            self._step(evaluator)  # root setup: not counted
            if self.config.expand_root:
                self.expand_root_children(evaluator)
        target = self.simulations + n
        while self.simulations < target:
            self._step(evaluator)

    def _step(self, evaluator: Evaluator) -> None:
        leaf = self.select()
        if leaf is None:
            return  # a terminal leaf, already backed up and counted
        policies, values = evaluator.evaluate([leaf])
        self.expand_and_backup(policies[0], float(values[0]))

    # --- Reading the answer and moving on ----------------------------------------

    def policy_target(self) -> np.ndarray:
        """The training target `π`: the root's visit distribution at `τ = 1`."""
        return visit_distribution(self.root, self.root.state.size)

    def choose_move(self, move_index: int) -> Cell:
        """The move to play in self-play, given the 0-based move number of the game.

        Before the temperature cutoff: sampled in proportion to visits
        (`τ = 1`), so games from the same network differ. From the cutoff on:
        the most visited move (`τ → 0`), so the tactical phase is played as
        well as the search can.
        """
        if move_index < self.config.cutoff(self.root.state.size):
            return sample_move(self.root, self.rng)
        return most_visited(self.root)

    def play(self, move: Cell) -> None:
        """Advance the root to `move`, keeping its subtree.

        The child's statistics are real simulations and are kept as a warm
        start, but the budget restarts: `simulations` goes back to 0 and the
        next move gets `n` fresh simulations on top of the inherited ones
        (plan 02, "Decisions"). In self-play the new root's priors, which are
        the network's own, get fresh Dirichlet noise.

        Raises:
            RuntimeError: A leaf is pending.
        """
        if self._pending is not None:
            raise RuntimeError("cannot play while a leaf is pending")
        if not self.root.expanded:
            raise RuntimeError("the root has not been expanded yet")
        self.root = reuse_subtree(self.root, move)
        self.simulations = 0
        # An unexpanded new root is set up by the next `select`, which applies
        # the noise itself. A finished game has nothing to noise.
        if self.config.root_noise and self.root.expanded:
            self._add_noise()

    def _add_noise(self) -> None:
        size = self.root.state.size
        add_dirichlet_noise(
            self.root, self.rng, self.config.dirichlet_epsilon, self.config.alpha(size)
        )

