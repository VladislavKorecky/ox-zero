"""Lockstep self-play: the current network plays itself and labels its own games.

Design: docs/design/training.md ("Self-play in lockstep", "Training data") and
docs/design/search.md ("Move selection", "Root noise", "Tree reuse").
Plan: docs/plans/04-training-pipeline.md, step 1.

Where training data comes from
------------------------------
AlphaZero has no human games. Each generation, the current network `f_θ`
plays games against itself, guided by MCTS. Every position `s_t` it meets
becomes one training example `(s_t, π_t, z_t)`:

- `π_t`, the **policy target**: the root's visit distribution after the
  search, `π(a) = N(s_t, a) / Σ_b N(s_t, b)`. The search is a policy
  improvement operator: PUCT starts from the network's prior `p` and, by
  looking ahead, ends with visit counts that are a better policy than `p`.
  Training `p` towards `π` distils the search back into the network.
  `π` is always the `τ = 1` distribution, *whatever temperature chose the
  move actually played*: the target should describe what the search found,
  including how good the second-best moves looked. `τ → 0` would collapse it
  to a one-hot vector and throw that information away. Temperature only
  decides which move is played (exploration vs. strength), never the target.
- `z_t`, the **value target**: the final result of the game from the point of
  view of the side to move at `s_t`: `+1` if that side went on to win, `-1` if
  it lost, `0` for a draw. Sides alternate every ply, so along one decisive
  game `z` flips sign every position. The value head learns to predict `z`,
  i.e. "who wins from here under self-play".

Why lockstep
------------
A network is far faster on one batch of 64 positions than on 64 batches of
one. A single search tree needs one leaf evaluated at a time (the next
selection depends on the last backup), so to fill a batch we either

- take many leaves from *one* tree, using virtual loss to push the parallel
  walks apart (AlphaGo/AlphaZero's distributed setup), which changes what the
  search computes; or
- take *one* leaf from each of *many* trees, one per game being played.

We do the second ("lockstep"): every step, every live tree selects one leaf,
the leaves go to the network in one `evaluate` call, and each tree gets its
own answer back. Each tree sees exactly the evaluations it would have asked
for alone, so lockstep changes the schedule and nothing else (test 4 pins
this). Virtual loss is a deferred upgrade (docs/design/upgrades.md).

Refilling
---------
`parallel` games are alive at once. When one ends, a fresh game takes its
slot until `games` have been started, so the batch stays full until the last
game has started, rather than one batch of `games` trees that shrinks as
games finish.

This module is torch-free: it talks to the network only through the
`Evaluator` protocol, like the search itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ox_zero.engine.encoding import NUM_PLANES, encode_batch
from ox_zero.engine.evaluator import Evaluator
from ox_zero.engine.search import SearchConfig, SearchTree
from ox_zero.game.rules import Cell, Mark, State, initial_state, is_terminal


@dataclass(frozen=True)
class SelfPlayConfig:
    """How much self-play one generation does.

    Attributes:
        games: Games played per generation.
        parallel: Trees stepped in lockstep; `min(parallel, games)` are alive
            at once, so it is also the largest batch the evaluator sees.
            PROVISIONAL (plan 04, "Provisional constants for 6x6").
        simulations: Full simulations per move: the search budget after each
            `play()`. Visits inherited from the reused subtree come on top.
    """

    games: int = 128
    parallel: int = 64
    simulations: int = 100

    def __post_init__(self) -> None:
        if self.games < 0:
            raise ValueError(f"games must be non-negative, got {self.games}")
        if self.parallel < 1:
            raise ValueError(f"parallel must be at least 1, got {self.parallel}")
        # At least one simulation, so the root has visits and therefore a
        # policy target and a move to choose.
        if self.simulations < 1:
            raise ValueError(f"simulations must be at least 1, got {self.simulations}")


@dataclass(frozen=True)
class Examples:
    """Training examples as a struct of arrays, one row per recorded position.

    Struct of arrays (not a list of `(state, π, z)` tuples) because training
    samples rows by fancy indexing and augments them with batched array
    transforms; storing `State` would mean re-encoding every sampled example
    on every step (plan 04, "Training examples store encoded planes").

    Attributes:
        planes: `uint8 [N, 3, S, S]`, the `encode` output (0/1) cast down to
            save memory; converted back to `float32` when sampled.
        pi: `float32 [N, S²]`, the policy target, flat board order.
        z: `float32 [N]`, the value target for the side to move.
    """

    planes: np.ndarray
    pi: np.ndarray
    z: np.ndarray

    def __len__(self) -> int:
        return int(self.z.shape[0])

    @property
    def size(self) -> int:
        """The board side `S`, read from the planes' last axis."""
        return int(self.planes.shape[-1])

    @staticmethod
    def empty(size: int) -> Examples:
        """No examples, with the right shapes and dtypes for an `S x S` board."""
        return Examples(
            planes=np.zeros((0, NUM_PLANES, size, size), dtype=np.uint8),
            pi=np.zeros((0, size * size), dtype=np.float32),
            z=np.zeros(0, dtype=np.float32),
        )

    @staticmethod
    def concatenate(parts: Sequence[Examples]) -> Examples:
        """All rows of `parts`, in order. Every part must share one board size.

        Raises:
            ValueError: `parts` is empty (there is no size to give the
                result) or the parts have different board sizes.
        """
        if not parts:
            raise ValueError("nothing to concatenate; use Examples.empty(size)")
        sizes = {part.size for part in parts}
        if len(sizes) != 1:
            raise ValueError(f"cannot concatenate examples of board sizes {sorted(sizes)}")
        return Examples(
            planes=np.concatenate([part.planes for part in parts]),
            pi=np.concatenate([part.pi for part in parts]),
            z=np.concatenate([part.z for part in parts]),
        )


@dataclass(frozen=True)
class GameRecord:
    """One finished self-play game: enough to replay it with `rules.play`.

    Attributes:
        moves: Every move, in order, starting from the empty board.
        winner: The player who completed an alternating line; `None` for a draw.
    """

    moves: tuple[Cell, ...]
    winner: Mark | None


@dataclass(frozen=True)
class SelfPlayResult:
    """What one generation of self-play produced.

    Attributes:
        examples: Every recorded position of every game, game by game in the
            order the games finished.
        games: The game records, in the same order as their examples.
        steps: Lockstep steps taken (at most one `evaluate` call each).
        simulations: Full simulations over all trees, for the sims/s metric.
    """

    examples: Examples
    games: list[GameRecord]
    steps: int
    simulations: int


def game_examples(states: Sequence[State], pis: Sequence[np.ndarray], final: State) -> Examples:
    """Label one finished game: planes, policy targets and `z` for every position.

    `states[i]` is the position before move `i` and `pis[i]` its policy target;
    `final` is the terminal position the game ended in.

    The `z` rule: `+1` if the side to move at `states[i]` is the winner, `-1`
    if the other side won, `0` for a draw (docs/design/training.md, "Training
    data"). The winner made the last move, so the last recorded position has
    the winner to move and gets `+1`; going backwards the side to move
    alternates every ply, so the signs alternate too.

    Raises:
        ValueError: `final` is not a finished game, or the lengths differ.
    """
    if not is_terminal(final):
        raise ValueError("final must be a finished game")
    if len(states) != len(pis):
        raise ValueError(f"{len(states)} states but {len(pis)} policy targets")
    if not states:
        return Examples.empty(final.size)

    # encode_batch() gives float32 0/1 planes; uint8 holds them exactly at a
    # quarter of the memory. (It cannot stack zero states, which is why the
    # empty game returned above.)
    planes = encode_batch(states).astype(np.uint8)
    pi = np.stack([np.asarray(p, dtype=np.float32) for p in pis])
    if final.winner is None:
        z = np.zeros(len(states), dtype=np.float32)
    else:
        z = np.array(
            [1.0 if state.to_move is final.winner else -1.0 for state in states],
            dtype=np.float32,
        )
    return Examples(planes=planes, pi=pi, z=z)


def lockstep_step(trees: Sequence[SearchTree], evaluator: Evaluator) -> int:
    """One select / batched evaluate / expand_and_backup over every tree.

    The core of lockstep: phase 1 (`select`) walks every tree down to a leaf,
    one `evaluate` call scores all the leaves together, phase 2
    (`expand_and_backup`) hands each tree its own row of the answer. A tree
    whose walk ended on a finished game already backed up the exact value in
    `select` and returned `None`; it takes no part in the batch. Fresh roots
    take part too: their first `select` returns the root itself, and the
    matching `expand_and_backup` is the (uncounted) root setup.

    Knows nothing about games, so the tournament code reuses it.

    Returns:
        The number of simulations completed this step: terminal leaves
        count, root setups do not. Read off each tree's own counter, which
        follows exactly that rule, rather than re-deriving it here.
    """
    before = [tree.simulations for tree in trees]

    # Phase 1: one leaf per tree. Keep the tree alongside its leaf so the
    # answers can be routed back by position in the batch.
    pending: list[tuple[SearchTree, State]] = []
    for tree in trees:
        leaf = tree.select()
        if leaf is not None:
            pending.append((tree, leaf))

    # Batched evaluation. Skipped entirely when no tree needs the network:
    # an empty batch is not a valid request (UniformEvaluator rejects it).
    if pending:
        policies, values = evaluator.evaluate([leaf for _, leaf in pending])
        # Phase 2: row i of the answer belongs to the i-th pending tree.
        for (tree, _), policy, value in zip(pending, policies, values):
            tree.expand_and_backup(policy, float(value))

    return sum(tree.simulations - count for tree, count in zip(trees, before))


@dataclass
class _Game:
    """One live self-play game: its search tree and the record so far.

    `states[i]` and `pis[i]` are the root position and the policy target at
    the moment move `i` was chosen, i.e. before `play`.
    """

    tree: SearchTree
    states: list[State] = field(default_factory=list)
    pis: list[np.ndarray] = field(default_factory=list)
    moves: list[Cell] = field(default_factory=list)


def self_play(
    evaluator: Evaluator,
    size: int,
    search: SearchConfig,
    config: SelfPlayConfig,
    rng: np.random.Generator,
) -> SelfPlayResult:
    """Play `config.games` games of the evaluator against itself, in lockstep.

    Every tree shares `rng` (Dirichlet noise and move sampling). The loop is
    single-threaded and visits the trees in a fixed order, so the draws from
    the one generator happen in a fixed order and a seed reproduces the whole
    generation (plan 04, "One RNG for everything").

    `search` is normally `SELF_PLAY`: Dirichlet noise at every root, so even
    a confident network keeps trying other moves, and `τ = 1` sampling for
    the first `cutoff` moves, so games from the same network differ.

    Raises:
        ValueError: `search.expand_root` is set. That analysis-only option
            evaluates every root child up front inside `SearchTree.simulate`;
            the lockstep driver steps trees through `select` /
            `expand_and_backup` and never runs it, so accepting the flag
            would silently search differently from what the config says.
    """
    if search.expand_root:
        raise ValueError("self_play does not support expand_root (an analysis-only option)")
    finished_examples: list[Examples] = []
    records: list[GameRecord] = []
    started = 0
    steps = 0
    simulations = 0

    def new_game() -> _Game:
        nonlocal started
        started += 1
        return _Game(SearchTree(initial_state(size), search, rng))

    live = [new_game() for _ in range(min(config.parallel, config.games))]

    while live:
        simulations += lockstep_step([game.tree for game in live], evaluator)
        steps += 1

        # Every tree that has spent its budget on the current move plays it.
        # Trees reach the budget at different steps (a terminal leaf costs no
        # network call but still counts, a reused subtree may be unexpanded
        # and need a setup step first), so check each one.
        still_live: list[_Game] = []
        for game in live:
            tree = game.tree
            if tree.simulations < config.simulations:
                still_live.append(game)
                continue

            # Record before `play`: the target is the root's distribution at
            # decision time. `play` keeps the chosen child's subtree as a warm
            # start, so after it the counts belong to the next position.
            game.states.append(tree.root.state)
            game.pis.append(tree.policy_target())
            # `choose_move` applies the temperature schedule: sampled in
            # proportion to visits before the cutoff, most visited after.
            move = tree.choose_move(len(game.moves))
            game.moves.append(move)
            tree.play(move)

            final = tree.root.state
            if not is_terminal(final):
                still_live.append(game)
                continue

            # Game over: label it, and refill the slot with a fresh game if
            # the generation still has games to start.
            finished_examples.append(game_examples(game.states, game.pis, final))
            records.append(GameRecord(moves=tuple(game.moves), winner=final.winner))
            if started < config.games:
                still_live.append(new_game())
        live = still_live

    examples = (
        Examples.concatenate(finished_examples) if finished_examples else Examples.empty(size)
    )
    return SelfPlayResult(
        examples=examples, games=records, steps=steps, simulations=simulations
    )
