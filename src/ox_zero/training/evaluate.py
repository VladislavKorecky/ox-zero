"""Checkpoint tournaments and the Elo fit: is the new generation stronger?

Design: docs/design/training.md ("Evaluation").
Plan: docs/plans/archive/04-training-pipeline.md, step 4 (Decisions rows "Tournament
protocol" and "Elo fit").

Why measure at all
------------------
The training losses say how well the network fits its own self-play data,
not whether it plays better: a network can fit stale or weak data perfectly.
The only direct measure of strength is to make generations play each other.
AlphaGo Zero gated every new network on a 55% win rate against the current
best; AlphaZero dropped the gate and always trains the latest network. We
follow AlphaZero (no gate) and use the matches purely as a measurement: each
new generation `g` plays the previous few checkpoints plus one long-range
"ladder" opponent, and a joint Elo fit over every match of the run turns the
scores into one rating per generation.

How a match is played
---------------------
- **Random first move, both colours.** Evaluation search is deterministic
  (see `EVALUATION`), so two fixed players would replay the same game over and
  over. Each pair of games starts from a random opening cell instead, and is
  played twice with the colours swapped. Pairing the colours cancels the
  first-move advantage (or disadvantage: 4x4 is an O win): both players get
  the same opening from both sides.
- **Two trees per game.** See `MatchGame`.
- **Lockstep.** All games of a match run at once. Each step, the trees of
  the games where `a` is to move go to `a` in one batch, then those where `b`
  is to move go to `b`, reusing `selfplay.lockstep_step`.

This module is torch-free: players are `Evaluator`s, like the search sees them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ox_zero.engine.evaluator import Evaluator
from ox_zero.engine.mcts import most_visited
from ox_zero.engine.search import SearchConfig, SearchTree
from ox_zero.game.rules import Cell, Mark, State, apply_move, initial_state, is_terminal
from ox_zero.training.selfplay import lockstep_step

# Evaluation search: no Dirichlet noise and the most visited move from move 0.
#
# Root noise and `τ = 1` sampling exist to make self-play *explore*: noise
# forces the search to try moves the network dislikes, sampling makes games
# from one network differ. Both deliberately weaken play. A tournament wants
# the opposite, each player at its true strength, so both are off; variety
# comes from the random opening move instead (module docstring).
EVALUATION = SearchConfig(root_noise=False, temperature_cutoff=0)

# The slope of the Elo curve, in points per Elo, scaled out of the Newton step.
# p = 1 / (1 + 10^(-d / 400)) has dp/dd = (ln 10 / 400) · p · (1 - p), so the
# Newton step on the Elo scale carries the factor 400 / ln 10 ≈ 173.7.
_ELO_PER_NAT = 400 / math.log(10)


@dataclass(frozen=True)
class EvalConfig:
    """Which opponents a new generation plays and how many games.

    Attributes:
        opponents: Nearest previous checkpoints played (`g-1`, `g-2`, ...);
            at least 1, so every generation is linked to the one before.
        ladder: Also play generation `g - ladder`, the anti-drift match (see
            `opponents_for`). `None` (or `<= 0`) turns it off.
        games_per_colour: Openings per pairing; each is played with both
            colours, so a pairing is `2 · games_per_colour` games.
        simulations: Search simulations per move, for both players.
    """

    opponents: int = 3
    ladder: int | None = 8
    games_per_colour: int = 20
    simulations: int = 100

    def __post_init__(self) -> None:
        # At least one nearest opponent: then every generation g plays g - 1,
        # so the match graph is one chain from the anchor (generation 0) to
        # the newest generation and the Bradley-Terry fit in `elo_ratings`
        # always has a connected graph. With 0, generation 1 would play no
        # match at all and its rating would be undefined (the fit raises).
        if self.opponents < 1:
            raise ValueError(f"opponents must be at least 1, got {self.opponents}")
        if self.games_per_colour < 1:
            raise ValueError(f"games_per_colour must be at least 1, got {self.games_per_colour}")
        if self.simulations < 1:
            raise ValueError(f"simulations must be at least 1, got {self.simulations}")


def opponents_for(generation: int, config: EvalConfig) -> list[int]:
    """The generations that `generation` plays in its tournament.

    `[g-1, ..., g-opponents]` clipped at 0, then the ladder opponent
    `g - ladder` if it exists and is not already listed.

    Why the ladder: with neighbour matches only, a generation's rating reaches
    the anchor (generation 0) through a chain of links, each with about 50 Elo
    of noise at 40 games. The errors add up like a random walk, so a late
    generation's absolute rating can drift by well over 100 Elo while the
    local curve looks fine. One direct match `ladder` generations back gives
    the fit a measurement across all those links at once and pins the chain.

    A ladder of `None` or `<= 0` means no ladder: `ladder = 0` would make a
    generation its own opponent, and its checkpoint does not exist yet during
    its tournament (it is written last).
    """
    nearest = [g for g in range(generation - 1, generation - 1 - config.opponents, -1) if g >= 0]
    if config.ladder is not None and config.ladder > 0:
        far = generation - config.ladder
        if far >= 0 and far not in nearest:
            nearest.append(far)
    return nearest


@dataclass(frozen=True)
class MatchResult:
    """The outcome of one pairing, from player `a`'s perspective.

    Attributes:
        wins, draws, losses: Games `a` won, drew, lost.
        openings: The first moves used, one per pair of games.
        as_x, as_o: `(wins, draws, losses)` for `a` in the games where it
            played X / O. The totals above are their sums. The split matters
            because the colours are not equal (4x4 is an O win): a 50% score
            can be "wins every O game, loses every X game", which says the
            players are equally strong in a very different way than 50% draws.
    """

    wins: int
    draws: int
    losses: int
    openings: tuple[Cell, ...]
    as_x: tuple[int, int, int]
    as_o: tuple[int, int, int]

    def __post_init__(self) -> None:
        totals = tuple(x + o for x, o in zip(self.as_x, self.as_o, strict=True))
        if totals != (self.wins, self.draws, self.losses):
            raise ValueError(
                f"colour split {self.as_x} + {self.as_o} does not sum to "
                f"(wins, draws, losses) = {(self.wins, self.draws, self.losses)}"
            )

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """`a`'s points per game, a draw counting ½."""
        return (self.wins + self.draws / 2) / self.games


class MatchGame:
    """One game between two evaluators, each searching in its own tree.

    Why two trees: with one shared tree, `play` would hand player `a` the
    subtree that `b` just built, expanded with `b`'s priors and backed up with
    `b`'s values. `a`'s search would start from its opponent's opinions every
    ply, and the measured strength would be a blend of both networks. So each
    side has its own tree, and nothing evaluated by one network ever enters
    the other's tree.

    Trees are keyed by side (`Mark.X`, `Mark.O`), never by evaluator identity:
    the same evaluator object may play both sides (a self-match), and it still
    gets two trees.

    Lazy creation: `SearchTree.play` refuses an unexpanded root, so the second
    player's tree must not be built (and left unexpanded) before the first
    player has moved. Each tree is created from the current position on its
    owner's first turn, via `tree`.

    Attributes:
        state: The current position.
        a, b: The two players.
        a_plays: The side `a` plays; `b` plays the other.
        trees: Each side's tree, `None` until that side's first turn.
        done: The game is over.
        final: The terminal position; valid once `done`.
    """

    def __init__(
        self,
        state: State,
        a: Evaluator,
        b: Evaluator,
        a_plays: Mark,
        simulations: int,
        search: SearchConfig,
        rng: np.random.Generator,
    ) -> None:
        if simulations < 1:
            # At least one simulation, so the mover's root has a visited child
            # to choose and the chosen child is expanded (see play_if_ready).
            raise ValueError(f"simulations must be at least 1, got {simulations}")
        self.state = state
        self.a = a
        self.b = b
        self.a_plays = a_plays
        self.simulations = simulations
        self.search = search
        self.rng = rng
        self.trees: dict[Mark, SearchTree | None] = {Mark.X: None, Mark.O: None}
        self.done = is_terminal(state)
        self.final = state

    @property
    def mover(self) -> Evaluator:
        """Whose turn it is: `a` or `b`."""
        return self.a if self.state.to_move is self.a_plays else self.b

    @property
    def tree(self) -> SearchTree:
        """The mover's tree, created from the current position on its first turn."""
        side = self.state.to_move
        tree = self.trees[side]
        if tree is None:
            tree = SearchTree(self.state, self.search, self.rng)
            self.trees[side] = tree
        return tree

    def play_if_ready(self) -> None:
        """Play the mover's move once its search has spent its budget.

        The move is the most visited root child (deterministic play), applied
        with `play(move)` to *both* existing trees, so each keeps its own
        statistics below the new position as a warm start:

        - the mover's root was expanded by its own search;
        - the opponent's root is the child its own search chose last turn,
          which was visited and therefore expanded (a visited non-terminal
          leaf is always expanded, and the most visited child is visited).

        If the new root of either tree is unexpanded (the reply was never
        visited in that tree), its owner's evaluator sets it up on the next
        `select`, so it is still filled with the owner's own opinion.
        """
        if self.done:
            return
        side = self.state.to_move
        tree = self.trees[side]
        if tree is None or tree.simulations < self.simulations:
            return

        move = most_visited(tree.root)
        self.state = apply_move(self.state, move)
        if is_terminal(self.state):
            # Nothing left to search; the trees are kept as they are.
            self.done = True
            self.final = self.state
            return

        for owner, owned in self.trees.items():
            if owned is None:
                continue
            if owned.root.expanded:
                owned.play(move)
            else:
                # Guard: should not happen (see the docstring: both roots are
                # expanded when a move arrives). If it ever does, `play` would
                # raise; rebuilding from the new position loses only the warm
                # start, never correctness.
                self.trees[owner] = SearchTree(self.state, self.search, self.rng)


def _openings(size: int, games_per_colour: int, rng: np.random.Generator) -> tuple[Cell, ...]:
    """`games_per_colour` first moves: distinct cells, cycled if the board runs out."""
    cells = size * size
    drawn = rng.choice(cells, size=min(games_per_colour, cells), replace=False)
    return tuple(
        divmod(int(drawn[i % len(drawn)]), size) for i in range(games_per_colour)
    )


def play_match(
    a: Evaluator,
    b: Evaluator,
    size: int,
    games_per_colour: int,
    simulations: int,
    rng: np.random.Generator,
    search: SearchConfig = EVALUATION,
) -> MatchResult:
    """Play `2 · games_per_colour` games between `a` and `b`, in lockstep.

    Game `i` and game `i + games_per_colour` start from the same opening
    move `openings[i]` (applied with `apply_move` before the game is built);
    `a` plays X in the first and O in the second. The opening is X's move, so
    the player to move first in the search is O.

    Each step: `lockstep_step` over the trees of the games where `a` is to
    move (one `a.evaluate` batch), then over those where `b` is to move (one
    `b.evaluate` batch), then `play_if_ready` on every game.
    """
    if games_per_colour < 1:
        raise ValueError(f"games_per_colour must be at least 1, got {games_per_colour}")
    openings = _openings(size, games_per_colour, rng)
    start = initial_state(size)
    games: list[MatchGame] = []
    for a_plays in (Mark.X, Mark.O):
        for opening in openings:
            games.append(
                MatchGame(apply_move(start, opening), a, b, a_plays, simulations, search, rng)
            )

    live = [game for game in games if not game.done]
    while live:
        # Split by side to move, not by `game.mover is a`: in a self-match
        # `a is b` and identity cannot tell the two apart. `game.tree`
        # creates a mover's tree lazily on its first turn.
        a_to_move = [game for game in live if game.state.to_move is game.a_plays]
        b_to_move = [game for game in live if game.state.to_move is not game.a_plays]
        a_trees = [game.tree for game in a_to_move]
        b_trees = [game.tree for game in b_to_move]
        if a_trees:
            lockstep_step(a_trees, a)
        if b_trees:
            lockstep_step(b_trees, b)
        for game in live:
            game.play_if_ready()
        live = [game for game in live if not game.done]

    # Tally per colour of `a`: [wins, draws, losses] for a-as-X and a-as-O.
    split: dict[Mark, list[int]] = {Mark.X: [0, 0, 0], Mark.O: [0, 0, 0]}
    for game in games:
        winner = game.final.winner
        if winner is None:
            outcome = 1  # draw
        elif winner is game.a_plays:
            outcome = 0  # a won
        else:
            outcome = 2  # a lost
        split[game.a_plays][outcome] += 1
    as_x = (split[Mark.X][0], split[Mark.X][1], split[Mark.X][2])
    as_o = (split[Mark.O][0], split[Mark.O][1], split[Mark.O][2])
    return MatchResult(
        wins=as_x[0] + as_o[0],
        draws=as_x[1] + as_o[1],
        losses=as_x[2] + as_o[2],
        openings=openings,
        as_x=as_x,
        as_o=as_o,
    )


def elo_difference(score: float) -> float:
    """The Elo difference implied by a single pairing's score.

    Inverts the Elo expectation `score = 1 / (1 + 10^(-d / 400))`:
    `d = -400 · log10(1 / score - 1)`. A perfect score (0 or 1) has no finite
    answer and returns `∓inf`; callers wanting a finite number apply the
    virtual draw first (`(points + ½) / (games + 1)`).
    """
    if score >= 1.0:
        return math.inf
    if score <= 0.0:
        return -math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_ratings(
    matches: Sequence[tuple[int, int, int, int, int]], anchor: int = 0
) -> dict[int, float]:
    """Bradley-Terry maximum-likelihood Elo ratings over every match given.

    Rows are `(player, opponent, wins, draws, losses)` from `player`'s
    perspective; players are generation numbers.

    The model (Bradley-Terry on the Elo scale): player `i` scores against `j`
    with probability `p_ij = 1 / (1 + 10^((r_j - r_i) / 400))`. The fit finds
    the ratings that make the observed points most likely, using every game of
    every pairing at once, so a rating is informed by all matches, not just
    the ones it played directly.

    Why an anchor: the likelihood depends only on rating *differences*, so
    adding a constant to everyone changes nothing and the maximum is a whole
    line, not a point. Fixing `anchor` (generation 0, the random network) at
    0 picks one point on that line and keeps the scale fixed as the run grows.

    Why a regulariser: a 40-0 sweep is best explained by an infinite rating
    gap, so maximum likelihood diverges. One virtual draw per pairing (½ point
    to each side, one extra game) is the standard fix: it acts like a weak
    prior that every pair is evenly matched, keeps every rating finite, and is
    negligible once real games outnumber it.

    The fit: Newton's method on one rating at a time, Gauss-Seidel style
    (each player's step uses the already-updated ratings of the players
    before it; updating everyone at once, Jacobi style, oscillates on
    chains). For player `i` with actual points `s_i` and expected points
    `E_i = Σ_j n_ij · p_ij`, the log-likelihood's gradient is
    `(ln 10 / 400) · (s_i - E_i)` and its curvature is
    `-(ln 10 / 400)² · Σ_j n_ij · p_ij · (1 - p_ij)`, so the Newton step is

        r_i += (400 / ln 10) · (s_i - E_i) / Σ_j n_ij · p_ij · (1 - p_ij)

    Without the `400 / ln 10` factor it would be the natural-log-scale step,
    about 170 times too small. Sweeps stop once no rating moves by 0.01 Elo,
    with a cap of 10,000 sweeps.

    Raises:
        ValueError: `anchor` is not one of the players, or some players are
            not connected to it through the match graph.
    """
    # Aggregate per unordered pair: games and points for the lower-numbered
    # player. Each pair then gets exactly one virtual draw.
    games: dict[tuple[int, int], float] = defaultdict(float)
    points: dict[tuple[int, int], float] = defaultdict(float)  # for pair[0]
    for player, opponent, wins, draws, losses in matches:
        if player == opponent:
            raise ValueError(f"player {player} cannot play itself")
        pair = (min(player, opponent), max(player, opponent))
        n = wins + draws + losses
        player_points = wins + draws / 2
        games[pair] += n
        points[pair] += player_points if player == pair[0] else n - player_points

    players = sorted({p for pair in games for p in pair})
    if anchor not in players:
        raise ValueError(f"anchor {anchor} has played no match; players are {players}")

    # Connectivity: the matches only measure differences along pairs that
    # played, so a player's rating relative to the anchor exists only if a
    # chain of matches links them. A group that never (even indirectly)
    # played the anchor floats freely: the likelihood is the same wherever
    # the group sits, and the fit would hand back whatever its start (0)
    # and the sweeps happened to give. Breadth-first search from the anchor
    # over the pairs finds everyone reachable; anyone else is an error.
    neighbours: dict[int, set[int]] = defaultdict(set)
    for i, j in games:
        neighbours[i].add(j)
        neighbours[j].add(i)
    reached = {anchor}
    frontier = [anchor]
    while frontier:
        node = frontier.pop()
        for other in neighbours[node] - reached:
            reached.add(other)
            frontier.append(other)
    disconnected = [p for p in players if p not in reached]
    if disconnected:
        raise ValueError(
            f"players {disconnected} are not connected to anchor {anchor} by any "
            "chain of matches, so their ratings are undetermined"
        )

    # Per player: (opponent, games, own points), virtual draw included.
    schedule: dict[int, list[tuple[int, float]]] = {p: [] for p in players}
    actual: dict[int, float] = {p: 0.0 for p in players}
    for (i, j), n in games.items():
        n_virtual = n + 1.0
        s_i = points[(i, j)] + 0.5
        schedule[i].append((j, n_virtual))
        schedule[j].append((i, n_virtual))
        actual[i] += s_i
        actual[j] += n_virtual - s_i

    ratings = {p: 0.0 for p in players}
    for _ in range(10_000):
        largest = 0.0
        for i in players:
            if i == anchor:
                continue
            expected = 0.0
            curvature = 0.0
            for j, n in schedule[i]:
                p = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
                expected += n * p
                curvature += n * p * (1.0 - p)
            step = _ELO_PER_NAT * (actual[i] - expected) / curvature
            ratings[i] += step  # in place: later players see it this sweep
            largest = max(largest, abs(step))
        if largest < 0.01:
            break
    return ratings
