"""Game statistics under random and greedy play, for several board sizes.

Plan 01 (docs/plans/01-game-experiments.md), step 4. The numbers feed the
open constants in docs/design/open-questions.md:

- game length and draw rate -> temperature cutoff, value head type, buffer size;
- X / O win rates -> first-player advantage, parity input plane;
- average legal moves per decision -> Dirichlet noise `alpha = 10 / avg`
  (docs/design/search.md). AlphaZero scales alpha inversely with the typical
  number of legal moves so the noise stays comparably "spiky" across games.

Two policies:

- random: a uniformly random legal move.
- greedy: one-ply lookahead. Win immediately if possible; otherwise avoid
  moves that hand the opponent an immediate win; otherwise random. A second,
  less silly data point, *not* a model of strong play.

Output: a Markdown table per policy on stdout and a JSON file with every
number. The JSON is deterministic for a given seed. Wall-clock numbers
(positions per second) depend on the machine, so they go to a separate
`*.timing.json` next to it, which is *not* expected to reproduce exactly.

Usage:
    uv run python scripts/experiments/random_play.py
    uv run python scripts/experiments/random_play.py --sizes 4 6 --games 500
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from ox_zero.game import (
    Cell,
    Mark,
    State,
    apply_move,
    initial_state,
    is_terminal,
    legal_moves,
)

DEFAULT_OUT = Path(__file__).parent / "results" / "random_play.json"

# Boards at least this large get `--greedy-games-large` greedy games instead
# of `--games`, to keep the default run to a few minutes.
LARGE_SIZE = 10

# The four line directions (see rules.py). A new line through a cell can only
# be completed at a cell 1 or 2 steps away from it along one of them.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

# Every call to `apply_move` creates one position. Counted so the timing file
# can report the throughput of the pure-Python rules.
_positions_generated = 0


def _apply(state: State, move: Cell) -> State:
    global _positions_generated
    _positions_generated += 1
    return apply_move(state, move)


type Policy = Callable[[State, random.Random], Cell]


def random_policy(state: State, rng: random.Random) -> Cell:
    return rng.choice(legal_moves(state))


def greedy_policy(state: State, rng: random.Random) -> Cell:
    """One-ply lookahead: take a win, else avoid giving one away, else random.

    How the "gives the opponent a win" check stays cheap: a cell is an
    immediate win for the opponent if placing their mark there completes an
    alternating line. Placing *our* mark on `m` can only
      - remove `m` itself from the opponent's winning cells (it is now full), and
      - create new ones, all lying on a line through `m`, at most 2 steps away.
    It cannot destroy any other winning cell, because a line through an empty
    winning cell `c` already has its other two cells filled. So the opponent's
    winning cells after `m` are (winning cells now, minus `m`) plus whatever
    the up-to-16 cells around `m` add. That avoids a full scan per candidate.
    """
    moves = legal_moves(state)
    children = {move: _apply(state, move) for move in moves}

    wins = [move for move in moves if children[move].winner is not None]
    if wins:
        return rng.choice(wins)

    # The opponent's winning cells on the current board, found by pretending
    # it is their turn. `dataclasses.replace` only flips `to_move`; the state
    # is never played on, just used to ask "would this mark complete a line?".
    as_opponent = dataclasses.replace(state, to_move=state.to_move.opponent)
    threats = {move for move in moves if _apply(as_opponent, move).winner is not None}

    safe = []
    for move in moves:
        if threats - {move}:
            continue  # an existing threat survives this move
        child = children[move]
        if any(_apply(child, cell).winner is not None for cell in _nearby_empty(child, move)):
            continue  # this move creates a new threat
        safe.append(move)

    # If every move loses immediately, any of them will do.
    return rng.choice(safe or moves)


def _nearby_empty(state: State, cell: Cell) -> list[Cell]:
    """Empty cells 1 or 2 steps from `cell` along any line direction."""
    row, col = cell
    size = state.size
    found = []
    for d_row, d_col in _DIRECTIONS:
        for step in (-2, -1, 1, 2):
            r, c = row + step * d_row, col + step * d_col
            if 0 <= r < size and 0 <= c < size and state[r, c] is None:
                found.append((r, c))
    return found


POLICIES: dict[str, Policy] = {"random": random_policy, "greedy": greedy_policy}
POLICY_LABELS = {
    "random": "uniformly random legal move",
    "greedy": "one-ply lookahead: take a win, avoid giving one, else random",
}


def play_game(size: int, policy: Policy, rng: random.Random) -> tuple[State, list[int]]:
    """Play one game; return the final state and the legal-move count at each decision."""
    state = initial_state(size)
    branching = []
    while not is_terminal(state):
        branching.append(len(legal_moves(state)))
        state = _apply(state, policy(state, rng))
    return state, branching


def run(size: int, policy_name: str, games: int, seed: int) -> tuple[dict, dict]:
    """Play `games` games; return (deterministic stats, timing stats)."""
    global _positions_generated
    # One generator per (size, policy) pair, seeded from a string. String
    # seeds are hashed with SHA-512 by `random.seed`, so they are stable
    # across runs and Python processes (unlike `hash()`, which is salted).
    rng = random.Random(f"{seed}/{size}/{policy_name}")
    policy = POLICIES[policy_name]

    lengths: list[int] = []
    outcomes = {"x_wins": 0, "o_wins": 0, "draws": 0}
    all_branching: list[int] = []
    first_quarter: list[int] = []
    last_quarter: list[int] = []

    _positions_generated = 0
    start = time.perf_counter()
    for _ in range(games):
        final, branching = play_game(size, policy, rng)
        length = final.move_count
        lengths.append(length)
        if final.winner is Mark.X:
            outcomes["x_wins"] += 1
        elif final.winner is Mark.O:
            outcomes["o_wins"] += 1
        else:
            outcomes["draws"] += 1

        all_branching.extend(branching)
        # Decision i (0-based) is in the first quarter if i < L/4 and in the
        # last quarter if i >= 3L/4. Written with integer maths to avoid
        # floating-point edge cases.
        first_quarter.extend(b for i, b in enumerate(branching) if 4 * i < length)
        last_quarter.extend(b for i, b in enumerate(branching) if 4 * i >= 3 * length)
    seconds = time.perf_counter() - start

    # Deciles by the "inclusive" method: P10 and P90 are the first and last.
    deciles = statistics.quantiles(lengths, n=10, method="inclusive")
    stats = {
        "size": size,
        "policy": policy_name,
        "games": games,
        "length": {
            "mean": _r(statistics.fmean(lengths)),
            "median": _r(statistics.median(lengths)),
            "min": min(lengths),
            "max": max(lengths),
            "p10": _r(deciles[0]),
            "p90": _r(deciles[-1]),
        },
        "outcomes": {key: _r(count / games) for key, count in outcomes.items()},
        "avg_legal_moves": {
            "overall": _r(statistics.fmean(all_branching)),
            "first_quarter": _r(statistics.fmean(first_quarter)),
            "last_quarter": _r(statistics.fmean(last_quarter)),
        },
    }
    timing = {
        "size": size,
        "policy": policy_name,
        "games": games,
        "seconds": round(seconds, 2),
        "positions_generated": _positions_generated,
        "positions_per_second": round(_positions_generated / seconds),
    }
    return stats, timing


def _r(value: float) -> float:
    """Round for stable, readable output."""
    return round(value, 4)


def markdown_table(results: list[dict], policy_name: str) -> str:
    header = (
        "| Size | Games | Mean length | Median | P10 | P90 | X win % | O win % | Draw % "
        "| Avg legal moves (all / first ¼ / last ¼) |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for s in results:
        if s["policy"] != policy_name:
            continue
        length, out, legal = s["length"], s["outcomes"], s["avg_legal_moves"]
        rows.append(
            f"| {s['size']}x{s['size']} | {s['games']} | {length['mean']:.1f} | {length['median']:g} "
            f"| {length['p10']:g} | {length['p90']:g} "
            f"| {100 * out['x_wins']:.1f} | {100 * out['o_wins']:.1f} | {100 * out['draws']:.1f} "
            f"| {legal['overall']:.1f} / {legal['first_quarter']:.1f} / {legal['last_quarter']:.1f} |"
        )
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6, 8, 12])
    parser.add_argument("--games", type=int, default=2000, help="games per size and policy")
    parser.add_argument(
        "--greedy-games-large",
        type=int,
        default=500,
        help=f"games for the greedy policy on boards {LARGE_SIZE}x{LARGE_SIZE} and up"
        " (its lookahead costs ~0.25 s per 12x12 game in pure Python)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["random", "greedy", "both"], default="both")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    policies = ["random", "greedy"] if args.policy == "both" else [args.policy]
    parameters = {
        "sizes": args.sizes,
        "games": args.games,
        "greedy_games_large": args.greedy_games_large,
        "large_size": LARGE_SIZE,
        "seed": args.seed,
        "policies": {name: POLICY_LABELS[name] for name in policies},
    }
    print(f"Parameters: {json.dumps(parameters)}\n")

    results, timings = [], []
    for policy_name in policies:
        for size in args.sizes:
            games = args.games
            if policy_name == "greedy" and size >= LARGE_SIZE:
                games = min(games, args.greedy_games_large)
            stats, timing = run(size, policy_name, games, args.seed)
            results.append(stats)
            timings.append(timing)

    for policy_name in policies:
        print(f"## {policy_name} ({POLICY_LABELS[policy_name]})\n")
        print(markdown_table(results, policy_name))
        print()

    print("## Throughput (machine-dependent)\n")
    print("| Policy | Size | Games | Seconds | Positions | Positions/s |\n|---|---|---|---|---|---|")
    for t in timings:
        print(
            f"| {t['policy']} | {t['size']}x{t['size']} | {t['games']} | {t['seconds']} "
            f"| {t['positions_generated']} | {t['positions_per_second']} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"parameters": parameters, "results": results}, indent=2) + "\n")
    timing_out = args.out.with_suffix(".timing.json")
    timing_out.write_text(json.dumps({"parameters": parameters, "timing": timings}, indent=2) + "\n")
    print(f"\nWrote {args.out} and {timing_out}")


if __name__ == "__main__":
    main()
