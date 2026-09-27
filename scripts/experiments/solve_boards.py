"""Solve the empty board exactly for small sizes.

Plan 01 (docs/plans/archive/01-game-experiments.md), step 4. For each size, runs the
exact negamax solver (`ox_zero.game.Solver`) on the empty board and reports:

- the game value, as "X wins" / "draw" / "O wins" (X moves first, so the
  solver's side-to-move value at the empty board is X's value);
- every optimal first move for X;
- how many distinct positions the solver had to cache to prove the value;
- the length of one optimal game, where both sides always play the *first*
  optimal move in board order. It is one line among many and the winner is
  not trying to win quickly, so treat it as a rough indicator of how long
  games between strong players might be, compared with random ones;
- the length distribution over `--sample-games` optimal games in which both
  sides pick uniformly at random among their optimal moves. Perfect play is
  not unique, so this is the honest version of the previous number: it shows
  how much of the board strong play actually uses. Caveat: in a lost
  position every move is "optimal" (they all lose), so the losing side plays
  uniformly at random and does not defend stubbornly. The sample therefore
  underestimates the length of games between two strong players; it is a
  floor, not an estimate.

The JSON is deterministic. Wall-clock times go to a separate
`*.timing.json`, which is *not* expected to reproduce exactly.

Usage:
    uv run python scripts/experiments/solve_boards.py
    uv run python scripts/experiments/solve_boards.py --sizes 5 --out /tmp/5x5.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from ox_zero.game import Solver, apply_move, format_cell, initial_state, is_terminal

DEFAULT_OUT = Path(__file__).parent / "results" / "solve_boards.json"

# The value at the empty board is for the side to move there, which is X.
VALUE_LABELS = {1: "X wins", 0: "draw", -1: "O wins"}


def solve(size: int, sample_games: int, seed: int) -> tuple[dict, dict]:
    """Solve one board size; return (deterministic results, timing)."""
    solver = Solver()
    start = initial_state(size)

    began = time.perf_counter()
    value = solver.value(start)
    solve_seconds = time.perf_counter() - began
    # Read before `best_moves` and the optimal line, which score extra
    # children and grow the cache: this is the cost of proving the value.
    positions_solved = solver.positions_solved

    began = time.perf_counter()
    best_first_moves = solver.best_moves(start)

    # Play out one optimal game: both sides take the first best move.
    state, line = start, []
    while not is_terminal(state):
        move = solver.best_moves(state)[0]
        line.append(move)
        state = apply_move(state, move)
    extra_seconds = time.perf_counter() - began

    # Sample optimal games with random tie-breaking. `best_moves` scores every
    # child, and the sample wanders off the lines the `+1` short-circuit
    # explored, so the cache keeps growing (about 20 s on 4x4).
    began = time.perf_counter()
    rng = random.Random(f"{seed}/{size}/optimal")  # string seeds are stable across processes
    lengths = []
    for _ in range(sample_games):
        state, k = start, 0
        while not is_terminal(state):
            state = apply_move(state, rng.choice(solver.best_moves(state)))
            k += 1
        lengths.append(k)
    sample_seconds = time.perf_counter() - began
    deciles = statistics.quantiles(lengths, n=10, method="inclusive")

    result = {
        "size": size,
        "value": value,
        "outcome": VALUE_LABELS[value],
        "best_first_moves": [format_cell(move) for move in best_first_moves],
        "positions_solved": positions_solved,
        "optimal_line": [format_cell(move) for move in line],
        "optimal_line_length": len(line),
        "optimal_line_winner": None if state.winner is None else state.winner.value,
        "optimal_play_sample": {
            "games": sample_games,
            "length": {
                "mean": round(statistics.fmean(lengths), 4),
                "median": round(statistics.median(lengths), 4),
                "min": min(lengths),
                "max": max(lengths),
                "p10": round(deciles[0], 4),
                "p90": round(deciles[-1], 4),
            },
            # Share of the board's cells filled in an average optimal game.
            "board_used": round(statistics.fmean(lengths) / size**2, 4),
        },
    }
    timing = {
        "size": size,
        "solve_seconds": round(solve_seconds, 3),
        "best_moves_and_line_seconds": round(extra_seconds, 3),
        "sample_seconds": round(sample_seconds, 3),
        "positions_solved_total": solver.positions_solved,
    }
    return result, timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--sample-games", type=int, default=500, help="optimal games with random tie-breaks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    parameters = {"sizes": args.sizes, "sample_games": args.sample_games, "seed": args.seed}
    print(f"Parameters: {json.dumps(parameters)}\n")

    results, timings = [], []
    for size in args.sizes:
        result, timing = solve(size, args.sample_games, args.seed)
        results.append(result)
        timings.append(timing)

    print(
        "| Size | Value | Best first moves | Positions solved | Solve time (s) "
        "| Optimal-play length | Sampled optimal length: mean (P10..P90) | Board used |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for r, t in zip(results, timings):
        moves = " ".join(r["best_first_moves"])
        if len(r["best_first_moves"]) == r["size"] ** 2:
            moves = f"all {r['size'] ** 2}"
        sample = r["optimal_play_sample"]
        print(
            f"| {r['size']}x{r['size']} | {r['outcome']} | {moves} | {r['positions_solved']} "
            f"| {t['solve_seconds']} | {r['optimal_line_length']} "
            f"| {sample['length']['mean']:.1f} ({sample['length']['p10']:g}..{sample['length']['p90']:g}) "
            f"| {100 * sample['board_used']:.0f}% |"
        )
    for r in results:
        print(f"\n{r['size']}x{r['size']} optimal line: {' '.join(r['optimal_line'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"parameters": parameters, "results": results}, indent=2) + "\n")
    timing_out = args.out.with_suffix(".timing.json")
    timing_out.write_text(json.dumps({"parameters": parameters, "timing": timings}, indent=2) + "\n")
    print(f"\nWrote {args.out} and {timing_out}")


if __name__ == "__main__":
    main()
