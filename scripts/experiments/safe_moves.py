"""How many safe moves are left as a game goes on, per board size.

A follow-up to plan 01 (docs/plans/archive/01-game-experiments.md), prompted by
its results. The random-play numbers count *legal* moves, but in OXOX most
legal moves soon lose on the spot: placing a mark next to another creates
alternating patterns the opponent can complete. What a player really chooses
from is the set of *safe* moves, those that do not hand the opponent an
immediate win. This script measures how that set shrinks with the move number.

Why it matters for the design (docs/design/open-questions.md):

- If the safe set shrinks to a handful of cells long before the board is
  full, the game is decided by who runs out of safe moves first. That is a
  counting (parity) problem, which convolutions cannot do on their own, and
  it argues for global pooling in the network (docs/design/upgrades.md).
- Games under any non-blundering policy end when the side to move has zero
  safe moves. The move number at which that happens under greedy play shows
  how long a player lasts *without* building structure. Players who build
  walls (thick same-mark blocks, which contain no alternating triple) keep
  their supply of safe moves and last much longer.
- Wasting search simulations on immediately losing moves is expensive when
  they are the majority. That is the case for proof propagation
  (MCTS-Solver) in the upgrades list.

Method: greedy games (take a win, else a uniformly random safe move, else a
uniformly random move), the same policy as in `random_play.py`. At every
decision where no win is available, record the number of safe moves. Per
move number `k` (0-based, so `S² - k` cells are empty) the script reports how
many games were still running, the mean and minimum safe count, the share of
decisions with *no* safe move (a forced loss), and the share where a win was
available instead.

Output: a Markdown table per size on stdout (every `--step`-th move number)
and a JSON file with every move number. The JSON is deterministic for a given
seed; wall-clock times go to a separate `*.timing.json`.

Usage:
    uv run python scripts/experiments/safe_moves.py
    uv run python scripts/experiments/safe_moves.py --sizes 6 --games 200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

# `random_play.py` sits next to this file, and running a script puts its
# directory first on `sys.path`, so a plain import works without a package.
from random_play import classify_moves

from ox_zero.game import apply_move, initial_state, is_terminal, legal_moves

DEFAULT_OUT = Path(__file__).parent / "results" / "safe_moves.json"


def run(size: int, games: int, seed: int) -> tuple[dict, dict]:
    """Play `games` greedy games; return (deterministic stats, timing)."""
    rng = random.Random(f"{seed}/{size}/safe")  # string seeds are stable across processes

    # Per move number: the safe counts recorded, and how often a win was available.
    safe_counts: dict[int, list[int]] = {}
    wins_available: dict[int, int] = {}
    lengths: list[int] = []

    start = time.perf_counter()
    for _ in range(games):
        state = initial_state(size)
        k = 0
        while not is_terminal(state):
            wins, safe = classify_moves(state)
            if wins:
                wins_available[k] = wins_available.get(k, 0) + 1
                move = rng.choice(wins)
            else:
                safe_counts.setdefault(k, []).append(len(safe))
                move = rng.choice(safe or legal_moves(state))
            state = apply_move(state, move)
            k += 1
        lengths.append(k)
    seconds = time.perf_counter() - start

    by_move = []
    for k in sorted(set(safe_counts) | set(wins_available)):
        counts = safe_counts.get(k, [])
        won = wins_available.get(k, 0)
        alive = len(counts) + won
        by_move.append(
            {
                "move": k,
                "legal": size * size - k,
                "games_alive": alive,
                "safe_mean": _r(statistics.fmean(counts)) if counts else None,
                "safe_min": min(counts) if counts else None,
                "safe_max": max(counts) if counts else None,
                # Of the decisions at this move number: forced losses and wins.
                "forced_loss_rate": _r(counts.count(0) / alive),
                "win_available_rate": _r(won / alive),
            }
        )

    stats = {
        "size": size,
        "games": games,
        "mean_length": _r(statistics.fmean(lengths)),
        "by_move": by_move,
    }
    timing = {"size": size, "games": games, "seconds": round(seconds, 2)}
    return stats, timing


def _r(value: float) -> float:
    return round(value, 4)


def markdown_table(stats: dict, step: int) -> str:
    header = (
        "| Move | Games alive | Legal | Safe: mean | Safe: min | Forced loss % | Win available % |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = []
    for row in stats["by_move"]:
        if row["move"] % step != 0:
            continue
        # Stop once fewer than a tenth of the games are still running.
        if row["games_alive"] < stats["games"] / 10:
            break
        mean = "-" if row["safe_mean"] is None else f"{row['safe_mean']:.1f}"
        low = "-" if row["safe_min"] is None else str(row["safe_min"])
        rows.append(
            f"| {row['move']} | {row['games_alive']} | {row['legal']} | {mean} | {low} "
            f"| {100 * row['forced_loss_rate']:.1f} | {100 * row['win_available_rate']:.1f} |"
        )
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 12])
    parser.add_argument("--games", type=int, default=500, help="greedy games per size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=0, help="print every STEP-th move; 0 = size // 3")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    parameters = {"sizes": args.sizes, "games": args.games, "seed": args.seed}
    print(f"Parameters: {json.dumps(parameters)}\n")

    results, timings = [], []
    for size in args.sizes:
        stats, timing = run(size, args.games, args.seed)
        results.append(stats)
        timings.append(timing)
        step = args.step or max(1, size // 3)
        print(f"## {size}x{size} ({stats['games']} greedy games, mean length {stats['mean_length']:.1f})\n")
        print(markdown_table(stats, step))
        print()

    print("## Wall time (machine-dependent)\n")
    for t in timings:
        print(f"- {t['size']}x{t['size']}: {t['games']} games in {t['seconds']} s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"parameters": parameters, "results": results}, indent=2) + "\n")
    timing_out = args.out.with_suffix(".timing.json")
    timing_out.write_text(json.dumps({"parameters": parameters, "timing": timings}, indent=2) + "\n")
    print(f"\nWrote {args.out} and {timing_out}")


if __name__ == "__main__":
    main()
