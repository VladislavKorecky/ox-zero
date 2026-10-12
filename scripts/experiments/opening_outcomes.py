"""Per-opening outcomes of near-strength network pairings, by opening class.

Unlike its neighbours this script measures trained networks, not the game
alone: it asks how informative each first move is as a benchmark opening.
Results are recorded in docs/experiments/02-benchmark-openings.md.

Why openings matter for the benchmark
-------------------------------------
The benchmark (`scripts/benchmark.py`) plays every cell as X's first move,
twice, with the colours swapped. Call the two games from one opening a *pair*.
If one player wins both games, the pair says that player is stronger: it is
*informative*. If each colour wins its game (X twice, or O twice), the pair
splits 1-1 whatever the players' strengths: it measures the opening, not the
players. An opening that is lost for X by force (see `forced_openings.py`)
splits 1-1 whenever both players find the win, so its pairs are informative
only when someone misses it.

Opening classes, by distance to the nearest edge: `corner` and `edge` (on the
edge), `ring2` (one step in; `forced_openings.py` proves these lost for X on
8x8 and 12x12, other sizes are unchecked), `interior` (everything else).

Method: every pairing of the given checkpoints (same board size), every cell
as the opening with both colours, the deterministic evaluation search of the
tournament (`training.evaluate.EVALUATION`) at `--simulations`. Per class: the
number of pairs, X's share of the games won, and the shares of informative,
O-won-both and X-won-both pairs.

Output: a Markdown table on stdout and a JSON file with every pair. The JSON
is deterministic for a given device; wall-clock times go to `*.timing.json`.

Usage:
    uv run python scripts/experiments/opening_outcomes.py --device mps \\
        runs/eight-a/gen_030.pt runs/eight-b/gen_030.pt runs/eight-c/gen_030.pt runs/eight-d/gen_030.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ox_zero.game.rules import Cell

DEFAULT_OUT = Path(__file__).parent / "results" / "opening_outcomes.json"
CLASSES = ("corner", "edge", "ring2", "interior")


def opening_class(cell: Cell, size: int) -> str:
    """`corner`, `edge`, `ring2` (one step in from the edge) or `interior`."""
    r, c = cell
    m = size - 1
    if r in (0, m) and c in (0, m):
        return "corner"
    depth = min(r, c, m - r, m - c)  # steps to the nearest edge
    return "edge" if depth == 0 else "ring2" if depth == 1 else "interior"


def pair_kind(a_as_x: int, a_as_o: int) -> str:
    """Classify one opening's pair from player `a`'s results (+1 / 0 / -1).

    `informative`: one player won both games. `o_both` / `x_both`: the same
    colour won both, so the pair split 1-1. `other`: a draw is involved.
    """
    if 0 in (a_as_x, a_as_o):
        return "other"
    if a_as_x == a_as_o:
        return "informative"
    return "o_both" if a_as_x == -1 else "x_both"


def x_wins(a_as_x: int, a_as_o: int) -> int:
    """Games of the pair won by the player with X: `a` winning as X, plus `a`
    losing as O (then `b`, as X, won)."""
    return (a_as_x == 1) + (a_as_o == -1)


def tabulate(rows: list[dict]) -> dict[str, dict]:
    """Per class (and `all`): pairs, X's win rate over games, and the shares
    of informative, O-won-both and X-won-both pairs."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["class"]].append(row)
        groups["all"].append(row)
    table = {}
    for name, group in groups.items():
        kinds = [pair_kind(r["a_as_x"], r["a_as_o"]) for r in group]
        n = len(group)
        table[name] = {
            "pairs": n,
            "x_win_rate": sum(x_wins(r["a_as_x"], r["a_as_o"]) for r in group) / (2 * n),
            "informative": kinds.count("informative") / n,
            "o_both": kinds.count("o_both") / n,
            "x_both": kinds.count("x_both") / n,
        }
    return table


def play_pairs(paths: list[Path], device_name: str, simulations: int) -> list[dict]:
    """Every pairing of the checkpoints, every opening with both colours."""
    # Imported here so the pure helpers above (and their tests) need no torch.
    from ox_zero.engine.network_evaluator import NetworkEvaluator, select_device
    from ox_zero.training.checkpoint import checkpoint_size, load_checkpoint
    from ox_zero.training.evaluate import EVALUATION, play_match

    device = select_device(device_name)
    sizes = {checkpoint_size(p) for p in paths}
    if len(sizes) != 1:
        raise SystemExit(f"checkpoints of different board sizes: {sorted(sizes)}")
    size = sizes.pop()
    players = {str(p): NetworkEvaluator(load_checkpoint(p, device).network, device) for p in paths}
    rows = []
    for a, b in itertools.combinations(players, 2):
        # Every cell as the opening, as in the benchmark; the generator only
        # orders them.
        result = play_match(
            players[a], players[b], size, size * size, simulations, np.random.default_rng(0), EVALUATION
        )
        n = len(result.openings)
        for i, cell in enumerate(result.openings):
            # Game i has `a` as X, game i + n the same opening with `a` as O.
            rows.append({
                "a": a, "b": b, "cell": list(cell), "class": opening_class(cell, size),
                "a_as_x": result.results[i], "a_as_o": result.results[i + n],
            })
        print(f"  {a} vs {b}: {result.wins}W {result.draws}D {result.losses}L", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", type=Path, nargs="+", help="two or more, one board size")
    parser.add_argument("--device", default="cpu", help="cpu | auto | mps | cuda")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        parser.error("need at least two checkpoints")
    if len({p.resolve() for p in args.checkpoints}) != len(args.checkpoints):
        parser.error("a checkpoint is given twice")

    parameters = {
        "checkpoints": [str(p) for p in args.checkpoints],
        "device": args.device,
        "simulations": args.simulations,
    }
    print(f"Parameters: {json.dumps(parameters)}\n")
    start = time.perf_counter()
    rows = play_pairs(args.checkpoints, args.device, args.simulations)
    seconds = round(time.perf_counter() - start, 1)
    table = tabulate(rows)

    print("\n| Opening class | Pairs | X wins | Informative | O won both | X won both |")
    print("|---|---|---|---|---|---|")
    for name in (*CLASSES, "all"):
        if name in table:
            t = table[name]
            print(
                f"| {name} | {t['pairs']} | {t['x_win_rate']:.1%} | {t['informative']:.1%} "
                f"| {t['o_both']:.1%} | {t['x_both']:.1%} |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"parameters": parameters, "table": table, "pairs": rows}, indent=2) + "\n"
    )
    timing_out = args.out.with_suffix(".timing.json")
    timing_out.write_text(json.dumps({"parameters": parameters, "seconds": seconds}, indent=2) + "\n")
    print(f"\nWrote {args.out} and {timing_out}")


if __name__ == "__main__":
    main()
