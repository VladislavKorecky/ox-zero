"""Compare training runs: one Markdown row per run, or one metric over generations.

Plan 06 (docs/plans/06-experiment-harness.md), step 4. Reads each run's
`metrics.jsonl` (and the cross-run benchmark rating when it exists) and prints
a table that can be pasted into an experiment file (docs/experiments/) as is.
It computes nothing new: dashboards are readers of the metrics files.

The `elo` column is the run's *own* tournament rating (anchored at that run's
generation 0, comparable within a run only). The `bench` column is the rating
of exactly the summarised checkpoint on the shared per-board-size scale from
`scripts/benchmark.py`, anchored at `uniform@100`; blank when unrated. Only
`bench` supports a claim that one run beat another.

A directory without `metrics.jsonl` is skipped with a notice, so
`compare_runs.py runs/*` works with `runs/benchmarks/` in the glob.

Usage:
    uv run python scripts/compare_runs.py runs/six-a runs/six-b
    uv run python scripts/compare_runs.py runs/six-a runs/six-b --at 20
    uv run python scripts/compare_runs.py runs/six-a runs/six-b --key loss_value
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ox_zero.training.benchmark import benchmark_dir, rating_table, read_rows
from ox_zero.training.checkpoint import checkpoint_size
from ox_zero.training.metrics import (
    MATCHES_FILE,
    METRICS_FILE,
    RATINGS_FILE,
    read_metrics,
    summarise,
)
from ox_zero.training.run import checkpoint_name

COLUMNS = (
    ("generation", "gen", "{:d}"),
    ("elo", "elo", "{:+.0f}"),
    ("loss_total", "loss", "{:.3f}"),
    ("loss_policy", "policy", "{:.3f}"),
    ("loss_value", "value", "{:.3f}"),
    ("mean_game_length", "len", "{:.1f}"),
    ("x_win_rate", "X win", "{:.0%}"),
    ("draw_rate", "draw", "{:.0%}"),
    ("wall_seconds_mean5", "s/gen", "{:.0f}"),
    ("simulations_per_second_mean5", "sim/s", "{:.0f}"),
)


def bench_rating(run_dir: Path, generation: int, benchmarks: Path) -> str:
    """The benchmark rating of `<run>/gen_NNN` as text, `""` when unrated."""
    checkpoint = run_dir / checkpoint_name(generation)
    if not checkpoint.is_file():
        return ""
    directory = benchmark_dir(benchmarks, checkpoint_size(checkpoint))
    name = f"{run_dir.resolve().name}/gen_{generation:03d}"
    ratings = directory / RATINGS_FILE
    if ratings.is_file():
        player = json.loads(ratings.read_text()).get("players", {}).get(name)
        if player:
            return f"{player['elo']:+.0f} [{player['low']:+.0f}, {player['high']:+.0f}]"
    matches = directory / MATCHES_FILE
    if matches.is_file():
        rows = read_rows(matches)
        if rows:
            try:
                elo = rating_table(rows).get(name)
            except ValueError:  # e.g. the anchor has no games: the column is optional
                return ""
            if elo is not None:
                return f"{elo:+.0f}"  # refit from the log: no interval
    return ""


def markdown(header: list[str], body: list[list[str]]) -> str:
    lines = [header, ["---"] * len(header), *body]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--at", type=int, help="summarise this generation (default: the last)")
    parser.add_argument("--key", help="print this metrics key per generation, one column per run")
    parser.add_argument("--benchmarks", type=Path, default=Path("runs/benchmarks"))
    args = parser.parse_args()

    runs: dict[str, tuple[Path, list[dict]]] = {}
    for directory in args.runs:
        path = directory / METRICS_FILE
        if not path.is_file():
            print(f"skipping {directory}: no {METRICS_FILE}", file=sys.stderr)
            continue
        name = directory.resolve().name
        if name in runs:
            parser.error(f"two run directories are both called {name!r}; rename one")
        runs[name] = (directory, read_metrics(path))
    if not runs:
        parser.error("none of the given directories has a metrics.jsonl")

    if args.key:
        generations = sorted({row["generation"] for _, rows in runs.values() for row in rows})
        by_run = {
            name: {row["generation"]: row.get(args.key) for row in rows}
            for name, (_, rows) in runs.items()
        }
        body = [
            [str(g)] + [
                "" if by_run[name].get(g) is None else f"{by_run[name][g]:.4g}" for name in runs
            ]
            for g in generations
        ]
        print(markdown(["gen", *runs], body))
        return

    body = []
    for name, (directory, rows) in runs.items():
        try:
            summary = summarise(rows, args.at)
        except ValueError as error:
            parser.error(f"{name}: {error}")
        body.append([
            name,
            *(fmt.format(summary[key]) for key, _, fmt in COLUMNS),
            bench_rating(directory, summary["generation"], args.benchmarks),
        ])
    print(markdown(["run", *(label for _, label, _ in COLUMNS), "bench"], body))


if __name__ == "__main__":
    main()
