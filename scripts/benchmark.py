"""Rate one checkpoint on the shared Elo scale: play it against the frozen reference pool.

Plan 06 (docs/plans/06-experiment-harness.md), step 3. A run's own tournament
rates generations against each other, anchored at that run's generation 0, so
two runs' ratings cannot be compared. This script plays one checkpoint against
every reference of its board size (`runs/benchmarks/<S>x<S>/pool/*.pt` plus the
uniform search at 100 and 400 simulations), appends the games to that size's
match log, refits one Bradley-Terry rating per player over the whole log
(anchored at `uniform@100` = 0 Elo) and prints the candidate's rating with a
bootstrap interval. See `ox_zero.training.benchmark` for the method.

Every pairing plays every cell of the board as the opening, once with each
colour, with the deterministic evaluation search: one pairing is one
measurement, so a pairing already in the log is skipped, never replayed.

A non-default *search* (an upgrade under test) is rated with `--tag NAME` plus
the flags that change it (`--c-init`, `--c-base`); the player is then
`<run>/gen_NNN@NAME`, which shares weights with the untagged player but is a
separate rating. References are always played with the baseline search.

The official log is CPU-only (float32 results differ between devices), so a
device other than `cpu` needs a scratch `--root`.

Usage:
    uv run python scripts/benchmark.py --model runs/six-a/gen_020.pt
    uv run python scripts/benchmark.py --model runs/x/gen_020.pt --tag c3 --c-init 3.0
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ox_zero.engine.network_evaluator import NetworkEvaluator, select_device
from ox_zero.training.benchmark import (
    ANCHOR,
    CANDIDATE_SIMULATIONS,
    BenchmarkRow,
    append_rows,
    benchmark_dir,
    candidate_name,
    check_consistency,
    file_sha256,
    list_pool,
    load_pool,
    pairings,
    read_rows,
    search_delta,
    write_ratings,
)
from ox_zero.training.checkpoint import checkpoint_size, load_checkpoint
from ox_zero.training.evaluate import EVALUATION, play_match
from ox_zero.training.metrics import MATCHES_FILE, RATINGS_FILE

DEFAULT_ROOT = Path("runs/benchmarks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="checkpoint to rate (sets the board size)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="benchmark directory")
    parser.add_argument("--device", default="cpu", help="cpu (the official log) | auto | mps | cuda")
    parser.add_argument("--tag", help="name suffix for a non-default search (needs --c-init or --c-base)")
    # Duplicated from scripts/train.py rather than shared through a helper
    # module: scripts are loaded by path in the tests, so imports between them
    # would not resolve.
    parser.add_argument("--c-init", type=float, default=None, help="PUCT c_init for the candidate")
    parser.add_argument("--c-base", type=float, default=None, help="PUCT c_base for the candidate")
    args = parser.parse_args()
    print(f"Parameters: {vars(args)}")

    # --- Usage errors, before any checkpoint is read ---
    given = {
        field: value
        for field, value in (("c_init", args.c_init), ("c_base", args.c_base))
        if value is not None
    }
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("--tag may contain only letters, digits, '_', '.' and '-'")
    if args.tag and not given:
        parser.error("--tag needs --c-init or --c-base (a tag names a non-default search)")
    if given and not args.tag:
        parser.error("--c-init / --c-base need --tag, which names the player they define")
    if args.device != "cpu" and args.root.resolve() == DEFAULT_ROOT.resolve():
        parser.error(
            "the official log is CPU-only (float32 results differ between devices); "
            "use --device cpu, or a scratch --root"
        )
    try:
        search = dataclasses.replace(EVALUATION, **given)
    except ValueError as error:
        parser.error(str(error))
    delta = search_delta(search)
    if args.tag and not delta:
        parser.error("the search flags given equal the defaults; drop --tag, or change a value")

    try:
        size = checkpoint_size(args.model)
    except ValueError as error:
        parser.error(str(error))
    directory = benchmark_dir(args.root, size)
    if not (directory / "pool").is_dir():
        parser.error(
            f"no reference pool at {directory / 'pool'}; see docs/experiments/pools.md "
            "for how a pool is seeded"
        )
    try:
        references = list_pool(directory)
    except ValueError as error:
        parser.error(str(error))

    matches_path = directory / MATCHES_FILE
    existing = read_rows(matches_path)
    device = select_device(args.device)
    devices = {row.device for row in existing} | {str(device)}
    if len(devices) > 1:
        parser.error(f"the log {matches_path} was played on {sorted(devices - {str(device)})}, not {device}")

    # --- Identity: before playing, so a bad name costs nothing ---
    name = candidate_name(args.model, args.tag)
    sha = file_sha256(args.model)
    probe = BenchmarkRow(
        a=name, b="(probe)", wins=0, draws=0, losses=0, as_x=(0, 0, 0), as_o=(0, 0, 0),
        openings=0, a_simulations=CANDIDATE_SIMULATIONS, b_simulations=0, device=str(device),
        date="", a_sha256=sha, b_sha256=None, a_search=delta,
    )
    try:
        check_consistency([*existing, probe], references)
    except ValueError as error:
        parser.error(str(error))

    todo = pairings(name, references, existing)
    for skipped in (r for r in references if r.name != name and r not in todo):
        print(f"Skipping {name} vs {skipped.name}: already in {matches_path}")
    if not todo:
        parser.error(f"every pairing of {name} is already logged in {matches_path}")

    # --- Play ---
    try:
        candidate = load_checkpoint(args.model, device)
        evaluators = load_pool(todo, size, device)
    except ValueError as error:
        sys.exit(f"error: {error}")
    player = NetworkEvaluator(candidate.network, device)
    print(f"{name} ({size}x{size}) vs {[r.name for r in todo]} on {device}")

    scores: dict[str, tuple[BenchmarkRow, float]] = {}
    for reference in todo:
        # Every cell of the board as the opening, so the full set is the
        # measurement; the generator only orders them, which does not matter.
        result = play_match(
            player, evaluators[reference.name], size, size * size,
            CANDIDATE_SIMULATIONS, np.random.default_rng(0), search,
            b_simulations=reference.simulations, b_search=EVALUATION,
        )
        row = BenchmarkRow(
            a=name, b=reference.name,
            wins=result.wins, draws=result.draws, losses=result.losses,
            as_x=result.as_x, as_o=result.as_o, openings=len(result.openings),
            a_simulations=CANDIDATE_SIMULATIONS, b_simulations=reference.simulations,
            device=str(device), date=datetime.now().isoformat(timespec="seconds"),
            a_sha256=sha, b_sha256=reference.sha256, a_search=delta,
        )
        try:
            check_consistency([*read_rows(matches_path), row], references)
            append_rows(matches_path, [row])
        except ValueError as error:
            sys.exit(f"error: {error}")
        scores[reference.name] = (row, result.score)
        print(f"  played {reference.name}: {result.wins}W {result.draws}D {result.losses}L", flush=True)

    # --- Rate and report ---
    rows = read_rows(matches_path)
    ratings = write_ratings(directory / RATINGS_FILE, rows)
    mine = ratings["players"][name]  # already fit and bootstrapped by write_ratings
    print(
        f"\n{name}: {mine['elo']:+.0f} Elo [{mine['low']:+.0f}, {mine['high']:+.0f}] "
        f"(95% bootstrap, anchor {ANCHOR} = 0)"
    )
    for reference_name, (row, score) in scores.items():
        # A sweep only bounds the true gap, so the score is marked.
        bound = "≥" if score == 1.0 else "≤" if score == 0.0 else " "
        print(
            f"  vs {reference_name:<16} {bound}{score:.3f}  "
            f"(X {row.as_x[0]}W {row.as_x[1]}D {row.as_x[2]}L, "
            f"O {row.as_o[0]}W {row.as_o[1]}D {row.as_o[2]}L)"
        )
    print("Pool:")
    for reference in references:
        info = ratings["players"].get(reference.name)
        if info:
            print(f"  {reference.name:<16} {info['elo']:+.0f} [{info['low']:+.0f}, {info['high']:+.0f}]")


if __name__ == "__main__":
    main()
