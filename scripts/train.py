"""Train an OXOX network by AlphaZero self-play: one run directory, resumable.

Plan 04 (docs/plans/04-training-pipeline.md), step 7. A thin wrapper around
`ox_zero.training.run.run`: it turns flags into a `RunConfig`, prints it,
starts (or resumes) the run, and prints one line per finished generation.

Every default is a provisional 6x6 constant from the plan's Decisions table
("Provisional constants for 6x6"). The run lives in `<root>/<name>/`
(`runs/` by default, not `checkpoints/`: the CLI auto-loads from there and is
12x12 only). Rerunning the same command resumes from the newest
`gen_NNN.pt`; raising `--generations` extends a finished run. Any other
changed flag is refused by `run` (a changed configuration is a new run).

Ctrl-C is safe at any moment: the checkpoint is written last in each
generation, so an interrupted generation is simply re-run on resume.

Usage:
    uv run python scripts/train.py --size 6
    uv run python scripts/train.py --name six-a --generations 20 --device mps
    tensorboard --logdir runs/6x6-seed0/tensorboard
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

from ox_zero.engine.network import NetworkConfig
from ox_zero.engine.network_evaluator import select_device
from ox_zero.engine.search import SELF_PLAY
from ox_zero.training.evaluate import EvalConfig
from ox_zero.training.metrics import METRICS_FILE, read_metrics
from ox_zero.training.run import (
    ConfigMismatchError,
    RunConfig,
    RunDirectoryError,
    checkpoint_name,
    latest_generation,
    run,
)
from ox_zero.training.selfplay import SelfPlayConfig
from ox_zero.training.trainer import TrainConfig


def format_row(row: dict[str, Any]) -> str:
    """One metrics row as a single compact line."""
    scores = " ".join(f"g{opp}:{score:.2f}" for opp, score in row["scores"].items())
    return (
        f"gen {row['generation']:3d} | "
        f"{row['games']} games, len {row['mean_game_length']:.1f}, "
        f"draw {row['draw_rate']:.0%}, X {row['x_win_rate']:.0%} | "
        f"{row['simulations_per_second']:.0f} sim/s | "
        f"loss {row['loss_total']:.3f} (p {row['loss_policy']:.3f}, v {row['loss_value']:.3f}) | "
        f"elo {row['elo']:+.0f} [{scores}] | "
        f"{row['wall_seconds']:.0f}s"
    )


class GenerationPrinter(threading.Thread):
    """Prints each generation's metrics row once that generation is committed.

    `run` returns only at the end, so this thread polls `metrics.jsonl`
    alongside it. A row is printed only once its `gen_NNN.pt` exists: the
    checkpoint is the run's commit marker, so a row written by a generation
    that is then interrupted (and which resume would drop) is never shown.
    Rows already committed before this invocation are skipped.
    """

    def __init__(self, run_dir: Path, after: int) -> None:
        super().__init__(daemon=True)
        self.run_dir = run_dir
        self.printed = after
        self.stop = threading.Event()

    def poll(self) -> None:
        path = self.run_dir / METRICS_FILE
        try:
            rows = read_metrics(path) if path.exists() else []
        except (OSError, ValueError):
            return  # caught mid-rewrite; try again on the next poll
        for row in rows:
            generation = row["generation"]
            if generation <= self.printed:
                continue
            if not (self.run_dir / checkpoint_name(generation)).is_file():
                break  # not committed yet
            print(format_row(row), flush=True)
            self.printed = generation

    def run(self) -> None:
        while not self.stop.wait(1.0):
            self.poll()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", help="run directory name (default: <size>x<size>-seed<seed>)")
    parser.add_argument("--size", type=int, default=6, help="board size")
    parser.add_argument("--generations", type=int, default=20, help="target generation")
    parser.add_argument("--seed", type=int, default=0, help="seeds the weights and the run's RNG")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="parent of run directories")
    parser.add_argument("--no-tensorboard", action="store_true", help="skip TensorBoard event files")
    # Self-play (SelfPlayConfig).
    parser.add_argument("--games", type=int, default=128, help="self-play games per generation")
    parser.add_argument("--parallel", type=int, default=64, help="games played in lockstep")
    parser.add_argument("--simulations", type=int, default=100, help="self-play simulations per move")
    # Training (TrainConfig).
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50, help="optimiser steps per generation")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--buffer-generations", type=int, default=10, help="replay buffer length K")
    # Network (NetworkConfig).
    parser.add_argument("--blocks", type=int, default=4, help="residual blocks")
    parser.add_argument("--filters", type=int, default=64, help="feature maps per convolution")
    # Tournament (EvalConfig).
    parser.add_argument("--eval-games", type=int, default=20, help="openings per pairing (x2 colours)")
    parser.add_argument(
        "--eval-opponents", type=int, default=3,
        help="nearest previous checkpoints played (at least 1)",
    )
    parser.add_argument("--eval-ladder", type=int, default=8, help="also play g - N; 0 turns it off")
    args = parser.parse_args()
    print(f"Parameters: {vars(args)}")

    name = args.name or f"{args.size}x{args.size}-seed{args.seed}"
    try:
        config = build_config(args, name)
    except ValueError as error:  # a flag out of range, e.g. --games 0
        parser.error(str(error))
    device = select_device(args.device)
    run_dir = args.root / name
    print(f"Run: {name} on {device}")
    print(f"Config: {config}")

    latest = latest_generation(run_dir)
    if latest is not None:
        print(f"Resuming from {run_dir / checkpoint_name(latest)}")
    else:
        print(f"Starting {run_dir}")

    printer = GenerationPrinter(run_dir, after=latest if latest is not None else 0)
    printer.start()
    try:
        run(config, args.root, device)
    except KeyboardInterrupt:
        stop_printer(printer)
        print(
            f"\nInterrupted. Run directory {run_dir} is intact; "
            f"rerun the same command to resume from generation {printer.printed + 1}.",
            file=sys.stderr,
        )
        sys.exit(130)
    except (ConfigMismatchError, RunDirectoryError) as error:
        # The two refusals a user can cause (a changed flag on resume, a
        # directory with an old run's files). Any other exception is a bug
        # and keeps its traceback (the `finally` still stops the printer).
        stop_printer(printer)
        sys.exit(f"error: {error}")
    finally:
        stop_printer(printer)
    print(f"Done: {run_dir} is at generation {latest_generation(run_dir)}")


def stop_printer(printer: GenerationPrinter) -> None:
    """Stop the printer thread, wait for it, then print what it has not yet.

    Idempotent (a second call finds nothing new to print), so every exit path
    of `main` can call it.
    """
    printer.stop.set()
    printer.join()
    printer.poll()  # the last generation, committed after the final poll


def build_config(args: argparse.Namespace, name: str) -> RunConfig:
    """The `RunConfig` the flags describe. Raises `ValueError` for a bad value."""
    return RunConfig(
        name=name,
        size=args.size,
        seed=args.seed,
        generations=args.generations,
        network=NetworkConfig(blocks=args.blocks, filters=args.filters),
        search=SELF_PLAY,
        selfplay=SelfPlayConfig(
            games=args.games, parallel=args.parallel, simulations=args.simulations
        ),
        train=TrainConfig(
            batch_size=args.batch_size,
            steps_per_generation=args.steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            buffer_generations=args.buffer_generations,
        ),
        eval=EvalConfig(
            opponents=args.eval_opponents,
            # The plan's flag convention: 0 means "no ladder opponent".
            ladder=args.eval_ladder if args.eval_ladder > 0 else None,
            games_per_colour=args.eval_games,
            simulations=args.simulations,
        ),
        tensorboard=not args.no_tensorboard,
    )


if __name__ == "__main__":
    main()
