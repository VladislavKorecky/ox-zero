"""Play one checkpoint against another (or against the uniform search) and report the score.

Plan 04 (docs/plans/04-training-pipeline.md), step 7. The same match the
training loop's tournament plays (`ox_zero.training.evaluate.play_match`,
noise-free `EVALUATION` search, random opening cells, every opening played
once with each colour), between any two players chosen by hand.

`--b uniform` is the no-model search the CLI runs without a checkpoint
(`UniformEvaluator`: equal priors, value 0), so it measures how much the
network adds over search alone.

The Elo gap is `elo_difference((points + 0.5) / (games + 1))`: one virtual
draw, the same regulariser the run's Bradley-Terry fit applies, so a clean
sweep prints a finite number instead of infinity.

Usage:
    uv run python scripts/evaluate.py --a runs/six-a/gen_020.pt --b runs/six-a/gen_000.pt
    uv run python scripts/evaluate.py --a runs/six-a/gen_020.pt --b uniform --device mps
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ox_zero.engine.evaluator import Evaluator, UniformEvaluator
from ox_zero.engine.network_evaluator import NetworkEvaluator, select_device
from ox_zero.training.checkpoint import load_checkpoint
from ox_zero.training.evaluate import EVALUATION, elo_difference, play_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True, help="checkpoint path (sets the board size)")
    parser.add_argument("--b", required=True, help="checkpoint path, or 'uniform'")
    parser.add_argument("--games", type=int, default=20, help="openings, each played with both colours")
    parser.add_argument("--simulations", type=int, default=100, help="search simulations per move")
    parser.add_argument("--seed", type=int, default=0, help="seeds the opening cells")
    parser.add_argument("--device", default="cpu", help="auto | cpu | mps | cuda")
    args = parser.parse_args()
    # Checked here, before any checkpoint is loaded: `play_match` would raise
    # on these too, but outside the error handling below, i.e. as a
    # traceback. At least one opening (no games means no score), and at
    # least one simulation (the root needs a visit to choose a move).
    for flag, value in (("--games", args.games), ("--simulations", args.simulations)):
        if value < 1:
            parser.error(f"{flag} must be at least 1, got {value}")
    print(f"Parameters: {vars(args)}")

    device = select_device(args.device)
    try:
        a = load_checkpoint(args.a, device)
        size = a.size
        player_a: Evaluator = NetworkEvaluator(a.network, device)
        if args.b == "uniform":
            player_b: Evaluator = UniformEvaluator()
        else:
            b = load_checkpoint(args.b, device)
            if b.size != size:
                sys.exit(f"error: {args.a} is {size}x{size} but {args.b} is {b.size}x{b.size}")
            player_b = NetworkEvaluator(b.network, device)
    except ValueError as error:  # load_checkpoint's one error type
        sys.exit(f"error: {error}")

    rng = np.random.default_rng(args.seed)
    result = play_match(
        player_a, player_b, size, args.games, args.simulations, rng, EVALUATION
    )

    print(f"{args.a} (a) vs {args.b} (b), {size}x{size}, {result.games} games")
    for colour, (wins, draws, losses) in (("X", result.as_x), ("O", result.as_o)):
        print(f"  a as {colour}: {wins} W  {draws} D  {losses} L")
    points = result.wins + result.draws / 2
    print(f"  total:   {result.wins} W  {result.draws} D  {result.losses} L")
    print(f"Score: {points:g}/{result.games} = {result.score:.3f} for a")
    gap = elo_difference((points + 0.5) / (result.games + 1))
    print(f"Elo gap (a - b, one virtual draw): {gap:+.0f}")


if __name__ == "__main__":
    main()
