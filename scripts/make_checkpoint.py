"""Write a checkpoint with random weights, so `--model` can be tried before training exists.

Plan 03 (docs/plans/03-cli-adapter.md), step 5. The weights are random
(seeded): the network's priors and values mean nothing, and searching with
them is *worse* than the CLI's no-model mode, because random priors steer
the search instead of leaving it neutral. The file exists only to exercise
the checkpoint format and the CLI's loading path by hand.

The default output is deliberately *outside* `checkpoints/`: the CLI loads
the newest checkpoint under `checkpoints/` automatically, and a random
network there would silently replace the uniform search on every run.

Usage:
    uv run python scripts/make_checkpoint.py
    uv run ox-zero analyze 5,5 6,6 5,6 --model checkpoints_random/gen_000.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.search import ANALYSIS
from ox_zero.game.rules import BOARD_SIZE
from ox_zero.training.checkpoint import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=BOARD_SIZE, help="board size (CLI: 12)")
    parser.add_argument("--blocks", type=int, default=4, help="residual blocks")
    parser.add_argument("--filters", type=int, default=64, help="feature maps per convolution")
    parser.add_argument("--value-hidden", type=int, default=256, help="value head hidden width")
    parser.add_argument("--seed", type=int, default=0, help="torch seed for the random weights")
    parser.add_argument("--out", type=Path, default=Path("checkpoints_random/gen_000.pt"))
    args = parser.parse_args()
    print(f"Parameters: {vars(args)}")

    # The CLI auto-loads only `gen_N.pt` files under checkpoints/ (the newest
    # generation). Resolved, so an absolute path or one through `..` counts.
    under_root = Path("checkpoints").resolve() in args.out.resolve().parents
    if under_root and re.fullmatch(r"gen_\d+\.pt", args.out.name):
        print("Warning: the CLI may load this automatically (newest gen_N.pt under checkpoints/).")

    # Seeding torch fixes the random initialisation, so the same arguments
    # always write the same weights.
    torch.manual_seed(args.seed)
    config = NetworkConfig(blocks=args.blocks, filters=args.filters, value_hidden=args.value_hidden)
    network = Network(args.size, config)
    save_checkpoint(args.out, network, generation=0, configs={"search": ANALYSIS})
    print(args.out)


if __name__ == "__main__":
    main()
