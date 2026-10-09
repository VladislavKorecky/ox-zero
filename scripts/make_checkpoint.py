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
from pathlib import Path

import torch

from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.search import ANALYSIS
from ox_zero.game.rules import BOARD_SIZE
from ox_zero.training.checkpoint import latest_checkpoint, save_checkpoint


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

    # Seeding torch fixes the random initialisation, so the same arguments
    # always write the same weights.
    torch.manual_seed(args.seed)
    config = NetworkConfig(blocks=args.blocks, filters=args.filters, value_hidden=args.value_hidden)
    network = Network(args.size, config)
    save_checkpoint(args.out, network, generation=0, configs={"search": ANALYSIS})
    print(args.out)

    # Ask the CLI's own rule rather than re-implementing it here. The file is
    # already saved, so a failure to scan checkpoints/ must not look like a
    # failed save.
    try:
        newest = latest_checkpoint(Path("checkpoints"))
    except OSError as error:
        print(f"Note: could not scan checkpoints/ ({error}).")
        return
    if newest is not None and newest.resolve() == args.out.resolve():
        print("Warning: this is now the newest checkpoint; the CLI will load it automatically.")


if __name__ == "__main__":
    main()
