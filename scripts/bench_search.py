"""Benchmark the search: simulations per second for one board size, evaluator and device.

Plan 02 (docs/plans/archive/02-engine-search-network.md), step 8. The first data
point for docs/design/engineering.md, "Performance plan": measure before
optimising anything.

It runs `ox_zero.engine.search.analyse` (analysis mode: batch size 1, every
root child evaluated up front) to `--simulations` on one position,
`--repeats` times, and reports:

- simulations per second (mean over repeats);
- evaluator calls per second: every `evaluate()` call, including the root
  setup and the one batched call for the root children;
- the fraction of wall time spent inside `evaluate()` (for the network
  evaluator: encoding, the forward pass, the device round trip). The rest is
  the pure-Python tree.

The network has random weights (seeded): speed does not depend on what the
weights are. One untimed warm-up search runs first, because the first
forward pass on a GPU backend pays one-off setup costs (kernel compilation,
memory pools) that are not what we want to measure.

A Markdown table row is printed last, to paste into a PR or a results table.

Usage:
    uv run python scripts/bench_search.py
    uv run python scripts/bench_search.py --size 12 --device mps
    uv run python scripts/bench_search.py --evaluator uniform --position 2,2 3,3 2,3
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections.abc import Sequence

import torch

from ox_zero.engine.evaluator import Evaluator, UniformEvaluator
from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.network_evaluator import NetworkEvaluator, select_device
from ox_zero.engine.search import analyse
from ox_zero.game import (
    State,
    apply_move,
    initial_state,
    is_terminal,
    legal_moves,
    parse_position,
    to_board_string,
)


class TimedEvaluator:
    """Wraps an evaluator and records how many calls it got and how long they took.

    The network evaluator ends every call with `.cpu().numpy()`, which waits
    for the device to finish, so wall time around the call is the real cost
    even on an asynchronous GPU backend.
    """

    def __init__(self, inner: Evaluator) -> None:
        self.inner = inner
        self.calls = 0
        self.seconds = 0.0

    def evaluate(self, states: Sequence[State]):
        began = time.perf_counter()
        result = self.inner.evaluate(states)
        self.seconds += time.perf_counter() - began
        self.calls += 1
        return result


def random_opening(size: int, moves: int, seed: int) -> State:
    """A seeded random position `moves` plies in, with no win in one for the side to move.

    A position with an immediate win is a bad benchmark: every simulation
    runs straight into that terminal child and the evaluator is never called
    again, so the tree barely grows. Retry until the game is on and quiet.
    """
    rng = random.Random(seed)
    while True:
        state = initial_state(size)
        for _ in range(moves):
            if is_terminal(state):
                break
            state = apply_move(state, rng.choice(legal_moves(state)))
        if is_terminal(state):
            continue
        if not any(apply_move(state, move).winner for move in legal_moves(state)):
            return state


def run_once(state: State, evaluator: Evaluator, simulations: int) -> tuple[float, int, float]:
    """One full analysis; returns (wall seconds, evaluate calls, seconds in evaluate)."""
    timed = TimedEvaluator(evaluator)
    began = time.perf_counter()
    for _ in analyse(state, timed, max_simulations=simulations):
        pass
    return time.perf_counter() - began, timed.calls, timed.seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--evaluator", choices=["network", "uniform"], default="network")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--position",
        nargs="*",
        default=None,
        metavar="ROW,COL",
        help="move list to analyse (default: a seeded random 4-move opening)",
    )
    args = parser.parse_args()

    if args.position is None:
        state = random_opening(args.size, 4, args.seed)
    else:
        state = parse_position(args.position, size=args.size)

    torch.manual_seed(args.seed)
    if args.evaluator == "network":
        device = select_device(args.device)
        config = NetworkConfig(blocks=args.blocks, filters=args.filters)
        evaluator: Evaluator = NetworkEvaluator(Network(args.size, config), device)
        device_name = device.type
        net_label = f"{args.blocks}x{args.filters}"
    else:
        evaluator = UniformEvaluator()
        device_name = "n/a"
        net_label = "n/a"

    print(f"size:        {args.size}x{args.size}")
    print(f"position:    {to_board_string(state)} ({state.to_move} to move, "
          f"{len(legal_moves(state))} legal moves)")
    print(f"evaluator:   {args.evaluator}")
    print(f"network:     {net_label}")
    print(f"device:      {device_name} (requested {args.device})")
    print(f"simulations: {args.simulations}, repeats: {args.repeats}, seed: {args.seed}")
    print(f"torch:       {torch.__version__}")

    run_once(state, evaluator, min(args.simulations, 20))  # warm-up, untimed

    sims_per_s, calls_per_s, eval_share = [], [], []
    for repeat in range(args.repeats):
        wall, calls, in_eval = run_once(state, evaluator, args.simulations)
        sims_per_s.append(args.simulations / wall)
        calls_per_s.append(calls / wall)
        eval_share.append(in_eval / wall)
        print(f"  run {repeat + 1}: {wall:.2f} s, {sims_per_s[-1]:.0f} sims/s, "
              f"{calls_per_s[-1]:.0f} evals/s, {100 * eval_share[-1]:.0f}% in evaluate")

    mean_sims = statistics.mean(sims_per_s)
    mean_calls = statistics.mean(calls_per_s)
    mean_share = statistics.mean(eval_share)
    print()
    print(f"simulations/s: {mean_sims:.0f}")
    print(f"evaluate calls/s: {mean_calls:.0f}")
    print(f"time in evaluate: {100 * mean_share:.0f}%")
    print()
    print("| Size | Evaluator | Network | Device | Simulations | Sims/s | Evals/s | In evaluate |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| {args.size}x{args.size} | {args.evaluator} | {net_label} | {device_name} "
          f"| {args.simulations} | {mean_sims:.0f} | {mean_calls:.0f} | {100 * mean_share:.0f}% |")


if __name__ == "__main__":
    main()
