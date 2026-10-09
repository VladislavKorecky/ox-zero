"""A stand-in engine with no intelligence, and the CLI's engine loader.

`PlaceholderEngine` lets the CLI run end to end before any real engine
exists. Its numbers mean nothing. It only imitates how the CLI expects
results to *behave*, so the interface can be exercised and tested:

- Scores are noisy at first and settle as the search count grows, so live
  views visibly converge. (The noise shrinks as 1/sqrt(n), like the standard
  error of an average over n samples.)
- It takes time. It sleeps to fake a throughput, so progress bars and live
  refreshes behave realistically.
- It is reproducible: the same seed and position give the same snapshots.

None of this is a model for how a real engine should work inside. A real
engine is connected by an adapter implementing `engine_port.Engine`, returned
from `load_engine` below.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

from ox_zero.cli.engine_port import Analysis, Engine
from ox_zero.game import Cell, State, is_terminal, legal_moves

# Simulations between snapshots. Small enough for smooth progress bars, large
# enough that producing snapshots is not the bottleneck.
_STEP = 50

# Standard deviation of the noise after the first step; it then shrinks as
# 1/sqrt(simulations).
_INITIAL_NOISE = 0.15


class PlaceholderEngine:
    """Deterministic pseudo-random scores that converge over time.

    Args:
        seed: Randomness seed. `None` picks a random one.
        rate: Fake throughput in simulations per second. `math.inf` disables
            the sleeping entirely (used by tests).
    """

    def __init__(self, seed: int | None = None, rate: float = 2000.0) -> None:
        self.seed = random.randrange(2**32) if seed is None else seed
        self.rate = rate

    def search(self, state: State, max_simulations: int | None = None) -> Iterator[Analysis]:
        if is_terminal(state):
            raise ValueError("the game is over; there is nothing to search")
        # A generator's body only runs on the first `next()`, so the check
        # above would be deferred too. Validate eagerly, then hand off.
        return self._search(state, max_simulations)

    def _search(self, state: State, max_simulations: int | None) -> Iterator[Analysis]:
        moves = legal_moves(state)
        # Seeding `random.Random` with a string is deterministic across runs
        # (it is hashed with SHA-512), unlike Python's salted `hash()`.
        rng = random.Random(f"{self.seed}|{state.board}")
        # The "true" score each move converges to. Moves near existing marks
        # get a bump so the heatmap has some shape rather than pure static.
        target = {move: _base_score(state, move, rng) for move in moves}
        # A fixed noise direction per move, scaled down over time. Reusing the
        # same direction (instead of fresh noise each step) makes the numbers
        # drift smoothly towards their targets instead of flickering.
        direction = {move: rng.gauss(0.0, 1.0) for move in moves}

        simulations = 0
        while max_simulations is None or simulations < max_simulations:
            step = _STEP if max_simulations is None else min(_STEP, max_simulations - simulations)
            if self.rate != math.inf:
                time.sleep(step / self.rate)
            simulations += step

            spread = _INITIAL_NOISE * math.sqrt(_STEP / simulations)
            wobble = rng.gauss(0.0, 0.2)  # a little extra jitter per snapshot
            scores = {
                move: _clamp(target[move] + spread * (direction[move] + wobble))
                for move in moves
            }
            # No visit counts here, so the "chosen" move is just the top score
            # (ties in board order, matching `Analysis.top`).
            chosen = max(scores, key=scores.__getitem__)
            yield Analysis(
                value=scores[chosen], scores=scores, simulations=simulations, chosen=chosen
            )


def load_engine(model: Path | None, seed: int | None) -> Engine:
    """The engine the CLI should use, for the `--model` and `--seed` flags.

    This is the one place a real engine gets connected. Until then it always
    returns a `PlaceholderEngine`. A given path is still checked for
    existence, so the `--model` flag already behaves as specified.

    Raises:
        FileNotFoundError: `model` was given but does not exist.
    """
    if model is not None and not model.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model}")
    return PlaceholderEngine(seed=seed)


def _base_score(state: State, move: Cell, rng: random.Random) -> float:
    """A made-up score: closer to existing marks is higher, plus randomness."""
    occupied = [divmod(i, state.size) for i, mark in enumerate(state.board) if mark is not None]
    if occupied:
        # Chebyshev distance (king moves) to the nearest mark.
        distance = min(max(abs(move[0] - r), abs(move[1] - c)) for r, c in occupied)
        closeness = 1.0 / distance
    else:
        # Empty board: favour the centre.
        centre = (state.size - 1) / 2
        closeness = 1.0 - max(abs(move[0] - centre), abs(move[1] - centre)) / state.size
    return _clamp(0.08 + 0.45 * closeness + rng.uniform(-0.05, 0.05))


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))
