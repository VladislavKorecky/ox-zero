"""The engine interface: what an engine produces and how callers drive it.

Everything outside `ox_zero.engine` (the CLI, later the GUI) talks to engines
only through this module, so swapping the placeholder `DummyEngine` for the
real AlphaZero search (roadmap step 3) should not change any caller.

How AlphaZero will fill these slots
-----------------------------------
AlphaZero picks moves with Monte Carlo tree search (MCTS). Each *simulation*
walks down the tree from the current position, choosing children by the PUCT
rule (Silver et al. 2017, "Mastering the game of Go without human knowledge"):

    a* = argmax_a  Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

When it reaches a leaf, it asks the neural network for a policy P (prior move
probabilities) and a value v (expected outcome), then backs v up the path.

- `Analysis.scores[move]` will be the mean backed-up value Q(s, a) of that
  move, rescaled from the network's [-1, 1] outcome range to a [0, 1] win
  probability for the side to move.
- `Analysis.value` will be the value of the position itself. Here it is the
  best move's score, which is what Q converges to under good play.
- `Analysis.simulations` is N(s), the number of simulations run so far. More
  simulations mean a deeper, more reliable search.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from ox_zero.game import Cell, State


@dataclass(frozen=True)
class Analysis:
    """A snapshot of an engine's opinion of one position.

    Attributes:
        value: Estimated win probability for the side to move, in [0, 1].
        scores: For every legal move, in board order, the estimated win
            probability for the side to move after playing it.
        simulations: How many search simulations produced this snapshot.
    """

    value: float
    scores: Mapping[Cell, float]
    simulations: int

    def top(self, n: int) -> list[tuple[Cell, float]]:
        """The `n` highest-scoring moves, best first; ties in board order."""
        # `sorted` is stable, and `scores` is in board order, so sorting by
        # descending score alone keeps tied moves in board order.
        ranked = sorted(self.scores.items(), key=lambda item: -item[1])
        return ranked[:n]

    @property
    def best(self) -> tuple[Cell, float]:
        """The engine's chosen move and its score."""
        return self.top(1)[0]


class Engine(Protocol):
    """Anything that can analyse a position.

    A search is exposed as an iterator of snapshots rather than a single
    blocking call, because the CLI needs three ways to consume it:

    - `analyze` with a fixed budget: run to the budget, keep the last one
      (and drive a progress bar from the ones in between);
    - `analyze --live`: keep reading and redraw every half second;
    - the sandbox: read in a background thread and abandon the iterator as
      soon as the position changes.

    The caller stops a search simply by not asking for the next snapshot.
    """

    def search(self, state: State, max_simulations: int | None = None) -> Iterator[Analysis]:
        """Yield snapshots with strictly increasing `simulations`.

        With `max_simulations`, the last snapshot has exactly that many
        simulations and the iterator ends. Without it, the search never ends
        on its own.

        Raises:
            ValueError: The game is already over; there is nothing to search.
        """
        ...


def analyze(engine: Engine, state: State, simulations: int) -> Analysis:
    """Run a search with a fixed budget and return the final snapshot."""
    if simulations < 1:
        raise ValueError(f"simulations must be positive, got {simulations}")
    last: Analysis | None = None
    for last in engine.search(state, max_simulations=simulations):
        pass
    assert last is not None  # a capped search always yields at least once
    return last
