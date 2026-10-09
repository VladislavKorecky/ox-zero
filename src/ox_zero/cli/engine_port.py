"""What the CLI needs from an engine: its *port* to the analysis backend.

This module belongs to the CLI, not to `ox_zero.engine`, on purpose. It
describes what the CLI displays and how it consumes results, and says nothing
about how an engine should work inside. That is the "ports and adapters"
pattern (also called dependency inversion): the consumer defines the
interface it needs, and whatever provides the analysis is connected through
a small adapter that translates into this shape. The engine package stays
free to have whatever design suits it, and never imports the CLI.

The real implementation is `SearchEngine` in `adapter.py`, which translates
the AlphaZero search (`ox_zero.engine.search.analyse`) into this shape.
`PlaceholderEngine` (`placeholder.py`) is a fast, fake implementation kept for
the CLI's own tests.

The two requirements, both fixed by the CLI specification (docs/cli.md):

- *What* is shown: a win probability for the side to move after each legal
  move, one for the position itself, and how much search produced them.
- *When* it is shown: `analyze --live` and the sandbox display results that
  improve while the search runs, so results must be available incrementally,
  and the CLI must be able to stop a search at any point.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from ox_zero.game import Cell, State


@dataclass(frozen=True)
class Analysis:
    """A snapshot of an engine's opinion of one position, as the CLI shows it.

    Attributes:
        value: Estimated win probability for the side to move, in [0, 1]
            (the eval bar).
        scores: For every legal move, in board order, the estimated win
            probability for the side to move after playing it (the numbers
            on the board).
        simulations: How much search produced this snapshot, shown in the
            status line and used as the `--simulations` budget.
        chosen: The move the engine would play. For the real engine that is
            the most visited root move, which can differ from the
            highest-scoring one: the visit count integrates both a move's
            value and how confident the search is in it, while a rarely
            visited move's score is an average of very few samples (noise).
            Required, with no default, so an adapter cannot forget it.
    """

    value: float
    scores: Mapping[Cell, float]
    simulations: int
    chosen: Cell

    def top(self, n: int) -> list[tuple[Cell, float]]:
        """The `n` highest-scoring moves, best first; ties in board order."""
        # `sorted` is stable, and `scores` is in board order, so sorting by
        # descending score alone keeps tied moves in board order.
        ranked = sorted(self.scores.items(), key=lambda item: -item[1])
        return ranked[:n]

    @property
    def best(self) -> tuple[Cell, float]:
        """The engine's chosen move and its score (not necessarily `top(1)`)."""
        return self.chosen, self.scores[self.chosen]


class Engine(Protocol):
    """Anything the CLI can analyse positions with.

    A search is consumed as an iterator of snapshots rather than a single
    blocking call, because the CLI reads it in three ways:

    - `analyze` / `best` with a fixed budget: read to the end, keep the last
      snapshot, and drive a progress bar from the ones in between;
    - `analyze --live`: keep reading and redraw every half second;
    - the sandbox: read in a background thread and abandon the iterator as
      soon as the position changes.

    The CLI stops a search simply by not asking for the next snapshot.
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
