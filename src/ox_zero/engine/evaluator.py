"""The evaluator seam: how the search asks "what are `p` and `v` here?".

Design: docs/design/engineering.md, "The evaluator seam".

AlphaZero's search needs one thing from the network, `(p, v) = f_θ(s)`:

- `p`, a prior over moves: which moves look worth searching. In PUCT it
  scales the exploration bonus `U(s,a)` (see `mcts.py`).
- `v`, a value in `[-1, 1]` for the side to move: the network's guess at the
  game result, used instead of playing the game out (AlphaGo's rollouts are
  gone in AlphaZero; the value head replaces them).

The search depends only on the `Evaluator` protocol below, never on torch.
That keeps the tree code testable with exact, scripted answers, and it is
also where batching happens: self-play collects one leaf from each of many
trees and calls `evaluate` once for all of them.

Return types (decided in plan 02): NumPy `float32` arrays.

- `Policies`: `[B, S²]`, flat board order (`row * S + col`). Exactly zero on
  illegal cells; legal cells sum to 1. Masking is the evaluator's job, so the
  search never sees an illegal prior.
- `Values`: `[B]`, in `[-1, 1]`, from the side to move's perspective.

This module stays torch-free. The torch-backed `NetworkEvaluator` lives in
`network_evaluator.py`, because `import torch` costs about half a second and
everything that only searches (tests, the tree, the CLI's placeholder path)
should not pay for it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ox_zero.engine.encoding import legal_mask
from ox_zero.game.rules import State

type Policies = np.ndarray
"""`float32 [B, S²]`: zero on illegal cells, legal cells sum to 1."""

type Values = np.ndarray
"""`float32 [B]`: in `[-1, 1]`, from the side to move's perspective."""


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can score a batch of positions for the search.

    A `Protocol` is structural typing: a class counts as an `Evaluator` if it
    has a matching `evaluate` method, without inheriting from anything.
    """

    def evaluate(self, states: Sequence[State]) -> tuple[Policies, Values]:
        """Priors and values for each state, in order. All states share one board size."""
        ...


class UniformEvaluator:
    """Equal priors over the legal moves and value 0 everywhere.

    "Knows nothing": with it, PUCT degenerates to a visit-balancing search
    whose only real information comes from terminal positions. That is
    exactly what makes it useful in tests: any tactic the search finds with
    it was found by the tree, not by an evaluator.
    """

    def evaluate(self, states: Sequence[State]) -> tuple[Policies, Values]:
        # [B, S²] booleans -> float32 0/1, then divide each row by its number
        # of legal moves. `keepdims` keeps the counts as a [B, 1] column so the
        # division broadcasts across each row. `maximum(.., 1)` only guards a
        # terminal state (no legal moves) against 0/0; its row stays all zero.
        masks = np.stack([legal_mask(state) for state in states]).astype(np.float32)
        counts = np.maximum(masks.sum(axis=1, keepdims=True), 1.0)
        return masks / counts, np.zeros(len(states), dtype=np.float32)


class TableEvaluator:
    """Exact scripted answers, for tests: `state -> (policy, value)`.

    States not in the table are passed to `fallback` (uniform by default).
    `State` is frozen and hashable, so it can be a dictionary key directly.
    The scripted policy is returned as given; the tests that script one are
    responsible for making it a valid prior.
    """

    def __init__(
        self,
        table: Mapping[State, tuple[np.ndarray, float]],
        fallback: Evaluator | None = None,
    ) -> None:
        self._table = dict(table)
        self._fallback = fallback if fallback is not None else UniformEvaluator()

    def evaluate(self, states: Sequence[State]) -> tuple[Policies, Values]:
        size = states[0].size if states else 0
        policies = np.zeros((len(states), size * size), dtype=np.float32)
        values = np.zeros(len(states), dtype=np.float32)

        # Split the batch into scripted and unscripted rows, answer the
        # unscripted ones with a single fallback call (the fallback may be a
        # real network, which prefers one batch to many), then scatter the
        # answers back into their original rows.
        missing: list[int] = []
        for row, state in enumerate(states):
            entry = self._table.get(state)
            if entry is None:
                missing.append(row)
            else:
                policies[row], values[row] = entry
        if missing:
            fallback_policies, fallback_values = self._fallback.evaluate(
                [states[row] for row in missing]
            )
            policies[missing] = fallback_policies
            values[missing] = fallback_values
        return policies, values
