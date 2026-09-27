"""JSON reports, in the shape specified in docs/cli.md (`--json`).

Kept separate from `render` because JSON must never depend on terminal
capabilities: it is the machine interface and stays byte-for-byte stable.
"""

from __future__ import annotations

from typing import Any

from ox_zero.cli.engine_port import Analysis
from ox_zero.game import Cell, State, is_draw, to_board_string

# Four decimals is more precision than any engine estimate deserves, and it
# keeps the output readable. Rounding also hides float noise like 0.47000001.
_DECIMALS = 4


def report_json(
    state: State,
    analysis: Analysis | None,
    *,
    top_n: int,
    include_simulations: bool = False,
) -> dict[str, Any]:
    """The `analyze --json` object. `analysis` is `None` for a finished game.

    With `include_simulations`, a leading `simulations` field is added, as in
    the JSON Lines stream of `analyze --live --json`.
    """
    report: dict[str, Any] = {}
    if include_simulations:
        report["simulations"] = analysis.simulations if analysis else 0
    finished = analysis is None
    report["board"] = to_board_string(state)
    report["to_move"] = None if finished else state.to_move.value
    report["value"] = None if finished else _round(analysis.value)
    report["moves"] = [] if finished else [_scored(c, s) for c, s in analysis.scores.items()]
    report["top"] = [] if finished else [_scored(c, s) for c, s in analysis.top(top_n)]
    report["result"] = _result(state)
    return report


def best_json(analysis: Analysis) -> dict[str, Any]:
    """The `best --json` object: `{"move": [5, 7], "score": 0.47}`."""
    return _scored(*analysis.best)


def finished_best_json(state: State) -> dict[str, Any]:
    """`best --json` for a finished game: no move, and the result."""
    return {"move": None, "score": None, "result": _result(state)}


def _result(state: State) -> str | None:
    if state.winner is not None:
        return state.winner.value
    return "draw" if is_draw(state) else None


def _scored(cell: Cell, score: float) -> dict[str, Any]:
    return {"move": list(cell), "score": _round(score)}


def _round(x: float) -> float:
    return round(x, _DECIMALS)
