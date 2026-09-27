"""The sandbox's state and command language, with no user interface.

Keeping this separate from the Textual screen means every command can be
unit-tested with plain function calls, and the screen stays a thin layer:
it feeds typed lines (or clicks and shortcuts) into `Session`, then redraws
from the session's state. It is the model half of a model/view split.

History model: `_states` is the list of positions from the starting position
to the current one, and `_moves[i]` is the move that led from `_states[i]` to
`_states[i + 1]`. `undo` moves the tail onto a redo stack, and `redo` moves it
back. Because states are immutable (see `ox_zero.game.rules`), keeping every
intermediate state is cheap and undo never has to recompute anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ox_zero.game import (
    Cell,
    IllegalMoveError,
    PositionError,
    State,
    apply_move,
    format_cell,
    initial_state,
    parse_cell,
    parse_position,
    to_board_string,
)

HELP = (
    "row,col play · undo · redo · pause · resume · load <position> · reset · export · quit"
    "   (keys: ctrl+z undo, ctrl+y redo, ctrl+p pause, ctrl+q quit; click a cell to play)"
)


@dataclass(frozen=True)
class Result:
    """What a command did, so the screen knows what to refresh.

    Attributes:
        message: Text for the message line, or `None` to clear it.
        error: The message is an error (drawn in red).
        position_changed: The engine must restart on the new position.
        quit: Leave the sandbox.
    """

    message: str | None = None
    error: bool = False
    position_changed: bool = False
    quit: bool = False


def _error(message: str) -> Result:
    return Result(message=message, error=True)


_CHANGED = Result(position_changed=True)


class Session:
    """A position with undo/redo history and the analysis pause switch."""

    def __init__(self, start: State, moves: Sequence[Cell] = ()) -> None:
        self._reset_to(start)
        for move in moves:
            self._push(move)
        self.paused = False

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> Session:
        """Start from command-line tokens (a move list or a board string).

        A move list is replayed from the empty board so its moves become undo
        history. A board string has no move order, so it becomes the start.

        Raises:
            PositionError: The position is invalid.
        """
        start, moves = _read_position(tokens)
        return cls(start, moves)

    # --- Read-only views ------------------------------------------------------

    @property
    def state(self) -> State:
        return self._states[-1]

    @property
    def history(self) -> list[Cell]:
        return list(self._moves)

    @property
    def last_move(self) -> Cell | None:
        return self._moves[-1] if self._moves else None

    def history_text(self) -> str:
        """Moves numbered in pairs, chess style: `1. 5,5 6,6  2. 5,6`.

        Numbering starts from the side that moved first in this history: after
        a board string with O to move, the first "pair" holds only O's move.
        """
        parts: list[str] = []
        for index, move in enumerate(self._moves):
            if index % 2 == 0:
                parts.append(f"{index // 2 + 1}. {format_cell(move)}")
            else:
                parts[-1] += f" {format_cell(move)}"
        return "  ".join(parts)

    # --- Commands -------------------------------------------------------------

    def execute(self, line: str) -> Result:
        """Run one line typed at the prompt. Never raises on bad input."""
        words = line.split()
        if not words:
            return Result()
        command, args = words[0].lower(), words[1:]

        if "," in command and not args:
            try:
                cell = parse_cell(command)
            except PositionError as error:
                return _error(str(error))
            return self.play(cell)

        match command:
            case "undo":
                return self.undo()
            case "redo":
                return self.redo()
            case "pause":
                if self.paused:
                    return Result(message="analysis is already paused")
                self.paused = True
                return Result()
            case "resume":
                if not self.paused:
                    return Result(message="analysis is already running")
                self.paused = False
                return Result()
            case "load":
                return self.load(args)
            case "reset":
                self._reset_to(initial_state(self.state.size))
                return _CHANGED
            case "export":
                return Result(message=to_board_string(self.state))
            case "help" | "?":
                return Result(message=HELP)
            case "quit" | "exit":
                return Result(quit=True)
            case _:
                return _error(f"unknown command {words[0]!r} (type help for the list)")

    def play(self, cell: Cell) -> Result:
        """Play `cell` for the side to move. Discards the redo history."""
        try:
            self._push(cell)
        except IllegalMoveError as error:
            return _error(str(error))
        self._redo.clear()
        return _CHANGED

    def undo(self) -> Result:
        if not self._moves:
            return _error("nothing to undo")
        self._redo.append(self._moves.pop())
        self._states.pop()
        return _CHANGED

    def redo(self) -> Result:
        if not self._redo:
            return _error("nothing to redo")
        self._push(self._redo.pop())
        return _CHANGED

    def load(self, tokens: Sequence[str]) -> Result:
        if not tokens:
            return _error("usage: load <move list or board string>")
        try:
            start, moves = _read_position(tokens)
        except PositionError as error:
            return _error(str(error))
        self._reset_to(start)
        for move in moves:
            self._push(move)
        return _CHANGED

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    # --- Internals ------------------------------------------------------------

    def _reset_to(self, start: State) -> None:
        self._states: list[State] = [start]
        self._moves: list[Cell] = []
        self._redo: list[Cell] = []

    def _push(self, move: Cell) -> None:
        self._states.append(apply_move(self.state, move))
        self._moves.append(move)


def _read_position(tokens: Sequence[str]) -> tuple[State, list[Cell]]:
    """Split a position into a starting state and the moves played from it.

    `parse_position` validates everything (and raises `PositionError`). For a
    move list, the tokens are then re-read as cells so they can be replayed
    as undoable history from the empty board.
    """
    state = parse_position(tokens)
    if tokens and "," in tokens[0]:
        return initial_state(state.size), [parse_cell(token) for token in tokens]
    return state, []
