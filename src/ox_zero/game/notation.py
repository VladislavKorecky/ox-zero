"""Text notation for OXOX positions: cells, move lists, and board strings.

The formats are specified in the README's "Positions and moves" section:

- A cell is `row,col`, zero-based, row 0 at the top, column 0 at the left.
- A position is either a *move list* (cells in the order played, X first) or a
  *board string* (`size * size` characters, row 0 first, `X`, `O`, or `_`).

This is still pure game logic, so it lives in `ox_zero.game`. The CLI uses it
to read positions from the command line, and later the GUI and any saved-game
tooling can share the same parser.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ox_zero.game.rules import (
    BOARD_SIZE,
    Cell,
    IllegalMoveError,
    Line,
    Mark,
    State,
    _completed_lines,
    apply_move,
    initial_state,
)

# Two non-negative integers separated by a comma; spaces around either part are
# allowed so `5, 7` works too. Negative numbers are rejected here rather than
# by `apply_move`, because "-1,3" is a typo, not a move off the board.
_CELL_RE = re.compile(r"\s*(\d+)\s*,\s*(\d+)\s*")

_EMPTY = "_"
_CHAR_TO_MARK: dict[str, Mark | None] = {"X": Mark.X, "O": Mark.O, _EMPTY: None}


class PositionError(ValueError):
    """A position or cell could not be read.

    Attributes:
        token_index: Which input token caused the error (0-based), when the
            input was a list of tokens. The CLI uses it to point a caret at
            the offending token. `None` when there is no single culprit.
    """

    def __init__(self, message: str, token_index: int | None = None) -> None:
        super().__init__(message)
        self.token_index = token_index


def parse_cell(text: str) -> Cell:
    """`"5,7"` -> `(5, 7)`. Only the syntax is checked, not the board bounds."""
    match = _CELL_RE.fullmatch(text)
    if match is None:
        raise PositionError(f"expected a cell as row,col (e.g. 5,7), got {text!r}")
    return int(match[1]), int(match[2])


def parse_position(tokens: Sequence[str], size: int = BOARD_SIZE) -> State:
    """Read a position given as command-line tokens.

    The format is auto-detected. A single token that is either board-sized or
    made only of board characters (`X`, `O`, `_`) is a board string; anything
    else is a move list. So `5` is reported as a malformed cell rather than as
    a board string of the wrong length. No tokens at all is the empty board.

    Raises:
        PositionError: With `token_index` set to the offending token.
    """
    if len(tokens) == 1 and _looks_like_board(tokens[0], size):
        try:
            return from_board_string(tokens[0], size)
        except PositionError as error:
            raise PositionError(str(error), token_index=0) from error

    state = initial_state(size)
    for index, token in enumerate(tokens):
        try:
            state = apply_move(state, parse_cell(token))
        except (PositionError, IllegalMoveError) as error:
            raise PositionError(str(error), token_index=index) from error
    return state


def to_board_string(state: State) -> str:
    """The position as a board string: row 0 first, `X`, `O`, or `_`."""
    return "".join(_EMPTY if mark is None else mark.value for mark in state.board)


def from_board_string(text: str, size: int = BOARD_SIZE) -> State:
    """Build a `State` from a board string, validating that it is reachable.

    A board string only says where the marks are, not in which order they were
    played. The rest of the state has to be reconstructed:

    - Side to move, from the mark counts. X moves first, so after a full
      round the counts are equal (X to move); mid-round X has one extra
      (O to move). Anything else cannot arise from alternating play.
    - Winner. In a real game the game ends on the move that completes the
      first alternating line, so every line on the board must have been
      completed by that single last move. The last mover is the side *not*
      to move. The position is reachable only if some mark of the last mover
      lies on every line; that cell is the winning move.

    Raises:
        PositionError: Wrong length, unknown character, impossible counts,
            or an unreachable decided position.
    """
    cells = size * size
    if len(text) != cells:
        raise PositionError(
            f"a board string must be {cells} characters ({size}x{size}), got {len(text)}"
        )
    for index, char in enumerate(text):
        if char not in _CHAR_TO_MARK:
            row, col = divmod(index, size)
            raise PositionError(
                f"unexpected character {char!r} at {row},{col} in board string"
                f" (use X, O, or {_EMPTY})"
            )

    board = tuple(_CHAR_TO_MARK[char] for char in text)
    x_count, o_count = text.count("X"), text.count("O")
    if x_count == o_count:
        to_move = Mark.X
    elif x_count == o_count + 1:
        to_move = Mark.O
    else:
        raise PositionError(
            f"impossible mark count: {x_count} X and {o_count} O"
            " (X moves first, so X has as many marks as O or one more)"
        )

    # Every alternating line on the board, found by asking `_completed_lines`
    # about every occupied cell. Each line shows up once per cell it contains,
    # so collect them in a set to deduplicate.
    lines: set[Line] = set()
    for index, mark in enumerate(board):
        if mark is not None:
            lines.update(_completed_lines(board, size, divmod(index, size)))

    winner: Mark | None = None
    winning_lines: tuple[Line, ...] = ()
    if lines:
        last_mover = to_move.opponent
        winning_cell = _common_cell(lines, board, size, last_mover)
        if winning_cell is None:
            raise PositionError(
                "unreachable position: no single move by the last mover"
                f" ({last_mover}) could have completed every line on the board"
            )
        winner = last_mover
        # Recompute from the winning cell, so the lines are in exactly the
        # form and order `apply_move` produces for the same game.
        winning_lines = _completed_lines(board, size, winning_cell)

    return State(
        size=size,
        board=board,
        to_move=to_move,
        move_count=x_count + o_count,
        winner=winner,
        winning_lines=winning_lines,
    )


def _common_cell(
    lines: set[Line], board: tuple[Mark | None, ...], size: int, mark: Mark
) -> Cell | None:
    """A cell holding `mark` that lies on every line in `lines`, if any."""
    shared = set.intersection(*(set(line) for line in lines))
    candidates = sorted(cell for cell in shared if board[cell[0] * size + cell[1]] is mark)
    return candidates[0] if candidates else None


def _looks_like_board(token: str, size: int) -> bool:
    return len(token) == size * size or set(token) <= _CHAR_TO_MARK.keys()
