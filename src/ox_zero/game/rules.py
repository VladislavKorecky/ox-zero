"""OXOX rules: board state, move generation, win and draw detection.

Design: stateless transitions
-----------------------------
The game is modelled as immutable `State` values plus pure functions that map
one state to the next:

    next_state = apply_move(state, move)

Nothing is ever modified in place. This is the functional style of JAX-era RL
environments (`pgx`, `gymnax`) and of `alpha-zero-general`'s `Game` class, not
the stateful `env.step()` of OpenAI Gym. It suits AlphaZero because Monte Carlo
tree search (MCTS) expands many hypothetical children from the same parent
node. With immutable states each tree node can simply hold its own `State`,
and exploring one branch can never corrupt another. There is no undo stack
and no defensive copying.

Immutable states are also hashable, so two move orders that reach the same
position (a *transposition*) compare and hash equal. That makes positions
usable as dictionary keys, e.g. for a transposition table or caching network
evaluations later.

This module is pure game logic: no NumPy, no tensors, no rewards. Converting a
`State` into network input belongs to `ox_zero.engine`.

Conventions
-----------
Cells are `(row, col)`, zero-based, row 0 at the top, column 0 at the left.
X always moves first.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

# 12x12 is the current (experimental) board size. Every function takes the
# size from the state, so other sizes work too; small boards are handy for
# tests and for exhaustively solving the game.
BOARD_SIZE = 12

type Cell = tuple[int, int]
type Line = tuple[Cell, Cell, Cell]

# The four line directions as (d_row, d_col) steps. Each direction also covers
# its opposite (right covers left, and so on), because a line of three read
# backwards is the same line.
_DIRECTIONS: tuple[Cell, ...] = (
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # diagonal, down-right
    (1, -1),  # anti-diagonal, down-left
)


class Mark(Enum):
    """A mark on the board, which is also the name of the player placing it."""

    X = "X"
    O = "O"

    @property
    def opponent(self) -> Mark:
        return Mark.O if self is Mark.X else Mark.X

    def __str__(self) -> str:
        return self.value


class IllegalMoveError(ValueError):
    """Raised when a move is off the board, on an occupied cell, or after the game ended."""


@dataclass(frozen=True, slots=True)
class State:
    """One OXOX position. Immutable: create new states with `apply_move`.

    Build states with `initial_state`, `apply_move`, or `play` rather than
    calling the constructor directly; those guarantee the fields are
    consistent with each other.

    Attributes:
        size: Board side length. The board has `size * size` cells.
        board: Cells in row-major order (row 0 left to right, then row 1, ...),
            each a `Mark` or `None` for empty. The cell `(row, col)` lives at
            flat index `row * size + col`. A flat tuple rather than nested
            tuples keeps copying and hashing cheap.
        to_move: The player whose turn it is.
        move_count: Number of marks on the board.
        winner: The player who completed an alternating line, or `None`.
        winning_lines: Every line the winning move completed (one move can
            complete several), each listed in board order. Empty if nobody
            has won.
    """

    size: int
    board: tuple[Mark | None, ...]
    to_move: Mark
    move_count: int = 0
    winner: Mark | None = None
    winning_lines: tuple[Line, ...] = ()

    def __getitem__(self, cell: Cell) -> Mark | None:
        """The mark at `(row, col)`, or `None` if the cell is empty: `state[5, 5]`."""
        row, col = cell
        return self.board[row * self.size + col]


def initial_state(size: int = BOARD_SIZE) -> State:
    """The empty board with X to move."""
    if size < 1:
        raise ValueError(f"board size must be positive, got {size}")
    return State(size=size, board=(None,) * (size * size), to_move=Mark.X)


def is_draw(state: State) -> bool:
    """The board is full and nobody completed a line."""
    return state.winner is None and state.move_count == state.size * state.size


def is_terminal(state: State) -> bool:
    """The game is over: somebody won, or the board is full."""
    return state.winner is not None or is_draw(state)


def legal_moves(state: State) -> list[Cell]:
    """Every empty cell in board order (left to right, top to bottom).

    Returns an empty list once the game is over, even if empty cells remain.
    """
    if is_terminal(state):
        return []
    size = state.size
    return [divmod(index, size) for index, mark in enumerate(state.board) if mark is None]


def apply_move(state: State, move: Cell) -> State:
    """Place the side to move's mark on `move` and return the resulting state.

    The input state is left untouched.

    Raises:
        IllegalMoveError: The game is over, the cell is off the board, or the
            cell is already occupied.
    """
    if is_terminal(state):
        raise IllegalMoveError(f"cannot play {format_cell(move)}: the game is over")

    row, col = move
    size = state.size
    if not (0 <= row < size and 0 <= col < size):
        raise IllegalMoveError(f"cannot play {format_cell(move)}: off the {size}x{size} board")

    index = row * size + col
    if state.board[index] is not None:
        raise IllegalMoveError(f"cannot play {format_cell(move)}: cell is occupied")

    mark = state.to_move
    # Tuples are immutable, so "placing" a mark means building a new tuple:
    # everything before the cell, the new mark, everything after it.
    board = state.board[:index] + (mark,) + state.board[index + 1 :]

    # Only lines through the new mark can have just been completed. Every
    # other line was already checked when its last cell was filled, and the
    # game would have ended then. So we check a handful of lines around one
    # cell instead of scanning the whole board.
    lines = _completed_lines(board, size, move)

    return State(
        size=size,
        board=board,
        to_move=mark.opponent,
        move_count=state.move_count + 1,
        # Whoever places the completing mark wins, regardless of who placed
        # the other two marks in the line.
        winner=mark if lines else None,
        winning_lines=lines,
    )


def play(moves: Iterable[Cell], size: int = BOARD_SIZE) -> State:
    """Replay `moves` from the empty board, X first, sides alternating."""
    state = initial_state(size)
    for move in moves:
        state = apply_move(state, move)
    return state


def _completed_lines(board: tuple[Mark | None, ...], size: int, cell: Cell) -> tuple[Line, ...]:
    """All alternating lines of three (`XOX` or `OXO`) that contain `cell`.

    How this works: for each direction `d`, the new cell can be the first,
    second, or third cell of a line of three. So the candidate lines start at
    `cell - 0*d`, `cell - 1*d`, and `cell - 2*d`, and each runs for three cells
    along `d`. That gives at most 4 directions x 3 offsets = 12 candidate lines.

    A candidate counts if it is fully on the board, fully occupied, and
    alternating. A line `a b c` alternates when the two ends match and the
    middle differs: `a == c != b`.

    Bounds are checked on `(row, col)`, never on the flat index. In flat terms,
    cells `(0, 11)` and `(1, 0)` are neighbours (indices 11 and 12), but on the
    board they are at opposite edges and must not form a line.

    Every direction has `d_row >= 0`, and where `d_row == 0` it has `d_col > 0`,
    so walking along `d` always moves forward in board order. Each returned
    line is therefore already sorted top-to-bottom, left-to-right.
    """
    row, col = cell
    found: list[Line] = []
    for d_row, d_col in _DIRECTIONS:
        for offset in range(3):
            r0, c0 = row - offset * d_row, col - offset * d_col
            r2, c2 = r0 + 2 * d_row, c0 + 2 * d_col
            # Both ends on the board implies the middle is too (lines are straight).
            if not (0 <= r0 < size and 0 <= c0 < size and 0 <= r2 < size and 0 <= c2 < size):
                continue
            line: Line = ((r0, c0), (r0 + d_row, c0 + d_col), (r2, c2))
            first, middle, last = (board[r * size + c] for r, c in line)
            if first is not None and middle is not None and first == last != middle:
                found.append(line)
    return tuple(found)


def format_cell(cell: Cell) -> str:
    """`(5, 7)` -> `"5,7"`, the notation used by the CLI."""
    return f"{cell[0]},{cell[1]}"
