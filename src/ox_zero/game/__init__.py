"""Pure OXOX game logic. Must not import from any other `ox_zero` subpackage."""

from ox_zero.game.notation import (
    PositionError,
    from_board_string,
    parse_cell,
    parse_position,
    to_board_string,
)
from ox_zero.game.rules import (
    BOARD_SIZE,
    Cell,
    IllegalMoveError,
    Line,
    Mark,
    State,
    apply_move,
    format_cell,
    initial_state,
    is_draw,
    is_terminal,
    legal_moves,
    play,
)
from ox_zero.game.solver import Solver

__all__ = [
    "BOARD_SIZE",
    "Cell",
    "IllegalMoveError",
    "Line",
    "Mark",
    "PositionError",
    "Solver",
    "State",
    "apply_move",
    "format_cell",
    "from_board_string",
    "initial_state",
    "is_draw",
    "is_terminal",
    "legal_moves",
    "parse_cell",
    "parse_position",
    "play",
    "to_board_string",
]
