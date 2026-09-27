"""Pure OXOX game logic. Must not import from any other `ox_zero` subpackage."""

from ox_zero.game.rules import (
    BOARD_SIZE,
    Cell,
    IllegalMoveError,
    Line,
    Mark,
    State,
    apply_move,
    initial_state,
    is_draw,
    is_terminal,
    legal_moves,
    play,
)

__all__ = [
    "BOARD_SIZE",
    "Cell",
    "IllegalMoveError",
    "Line",
    "Mark",
    "State",
    "apply_move",
    "initial_state",
    "is_draw",
    "is_terminal",
    "legal_moves",
    "play",
]
