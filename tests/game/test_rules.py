"""Tests for the OXOX rules: board state, move generation, win and draw detection.

Coordinates are `(row, col)`, zero-based, row 0 at the top (see README).
Positions are mostly built with `play(moves)`, which replays a move list from
the empty board: X moves first, then the sides alternate.
"""

import random

import pytest

from ox_zero.game import (
    BOARD_SIZE,
    IllegalMoveError,
    Mark,
    State,
    apply_move,
    initial_state,
    is_draw,
    is_terminal,
    legal_moves,
    play,
)


# --- Initial state -----------------------------------------------------------


def test_default_board_is_12_by_12():
    assert BOARD_SIZE == 12
    assert initial_state().size == 12


def test_initial_state_is_empty_with_x_to_move():
    state = initial_state()
    assert state.to_move is Mark.X
    assert state.move_count == 0
    assert all(state[row, col] is None for row in range(12) for col in range(12))
    assert state.winner is None
    assert not is_terminal(state)


def test_every_cell_is_legal_on_empty_board_in_board_order():
    # Board order: left to right, top to bottom (README, `moves` in --json).
    expected = [(row, col) for row in range(12) for col in range(12)]
    assert legal_moves(initial_state()) == expected


def test_board_size_is_configurable():
    state = initial_state(size=5)
    assert state.size == 5
    assert len(legal_moves(state)) == 25


def test_board_size_must_be_positive():
    with pytest.raises(ValueError):
        initial_state(size=0)


# --- Applying moves ----------------------------------------------------------


def test_apply_move_places_mark_of_side_to_move_and_passes_turn():
    after = apply_move(initial_state(), (5, 5))
    assert after[5, 5] is Mark.X
    assert after.to_move is Mark.O
    assert after.move_count == 1

    after = apply_move(after, (6, 6))
    assert after[6, 6] is Mark.O
    assert after.to_move is Mark.X


def test_apply_move_does_not_modify_the_input_state():
    # The core stateless property: tree search expands many children from
    # one parent, so the parent must survive untouched.
    before = initial_state()
    apply_move(before, (5, 5))
    assert before[5, 5] is None
    assert before.to_move is Mark.X
    assert before.move_count == 0


def test_state_is_immutable():
    state = initial_state()
    with pytest.raises(AttributeError):
        state.to_move = Mark.O  # type: ignore[misc]


def test_play_replays_a_move_list():
    state = play([(5, 5), (6, 6), (5, 6)])
    assert state[5, 5] is Mark.X
    assert state[6, 6] is Mark.O
    assert state[5, 6] is Mark.X
    assert state.to_move is Mark.O
    assert state.move_count == 3


def test_play_accepts_a_board_size():
    assert play([(0, 0)], size=4).size == 4


def test_occupied_cells_are_not_legal():
    state = play([(5, 5), (6, 6)])
    moves = legal_moves(state)
    assert (5, 5) not in moves
    assert (6, 6) not in moves
    assert len(moves) == 144 - 2


def test_same_position_by_different_move_orders_is_equal_and_hashes_equal():
    # Transpositions: equal positions compare and hash equal, so they can be
    # used as dictionary keys (e.g. a transposition table in tree search).
    a = play([(5, 5), (6, 6), (5, 6)])
    b = play([(5, 6), (6, 6), (5, 5)])
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# --- Illegal moves -----------------------------------------------------------


def test_playing_on_an_occupied_cell_is_illegal():
    state = play([(5, 5)])
    with pytest.raises(IllegalMoveError):
        apply_move(state, (5, 5))


@pytest.mark.parametrize("cell", [(-1, 0), (0, -1), (12, 0), (0, 12), (12, 12)])
def test_playing_off_the_board_is_illegal(cell):
    with pytest.raises(IllegalMoveError):
        apply_move(initial_state(), cell)


def test_playing_after_the_game_is_over_is_illegal():
    state = play([(5, 5), (6, 6), (5, 7), (5, 6)])  # O wins
    with pytest.raises(IllegalMoveError):
        apply_move(state, (0, 0))


def test_illegal_move_error_is_a_value_error():
    assert issubclass(IllegalMoveError, ValueError)


# --- Winning -----------------------------------------------------------------


def test_completing_xox_wins_for_the_player_who_completes_it():
    # X at 5,5 and 5,7, then O fills the gap: X O X, completed by O.
    state = play([(5, 5), (6, 6), (5, 7), (5, 6)])
    assert state.winner is Mark.O
    assert is_terminal(state)
    assert not is_draw(state)


def test_ownership_of_the_other_marks_does_not_matter():
    # The winning line may include the opponent's marks. Here O wins with
    # O X O on row 1, and the X in the middle was placed by X.
    #   X: 1,1  5,5 (filler)    O: 1,0  1,2
    state = play([(1, 1), (1, 0), (5, 5), (1, 2)])
    assert state.winner is Mark.O


def test_x_can_win_by_completing_oxo():
    # O at 3,3 and 3,5, X drops into the middle: O X O, completed by X.
    state = play([(0, 0), (3, 3), (0, 11), (3, 5), (3, 4)])
    assert state.winner is Mark.X


def test_vertical_line_wins():
    # Column 4: X at 2,4, O at 3,4, X at 4,4.
    state = play([(2, 4), (3, 4), (4, 4)])
    assert state.winner is Mark.X
    assert state.winning_lines == (((2, 4), (3, 4), (4, 4)),)


def test_diagonal_line_wins():
    # Down-right diagonal: X at 2,2, O at 3,3, X at 4,4.
    state = play([(2, 2), (3, 3), (4, 4)])
    assert state.winner is Mark.X
    assert state.winning_lines == (((2, 2), (3, 3), (4, 4)),)


def test_anti_diagonal_line_wins():
    # Down-left diagonal: X at 2,6, O at 3,5, X at 4,4.
    state = play([(2, 6), (3, 5), (4, 4)])
    assert state.winner is Mark.X
    assert state.winning_lines == (((2, 6), (3, 5), (4, 4)),)


def test_winning_move_may_be_any_cell_of_the_line():
    # Start of line: O at 5,6 and X at 5,7 already there, X plays 5,5.
    start = play([(5, 7), (5, 6), (5, 5)])
    # Middle: X at 5,5 and 5,7, O plays 5,6.
    middle = play([(5, 5), (0, 0), (5, 7), (5, 6)])
    # End: X at 5,5, O at 5,6, X plays 5,7.
    end = play([(5, 5), (5, 6), (5, 7)])
    for state in (start, middle, end):
        assert state.winning_lines == (((5, 5), (5, 6), (5, 7)),)


def test_winning_line_cells_are_in_board_order():
    # The line is reported top-to-bottom, left-to-right regardless of which
    # cell was played last, so the same line always prints the same way.
    state = play([(4, 4), (3, 5), (2, 6)])
    assert state.winning_lines == (((2, 6), (3, 5), (4, 4)),)


def test_same_marks_in_a_row_do_not_win():
    # X X X is a Tic-tac-toe win but not an OXOX win.
    state = play([(5, 5), (0, 0), (5, 6), (11, 11), (5, 7)])
    assert state.winner is None
    assert not is_terminal(state)


def test_two_of_a_kind_then_opposite_does_not_win():
    # X X O on row 5.
    state = play([(5, 5), (6, 6), (5, 6), (5, 7)])
    assert state.winner is None


def test_lines_do_not_wrap_around_the_board_edge():
    # 0,10 / 0,11 / 1,0 are consecutive in row-major order but not a line.
    state = play([(0, 10), (0, 11), (1, 0)])
    assert state.winner is None
    # Same for the left edge going backwards: 1,1 / 1,0 / 0,11.
    state = play([(1, 1), (1, 0), (0, 11)])
    assert state.winner is None


def test_patterns_with_a_gap_do_not_win():
    # X _ O _ X is not a line of three.
    state = play([(5, 3), (5, 5), (5, 7)])
    assert state.winner is None


def test_one_move_can_complete_several_lines():
    # X at 5,5 completes row 5 (X O X at 5,3 5,4 5,5) and column 5
    # (X O X at 3,5 4,5 5,5) at the same time.
    #   X: 5,3  3,5  5,5     O: 5,4  4,5
    state = play([(5, 3), (5, 4), (3, 5), (4, 5), (5, 5)])
    assert state.winner is Mark.X
    assert set(state.winning_lines) == {
        ((5, 3), (5, 4), (5, 5)),
        ((3, 5), (4, 5), (5, 5)),
    }


def test_no_legal_moves_after_a_win():
    state = play([(2, 4), (3, 4), (4, 4)])
    assert legal_moves(state) == []


# --- Draws -------------------------------------------------------------------


# A full 3x3 board with no alternating line anywhere:
#   X X X
#   X X O
#   O O O
DRAWN_3X3 = [(0, 0), (1, 2), (0, 1), (2, 0), (0, 2), (2, 1), (1, 0), (2, 2), (1, 1)]


def test_full_board_without_a_line_is_a_draw():
    state = play(DRAWN_3X3, size=3)
    assert state.winner is None
    assert is_draw(state)
    assert is_terminal(state)
    assert legal_moves(state) == []


def test_board_that_is_not_full_is_not_a_draw():
    state = play(DRAWN_3X3[:-1], size=3)
    assert not is_draw(state)
    assert not is_terminal(state)


def test_winning_on_the_last_empty_cell_is_a_win_not_a_draw():
    # Board after all moves (3x3):
    #   O X X
    #   X X O
    #   X O O
    # The final move, X at 1,1, fills the board and also completes O X O on
    # the main diagonal (0,0 1,1 2,2). A win takes precedence over a draw.
    moves = [(0, 1), (0, 0), (0, 2), (1, 2), (1, 0), (2, 1), (2, 0), (2, 2), (1, 1)]
    state = play(moves, size=3)
    assert state.winner is Mark.X
    assert not is_draw(state)
    assert is_terminal(state)


# --- Mark --------------------------------------------------------------------


def test_mark_opponent():
    assert Mark.X.opponent is Mark.O
    assert Mark.O.opponent is Mark.X


def test_mark_str_is_its_letter():
    assert str(Mark.X) == "X"
    assert str(Mark.O) == "O"


def test_state_type_is_exported():
    assert isinstance(initial_state(), State)


# --- Cross-check against a brute-force reference -----------------------------


def _reference_lines(state: State) -> set:
    """Every alternating line on the whole board, found by scanning every cell."""
    size = state.size
    found = set()
    for row in range(size):
        for col in range(size):
            for d_row, d_col in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                line = tuple((row + k * d_row, col + k * d_col) for k in range(3))
                if all(0 <= r < size and 0 <= c < size for r, c in line):
                    a, b, c = (state[cell] for cell in line)
                    if a is not None and b is not None and a == c != b:
                        found.add(line)
    return found


@pytest.mark.parametrize("size", [3, 4, 6, 12])
def test_random_games_agree_with_brute_force_scan(size):
    # Play random games. After every move, the incremental win check (only
    # lines through the new cell) must agree with a full-board scan: the game
    # continues while the board has no alternating line, and ends exactly
    # when the first ones appear.
    rng = random.Random(size)
    for _ in range(100):
        state = initial_state(size)
        while not is_terminal(state):
            state = apply_move(state, rng.choice(legal_moves(state)))
            assert set(state.winning_lines) == _reference_lines(state)
            assert (state.winner is not None) == bool(state.winning_lines)
