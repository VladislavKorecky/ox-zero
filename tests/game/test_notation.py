"""Tests for position notation: cells, move lists, and board strings.

The formats are specified in the README's "Positions and moves" section.
Cells are `row,col`, zero-based, row 0 at the top.
"""

import pytest

from ox_zero.game import (
    Mark,
    PositionError,
    format_cell,
    from_board_string,
    initial_state,
    parse_cell,
    parse_position,
    play,
    to_board_string,
)


def board_string(marks: dict[tuple[int, int], str], size: int = 12) -> str:
    """Build a board string with the given marks and `_` everywhere else."""
    cells = ["_"] * (size * size)
    for (row, col), mark in marks.items():
        cells[row * size + col] = mark
    return "".join(cells)


# --- Cells -------------------------------------------------------------------


def test_parse_cell():
    assert parse_cell("5,7") == (5, 7)
    assert parse_cell("0,11") == (0, 11)


def test_parse_cell_tolerates_spaces_around_the_comma():
    assert parse_cell(" 5 , 7 ") == (5, 7)


@pytest.mark.parametrize("text", ["", "5", "5,", ",7", "5;7", "a,b", "-1,3", "5,7,1", "5.7"])
def test_parse_cell_rejects_malformed_input(text):
    with pytest.raises(PositionError):
        parse_cell(text)


def test_format_cell():
    assert format_cell((5, 7)) == "5,7"


def test_parse_and_format_round_trip():
    assert parse_cell(format_cell((10, 3))) == (10, 3)


# --- Move lists --------------------------------------------------------------


def test_no_tokens_is_the_empty_board():
    assert parse_position([]) == initial_state()


def test_move_list_is_replayed_x_first():
    assert parse_position(["5,5", "6,6", "5,6"]) == play([(5, 5), (6, 6), (5, 6)])


def test_malformed_move_reports_its_token_index():
    with pytest.raises(PositionError) as info:
        parse_position(["5,5", "oops", "5,6"])
    assert info.value.token_index == 1


def test_single_malformed_cell_is_not_mistaken_for_a_board_string():
    with pytest.raises(PositionError, match="row,col"):
        parse_position(["5"])


def test_illegal_move_reports_its_token_index_and_reason():
    with pytest.raises(PositionError) as info:
        parse_position(["5,5", "6,6", "5,5"])
    assert info.value.token_index == 2
    assert "occupied" in str(info.value)


def test_off_board_move_is_rejected():
    with pytest.raises(PositionError) as info:
        parse_position(["12,0"])
    assert info.value.token_index == 0


def test_move_after_game_over_is_rejected():
    with pytest.raises(PositionError) as info:
        parse_position(["5,5", "6,6", "5,7", "5,6", "0,0"])
    assert info.value.token_index == 4


# --- Board strings -----------------------------------------------------------


def test_empty_board_string():
    assert to_board_string(initial_state()) == "_" * 144


def test_board_string_is_row_major():
    state = play([(0, 1), (1, 0)])
    s = to_board_string(state)
    assert s[1] == "X"
    assert s[12] == "O"
    assert s.count("_") == 142


def test_board_string_round_trip_matches_move_list():
    state = play([(5, 5), (6, 6), (5, 6)])
    assert from_board_string(to_board_string(state)) == state


def test_parse_position_detects_a_board_string():
    state = play([(5, 5), (6, 6), (5, 6)])
    assert parse_position([to_board_string(state)]) == state


def test_side_to_move_comes_from_mark_counts():
    assert from_board_string(board_string({(0, 0): "X"})).to_move is Mark.O
    assert from_board_string(board_string({(0, 0): "X", (0, 5): "O"})).to_move is Mark.X


@pytest.mark.parametrize(
    "marks",
    [
        {(0, 0): "O"},  # O first
        {(0, 0): "X", (0, 5): "X"},  # two extra X
    ],
)
def test_impossible_mark_counts_are_rejected(marks):
    with pytest.raises(PositionError, match="count"):
        from_board_string(board_string(marks))


def test_wrong_length_is_rejected():
    with pytest.raises(PositionError, match="144"):
        from_board_string("_" * 143)


def test_bad_character_is_rejected():
    with pytest.raises(PositionError, match="character"):
        from_board_string("_" * 10 + "Z" + "_" * 133)


def test_board_string_accepts_other_sizes():
    state = from_board_string("X" + "_" * 24, size=5)
    assert state.size == 5
    assert state[0, 0] is Mark.X


def test_parse_position_reports_a_bad_board_string_at_token_zero():
    with pytest.raises(PositionError) as info:
        parse_position(["O" + "_" * 143])
    assert info.value.token_index == 0


# --- Board strings: decided positions ----------------------------------------


def test_decided_board_string_credits_the_last_mover():
    # X O X across row 5 with one extra O elsewhere: equal counts, so X is to
    # move and O moved last. O must have placed the completing mark.
    state = from_board_string(
        board_string({(5, 5): "X", (5, 6): "O", (5, 7): "X", (6, 6): "O"})
    )
    assert state.winner is Mark.O
    assert state.winning_lines == (((5, 5), (5, 6), (5, 7)),)


def test_decided_board_string_equals_the_same_game_played_out():
    moves = [(5, 5), (6, 6), (5, 7), (5, 6)]
    state = play(moves)
    assert from_board_string(to_board_string(state)) == state


def test_line_containing_a_last_mover_mark_is_reachable():
    # O X O across row 0 plus a stray X: equal counts, X to move, O moved last.
    # Either O on the line could have been the completing move.
    s = board_string({(0, 0): "O", (0, 1): "X", (0, 2): "O", (5, 5): "X"})
    assert from_board_string(s).winner is Mark.O


def test_lines_sharing_no_last_mover_mark_are_unreachable():
    # Two separate X O X lines and equal counts: O moved last. One move
    # completes every line it is on, so a single O would have to lie on both
    # lines. The O's are on different lines, so no single move explains this.
    bad = board_string(
        {
            (0, 0): "X", (0, 1): "O", (0, 2): "X",
            (9, 0): "X", (9, 1): "O", (9, 2): "X",
            (5, 5): "O", (5, 7): "O",
        }
    )
    with pytest.raises(PositionError, match="unreachable"):
        from_board_string(bad)


def test_two_lines_through_one_last_mover_mark_is_reachable():
    # Row 5 reads X O X (5,4 5,5 5,6) and column 6 reads X O X (5,6 6,6 7,6).
    # The lines share the X at 5,6.
    marks = {
        (5, 4): "X", (5, 5): "O", (5, 6): "X",
        (6, 6): "O", (7, 6): "X",
        (0, 0): "O",
    }
    # With the stray O the counts are equal, so O moved last, but no O lies
    # on both lines: unreachable.
    with pytest.raises(PositionError, match="unreachable"):
        from_board_string(board_string(marks))
    # Without it X moved last, and X at 5,6 completed both lines at once.
    del marks[(0, 0)]
    state = from_board_string(board_string(marks))
    assert state.winner is Mark.X
    assert len(state.winning_lines) == 2


def test_full_board_without_lines_is_a_draw():
    # Draw fixture from test_rules: X X X / X X O / O O O on 3x3.
    state = from_board_string("XXXXXOOOO", size=3)
    assert state.winner is None
    assert state.move_count == 9
