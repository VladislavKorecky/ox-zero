"""Tests for the sandbox's command logic, independent of the screen.

The commands are specified in the README's `sandbox` section.
"""

import pytest

from ox_zero.cli.sandbox.session import Session
from ox_zero.game import PositionError, initial_state, play, to_board_string


def session(*moves: str) -> Session:
    return Session.from_tokens(list(moves))


# --- Starting positions ------------------------------------------------------


def test_starts_on_the_empty_board():
    s = session()
    assert s.state == initial_state()
    assert s.history == []


def test_starts_from_a_move_list_with_history():
    s = session("5,5", "6,6")
    assert s.state == play([(5, 5), (6, 6)])
    assert s.history == [(5, 5), (6, 6)]
    assert s.last_move == (6, 6)


def test_starts_from_a_board_string_without_history():
    board = to_board_string(play([(5, 5), (6, 6)]))
    s = session(board)
    assert s.state == play([(5, 5), (6, 6)])
    assert s.history == []
    assert s.last_move is None


def test_invalid_start_raises():
    with pytest.raises(PositionError):
        session("5,5", "5,5")


# --- Playing moves -----------------------------------------------------------


def test_playing_a_move():
    s = session()
    result = s.execute("5,5")
    assert result.position_changed
    assert result.message is None and not result.error
    assert s.history == [(5, 5)]
    assert s.last_move == (5, 5)


def test_illegal_move_is_an_error_message_not_an_exception():
    s = session("5,5")
    result = s.execute("5,5")
    assert result.error
    assert "occupied" in result.message
    assert not result.position_changed
    assert s.history == [(5, 5)]


def test_move_after_game_over_is_an_error():
    s = session("5,5", "6,6", "5,7", "5,6")
    assert s.execute("0,0").error


def test_play_method_for_clicks():
    s = session()
    assert s.play((3, 4)).position_changed
    assert s.history == [(3, 4)]


# --- Undo / redo -------------------------------------------------------------


def test_undo_and_redo():
    s = session("5,5", "6,6")
    assert s.execute("undo").position_changed
    assert s.history == [(5, 5)]
    assert s.execute("redo").position_changed
    assert s.history == [(5, 5), (6, 6)]


def test_undo_with_nothing_to_undo():
    result = session().execute("undo")
    assert result.error and not result.position_changed


def test_redo_with_nothing_to_redo():
    result = session("5,5").execute("redo")
    assert result.error and not result.position_changed


def test_new_move_discards_redo_history():
    s = session("5,5", "6,6")
    s.execute("undo")
    s.execute("7,7")
    assert s.execute("redo").error
    assert s.history == [(5, 5), (7, 7)]


def test_undo_stops_at_a_loaded_board_string():
    board = to_board_string(play([(5, 5), (6, 6)]))
    s = session(board)
    s.execute("0,0")
    s.execute("undo")
    assert s.state == play([(5, 5), (6, 6)])
    assert s.execute("undo").error


# --- Pause / resume ----------------------------------------------------------


def test_pause_and_resume():
    s = session()
    assert not s.paused
    s.execute("pause")
    assert s.paused
    s.execute("resume")
    assert not s.paused


def test_toggle_pause():
    s = session()
    s.toggle_pause()
    assert s.paused
    s.toggle_pause()
    assert not s.paused


def test_moves_still_play_while_paused():
    s = session()
    s.execute("pause")
    assert s.execute("5,5").position_changed


# --- Load / reset / export ---------------------------------------------------


def test_load_a_move_list():
    s = session("0,0")
    assert s.execute("load 5,5 6,6").position_changed
    assert s.history == [(5, 5), (6, 6)]


def test_load_a_board_string():
    board = to_board_string(play([(5, 5)]))
    s = session()
    s.execute("load " + board)
    assert s.state == play([(5, 5)])
    assert s.history == []


def test_load_clears_redo():
    s = session("5,5")
    s.execute("undo")
    s.execute("load 1,1")
    assert s.execute("redo").error


def test_bad_load_keeps_the_position():
    s = session("5,5")
    result = s.execute("load 1,1 1,1")
    assert result.error
    assert s.history == [(5, 5)]


def test_load_without_a_position_is_an_error():
    assert session().execute("load").error


def test_reset():
    s = session("5,5", "6,6")
    assert s.execute("reset").position_changed
    assert s.state == initial_state()
    assert s.history == []


def test_export_shows_the_board_string():
    s = session("5,5")
    result = s.execute("export")
    assert result.message == to_board_string(s.state)
    assert not result.error


# --- Other input -------------------------------------------------------------


@pytest.mark.parametrize("line", ["quit", "exit", "QUIT"])
def test_quit(line):
    assert session().execute(line).quit


def test_blank_line_does_nothing():
    result = session().execute("   ")
    assert not result.position_changed and not result.error and result.message is None


def test_unknown_command():
    result = session().execute("fly")
    assert result.error
    assert "help" in result.message


def test_help_lists_commands():
    result = session().execute("help")
    assert "undo" in result.message and "export" in result.message


def test_history_text_numbers_move_pairs():
    assert session("5,5", "6,6", "5,6").history_text() == "1. 5,5 6,6  2. 5,6"
    assert session().history_text() == ""
