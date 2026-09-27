"""Tests for the Rich renderables shared by `analyze`, live analysis, and the
sandbox. Layout is checked on plain text; colour on a few spot-checked styles.
"""

from rich.console import Console

from ox_zero.cli import render
from ox_zero.engine import Analysis
from ox_zero.game import initial_state, legal_moves, play

POSITION = play([(5, 5), (6, 6), (5, 6)])


def plain(renderable) -> str:
    console = Console(width=120, color_system=None, record=True, file=open("/dev/null", "w"))
    console.print(renderable)
    return console.export_text()


def flat_analysis(state, score: float = 0.1, overrides=None) -> Analysis:
    scores = {move: score for move in legal_moves(state)}
    scores.update(overrides or {})
    return Analysis(value=max(scores.values()), scores=scores, simulations=800)


def style_at(text, offset):
    return text.get_style_at_offset(Console(), offset)


# --- Board -------------------------------------------------------------------


def test_board_layout_matches_the_readme():
    analysis = flat_analysis(POSITION, 0.08, {(5, 7): 0.47})
    lines = plain(render.board(POSITION, analysis)).splitlines()
    assert lines[0] == "       0   1   2   3   4   5   6   7   8   9  10  11"
    assert lines[6] == "  5    8   8   8   8   8   X   X  47   8   8   8   8"
    assert lines[7] == "  6    8   8   8   8   8   8   O   8   8   8   8   8"
    assert lines[12] == " 11    8   8   8   8   8   8   8   8   8   8   8   8"
    assert len(lines) == 13


def test_board_rounds_scores_to_whole_percent_and_fits_100():
    analysis = flat_analysis(POSITION, 0.0, {(0, 0): 1.0, (0, 1): 0.555})
    row0 = plain(render.board(POSITION, analysis)).splitlines()[1]
    assert row0.startswith("  0  100  56   0")


def test_board_without_analysis_shows_dots():
    row5 = plain(render.board(POSITION, None)).splitlines()[6]
    assert row5 == "  5    .   .   .   .   .   X   X   .   .   .   .   ."


def test_board_takes_its_size_from_the_state():
    lines = plain(render.board(initial_state(5), None)).splitlines()
    assert lines[0] == "       0   1   2   3   4"
    assert len(lines) == 6


def test_empty_cells_get_a_heat_background_that_rises_with_score():
    analysis = flat_analysis(POSITION, 0.0, {(0, 1): 1.0})
    text = render.board(POSITION, analysis)
    # Line 0 of the Text is the column header; board row 0 starts after it.
    offset_row1 = len(text.plain.splitlines()[0]) + 1
    low = style_at(text, offset_row1 + 5)   # cell 0,0
    high = style_at(text, offset_row1 + 9)  # cell 0,1
    assert low.bgcolor is not None and high.bgcolor is not None
    assert low.bgcolor != high.bgcolor


def test_marks_have_distinct_colours():
    text = render.board(POSITION, None)
    lines = text.plain.splitlines(keepends=True)
    row5 = sum(len(line) for line in lines[:6])
    row6 = sum(len(line) for line in lines[:7])
    x = style_at(text, row5 + 4 + 5 * 4 + 3)
    o = style_at(text, row6 + 4 + 6 * 4 + 3)
    assert text.plain[row5 + 4 + 5 * 4 + 3] == "X"
    assert text.plain[row6 + 4 + 6 * 4 + 3] == "O"
    assert x.color != o.color
    assert x.bold and o.bold


def test_last_move_is_underlined():
    text = render.board(POSITION, None, last_move=(5, 6))
    lines = text.plain.splitlines(keepends=True)
    row5 = sum(len(line) for line in lines[:6])
    assert style_at(text, row5 + 4 + 6 * 4 + 3).underline


def test_winning_line_is_highlighted():
    finished = play([(5, 5), (6, 6), (5, 7), (5, 6)])
    text = render.board(finished, None)
    lines = text.plain.splitlines(keepends=True)
    row5 = sum(len(line) for line in lines[:6])
    row6 = sum(len(line) for line in lines[:7])
    on_line = style_at(text, row5 + 4 + 6 * 4 + 3)
    off_line = style_at(text, row6 + 4 + 6 * 4 + 3)
    assert on_line.bgcolor is not None
    assert off_line.bgcolor is None


# --- Eval bar, top list, status ----------------------------------------------


def test_eval_bar_matches_the_readme_example():
    assert plain(render.eval_bar(0.47, POSITION.to_move)).rstrip("\n") == (
        "Eval ███████████▊░░░░░░░░░░░░░ 47% for O"
    )


def test_eval_bar_extremes():
    assert "█" * 25 in plain(render.eval_bar(1.0, POSITION.to_move))
    assert "░" * 25 in plain(render.eval_bar(0.0, POSITION.to_move))


def test_top_list_matches_the_readme_layout():
    analysis = flat_analysis(POSITION, 0.1, {(5, 7): 0.47, (4, 6): 0.38, (6, 5): 0.36})
    assert plain(render.top_list(analysis, 3)).splitlines() == [
        "Top 3",
        "  1. 5,7   47%",
        "  2. 4,6   38%",
        "  3. 6,5   36%",
    ]


def test_top_list_aligns_two_digit_coordinates():
    analysis = flat_analysis(POSITION, 0.1, {(10, 11): 0.5})
    assert plain(render.top_list(analysis, 1)).splitlines()[1] == "  1. 10,11 50%"


def test_status_line():
    assert plain(render.status_line(POSITION)).strip() == "O to move (3 marks on board)"
    assert plain(render.status_line(play([(0, 0)]))).strip() == "O to move (1 mark on board)"


def test_game_over_win():
    finished = play([(5, 5), (6, 6), (5, 7), (5, 6)])
    assert plain(render.game_over(finished)).strip() == (
        "Game over: O wins (X O X at 5,5 5,6 5,7)"
    )


def test_game_over_draw():
    drawn = play([(0, 0), (1, 2), (0, 1), (2, 0), (0, 2), (2, 1), (1, 0), (2, 2), (1, 1)], size=3)
    assert plain(render.game_over(drawn)).strip() == "Game over: draw (board full)"


def test_report_combines_the_sections():
    analysis = flat_analysis(POSITION, 0.1, {(5, 7): 0.47})
    text = plain(render.report(POSITION, analysis, top_n=3))
    assert text.startswith("O to move (3 marks on board)")
    assert "Eval " in text
    assert "Top 3" in text
