"""Tests for the JSON report shape specified in the README (`--json`)."""

import json

from ox_zero.cli.report import best_json, report_json
from ox_zero.engine import Analysis
from ox_zero.game import legal_moves, play, to_board_string

POSITION = play([(5, 5), (6, 6), (5, 6)])


def analysis_for(state) -> Analysis:
    scores = {move: 0.1 for move in legal_moves(state)}
    scores[(5, 7)] = 0.47
    scores[(4, 6)] = 0.38
    return Analysis(value=0.47, scores=scores, simulations=800)


def test_report_has_the_readme_shape():
    report = report_json(POSITION, analysis_for(POSITION), top_n=2)
    assert list(report) == ["board", "to_move", "value", "moves", "top", "result"]
    assert report["board"] == to_board_string(POSITION)
    assert report["to_move"] == "O"
    assert report["value"] == 0.47
    assert report["result"] is None
    assert report["top"] == [
        {"move": [5, 7], "score": 0.47},
        {"move": [4, 6], "score": 0.38},
    ]


def test_moves_list_every_legal_move_in_board_order():
    report = report_json(POSITION, analysis_for(POSITION), top_n=3)
    assert [tuple(m["move"]) for m in report["moves"]] == legal_moves(POSITION)


def test_live_report_puts_simulations_first():
    report = report_json(POSITION, analysis_for(POSITION), top_n=3, include_simulations=True)
    assert list(report)[0] == "simulations"
    assert report["simulations"] == 800


def test_finished_game_report():
    finished = play([(5, 5), (6, 6), (5, 7), (5, 6)])
    report = report_json(finished, None, top_n=3)
    assert report["result"] == "O"
    assert report["moves"] == []
    assert report["top"] == []
    assert report["value"] is None
    assert report["to_move"] is None


def test_scores_are_rounded_for_readability():
    scores = {move: 0.123456789 for move in legal_moves(POSITION)}
    report = report_json(POSITION, Analysis(0.123456789, scores, 1), top_n=1)
    assert report["value"] == 0.1235


def test_report_is_json_serialisable():
    json.dumps(report_json(POSITION, analysis_for(POSITION), top_n=3))


def test_best_json():
    assert best_json(analysis_for(POSITION)) == {"move": [5, 7], "score": 0.47}
