"""Tests for `analyze --live` (README, "Live analysis")."""

import io
import json
import math

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from ox_zero.cli import app as app_module
from ox_zero.cli.app import app
from ox_zero.cli.live import run_live
from ox_zero.cli.engine_port import Analysis
from ox_zero.cli.placeholder import PlaceholderEngine
from ox_zero.game import play

POSITION = play([(5, 5), (6, 6), (5, 6)])


def fast_engine() -> PlaceholderEngine:
    return PlaceholderEngine(seed=0, rate=math.inf)


def json_lines(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def live(engine, *, cap=None, json_output=True, interval=0.0, out=None):
    run_live(
        engine,
        POSITION,
        max_simulations=cap,
        top_n=3,
        last_move=None,
        json_output=json_output,
        out=out or Console(file=io.StringIO()),
        interval=interval,
    )


def test_json_lines_stream_reports_with_increasing_simulations(capsys):
    live(fast_engine(), cap=500)
    reports = json_lines(capsys)
    assert len(reports) > 2
    counts = [report["simulations"] for report in reports]
    assert counts == sorted(counts)
    assert list(reports[0])[:2] == ["simulations", "board"]


def test_last_line_is_the_final_report_at_the_cap(capsys):
    # With a long refresh interval only the first snapshot is due, but the
    # final report at the cap must still be written.
    live(fast_engine(), cap=3000, interval=60.0)
    reports = json_lines(capsys)
    assert [r["simulations"] for r in reports] == [50, 3000]


def test_refreshes_are_throttled(capsys):
    live(fast_engine(), cap=5000, interval=60.0)
    assert len(json_lines(capsys)) == 2


class InterruptedEngine:
    """Yields a few snapshots, then behaves as if Ctrl-C was pressed."""

    def search(self, state, max_simulations=None):
        for n in (100, 200):
            yield Analysis(value=0.5, scores={(0, 0): 0.5}, simulations=n)
        raise KeyboardInterrupt


def test_ctrl_c_ends_the_stream_cleanly(capsys):
    with pytest.raises(typer.Exit) as info:
        live(InterruptedEngine())
    assert info.value.exit_code == 130
    assert [r["simulations"] for r in json_lines(capsys)] == [100, 200]


def test_text_mode_renders_the_report_with_simulation_count():
    buffer = io.StringIO()
    live(fast_engine(), cap=400, json_output=False, out=Console(file=buffer, width=100))
    text = buffer.getvalue()
    assert "O to move (3 marks on board)" in text
    assert "400 simulations" in text
    assert "Top 3" in text


def test_cli_live_json_until_cap(monkeypatch):
    monkeypatch.setattr(app_module, "load_engine", lambda model, seed: fast_engine())
    result = CliRunner().invoke(
        app, ["analyze", "5,5", "6,6", "5,6", "--live", "--json", "--simulations", "1000"]
    )
    assert result.exit_code == 0
    reports = [json.loads(line) for line in result.stdout.splitlines()]
    assert reports[-1]["simulations"] == 1000
