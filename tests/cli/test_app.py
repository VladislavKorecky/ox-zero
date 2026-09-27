"""End-to-end tests of `ox-zero analyze` and `ox-zero best` through Typer's
test runner. The README's Usage section is the spec.

The runner's stdout is not a terminal, so all output here is plain text:
exactly what a pipe would receive.
"""

import json
import math

import pytest
from typer.testing import CliRunner

from ox_zero.cli import app as app_module
from ox_zero.cli.app import app
from ox_zero.engine import DummyEngine
from ox_zero.game import play, to_board_string

runner = CliRunner()

POSITION = ["5,5", "6,6", "5,6"]
FINISHED = ["5,5", "6,6", "5,7", "5,6"]


@pytest.fixture(autouse=True)
def fast_engine(monkeypatch):
    """Swap in a dummy without the fake throughput limit, so tests don't sleep."""
    calls = []

    def load(model, seed):
        calls.append((model, seed))
        if model is not None and not model.exists():
            raise FileNotFoundError(f"model checkpoint not found: {model}")
        return DummyEngine(seed=0 if seed is None else seed, rate=math.inf)

    monkeypatch.setattr(app_module, "load_engine", load)
    return calls


def run(*args):
    return runner.invoke(app, list(args))


# --- analyze -----------------------------------------------------------------


def test_analyze_prints_the_report():
    result = run("analyze", *POSITION)
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert out.startswith("O to move (3 marks on board)")
    assert "       0   1   2   3   4   5   6   7   8   9  10  11" in out
    assert "Eval " in out and "for O" in out
    assert "Top 3" in out


def test_analyze_top_sets_the_candidate_count():
    out = run("analyze", *POSITION, "--top", "5").stdout
    assert "Top 5" in out
    assert "  5. " in out


def test_analyze_accepts_a_board_string():
    board = to_board_string(play([(5, 5), (6, 6), (5, 6)]))
    result = run("analyze", board)
    assert result.exit_code == 0
    assert result.stdout.startswith("O to move (3 marks on board)")


def test_analyze_json_has_the_readme_shape():
    result = run("analyze", *POSITION, "--json")
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert list(report) == ["board", "to_move", "value", "moves", "top", "result"]
    assert len(report["moves"]) == 141
    assert len(report["top"]) == 3


def test_seed_makes_output_reproducible():
    a = run("analyze", *POSITION, "--json", "--seed", "3").stdout
    b = run("analyze", *POSITION, "--json", "--seed", "3").stdout
    assert a == b


def test_seed_and_model_reach_the_engine_loader(fast_engine, tmp_path):
    checkpoint = tmp_path / "m.pt"
    checkpoint.write_bytes(b"")
    run("analyze", *POSITION, "--seed", "9", "--model", str(checkpoint))
    assert fast_engine[-1] == (checkpoint, 9)


def test_empty_position_analyses_the_empty_board():
    result = run("analyze")
    assert result.exit_code == 0
    assert result.stdout.startswith("X to move (0 marks on board)")


def test_analyze_finished_game_reports_the_result_without_searching(fast_engine):
    result = run("analyze", *FINISHED)
    assert result.exit_code == 0
    assert result.stdout.startswith("Game over: O wins (X O X at 5,5 5,6 5,7)")
    assert "  5    .   .   .   .   .   X   O   X" in result.stdout
    assert fast_engine == []


def test_analyze_finished_game_json():
    report = json.loads(run("analyze", *FINISHED, "--json").stdout)
    assert report["result"] == "O"


def test_placeholder_engine_notice_goes_to_stderr():
    result = run("analyze", *POSITION, "--json")
    assert "placeholder" in result.stderr.lower()
    json.loads(result.stdout)  # stdout stays clean JSON


# --- errors ------------------------------------------------------------------


def test_illegal_move_shows_a_caret_under_the_token():
    result = run("analyze", "5,5", "6,6", "5,5")
    assert result.exit_code == 2
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert lines[0] == "Error: cannot play 5,5: cell is occupied"
    assert lines[1] == "  5,5 6,6 5,5"
    assert lines[2] == "          ^^^"


def test_bad_board_string_is_an_error():
    result = run("analyze", "O" + "_" * 143)
    assert result.exit_code == 2
    assert "count" in result.stderr


def test_missing_model_is_an_error(tmp_path):
    result = run("analyze", *POSITION, "--model", str(tmp_path / "missing.pt"))
    assert result.exit_code == 2
    assert "not found" in result.stderr


@pytest.mark.parametrize("flag", ["--top", "--simulations"])
def test_counts_must_be_positive(flag):
    assert run("analyze", *POSITION, flag, "0").exit_code == 2


# --- best --------------------------------------------------------------------


def test_best_prints_only_the_move():
    result = run("best", *POSITION)
    assert result.exit_code == 0
    assert result.stdout.strip().count("\n") == 0
    row, col = map(int, result.stdout.strip().split(","))
    assert play([(5, 5), (6, 6), (5, 6)])[row, col] is None


def test_best_agrees_with_analyze_top_move():
    best = run("best", *POSITION, "--seed", "4").stdout.strip()
    report = json.loads(run("analyze", *POSITION, "--seed", "4", "--json").stdout)
    assert best == "{},{}".format(*report["top"][0]["move"])


def test_best_json():
    result = run("best", *POSITION, "--json")
    data = json.loads(result.stdout)
    assert list(data) == ["move", "score"]


def test_best_on_a_finished_game_exits_1():
    result = run("best", *FINISHED)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Game over: O wins" in result.stderr


def test_best_json_on_a_finished_game():
    result = run("best", *FINISHED, "--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"move": None, "score": None, "result": "O"}


@pytest.mark.parametrize("flag", ["--live", "--top"])
def test_best_rejects_flags_that_do_not_apply(flag):
    args = [flag] if flag == "--live" else [flag, "3"]
    assert run("best", *POSITION, *args).exit_code == 2


# --- sandbox -----------------------------------------------------------------


@pytest.fixture
def launched(monkeypatch):
    """Capture what `sandbox` would launch instead of taking over the terminal."""
    calls = []
    monkeypatch.setattr(
        app_module, "run_sandbox", lambda session, engine, top_n: calls.append((session, top_n))
    )
    return calls


def test_sandbox_launches_with_the_position(launched):
    result = run("sandbox", *POSITION, "--top", "5")
    assert result.exit_code == 0, result.output
    session, top_n = launched[0]
    assert session.history == [(5, 5), (6, 6), (5, 6)]
    assert top_n == 5


def test_sandbox_without_a_position_starts_empty(launched):
    run("sandbox")
    assert launched[0][0].history == []


def test_sandbox_rejects_an_invalid_position(launched):
    result = run("sandbox", "5,5", "5,5")
    assert result.exit_code == 2
    assert "^^^" in result.stderr
    assert launched == []


@pytest.mark.parametrize("flag", ["--json", "--live", "--simulations"])
def test_sandbox_rejects_flags_that_do_not_apply(launched, flag):
    args = [flag, "5"] if flag == "--simulations" else [flag]
    assert run("sandbox", *args).exit_code == 2
