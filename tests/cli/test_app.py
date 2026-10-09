"""End-to-end tests of `ox-zero analyze` and `ox-zero best` through Typer's
test runner. docs/cli.md is the spec.

The runner's stdout is not a terminal, so all output here is plain text:
exactly what a pipe would receive.
"""

import json
import math

import pytest
import torch
from typer.testing import CliRunner

from ox_zero.cli import adapter as adapter_module
from ox_zero.cli import app as app_module
from ox_zero.cli.adapter import SearchEngine, load_engine
from ox_zero.cli.app import app
from ox_zero.cli.placeholder import PlaceholderEngine
from ox_zero.engine import network_evaluator
from ox_zero.engine.evaluator import UniformEvaluator
from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.network_evaluator import NetworkEvaluator
from ox_zero.game import play, to_board_string
from ox_zero.training.checkpoint import save_checkpoint

runner = CliRunner()

POSITION = ["5,5", "6,6", "5,6"]
FINISHED = ["5,5", "6,6", "5,7", "5,6"]


@pytest.fixture(autouse=True)
def fast_engine(request, monkeypatch):
    """Swap in a placeholder engine without the fake throughput limit, so tests don't sleep.

    The CLI tests are about output, not about which engine gets loaded, so
    they run on the fast, deterministic placeholder. Tests marked
    `real_engine` opt out and exercise the real loader.
    """
    calls = []
    if request.node.get_closest_marker("real_engine"):
        return calls

    def load(model, seed, device="cpu"):
        calls.append((model, seed))
        if model is not None and not model.exists():
            raise FileNotFoundError(f"model checkpoint not found: {model}")
        engine = PlaceholderEngine(seed=0 if seed is None else seed, rate=math.inf)
        return engine, "Using a test placeholder engine."

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


def test_engine_notice_goes_to_stderr():
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


# --- Loading the real engine ---------------------------------------------------
# These tests opt out of the placeholder fixture. They run with the working
# directory in `tmp_path`, so the default `checkpoints/` root is empty unless
# a test writes into it.

TINY = NetworkConfig(blocks=1, filters=4, value_hidden=8)


def write_checkpoint(path, size=12, generation=0, seed=0):
    torch.manual_seed(seed)
    save_checkpoint(path, Network(size, TINY), generation=generation)
    return path


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.real_engine
def test_no_model_searches_with_uniform_priors(workdir):
    engine, notice = load_engine(None, None, root=workdir / "none")
    assert isinstance(engine, SearchEngine)
    assert isinstance(engine.evaluator, UniformEvaluator)
    assert "uniform priors" in notice


@pytest.mark.real_engine
def test_no_model_cli_finds_a_win_in_one(workdir):
    # X at 0,0 and O at 0,1: X completes X O X at 0,2. Root expansion scores
    # all 142 moves up front and the winning child is a finished game with
    # Q = 1, so the uniform search finds it with a small budget.
    result = run("best", "0,0", "0,1", "--simulations", "50")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "0,2"
    assert "uniform priors" in result.stderr


@pytest.mark.real_engine
def test_model_flag_loads_the_checkpoint(workdir):
    path = write_checkpoint(workdir / "model.pt", generation=7)
    result = run("analyze", *POSITION, "--model", str(path), "--simulations", "20", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert len(report["moves"]) == 141
    assert str(path) in result.stderr
    assert "generation 7" in result.stderr

    engine, _ = load_engine(path, None)
    assert isinstance(engine.evaluator, NetworkEvaluator)
    assert engine.evaluator.device.type == "cpu"
    assert engine.size == 12


@pytest.mark.real_engine
def test_missing_model_path_is_an_error_from_the_real_loader(workdir):
    with pytest.raises(FileNotFoundError):
        load_engine(workdir / "missing.pt", None)


@pytest.mark.real_engine
def test_newest_checkpoint_is_picked(workdir):
    root = workdir / "checkpoints"
    write_checkpoint(root / "run" / "gen_001.pt", generation=1, seed=1)
    newest = write_checkpoint(root / "run" / "gen_002.pt", generation=2, seed=2)

    engine, notice = load_engine(None, None, root=root)
    assert "generation 2" in notice and str(newest) in notice
    # Same weights as the generation-2 file, not the generation-1 one.
    expected = torch.load(newest, weights_only=True)["model_state"]
    for key, value in engine.evaluator.network.state_dict().items():
        assert torch.equal(value.cpu(), expected[key]), key


@pytest.mark.real_engine
def test_default_root_is_checkpoints_in_the_working_directory(workdir):
    write_checkpoint(workdir / "checkpoints" / "gen_003.pt", generation=3)
    result = run("best", *POSITION, "--simulations", "5")
    assert result.exit_code == 0, result.output
    assert "generation 3" in result.stderr


@pytest.mark.real_engine
@pytest.mark.parametrize("command", ["analyze", "best", "sandbox"])
def test_board_size_mismatch_is_a_usage_error(workdir, launched, command):
    path = write_checkpoint(workdir / "small.pt", size=4)
    result = run(command, *POSITION, "--model", str(path))
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "4x4" in result.stderr and "12x12" in result.stderr
    assert launched == []


@pytest.mark.real_engine
def test_sandbox_checks_the_size_even_from_a_finished_position(workdir, launched):
    # Undo from a finished game reaches positions the background search
    # would then try, so the size must be checked before launch regardless.
    path = write_checkpoint(workdir / "small.pt", size=4)
    result = run("sandbox", *FINISHED, "--model", str(path))
    assert result.exit_code == 2
    assert "4x4" in result.stderr
    assert launched == []


@pytest.mark.real_engine
def test_device_cpu_is_accepted(workdir):
    path = write_checkpoint(workdir / "model.pt")
    result = run("best", *POSITION, "--model", str(path), "--device", "cpu", "--simulations", "3")
    assert result.exit_code == 0, result.output


@pytest.mark.real_engine
def test_device_auto_asks_select_device(workdir, monkeypatch):
    asked = []

    def fake_select_device(preference=None):
        asked.append(preference)
        return torch.device("cpu")

    monkeypatch.setattr(network_evaluator, "select_device", fake_select_device)
    path = write_checkpoint(workdir / "model.pt")
    result = run("best", *POSITION, "--model", str(path), "--device", "auto", "--simulations", "3")
    assert result.exit_code == 0, result.output
    assert asked == [None]


@pytest.mark.real_engine
@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_unavailable_device_is_a_usage_error(workdir):
    path = write_checkpoint(workdir / "model.pt")
    result = run("best", *POSITION, "--model", str(path), "--device", "cuda")
    assert result.exit_code == 2
    assert "cuda" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.real_engine
def test_invalid_device_is_rejected_and_help_lists_the_choices(workdir):
    assert run("best", *POSITION, "--device", "tpu").exit_code == 2
    help_text = run("best", "--help").stdout
    for choice in ("auto", "cpu", "mps", "cuda"):
        assert choice in help_text


@pytest.mark.real_engine
def test_device_is_ignored_without_a_model(workdir):
    result = run("best", "0,0", "0,1", "--device", "mps", "--simulations", "5")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "0,2"


@pytest.mark.real_engine
def test_the_cli_loader_is_the_adapter(workdir):
    assert app_module.load_engine is adapter_module.load_engine
