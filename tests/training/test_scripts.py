"""Tests for the command-line scripts' flag validation (scripts/train.py, scripts/evaluate.py).

A bad flag value must end in argparse's usage error (exit status 2, a one-line
`error:` message), never in a traceback from deep inside the run or the match.
The scripts run in a subprocess, exactly as a user starts them; every case
here is rejected before any checkpoint is read or any game is played.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from ox_zero.engine.search import SELF_PLAY
from ox_zero.training.evaluate import EvalConfig
from ox_zero.training.run import RunConfig

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

# Tiny training settings, so a regression (a bad flag no longer rejected)
# costs a second of training instead of hanging the suite on the 6x6 defaults.
TINY_TRAIN = [
    "--generations", "1", "--games", "1", "--parallel", "1", "--simulations", "1",
    "--batch-size", "1", "--steps", "1", "--blocks", "1", "--filters", "1",
    "--eval-games", "1", "--eval-ladder", "0", "--no-tensorboard", "--device", "cpu",
]


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("flag", ["--games", "--simulations"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_evaluate_rejects_non_positive_counts(tmp_path, flag, value):
    # The checkpoint does not exist: the flag check must come before loading.
    result = run_script(
        "evaluate.py", "--a", str(tmp_path / "missing.pt"), "--b", "uniform", flag, value
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert flag in result.stderr and "at least 1" in result.stderr


@pytest.mark.parametrize(
    "args, message",
    [
        (["--size", "2"], "size"),
        (["--generations", "-1"], "generations"),
        (["--name", "a/b"], "name"),
        (["--name", "benchmarks"], "benchmarks"),
        (["--name", "Benchmarks"], "benchmarks"),
        (["--temperature-cutoff", "-1"], "temperature_cutoff"),
        (["--dirichlet-epsilon", "1.5"], "dirichlet_epsilon"),
        (["--dirichlet-alpha", "0"], "dirichlet_alpha"),
        (["--c-base", "0"], "c_base"),
        (["--eval-simulations", "0"], "simulations"),
    ],
)
def test_train_rejects_bad_run_flags(tmp_path, args, message):
    # Later flags win in argparse, so `args` overrides TINY_TRAIN's values.
    result = run_script("train.py", "--root", str(tmp_path), *TINY_TRAIN, *args)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert message in result.stderr
    assert list(tmp_path.iterdir()) == []  # no run directory was created


# --- build_config (scripts/train.py loaded in-process) --------------------------


def load_script(name: str):
    """Import `scripts/<name>.py` as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_args(**overrides) -> argparse.Namespace:
    """The Namespace `train.py`'s parser gives for no flags, with `overrides`."""
    module = load_script("train.py")
    parser = module.make_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_build_config_defaults_equal_the_plan_04_configuration():
    # The test that keeps runs/six-a resumable: no new flag may change a field.
    args = train_args(name="x")
    config = load_script("train.py").build_config(args, "x")
    assert config.search == SELF_PLAY
    assert config.eval == EvalConfig(
        opponents=args.eval_opponents,
        ladder=args.eval_ladder,
        games_per_colour=args.eval_games,
        simulations=args.simulations,
    )
    assert isinstance(config, RunConfig)


def test_build_config_each_new_flag_reaches_its_field():
    module = load_script("train.py")
    config = module.build_config(
        train_args(
            temperature_cutoff=5,
            c_init=2.0,
            c_base=100.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.5,
            eval_simulations=400,
        ),
        "x",
    )
    assert config.search.temperature_cutoff == 5
    assert config.search.c_init == 2.0
    assert config.search.c_base == 100.0
    assert config.search.dirichlet_alpha == 0.3
    assert config.search.dirichlet_epsilon == 0.5
    assert config.search.root_noise
    assert config.eval.simulations == 400
    assert config.selfplay.simulations == 100  # untouched


def test_build_config_rejects_bad_search_values():
    module = load_script("train.py")
    with pytest.raises(ValueError):
        module.build_config(train_args(temperature_cutoff=-1), "x")


# --- scripts/benchmark.py (plan 06, step 3) ---------------------------------------


def make_checkpoint(path: Path, seed: int, size: int = 3) -> Path:
    """A tiny random 3x3 checkpoint; the seed makes distinct bytes."""
    import torch

    from ox_zero.engine.network import Network, NetworkConfig
    from ox_zero.training.checkpoint import save_checkpoint

    torch.manual_seed(seed)
    network = Network(size, NetworkConfig(blocks=1, filters=4, value_hidden=8))
    save_checkpoint(path, network, generation=1)
    return path


def bench(root: Path, model: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return run_script("benchmark.py", "--model", str(model), "--root", str(root), *flags)


def log_rows(root: Path, size: int = 3) -> list[dict]:
    import json

    path = root / f"{size}x{size}" / "matches.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def without_date(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "date"} for row in rows]


@pytest.fixture
def bench_root(tmp_path: Path) -> Path:
    root = tmp_path / "bench"
    make_checkpoint(root / "3x3" / "pool" / "p-gen_001.pt", seed=1)
    return root


def test_benchmark_plays_every_reference_once(tmp_path, bench_root):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    result = bench(bench_root, candidate)
    assert result.returncode == 0, result.stderr
    rows = log_rows(bench_root)
    assert sorted(r["b"] for r in rows) == ["p/gen_001", "uniform@100", "uniform@400"]
    assert all(r["a"] == "cand/gen_001" and r["openings"] == 9 for r in rows)
    assert all(r["wins"] + r["draws"] + r["losses"] == 18 for r in rows)
    import json

    ratings = json.loads((bench_root / "3x3" / "ratings.json").read_text())
    assert "cand/gen_001" in ratings["players"]
    assert "cand/gen_001" in result.stdout


def test_benchmark_is_repeatable_across_roots(tmp_path, bench_root):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    other = tmp_path / "bench2"
    make_checkpoint(other / "3x3" / "pool" / "p-gen_001.pt", seed=1)
    assert bench(bench_root, candidate).returncode == 0
    assert bench(other, candidate).returncode == 0
    assert without_date(log_rows(bench_root)) == without_date(log_rows(other))


def test_benchmark_rerun_plays_nothing_and_leaves_the_log_alone(tmp_path, bench_root):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    assert bench(bench_root, candidate).returncode == 0
    before = log_rows(bench_root)
    again = bench(bench_root, candidate)
    assert again.returncode == 2
    assert "already" in again.stderr and "Traceback" not in again.stderr
    assert log_rows(bench_root) == before


def test_benchmark_plays_only_a_new_reference(tmp_path, bench_root):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    assert bench(bench_root, candidate).returncode == 0
    make_checkpoint(bench_root / "3x3" / "pool" / "q-gen_002.pt", seed=3)
    assert bench(bench_root, candidate).returncode == 0
    assert [r["b"] for r in log_rows(bench_root)][-1] == "q/gen_002"
    assert len(log_rows(bench_root)) == 4


def test_a_pool_member_skips_itself(bench_root):
    make_checkpoint(bench_root / "3x3" / "pool" / "q-gen_002.pt", seed=3)
    member = bench_root / "3x3" / "pool" / "p-gen_001.pt"
    assert bench(bench_root, member).returncode == 0
    assert sorted(r["b"] for r in log_rows(bench_root)) == ["q/gen_002", "uniform@100", "uniform@400"]


def test_a_tagged_candidate_plays_its_untagged_entry(bench_root):
    make_checkpoint(bench_root / "3x3" / "pool" / "q-gen_002.pt", seed=3)
    member = bench_root / "3x3" / "pool" / "p-gen_001.pt"
    result = bench(bench_root, member, "--tag", "t", "--c-init", "3.0")
    assert result.returncode == 0, result.stderr
    rows = log_rows(bench_root)
    assert "p/gen_001" in [r["b"] for r in rows]
    assert all(r["a"] == "p/gen_001@t" and r["a_search"] == {"c_init": 3.0} for r in rows)


def test_the_same_name_with_different_bytes_is_refused(tmp_path, bench_root):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    assert bench(bench_root, candidate).returncode == 0
    make_checkpoint(candidate, seed=9)  # retrained, same name
    result = bench(bench_root, candidate)
    assert result.returncode == 2 and "cand/gen_001" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_same_bytes_under_a_new_name_is_refused(tmp_path, bench_root):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    assert bench(bench_root, candidate).returncode == 0
    copy = tmp_path / "other" / "gen_001.pt"
    copy.parent.mkdir()
    copy.write_bytes(candidate.read_bytes())
    result = bench(bench_root, copy)
    assert result.returncode == 2 and "other/gen_001" in result.stderr


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--tag", "t"], "--c-init"),
        (["--c-init", "3.0"], "--tag"),
        (["--tag", "t", "--c-init", "1.25"], "equal"),  # the defaults: a disguised self-match
    ],
)
def test_benchmark_flag_errors(tmp_path, bench_root, flags, message):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    result = bench(bench_root, candidate, *flags)
    assert result.returncode == 2 and message in result.stderr
    assert log_rows(bench_root) == []


def test_benchmark_refuses_a_non_cpu_device_on_the_default_root(tmp_path):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    result = run_script("benchmark.py", "--model", str(candidate), "--device", "mps")
    assert result.returncode == 2 and "cpu" in result.stderr


def test_benchmark_without_a_pool_names_the_manifest(tmp_path):
    candidate = make_checkpoint(tmp_path / "cand" / "gen_001.pt", seed=2)
    result = bench(tmp_path / "empty", candidate)
    assert result.returncode == 2 and "docs/experiments/pools.md" in result.stderr
