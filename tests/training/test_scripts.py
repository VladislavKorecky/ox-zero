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
