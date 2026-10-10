"""Tests for the command-line scripts' flag validation (scripts/train.py, scripts/evaluate.py).

A bad flag value must end in argparse's usage error (exit status 2, a one-line
`error:` message), never in a traceback from deep inside the run or the match.
The scripts run in a subprocess, exactly as a user starts them; every case
here is rejected before any checkpoint is read or any game is played.
"""

import subprocess
import sys
from pathlib import Path

import pytest

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
    ],
)
def test_train_rejects_bad_run_flags(tmp_path, args, message):
    # Later flags win in argparse, so `args` overrides TINY_TRAIN's values.
    result = run_script("train.py", "--root", str(tmp_path), *TINY_TRAIN, *args)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert message in result.stderr
    assert list(tmp_path.iterdir()) == []  # no run directory was created
