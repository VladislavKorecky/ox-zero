"""Tests for scripts/experiments/opening_outcomes.py: per-opening match outcomes.

The script plays near-strength networks against each other from every opening
and tabulates, per opening class, X's win rate and how often an opening's pair
of games is *informative* (one player wins both). These tests pin the opening
classes and the pair arithmetic; the matches themselves are
`training.evaluate.play_match`, tested in tests/training.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "opening_outcomes.py"


@pytest.fixture(scope="module")
def oo():
    spec = importlib.util.spec_from_file_location("opening_outcomes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opening_outcomes"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "size, counts",
    [(8, {"corner": 4, "edge": 24, "ring2": 20, "interior": 16}),
     (12, {"corner": 4, "edge": 40, "ring2": 36, "interior": 64})],
)
def test_opening_classes_partition_the_board(oo, size, counts):
    classes = Counter(oo.opening_class((r, c), size) for r in range(size) for c in range(size))
    assert classes == counts


def test_opening_class_examples(oo):
    assert oo.opening_class((0, 0), 8) == "corner"
    assert oo.opening_class((0, 3), 8) == "edge"
    assert oo.opening_class((1, 1), 8) == "ring2"
    assert oo.opening_class((6, 3), 8) == "ring2"
    assert oo.opening_class((3, 4), 8) == "interior"


@pytest.mark.parametrize(
    "a_as_x, a_as_o, kind",
    [(1, 1, "informative"),     # a won both games
     (-1, -1, "informative"),   # b won both games
     (-1, 1, "o_both"),         # a lost as X, won as O: O won both
     (1, -1, "x_both"),         # X won both
     (0, 1, "other")],          # a draw is in the pair
)
def test_pair_kind(oo, a_as_x, a_as_o, kind):
    assert oo.pair_kind(a_as_x, a_as_o) == kind


def test_x_wins_in_a_pair(oo):
    # a as X won (+1): one X win. a as O lost (-1): b won as X, another.
    assert oo.x_wins(1, -1) == 2
    assert oo.x_wins(-1, 1) == 0
    assert oo.x_wins(0, 0) == 0


def test_tabulate_rates_per_class(oo):
    rows = [
        {"class": "edge", "a_as_x": 1, "a_as_o": 1},     # informative, 1 X win
        {"class": "edge", "a_as_x": -1, "a_as_o": 1},    # O both, 0 X wins
        {"class": "ring2", "a_as_x": 1, "a_as_o": -1},   # X both, 2 X wins
    ]
    table = oo.tabulate(rows)
    assert table["edge"] == {"pairs": 2, "x_win_rate": 0.25, "informative": 0.5, "o_both": 0.5, "x_both": 0.0}
    assert table["ring2"]["x_both"] == 1.0 and table["ring2"]["x_win_rate"] == 1.0
    assert table["all"]["pairs"] == 3


def test_duplicate_checkpoints_are_refused(oo, tmp_path):
    import subprocess
    path = tmp_path / "x.pt"
    path.write_bytes(b"")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), str(path)], capture_output=True, text=True
    )
    assert proc.returncode != 0 and "twice" in proc.stderr
