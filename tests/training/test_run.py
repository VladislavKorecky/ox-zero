"""Tests for the generation loop: `run`, checkpoints per generation, resume.

One generation of AlphaZero training is: self-play with the current network,
train on the replay buffer, play a tournament against earlier checkpoints,
log the numbers, save the buffer, and finally write `gen_NNN.pt`. The
checkpoint is written *last* so it acts as the commit marker: a generation
whose checkpoint exists happened completely, and anything a crash left
behind above the latest checkpoint is thrown away on resume.

These tests pin the run directory layout, the metrics row, the plan 03
handshake (the CLI loads what training writes), and above all resume: a run
stopped and restarted must produce exactly the same weights as one that ran
straight through (one RNG, buffer and optimiser state restored), and must
refuse to continue under a changed configuration.

Plan: docs/plans/04-training-pipeline.md, step 6. Tiny everything, CPU only.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ox_zero.cli.adapter import load_engine
from ox_zero.engine.network import NetworkConfig
from ox_zero.engine.search import SELF_PLAY, SearchConfig
from ox_zero.game.rules import apply_move, initial_state
from ox_zero.training.checkpoint import load_checkpoint
from ox_zero.training.evaluate import EvalConfig
from ox_zero.training.metrics import read_metrics
import ox_zero.training.run as run_module
from ox_zero.training.run import (
    ConfigMismatchError,
    RunConfig,
    RunDirectoryError,
    RunIdentity,
    checkpoint_name,
    latest_generation,
    run,
)
from ox_zero.training.selfplay import SelfPlayConfig
from ox_zero.training.trainer import TrainConfig

CPU = torch.device("cpu")

# The keys every metrics row carries, exactly (plan 04, step 6, test 2).
METRICS_KEYS = {
    "generation",
    "games",
    "examples",
    "buffer_size",
    "mean_game_length",
    "draw_rate",
    "x_win_rate",
    "selfplay_seconds",
    "selfplay_steps",
    "simulations_per_second",
    "train_seconds",
    "loss_total",
    "loss_policy",
    "loss_value",
    "eval_seconds",
    "elo",
    "scores",
    "wall_seconds",
}
# Wall-clock measurements differ between any two runs; everything else in a
# metrics row is a deterministic function of the seed.
TIMING_KEYS = {
    "selfplay_seconds",
    "simulations_per_second",
    "train_seconds",
    "eval_seconds",
    "wall_seconds",
}


def tiny(name: str = "tiny", generations: int = 2, seed: int = 0, **changes) -> RunConfig:
    """The plan's tiny configuration: 4x4, a 1-block network, a handful of games."""
    config = RunConfig(
        name=name,
        size=4,
        seed=seed,
        generations=generations,
        network=NetworkConfig(1, 4, 8),
        selfplay=SelfPlayConfig(games=4, parallel=4, simulations=6),
        train=TrainConfig(batch_size=8, steps_per_generation=2, buffer_generations=2),
        eval=EvalConfig(opponents=1, ladder=None, games_per_colour=1, simulations=4),
        tensorboard=False,
    )
    return replace(config, **changes)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pairings(run_dir: Path) -> list[tuple[int, int]]:
    return [(row["generation"], row["opponent"]) for row in read_jsonl(run_dir / "matches.jsonl")]


def ratings(run_dir: Path) -> dict[str, float]:
    return json.loads((run_dir / "ratings.json").read_text())


# 1. End to end ---------------------------------------------------------------


def test_end_to_end_writes_the_run_directory(tmp_path):
    config = tiny(generations=2)
    run_dir = run(config, root=tmp_path, device=CPU)

    assert run_dir == tmp_path / config.name
    for name in ("gen_000.pt", "gen_001.pt", "gen_002.pt"):
        assert (run_dir / name).is_file()
    assert sorted(p.name for p in (run_dir / "buffer").iterdir()) == ["gen_001.npz", "gen_002.npz"]

    rows = read_metrics(run_dir / "metrics.jsonl")
    assert [row["generation"] for row in rows] == [1, 2]
    assert pairings(run_dir) == [(1, 0), (2, 1)]
    table = ratings(run_dir)
    assert set(table) == {"0", "1", "2"}
    assert table["0"] == 0.0  # generation 0 is the anchor, 0 Elo by definition

    for generation in range(3):
        checkpoint = load_checkpoint(run_dir / f"gen_{generation:03d}.pt")
        assert checkpoint.generation == generation
        assert checkpoint.size == config.size
        configs = checkpoint.configs
        # Dataclasses are stored as field dicts (weights_only-safe) and rebuild exactly.
        assert SearchConfig(**configs["search"]) == config.search
        assert SelfPlayConfig(**configs["selfplay"]) == config.selfplay
        assert TrainConfig(**configs["train"]) == config.train
        assert EvalConfig(**configs["eval"]) == config.eval
        assert RunIdentity(**configs["run"]) == RunIdentity(config.name, config.seed)
        assert checkpoint.optimizer_state is not None
        assert checkpoint.rng is not None

    # The plan 03 handshake: what training writes, the CLI reads.
    engine, notice = load_engine(None, None, root=run_dir)
    assert "gen_002.pt" in notice
    state = apply_move(initial_state(4), (1, 1))
    analyses = list(engine.search(state, max_simulations=8))
    assert analyses and analyses[-1].chosen is not None


# 2. Metrics keys -------------------------------------------------------------


def test_metrics_rows_have_exactly_the_documented_keys(tmp_path):
    run_dir = run(tiny(generations=2), root=tmp_path, device=CPU)
    rows = read_metrics(run_dir / "metrics.jsonl")
    assert len(rows) == 2
    for row in rows:
        assert set(row) == METRICS_KEYS
        # Opponent generations become string keys in the JSON round trip.
        assert set(row["scores"]) == {str(row["generation"] - 1)}
        assert row["games"] == 4
        assert row["examples"] > 0
        assert 0.0 <= row["draw_rate"] <= 1.0
        assert 0.0 <= row["x_win_rate"] <= 1.0
    assert rows[1]["elo"] == ratings(run_dir)["2"]


# 3. Resume continues ---------------------------------------------------------


def test_resume_continues_to_the_new_target(tmp_path):
    config = tiny(generations=2)
    run_dir = run(config, root=tmp_path, device=CPU)
    first_mtime = os.stat(run_dir / "gen_001.pt").st_mtime_ns

    run(replace(config, generations=4), root=tmp_path, device=CPU)

    assert os.stat(run_dir / "gen_001.pt").st_mtime_ns == first_mtime  # not re-run
    assert (run_dir / "gen_004.pt").is_file()
    assert [row["generation"] for row in read_metrics(run_dir / "metrics.jsonl")] == [1, 2, 3, 4]
    assert set(ratings(run_dir)) == {"0", "1", "2", "3", "4"}


# 4. Resume is exact ----------------------------------------------------------


def test_resume_is_exact(tmp_path):
    """Stopping after generation 2 and resuming changes nothing.

    This is the test for the RNG restore (one NumPy generator plus torch's
    CPU state), the optimiser restore (Adam's moments and step counts) and
    the buffer restore (the same examples, in the same order, so the same
    random draws pick the same rows) together.
    """
    straight = run(tiny(name="straight", generations=4, seed=7), root=tmp_path, device=CPU)
    run(tiny(name="split", generations=2, seed=7), root=tmp_path, device=CPU)
    split = run(tiny(name="split", generations=4, seed=7), root=tmp_path, device=CPU)

    a = load_checkpoint(straight / "gen_004.pt").network.state_dict()
    b = load_checkpoint(split / "gen_004.pt").network.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"tensor {key} differs after resume"

    rows_a = read_metrics(straight / "metrics.jsonl")
    rows_b = read_metrics(split / "metrics.jsonl")
    assert len(rows_a) == len(rows_b) == 4
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        for key in METRICS_KEYS - TIMING_KEYS:
            assert row_a[key] == row_b[key], f"generation {row_a['generation']}: {key} differs"


# 5. The checkpoint is the commit marker --------------------------------------


def test_a_generation_without_its_checkpoint_is_rerun(tmp_path):
    config = tiny(generations=2)
    run_dir = run(config, root=tmp_path, device=CPU)
    # Simulate a crash between generation 2's buffer save and its checkpoint:
    # its metrics row, match row and buffer file exist, its checkpoint does not.
    (run_dir / "gen_002.pt").unlink()

    run(replace(config, generations=3), root=tmp_path, device=CPU)

    rows = read_metrics(run_dir / "metrics.jsonl")
    assert [row["generation"] for row in rows] == [1, 2, 3]  # the stale row 2 was dropped
    assert pairings(run_dir) == [(1, 0), (2, 1), (3, 2)]
    assert set(ratings(run_dir)) == {"0", "1", "2", "3"}
    assert (run_dir / "gen_002.pt").is_file()
    assert sorted(p.name for p in (run_dir / "buffer").iterdir()) == ["gen_002.npz", "gen_003.npz"]
    # The re-run's examples replaced the stale file instead of being appended to it.
    with np.load(run_dir / "buffer" / "gen_002.npz") as data:
        assert len(data["z"]) == rows[1]["examples"]


# 6. Resume refuses a changed config ------------------------------------------


def test_resume_refuses_a_changed_config(tmp_path):
    config = tiny(generations=2)
    run(config, root=tmp_path, device=CPU)

    with pytest.raises(ValueError, match="learning_rate"):
        run(replace(config, train=replace(config.train, learning_rate=0.5)), root=tmp_path,
            device=CPU)
    with pytest.raises(ValueError, match="seed"):
        run(replace(config, seed=1), root=tmp_path, device=CPU)
    with pytest.raises(ValueError, match="filters"):
        run(replace(config, network=NetworkConfig(1, 8, 8)), root=tmp_path, device=CPU)
    with pytest.raises(ValueError, match="size"):
        run(replace(config, size=5), root=tmp_path, device=CPU)
    with pytest.raises(ValueError, match="simulations"):
        run(replace(config, eval=replace(config.eval, simulations=5)), root=tmp_path, device=CPU)

    # Exempt: the target generation and the TensorBoard mirror.
    run(replace(config, tensorboard=True), root=tmp_path, device=CPU)
    run(replace(config, generations=3), root=tmp_path, device=CPU)
    assert (tmp_path / config.name / "gen_003.pt").is_file()


# 7. Missing buffer -----------------------------------------------------------


def test_missing_buffer_warns_and_continues(tmp_path, capsys):
    config = tiny(generations=2)
    run_dir = run(config, root=tmp_path, device=CPU)
    capsys.readouterr()
    for path in (run_dir / "buffer").iterdir():
        path.unlink()
    (run_dir / "buffer").rmdir()

    run(replace(config, generations=3), root=tmp_path, device=CPU)

    assert "buffer" in capsys.readouterr().err.lower()
    assert (run_dir / "gen_003.pt").is_file()


def test_no_warning_when_only_generation_zero_exists(tmp_path, capsys):
    """Interrupted during generation 1: only gen_000.pt, and no buffer yet.

    The buffer is first written after generation 1, so nothing was lost and
    a warning would be a false alarm.
    """
    config = tiny(generations=1)
    run_dir = run(config, root=tmp_path, device=CPU)
    (run_dir / "gen_001.pt").unlink()
    for path in (run_dir / "buffer").iterdir():
        path.unlink()
    (run_dir / "buffer").rmdir()
    capsys.readouterr()

    run(config, root=tmp_path, device=CPU)

    assert capsys.readouterr().err == ""
    assert (run_dir / "gen_001.pt").is_file()
    assert [row["generation"] for row in read_metrics(run_dir / "metrics.jsonl")] == [1]


# 8. The ladder plays ---------------------------------------------------------


def test_the_ladder_opponent_plays(tmp_path):
    """The ladder (generation g - ladder) is the anti-drift match: one long-range
    measurement per generation, so the Elo chain is not only neighbour links."""
    config = tiny(
        generations=3,
        eval=EvalConfig(opponents=1, ladder=2, games_per_colour=1, simulations=4),
    )
    run_dir = run(config, root=tmp_path, device=CPU)

    assert pairings(run_dir) == [(1, 0), (2, 1), (2, 0), (3, 2), (3, 1)]
    rows = read_metrics(run_dir / "metrics.jsonl")
    assert set(rows[2]["scores"]) == {"2", "1"}


def test_defaults_match_the_interface():
    config = RunConfig(name="x", size=6, seed=0, generations=1)
    assert config.search == SELF_PLAY
    assert config.network == NetworkConfig()
    assert config.tensorboard is True


# 9. A fresh run refuses a directory with a previous run's leftovers ----------


@pytest.mark.parametrize("leftover", ["metrics.jsonl", "matches.jsonl", "ratings.json", "buffer"])
def test_fresh_run_refuses_leftovers_without_a_checkpoint(tmp_path, leftover):
    """No `gen_*.pt` but old logs: starting fresh would append the new run's
    rows to the old run's, mixing two experiments in one curve. Nothing is
    deleted; the user decides."""
    run_dir = tmp_path / "tiny"
    run_dir.mkdir()
    if leftover == "buffer":
        (run_dir / "buffer").mkdir()
        (run_dir / "buffer" / "gen_001.npz").write_bytes(b"old")
    else:
        (run_dir / leftover).write_text("{}\n")

    with pytest.raises(RunDirectoryError, match="previous run"):
        run(tiny(generations=1), root=tmp_path, device=CPU)
    # A ValueError too, so callers catching ValueError keep working.
    assert issubclass(RunDirectoryError, ValueError)
    # Nothing was deleted, and no checkpoint was written.
    assert (run_dir / leftover).exists()
    assert latest_generation(run_dir) is None


def test_fresh_run_accepts_an_empty_directory(tmp_path):
    (tmp_path / "tiny").mkdir()
    (tmp_path / "tiny" / "buffer").mkdir()  # an empty buffer/ holds nothing
    run_dir = run(tiny(generations=1), root=tmp_path, device=CPU)
    assert (run_dir / "gen_001.pt").is_file()


# 10. Resume mismatch has its own exception type -------------------------------


def test_resume_mismatch_raises_config_mismatch_error(tmp_path):
    config = tiny(generations=1)
    run(config, root=tmp_path, device=CPU)
    with pytest.raises(ConfigMismatchError, match="seed"):
        run(replace(config, seed=1), root=tmp_path, device=CPU)
    assert issubclass(ConfigMismatchError, ValueError)


# 11. Checkpoint naming helpers --------------------------------------------------


def test_checkpoint_name_and_latest_generation(tmp_path):
    assert checkpoint_name(7) == "gen_007.pt"
    assert checkpoint_name(1234) == "gen_1234.pt"
    assert latest_generation(tmp_path / "missing") is None
    assert latest_generation(tmp_path) is None
    (tmp_path / "gen_002.pt").write_bytes(b"")
    (tmp_path / "gen_010.pt").mkdir()  # not a file: ignored
    (tmp_path / "gen_x.pt").write_bytes(b"")  # not a number: ignored
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "gen_009.pt").write_bytes(b"")  # not recursive
    assert latest_generation(tmp_path) == 2

