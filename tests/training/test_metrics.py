"""Tests for the metrics writer: one JSON Lines row per generation, mirrored
to TensorBoard.

Design: docs/design/training.md ("Logging": plain files, dashboards are
readers) and docs/plans/archive/04-training-pipeline.md (Decisions "Metrics format"
and "The checkpoint is the commit marker"; step 5).
"""

import json

import numpy as np
import pytest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ox_zero.training.metrics import MetricsWriter, read_metrics


def _row(generation: int, loss: float = 1.0) -> dict:
    """A small metrics row shaped like the real one: scalars plus the nested
    per-opponent `scores` dict (opponent generation -> score)."""
    return {
        "generation": generation,
        "loss_total": loss,
        "draw_rate": 0.25,
        "elo": 12.5,
        "scores": {generation - 1: 0.75},
    }


def _tensorboard_scalars(directory) -> EventAccumulator:
    """Read every event file under `directory / "tensorboard"` back.

    The accumulator is pointed at the directory, not one file, so it sees all
    sessions in order and applies a purge when a later session starts.
    """
    accumulator = EventAccumulator(str(directory / "tensorboard"))
    accumulator.Reload()
    return accumulator


def test_rows_append_and_read_back(tmp_path):
    writer = MetricsWriter(tmp_path, tensorboard=False)
    writer.write(_row(1, loss=2.0))
    writer.write(_row(2, loss=1.5))
    writer.close()

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2

    rows = read_metrics(tmp_path / "metrics.jsonl")
    # JSON object keys are always strings, so the nested scores dict comes
    # back keyed by "0" and "1", not by the ints that went in.
    assert rows == [
        {"generation": 1, "loss_total": 2.0, "draw_rate": 0.25, "elo": 12.5, "scores": {"0": 0.75}},
        {"generation": 2, "loss_total": 1.5, "draw_rate": 0.25, "elo": 12.5, "scores": {"1": 0.75}},
    ]


def test_partial_last_line_is_skipped(tmp_path):
    writer = MetricsWriter(tmp_path, tensorboard=False)
    writer.write(_row(1))
    writer.write(_row(2))
    writer.close()

    # Simulate a crash mid-row: chop the file inside the second line.
    path = tmp_path / "metrics.jsonl"
    text = path.read_text()
    first_end = text.index("\n") + 1
    path.write_text(text[: first_end + 10])

    rows = read_metrics(path)
    assert [row["generation"] for row in rows] == [1]


def test_tensorboard_on_writes_flattened_scalars(tmp_path):
    writer = MetricsWriter(tmp_path, tensorboard=True)
    row = _row(4, loss=0.5)
    row["scores"] = {3: 0.6, 2: 0.9}
    row["note"] = "not a number"  # non-numeric values are skipped, not logged
    writer.write(row)
    writer.close()

    event_files = list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
    assert event_files

    tags = set(_tensorboard_scalars(tmp_path).Tags()["scalars"])
    assert {"loss_total", "draw_rate", "elo", "scores/vs_gen_003", "scores/vs_gen_002"} <= tags
    assert "note" not in tags
    # The step axis is the generation, so the row's own key is not a curve.
    assert "generation" not in tags

    accumulator = _tensorboard_scalars(tmp_path)
    (event,) = accumulator.Scalars("scores/vs_gen_003")
    assert event.step == 4
    assert abs(event.value - 0.6) < 1e-6


def test_tensorboard_off_writes_no_directory(tmp_path):
    writer = MetricsWriter(tmp_path, tensorboard=False)
    writer.write(_row(1))
    writer.close()
    assert not (tmp_path / "tensorboard").exists()
    # The JSON Lines file is still written: TensorBoard is only a reader.
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["generation"] == 1


def test_purge_on_resume_drops_stale_points(tmp_path):
    writer = MetricsWriter(tmp_path, tensorboard=True)
    for generation in (1, 2, 3):
        writer.write(_row(generation, loss=float(generation)))
    writer.close()

    # Resume: generation 3 had no checkpoint, so it is re-run.
    writer = MetricsWriter(tmp_path, tensorboard=True, purge_from=3)
    writer.write(_row(3, loss=30.0))
    writer.close()

    events = _tensorboard_scalars(tmp_path).Scalars("loss_total")
    at_three = [event for event in events if event.step == 3]
    assert len(at_three) == 1
    assert at_three[0].value == 30.0
    assert [event.step for event in events] == [1, 2, 3]


def test_reopening_after_a_crash_truncates_the_partial_line(tmp_path):
    writer = MetricsWriter(tmp_path, tensorboard=False)
    writer.write(_row(1))
    writer.write(_row(2))
    writer.close()

    # A crash mid-row leaves the second line cut off, with no newline.
    path = tmp_path / "metrics.jsonl"
    text = path.read_text()
    first_end = text.index("\n") + 1
    path.write_text(text[: first_end + 10])

    # The resumed run reopens the file and writes generation 2 again. Without
    # repair the new row would be glued onto the fragment, burying a corrupt
    # line mid-file, and read_metrics would raise from then on.
    writer = MetricsWriter(tmp_path, tensorboard=False)
    writer.write(_row(2, loss=9.0))
    writer.close()
    rows = read_metrics(path)
    assert [row["generation"] for row in rows] == [1, 2]
    assert rows[1]["loss_total"] == 9.0


def test_reopening_completes_a_whole_row_missing_only_its_newline(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(_row(1)) + "\n" + json.dumps(_row(2)))
    writer = MetricsWriter(tmp_path, tensorboard=False)
    writer.write(_row(3))
    writer.close()
    assert [row["generation"] for row in read_metrics(path)] == [1, 2, 3]


@pytest.mark.parametrize("tensorboard", [False, True])
@pytest.mark.parametrize("bad", [{}, {"generation": "three"}, {"generation": 2.5}, {"generation": True}])
def test_write_validates_the_generation_before_writing(tmp_path, tensorboard, bad):
    writer = MetricsWriter(tmp_path, tensorboard=tensorboard)
    writer.write(_row(1))
    row = {"loss_total": 1.0, **bad}
    with pytest.raises(ValueError):
        writer.write(row)
    writer.close()
    # Nothing was appended: the file still holds only the good row.
    assert [r["generation"] for r in read_metrics(tmp_path / "metrics.jsonl")] == [1]


@pytest.mark.parametrize("tensorboard", [False, True])
def test_write_accepts_numpy_scalars(tmp_path, tensorboard):
    # The run loop computes many values with NumPy (np.mean, array indexing),
    # so they arrive as np.int64 / np.float32 rather than int / float. The
    # generation check accepts NumPy integers, so the row must serialise too:
    # `json.dumps` cannot encode NumPy scalars, also inside the nested dict.
    writer = MetricsWriter(tmp_path, tensorboard=tensorboard)
    writer.write(
        {
            "generation": np.int64(3),
            "loss_total": np.float32(0.5),
            "games": np.int32(7),
            "scores": {2: np.float32(0.75)},
        }
    )
    writer.close()

    (row,) = read_metrics(tmp_path / "metrics.jsonl")
    assert row == {"generation": 3, "loss_total": 0.5, "games": 7, "scores": {"2": 0.75}}
    # Plain Python types after the round trip, not floats-for-ints.
    assert type(row["generation"]) is int
    assert type(row["games"]) is int
    if tensorboard:
        events = _tensorboard_scalars(tmp_path).Scalars("loss_total")
        assert [(e.step, e.value) for e in events] == [(3, 0.5)]
