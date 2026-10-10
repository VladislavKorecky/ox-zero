"""Metrics: one JSON Lines row per trained generation, mirrored to TensorBoard.

Design: docs/design/training.md ("Logging") and docs/plans/04-training-pipeline.md
(Decisions "Metrics format", "The checkpoint is the commit marker",
"TensorBoard is a regular dependency").

Plain files first, dashboards are readers
-----------------------------------------
The training loop's record of a run is `metrics.jsonl`: one JSON object per
line, one line per generation (losses, Elo, self-play throughput, game
length, draw rate, buffer size, wall time, per-opponent scores). JSON Lines
was chosen over CSV and over a single JSON document because:

- it *appends*: writing generation 17 never rewrites generations 1-16;
- it survives a crash mid-row: a half-written last line is just skipped by
  `read_metrics`, everything before it is intact;
- it nests: the per-opponent `scores` dict is one field, not a variable
  number of CSV columns.

TensorBoard is a *reader's convenience* layered on top: live curves while a
run is going. The writer mirrors each numeric value of the row as a scalar
whose x-axis ("step") is the row's `generation`. The training loop never
reads TensorBoard back; if the event files were deleted, nothing would break.

Resume and purge
----------------
A generation only "happened" once its checkpoint `gen_NNN.pt` exists (the
checkpoint is written last, so it acts as the commit marker). If the run
crashed after writing generation N's metrics but before its checkpoint, the
resumed run re-runs generation N. The JSON file is cleaned by the run loop
(it drops rows above the latest checkpoint). TensorBoard's event files are
append-only, so instead the writer is reopened with `purge_from=N`, which
becomes `SummaryWriter(purge_step=N)`: the new event file starts with a
session-start marker at step N, and TensorBoard's reader discards every
point at step >= N from earlier files when it sees that marker. The curve
then shows one point per generation, the re-run's.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

# `torch.utils.tensorboard` is imported lazily inside `MetricsWriter.__init__`
# (plan 04: "Lazy import keeps `import ox_zero.training` cheap"). Importing
# it pulls in tensorboard and protobuf; tests and `--no-tensorboard` runs
# should not pay for that. Under TYPE_CHECKING the name exists for annotations.
if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

METRICS_FILE = "metrics.jsonl"
TENSORBOARD_DIR = "tensorboard"


class MetricsWriter:
    """Appends one metrics row per generation to `directory / metrics.jsonl`,
    and optionally mirrors its numeric values to TensorBoard.

    Args:
        directory: The run directory (`runs/<name>/`). Created if missing.
        tensorboard: Also write TensorBoard event files under
            `directory / "tensorboard"`. Off in tests; on by default in
            `scripts/train.py`.
        purge_from: On resume, the first generation that will be re-run.
            Passed to the `SummaryWriter` as `purge_step` so stale TensorBoard
            points at that generation and above are discarded by the reader.
            `None` for a fresh run (or when nothing needs purging).
    """

    def __init__(self, directory: Path, tensorboard: bool, purge_from: int | None = None) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._path = self._directory / METRICS_FILE
        _repair_partial_last_line(self._path)
        self._tensorboard: SummaryWriter | None = None
        if tensorboard:
            # Lazy import: see the module-level note above.
            from torch.utils.tensorboard import SummaryWriter

            self._tensorboard = SummaryWriter(
                log_dir=str(self._directory / TENSORBOARD_DIR),
                purge_step=purge_from,
            )

    def write(self, row: Mapping[str, Any]) -> None:
        """Append `row` as one JSON line, then mirror it to TensorBoard.

        The line is written in one `write` call and flushed when the file
        closes, so a crash leaves at most one partial line at the end, which
        `read_metrics` skips. JSON turns integer dict keys (the opponent
        generations in `scores`) into strings; readers must expect that.

        Raises:
            ValueError: `row` has no integer `generation`. Checked before
                anything is written, so a bad row never leaves a JSON line
                without its TensorBoard points (or a row the run loop's
                purge-by-generation could not place).
        """
        generation = row.get("generation")
        # `numbers.Integral` accepts Python and NumPy integers; bool is an
        # int subclass but `True` is not a generation number.
        if not isinstance(generation, numbers.Integral) or isinstance(generation, bool):
            raise ValueError(f"a metrics row needs an integer 'generation', got {generation!r}")

        line = json.dumps(row, sort_keys=False)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

        if self._tensorboard is not None:
            # The TensorBoard x-axis is the generation: one point per curve
            # per generation, aligned across all metrics.
            step = int(generation)
            for tag, value in _scalars(row):
                if tag == "generation":
                    continue  # it is the step axis, not a curve
                self._tensorboard.add_scalar(tag, value, global_step=step)
            # Push the events to disk now, so a live `tensorboard --logdir`
            # sees each generation as it completes rather than in batches.
            self._tensorboard.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer (the JSON file is opened
        and closed on every `write`, so it needs nothing here)."""
        if self._tensorboard is not None:
            self._tensorboard.close()
            self._tensorboard = None


def _repair_partial_last_line(path: Path) -> None:
    """Make `path` end on a complete line before anything is appended to it.

    Every `write` ends its line with `\n`, so a file whose last byte is not
    `\n` was cut off by a crash. Appending to it as-is would glue the next
    row onto the fragment; the corrupt line would then no longer be the
    *last* one, and `read_metrics` (which only forgives a partial last line)
    would raise on every read from then on. So on open:

    - if the unterminated tail parses as JSON, only the newline was lost:
      add it, keeping the row;
    - otherwise it is half a row: truncate the file back to the end of the
      last complete line.
    """
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    # Index just past the last newline (0 if the only line is partial).
    start = data.rfind(b"\n") + 1
    try:
        json.loads(data[start:])
    except (json.JSONDecodeError, UnicodeDecodeError):
        with path.open("r+b") as file:
            file.truncate(start)
    else:
        with path.open("ab") as file:
            file.write(b"\n")


def _scalars(row: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, float]]:
    """Flatten a metrics row into `(tag, number)` pairs for TensorBoard.

    TensorBoard scalars are flat `tag -> number` series; a `/` in a tag
    groups curves into one panel. So the nested `scores` dict
    `{3: 0.6, 2: 0.9}` becomes `scores/vs_gen_003` and `scores/vs_gen_002`.
    Integer keys (opponent generations, either as ints or as the strings a
    JSON round trip produces) are named `vs_gen_NNN`, matching the
    checkpoint file names `gen_NNN.pt`; other keys are used as they are.

    Only real numbers are yielded. Strings, lists, `None` and booleans are
    skipped: they have no meaningful curve (a bool would plot as 0/1, which
    the row has no use for).
    """
    for key, value in row.items():
        name = _tag_name(key)
        tag = f"{prefix}/{name}" if prefix else name
        if isinstance(value, Mapping):
            yield from _scalars(value, tag)
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            yield tag, float(value)


def _tag_name(key: Any) -> str:
    """`3` or `"3"` -> `"vs_gen_003"`; any other key -> `str(key)`."""
    if isinstance(key, int) and not isinstance(key, bool):
        return f"vs_gen_{key:03d}"
    if isinstance(key, str) and key.isdigit():
        return f"vs_gen_{int(key):03d}"
    return str(key)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Read every complete row of a `metrics.jsonl` file, in order.

    A crash during `write` can leave the last line cut off mid-object. That
    line fails to parse and is skipped; every line before it is a complete
    row. Only the *last* line may be partial (rows are appended one at a
    time), so an unparseable line anywhere else means the file was damaged
    some other way, and that is raised rather than silently dropped.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # the partial last line of a crashed write
            raise
    return rows
