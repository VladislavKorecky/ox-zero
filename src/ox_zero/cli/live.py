"""`analyze --live`: keep searching and stream refreshed reports.

The engine's `search()` yields snapshots far more often than anyone can read
them (the placeholder engine yields every 50 simulations). This module *throttles* them:
it keeps consuming snapshots but only emits one when `interval` seconds have
passed since the last emit, roughly twice a second by default.

- Text mode redraws one report in place with `rich.live.Live`.
- JSON mode writes JSON Lines: one complete report object per line, so a
  consumer can simply keep the most recent line.

The loop ends when the `--simulations` cap is reached (the final snapshot is
always emitted, even if a refresh isn't due yet) or on Ctrl-C.
"""

from __future__ import annotations

import json
import time

import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text

from ox_zero.cli import render
from ox_zero.cli.report import report_json
from ox_zero.cli.engine_port import Analysis, Engine
from ox_zero.game import Cell, State

REFRESH_INTERVAL = 0.5

# Shells report "terminated by Ctrl-C" as 128 + SIGINT (2).
INTERRUPTED = 130


def run_live(
    engine: Engine,
    state: State,
    *,
    max_simulations: int | None,
    top_n: int,
    last_move: Cell | None,
    json_output: bool,
    out: Console,
    interval: float = REFRESH_INTERVAL,
) -> None:
    """Stream reports until the cap or Ctrl-C (then exit with status 130)."""

    def status(analysis: Analysis, done: bool) -> Text:
        count = f"{analysis.simulations:,}"
        if max_simulations is not None:
            count += f" / {max_simulations:,}"
        label = "done" if done else "analysing"
        return Text.assemble((f"{label}: ", "bold cyan"), (f"{count} simulations", "cyan"))

    def renderable(analysis: Analysis, done: bool = False):
        return render.report(
            state, analysis, top_n=top_n, last_move=last_move, status=status(analysis, done)
        )

    def emit_json(analysis: Analysis) -> None:
        report = report_json(state, analysis, top_n=top_n, include_simulations=True)
        # Flush every line: a consumer reading the pipe should see each report
        # as soon as it is produced, not when a buffer happens to fill up.
        print(json.dumps(report), flush=True)

    # Redrawing in place only makes sense on a terminal. Anywhere else (a
    # file, a pipe) the text report is printed once, at the end.
    live = Live(console=out, auto_refresh=False) if not json_output and out.is_terminal else None

    last: Analysis | None = None  # newest snapshot from the engine
    emitted: Analysis | None = None  # newest snapshot actually shown
    last_emit = -float("inf")
    interrupted = False
    try:
        if live:
            live.start()
        for last in engine.search(state, max_simulations=max_simulations):
            now = time.monotonic()
            if now - last_emit < interval:
                continue
            last_emit, emitted = now, last
            if json_output:
                emit_json(last)
            elif live:
                live.update(renderable(last), refresh=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if live:
            if last is not None:
                live.update(renderable(last, done=not interrupted), refresh=True)
            live.stop()

    if json_output:
        # The final snapshot at the cap may have been skipped by the throttle;
        # the README promises the last line is the final report.
        if not interrupted and last is not None and last is not emitted:
            emit_json(last)
    elif not live and last is not None:
        out.print(renderable(last, done=not interrupted))

    if interrupted:
        raise typer.Exit(code=INTERRUPTED)
