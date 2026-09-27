"""The `ox-zero` command: `analyze`, `best`, and `sandbox`.

The README's Usage section is the specification; this module implements it.

Output conventions (README, "Output conventions"):
- Results go to stdout; progress bars, notices, and errors go to stderr, so
  piping `best` or `--json` output always yields clean data.
- Styling comes from Rich, which drops it by itself when the stream is not a
  terminal or `NO_COLOR` is set. The code never checks for colour support.
- Invalid input exits with status 2 (the usual "usage error" status, which
  Typer also uses for bad flags).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from ox_zero.cli import render
from ox_zero.cli.report import best_json, finished_best_json, report_json
from ox_zero.cli.sandbox.app import run_sandbox
from ox_zero.cli.sandbox.session import Session
from ox_zero.engine import Analysis, DummyEngine, Engine, load_engine
from ox_zero.game import Cell, PositionError, State, is_terminal, parse_cell, parse_position

# AlphaZero's playout budget per move in the original paper. The dummy engine
# "runs" at ~2000 simulations per second, so this takes about 0.4 s.
DEFAULT_SIMULATIONS = 800

USAGE_ERROR = 2

app = typer.Typer(
    name="ox-zero",
    help="Analyse OXOX positions with an AlphaZero-style engine.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


# --- Shared options ----------------------------------------------------------
# `Annotated` aliases keep each command signature short and guarantee that a
# flag has the same help text and validation everywhere it appears.

PositionArg = Annotated[
    list[str] | None,
    typer.Argument(
        help="Move list (e.g. [bold]5,5 6,6 5,6[/]) or a 144-character board string."
        " Empty means the empty board.",
        show_default=False,
    ),
]
SimulationsOpt = Annotated[
    int | None,
    typer.Option(
        "--simulations",
        min=1,
        help=f"Tree-search playouts. Default {DEFAULT_SIMULATIONS}; with --live, an optional cap.",
        show_default=False,
    ),
]
ModelOpt = Annotated[
    Path | None,
    typer.Option("--model", help="Checkpoint to load. Defaults to the newest in checkpoints/."),
]
TopOpt = Annotated[int, typer.Option("--top", min=1, help="Number of candidate moves to list.")]
SeedOpt = Annotated[int | None, typer.Option("--seed", help="Fix randomness for reproducible output.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output instead of text.")]


# --- Commands ----------------------------------------------------------------


@app.command()
def analyze(
    position: PositionArg = None,
    simulations: SimulationsOpt = None,
    live: Annotated[
        bool, typer.Option("--live", help="Keep analysing and stream reports until interrupted.")
    ] = False,
    model: ModelOpt = None,
    top: TopOpt = 3,
    seed: SeedOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Evaluate a position and score every legal move."""
    out, err = _consoles()
    tokens = position or []
    state = _parse(tokens, err)
    last_move = _last_move(tokens)

    if is_terminal(state):
        # Nothing to search: report the result straight away.
        if json_output:
            _print_json(report_json(state, None, top_n=top))
        else:
            out.print(render.report(state, None, last_move=last_move))
        return

    engine = _load(model, seed, err)

    if live:
        from ox_zero.cli.live import run_live

        run_live(
            engine,
            state,
            max_simulations=simulations,
            top_n=top,
            last_move=last_move,
            json_output=json_output,
            out=out,
        )
        return

    analysis = _search_with_progress(engine, state, simulations or DEFAULT_SIMULATIONS, err)
    if json_output:
        _print_json(report_json(state, analysis, top_n=top))
    else:
        out.print(render.report(state, analysis, top_n=top, last_move=last_move))


@app.command()
def best(
    position: PositionArg = None,
    simulations: SimulationsOpt = None,
    model: ModelOpt = None,
    seed: SeedOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Print only the engine's chosen move."""
    _, err = _consoles()
    tokens = position or []
    state = _parse(tokens, err)

    if is_terminal(state):
        # No move exists. Exit 1 so a script like `move=$(ox-zero best ...)`
        # notices instead of silently getting an empty string.
        if json_output:
            _print_json(finished_best_json(state))
        else:
            err.print(render.game_over(state))
        raise typer.Exit(code=1)

    engine = _load(model, seed, err)
    analysis = _search_with_progress(engine, state, simulations or DEFAULT_SIMULATIONS, err)
    if json_output:
        _print_json(best_json(analysis))
    else:
        # Plain `print`, not Rich: this output is meant for other programs.
        row, col = analysis.best[0]
        print(f"{row},{col}")


@app.command()
def sandbox(
    position: PositionArg = None,
    model: ModelOpt = None,
    top: TopOpt = 3,
    seed: SeedOpt = None,
) -> None:
    """Interactive analysis screen: play moves, watch the engine think."""
    _, err = _consoles()
    tokens = position or []
    try:
        session = Session.from_tokens(tokens)
    except PositionError as error:
        _fail(err, str(error), tokens, error.token_index)
    engine = _load(model, seed, err)
    run_sandbox(session, engine, top)


# --- Helpers -----------------------------------------------------------------


def _consoles() -> tuple[Console, Console]:
    """Fresh stdout and stderr consoles.

    Created per command rather than at import time, so they bind to whatever
    `sys.stdout` / `sys.stderr` are at that moment (the test runner swaps
    them). `highlight=False` stops Rich from colouring numbers in plain
    strings on its own; all styling here is deliberate.
    """
    return Console(highlight=False), Console(stderr=True, highlight=False)


def _parse(tokens: Sequence[str], err: Console) -> State:
    """Parse the position, or print a caret-style error and exit with status 2."""
    try:
        return parse_position(tokens)
    except PositionError as error:
        _fail(err, str(error), tokens, error.token_index)


def _last_move(tokens: Sequence[str]) -> Cell | None:
    """The last move of a move list (to underline on the board), if known.

    Only called after parsing succeeded. A board string has no commas and no
    move order, so there is no last move to show.
    """
    if tokens and "," in tokens[-1]:
        return parse_cell(tokens[-1])
    return None


def _load(model: Path | None, seed: int | None, err: Console) -> Engine:
    try:
        engine = load_engine(model, seed)
    except FileNotFoundError as error:
        _fail(err, str(error))
    if isinstance(engine, DummyEngine):
        err.print(
            "[dim]Using the placeholder engine (no trained model yet): scores are not meaningful.[/]"
        )
    return engine


def _fail(
    err: Console,
    message: str,
    tokens: Sequence[str] = (),
    token_index: int | None = None,
) -> NoReturn:
    """Print `Error: ...`, echo the input with a caret under the bad token, exit 2.

    How the caret is placed: the tokens are echoed joined by single spaces,
    so token `i` starts after the lengths of tokens 0..i-1 plus one space
    each. Very long tokens (board strings) are not echoed; the message
    already says what is wrong with them.
    """
    err.print(Text.assemble(("Error: ", "bold red"), message))
    if token_index is not None and len(tokens[token_index]) <= 20:
        offset = sum(len(token) + 1 for token in tokens[:token_index])
        err.print(Text("  " + " ".join(tokens)))
        err.print(Text("  " + " " * offset + "^" * len(tokens[token_index]), style="bold red"))
    raise typer.Exit(code=USAGE_ERROR)


def _print_json(data: dict[str, Any]) -> None:
    # Plain `print`: JSON must never pass through Rich's styling.
    print(json.dumps(data))


class _RateColumn(ProgressColumn):
    """Simulations per second, e.g. `2,013 sims/s`."""

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        return Text(f"{speed:,.0f} sims/s" if speed else "", style="dim")


def _search_with_progress(
    engine: Engine, state: State, simulations: int, err: Console
) -> Analysis:
    """Run a fixed-budget search, with a transient progress bar on stderr.

    The bar is only drawn when stderr is a terminal; in a pipe or a log file
    it would be noise. `transient=True` erases it once the search finishes,
    so only the report remains on screen.
    """
    snapshots = engine.search(state, max_simulations=simulations)
    last: Analysis | None = None
    if not err.is_terminal:
        for last in snapshots:
            pass
    else:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]Searching"),
            BarColumn(),
            MofNCompleteColumn(),
            _RateColumn(),
            TimeRemainingColumn(),
            console=err,
            transient=True,
        )
        with progress:
            task = progress.add_task("search", total=simulations)
            for last in snapshots:
                progress.update(task, completed=last.simulations)
    assert last is not None
    return last


def main() -> None:
    """Console-script entry point (`ox-zero`, see pyproject.toml)."""
    app()
