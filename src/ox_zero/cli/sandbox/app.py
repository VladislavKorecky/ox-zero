"""The sandbox screen: a Textual app around a `Session` and an `Engine`.

Textual is a framework for full-screen terminal apps. The pieces used here:

- *Widgets* (`Static`, `Input`, `Footer`) laid out with CSS-like rules.
- *Bindings*: key shortcuts mapped to `action_*` methods.
- *Workers*: background tasks. The engine runs in a thread worker so a long
  search never freezes the screen. Threads cannot touch widgets directly;
  the worker hands each snapshot to the UI thread with `call_from_thread`.

Engine lifecycle: every time the position changes, or analysis is paused or
resumed, `_restart_analysis` cancels the running search and, unless paused or
finished, starts a new one on the current position. Python threads cannot be
killed, so cancelling only sets a flag; the worker checks it after every
snapshot and returns. Snapshots from a stale search that slip through are
dropped by comparing their position with the current one.
"""

from __future__ import annotations

import time
from typing import ClassVar

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Footer, Input, Static
from textual.worker import get_current_worker

from ox_zero.cli import render
from ox_zero.cli.sandbox.session import Result, Session
from ox_zero.engine import Analysis, Engine
from ox_zero.game import Cell, State, is_terminal

# How often the running search pushes a snapshot to the screen. Faster than
# `analyze --live` (0.5 s) because here the user is actively watching.
REFRESH_INTERVAL = 0.25

ENGINE_GROUP = "engine"


class BoardView(Static):
    """The heatmap board. Clicking an empty cell posts `CellClicked`."""

    class CellClicked(Message):
        def __init__(self, cell: Cell) -> None:
            super().__init__()
            self.cell = cell

    def on_click(self, event: events.Click) -> None:
        """Turn a click position into a board cell.

        The board text (see `render.board`) has the column header on line 0
        and a 4-character row label before the cells, each cell being
        `render.CELL_WIDTH` characters wide. So inverting the layout:
        `row = y - 1` and `col = (x - 4) // 4`. Clicks on the labels or
        outside the grid are ignored.
        """
        offset = event.get_content_offset(self)
        if offset is None:
            return
        size = self.app.session.state.size  # type: ignore[attr-defined]
        row = offset.y - 1
        col, _ = divmod(offset.x - render.CELL_WIDTH, render.CELL_WIDTH)
        if 0 <= row < size and offset.x >= render.CELL_WIDTH and 0 <= col < size:
            self.post_message(self.CellClicked((row, col)))


class SandboxApp(App[None]):
    """Interactive analysis: play moves for both sides, watch the scores settle."""

    CSS = """
    Screen {
        padding: 1 2 0 2;
    }
    #status {
        height: 1;
        margin-bottom: 1;
    }
    #main {
        height: auto;
    }
    #board {
        width: auto;
        margin-right: 4;
    }
    #side {
        width: 1fr;
    }
    #message {
        height: auto;
        min-height: 1;
        margin-top: 1;
    }
    #prompt {
        dock: bottom;
        margin-bottom: 1;
    }
    """

    TITLE = "ox-zero sandbox"

    # ctrl+p opens Textual's command palette by default; the sandbox uses it
    # for pause instead.
    ENABLE_COMMAND_PALETTE = False

    # `priority=True` makes the app handle these keys even while the prompt
    # has focus, instead of the input field swallowing them.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+z", "undo", "Undo", priority=True),
        Binding("ctrl+y", "redo", "Redo", priority=True),
        Binding("ctrl+p", "toggle_pause", "Pause/resume", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, session: Session, engine: Engine, top_n: int = 3) -> None:
        super().__init__()
        self.session = session
        self.engine = engine
        self.top_n = top_n
        # The latest snapshot for the *current* position, or None while
        # waiting for the first one, paused, or the game is over.
        self.analysis: Analysis | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        with Horizontal(id="main"):
            yield BoardView(id="board")
            yield Static(id="side")
        yield Static(id="message")
        yield Input(placeholder="row,col to play  ·  help for commands", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        self._restart_analysis()

    # --- Input ----------------------------------------------------------------

    @on(Input.Submitted, "#prompt")
    def _submitted(self, event: Input.Submitted) -> None:
        event.input.clear()
        self._apply(self.session.execute(event.value))

    @on(BoardView.CellClicked)
    def _clicked(self, event: BoardView.CellClicked) -> None:
        self._apply(self.session.play(event.cell))

    def action_undo(self) -> None:
        self._apply(self.session.undo())

    def action_redo(self) -> None:
        self._apply(self.session.redo())

    def action_quit(self) -> None:  # type: ignore[override]
        # Stop the engine first, so no snapshot arrives while the screen is
        # being torn down. (Overrides the App's built-in ctrl+q action.)
        self.workers.cancel_group(self, ENGINE_GROUP)
        self._engine_running = False
        self.exit()

    def action_toggle_pause(self) -> None:
        self.session.toggle_pause()
        self._apply(Result())

    def _apply(self, result: Result) -> None:
        """Update the screen after a command."""
        if result.quit:
            self.action_quit()
            return
        self._show_message(result)
        # Pausing, resuming, and position changes all need the engine
        # restarted (or stopped). Restarting when nothing changed would throw
        # away a deep search, so check whether the engine's job changed.
        if result.position_changed or self._engine_should_run() != self._engine_running:
            self._restart_analysis()
        else:
            self._redraw()

    # --- Engine ---------------------------------------------------------------

    _engine_running = False

    def _engine_should_run(self) -> bool:
        return not self.session.paused and not is_terminal(self.session.state)

    def _restart_analysis(self) -> None:
        self.workers.cancel_group(self, ENGINE_GROUP)
        self.analysis = None
        self._engine_running = self._engine_should_run()
        if self._engine_running:
            self._search(self.session.state)
        self._redraw()

    @work(thread=True, exclusive=True, group=ENGINE_GROUP)
    def _search(self, state: State) -> None:
        """Runs in a background thread: stream snapshots to the UI thread.

        Caveat for the real engine (roadmap step 3): a thread shares Python's
        GIL with the UI. The dummy sleeps between steps, which releases it;
        a pure-Python MCTS loop that never blocks would starve the screen
        and make it laggy. It will need to yield regularly, or run in a
        separate process.
        """
        worker = get_current_worker()
        last_emit = -float("inf")
        for analysis in self.engine.search(state):
            if worker.is_cancelled:
                return
            now = time.monotonic()
            if now - last_emit >= REFRESH_INTERVAL:
                last_emit = now
                try:
                    self.call_from_thread(self._receive, state, analysis)
                except RuntimeError:
                    return  # the app is shutting down

    def _receive(self, state: State, analysis: Analysis) -> None:
        """UI thread: accept a snapshot unless it belongs to an old position."""
        if state != self.session.state or not self._engine_running:
            return
        self.analysis = analysis
        try:
            self._redraw()
        except NoMatches:
            # The app is closing and its widgets are already gone. This can
            # happen when a snapshot was queued just before shutdown.
            pass

    # --- Drawing --------------------------------------------------------------

    def _redraw(self) -> None:
        state = self.session.state
        self.query_one("#status", Static).update(self._status())
        self.query_one("#board", BoardView).update(
            render.board(state, self.analysis, top_n=self.top_n, last_move=self.session.last_move)
        )
        self.query_one("#side", Static).update(self._side())

    def _status(self) -> Text:
        state = self.session.state
        if is_terminal(state):
            return render.game_over(state)
        if self.session.paused:
            right = Text("paused", style="bold yellow")
        elif self.analysis is None:
            right = Text("analysing…", style="cyan")
        else:
            right = Text.assemble(
                ("analysing: ", "bold cyan"), (f"{self.analysis.simulations:,} simulations", "cyan")
            )
        return render.status_line(state, right)

    def _side(self) -> Text:
        history = Text()
        history.append("Moves: ", "bold")
        history.append(self.session.history_text() or "—", "" if self.session.history else "dim")
        if self.analysis is None:
            return history
        return Text("\n\n").join(
            [
                render.eval_bar(self.analysis.value, self.session.state.to_move),
                render.top_list(self.analysis, self.top_n),
                history,
            ]
        )

    def _show_message(self, result: Result) -> None:
        message = Text(result.message or "", style="bold red" if result.error else "")
        self.query_one("#message", Static).update(message)


def run_sandbox(session: Session, engine: Engine, top_n: int) -> None:
    SandboxApp(session, engine, top_n).run()
