"""Tests for the sandbox screen, driven through Textual's Pilot.

`App.run_test()` runs the app headless; `pilot.press` types keys and
`pilot.click` clicks widgets. The engine runs in a background thread, so
tests that need scores wait for them with `wait_for`.
"""

import asyncio
import threading

import pytest
from textual.widgets import Static

from ox_zero.cli import render
from ox_zero.cli.sandbox.app import BoardView, SandboxApp
from ox_zero.cli.sandbox.session import Session
from ox_zero.cli.placeholder import PlaceholderEngine
from ox_zero.game import to_board_string


def make_app(*moves: str) -> SandboxApp:
    # The realistic rate, not an unthrottled one: a thread that never sleeps
    # hogs Python's GIL and starves the UI's event loop, making every test
    # take seconds.
    engine = PlaceholderEngine(seed=0, rate=2000)
    return SandboxApp(Session.from_tokens(list(moves)), engine, top_n=3)


def text(app: SandboxApp, widget_id: str) -> str:
    return str(app.query_one(f"#{widget_id}", Static).content)


async def wait_for(pilot, condition, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await pilot.pause(0.02)


async def type_command(pilot, command: str) -> None:
    await pilot.press(*command, "enter")


# Cell (row, col) sits at content offset x = 4 + 4*col (+2 lands on the digits),
# y = 1 + row (row 0 is the column header).
def cell_offset(row: int, col: int) -> tuple[int, int]:
    return 4 + 4 * col + 2, 1 + row


async def test_starts_with_the_position_and_analyses_it():
    app = make_app("5,5", "6,6", "5,6")
    async with app.run_test(size=(120, 40)) as pilot:
        assert text(app, "status").startswith("O to move (3 marks on board)")
        await wait_for(pilot, lambda: app.analysis is not None)
        await wait_for(pilot, lambda: "simulations" in text(app, "status"))
        assert "Top 3" in text(app, "side")
        assert "1. 5,5 6,6  2. 5,6" in text(app, "side")


async def test_typing_a_move_plays_it():
    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await type_command(pilot, "5,5")
        assert app.session.history == [(5, 5)]
        assert text(app, "status").startswith("O to move (1 mark on board)")


async def test_bad_input_shows_an_error_message():
    app = make_app("5,5")
    async with app.run_test(size=(120, 40)) as pilot:
        await type_command(pilot, "5,5")
        assert "occupied" in text(app, "message")


async def test_export_shows_the_board_string():
    app = make_app("5,5")
    async with app.run_test(size=(120, 40)) as pilot:
        await type_command(pilot, "export")
        assert to_board_string(app.session.state) in text(app, "message").replace("\n", "")


async def test_clicking_a_cell_plays_it():
    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click(BoardView, offset=cell_offset(3, 7))
        assert app.session.history == [(3, 7)]


async def test_clicking_the_labels_does_nothing():
    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click(BoardView, offset=(1, 5))  # row label
        await pilot.click(BoardView, offset=(10, 0))  # column header
        assert app.session.history == []


async def test_undo_and_redo_shortcuts():
    app = make_app("5,5", "6,6")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+z")
        assert app.session.history == [(5, 5)]
        await pilot.press("ctrl+y")
        assert app.session.history == [(5, 5), (6, 6)]


async def test_pause_blanks_the_scores_and_resume_restores_them():
    app = make_app("5,5")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(pilot, lambda: app.analysis is not None)
        await pilot.press("ctrl+p")
        assert app.session.paused
        assert app.analysis is None
        assert "paused" in text(app, "status")
        await type_command(pilot, "resume")
        await wait_for(pilot, lambda: app.analysis is not None)


async def test_new_position_restarts_analysis_from_scratch():
    app = make_app("5,5")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(pilot, lambda: app.analysis is not None)
        await type_command(pilot, "6,6")
        # Scores for the old position must never be shown for the new one.
        assert app.analysis is None or (5, 5) not in app.analysis.scores
        await wait_for(pilot, lambda: app.analysis is not None)
        assert (6, 6) not in app.analysis.scores


async def test_game_over_is_shown_in_the_status_line():
    app = make_app("5,5", "6,6", "5,7")
    async with app.run_test(size=(120, 40)) as pilot:
        await type_command(pilot, "5,6")
        assert text(app, "status").startswith("Game over: O wins")
        assert app.analysis is None


@pytest.mark.parametrize("keys", [("q", "u", "i", "t", "enter"), ("ctrl+q",)])
async def test_quit(keys):
    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press(*keys)
        await pilot.pause()
        assert not app.is_running


# --- Hide / show -------------------------------------------------------------
#
# Hiding is a display switch: the search must keep running underneath. A
# restarted search emits its first snapshot at once, so "no restart" cannot be
# shown by looking at `app.analysis`; instead `spy_on_restarts` counts calls.


def spy_on_restarts(app: SandboxApp, monkeypatch) -> list[int]:
    """Count `_restart_analysis` calls (patch before `run_test()`)."""
    calls = [0]
    original = app._restart_analysis

    def counting() -> None:
        calls[0] += 1
        original()

    monkeypatch.setattr(app, "_restart_analysis", counting)
    return calls


async def hide_and_wait_for_a_hidden_snapshot(pilot, app: SandboxApp) -> None:
    """Press ctrl+t, then wait for a snapshot that arrived while hidden."""
    await wait_for(pilot, lambda: app.analysis is not None)
    before = app.analysis.simulations
    await pilot.press("ctrl+t")
    assert app.session.hidden
    await wait_for(pilot, lambda: app.analysis.simulations > before)


class GatedEngine:
    """Wraps an engine; `search()` yields nothing until `gate` is set."""

    def __init__(self, inner, gate: threading.Event) -> None:
        self.inner = inner
        self.gate = gate

    def search(self, state, max_simulations=None):
        self.gate.wait(timeout=5)
        return self.inner.search(state, max_simulations)


async def test_hiding_keeps_the_search_running(monkeypatch):
    app = make_app("5,5")
    restarts = spy_on_restarts(app, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(pilot, lambda: app.analysis is not None)
        baseline = restarts[0]
        await hide_and_wait_for_a_hidden_snapshot(pilot, app)
        assert restarts[0] == baseline


async def test_hiding_blanks_the_display_as_snapshots_keep_arriving():
    app = make_app("5,5")
    async with app.run_test(size=(120, 40)) as pilot:
        await hide_and_wait_for_a_hidden_snapshot(pilot, app)
        blank = render.board(app.session.state, None, last_move=app.session.last_move)
        assert text(app, "board") == str(blank)
        assert "1. 5,5" in text(app, "side")
        assert "Top 3" not in text(app, "side")
        assert "Eval" not in text(app, "side")


async def test_hidden_status_line_shows_the_count_only_after_a_snapshot():
    gate = threading.Event()
    app = SandboxApp(
        Session.from_tokens([]), GatedEngine(PlaceholderEngine(seed=0, rate=2000), gate), top_n=3
    )
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+t")
            assert "hidden…" in text(app, "status")
            assert "analysing" not in text(app, "status")
            gate.set()
            await wait_for(pilot, lambda: app.analysis is not None)
            assert "hidden: " in text(app, "status")
            assert "simulations" in text(app, "status")
            assert "analysing" not in text(app, "status")
    finally:
        gate.set()  # never leave the worker thread blocked


@pytest.mark.parametrize("how", ["command", "key"])
async def test_showing_is_instant_and_does_not_restart(how, monkeypatch):
    app = make_app("5,5")
    restarts = spy_on_restarts(app, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        await hide_and_wait_for_a_hidden_snapshot(pilot, app)
        baseline = restarts[0]
        if how == "command":
            await type_command(pilot, "show")
        else:
            await pilot.press("ctrl+t")
        assert not app.session.hidden
        assert restarts[0] == baseline
        assert "Top 3" in text(app, "side")


async def test_hidden_survives_a_move():
    app = make_app("5,5")
    async with app.run_test(size=(120, 40)) as pilot:
        await hide_and_wait_for_a_hidden_snapshot(pilot, app)
        await type_command(pilot, "6,6")
        # Right after the move `analysis` is None, so wait for a snapshot of
        # the new position; only then does the assertion prove anything.
        await wait_for(
            pilot, lambda: app.analysis is not None and (6, 6) not in app.analysis.scores
        )
        assert "hidden: " in text(app, "status")
        assert "Top 3" not in text(app, "side")


async def test_pause_and_hide_together():
    app = make_app("5,5")
    async with app.run_test(size=(120, 40)) as pilot:
        await hide_and_wait_for_a_hidden_snapshot(pilot, app)
        await pilot.press("ctrl+p")
        assert "paused" in text(app, "status")
        assert app.analysis is None
        assert "Top 3" not in text(app, "side")
        await type_command(pilot, "resume")
        await wait_for(pilot, lambda: app.analysis is not None)
        assert app.session.hidden
        assert "hidden: " in text(app, "status")


async def test_footer_lists_hide_show():
    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert any(b.description == "Hide/show" for b in app.BINDINGS)
