# Plan 07: Hide the analysis in the sandbox

| | |
|---|---|
| Status | Draft (2026-10-12). The choices below were approved by Vláďa on 2026-10-12 ([Approvals](#approvals-2026-10-12)). |
| Branches | Plan: `plan/sandbox-hide`. Implementation: `feat/sandbox-hide`. |
| Design references | [docs/cli.md](../cli.md#sandbox) (the sandbox spec this plan extends), [architecture.md](../architecture.md#sandbox-logic-separate-from-the-screen) (session/screen split) |
| Depends on | Nothing beyond `main`. Touches only `cli/sandbox/` and `docs/cli.md`. |

## Goal

Let a human play a fair game against the engine in the sandbox. The user plays their own moves and relays the engine's moves by hand: on the engine's turn they reveal the analysis, note its best move, hide the analysis again, and only then play the move. Hiding before playing matters. Playing the move restarts the search on the user's own turn, and its first scores reach the screen as soon as the engine yields its first snapshot (`_search` starts with `last_emit = -inf`, so `REFRESH_INTERVAL` only throttles the later ones). There is effectively no window in which to type `hide`. While the analysis is hidden, **the search keeps running**. Only the display is suppressed, so when the user reveals it the engine has been thinking all along and nothing restarts.

This is different from `pause`, which stops the search. With `pause`, revealing on the engine's turn would mean starting the search from zero at that moment.

## Deliverables

1. `docs/cli.md`: the `hide` / `show` commands, the `ctrl+t` shortcut, the hidden status line, and a short "playing against the engine" paragraph (spec first, per CLAUDE.md).
2. `cli/sandbox/session.py`: a `hidden` flag, `hide` / `show` commands, `toggle_hidden()`, the updated `HELP` text, and a `Session` class docstring that names both switches (today it says "the analysis pause switch").
3. `cli/sandbox/app.py`: the `ctrl+t` binding, drawing with the analysis suppressed while hidden, the hidden status text, and updated comments: the module's engine-lifecycle docstring (hiding never touches the engine) and the `self.analysis` comment in `__init__` (it stays populated while hidden).
4. Tests in `tests/cli/sandbox/test_session.py` and `tests/cli/sandbox/test_sandbox_app.py`.

## Out of scope

- An engine opponent that plays its own moves (`--play o` or similar). The user relays moves by hand. That opponent may be a future plan; this one adds nothing that would get in its way.
- A `--hidden` startup flag (declined, see Approvals). You type `hide` once.
- Any change to the engine, `render.py`, `analyze`, `best`, or the JSON output.
- Persisting the hidden state anywhere beyond the running session.

## Decisions made in this plan

| Decision | Choice | Why |
|---|---|---|
| **Names and key** | Commands `hide` and `show`; shortcut `ctrl+t` ("toggle") toggles them. Footer label `Hide/show`. | `ctrl+h` arrives as backspace in most terminals. `ctrl+a/e/d/k/w/u/x/c/v` are taken by Textual's `Input` line editing (checked against Textual 8.2.8's `Input.BINDINGS`). `ctrl+t` is free there too, and unlike `ctrl+b` (tmux's default prefix) or `ctrl+a` (GNU screen's) it isn't taken by a terminal multiplexer. |
| **Hidden is independent of paused** | Two separate flags on `Session`. Hiding never stops the search, and pausing never changes `hidden`. While both are set, the status line says `paused`, since nothing is being computed. | The point of the feature is that the search continues. Coupling the two flags would bring back the "restart on reveal" problem. |
| **What gets hidden** | Everything derived from the analysis: the scores and heatmap in the cells, the top-N emphasis, the eval bar, and the top-N list. The board looks exactly as it does while paused (empty cells show `.`). | Any of these would give the best move away. |
| **What stays visible** | Marks, the last-move underline, the move history, the game-over line and the winning-line highlight, and the **simulation count**: `hidden: 12,400 simulations` (`hidden…` before the first snapshot), in the same format as `analysing: 12,400 simulations` / `analysing…`. Style: bold magenta for `hidden: ` and magenta for the count, so it is distinct at a glance from `analysing` (cyan) and `paused` (yellow). | None of these reveals a move. The count tells the user whether the engine has thought long enough before they reveal. |
| **Hidden persists** | `hidden` survives `play`, `undo`, `redo`, `load` and `reset`. Only `hide`, `show` and `ctrl+t` change it. | You stay blind for the whole game, not just for one move. |
| **Repeat commands** | `hide` while hidden says `analysis is already hidden`; `show` while shown says `analysis is already shown`. Neither is an error, the same as `pause` and `resume` today. | Consistent with the existing commands. |
| **Showing never restarts the search** | The app keeps storing every snapshot in `self.analysis` while hidden. Drawing goes through one accessor that returns `None` while hidden. `hide` and `show` don't change `_engine_should_run()`, so `_apply`'s existing "did the engine's job change?" check takes the redraw-only branch. | A reveal shows the deep search at once. Restarting would throw it away, which is the whole bug in using `pause` for this. |

## Interfaces

```python
# cli/sandbox/session.py
class Session:
    hidden: bool                     # False on construction
    def toggle_hidden(self) -> None
    # execute("hide") / execute("show"): set the flag, return Result() (or the
    # "already" message). position_changed stays False.

HELP  # gains "hide · show" and "ctrl+t hide/show"

# cli/sandbox/app.py
class SandboxApp:
    BINDINGS  # + Binding("ctrl+t", "toggle_hidden", "Hide/show", priority=True)
    def action_toggle_hidden(self) -> None
    @property
    def _shown_analysis(self) -> Analysis | None   # None while hidden, else self.analysis
    # _redraw and _side read _shown_analysis; _status reads self.analysis for the count.
```

`self.analysis` keeps its current meaning: the latest snapshot for the current position. Tests rely on it being populated while hidden.

## Steps

Test-driven: write each step's tests, run `uv run pytest tests/cli/sandbox -q` and see them fail, then implement until they pass. Comment as CLAUDE.md asks. In particular, the app comment should say *why* hiding is a display-only switch (the search state is the expensive thing, and a reveal should cost nothing).

### Step 1: spec (`docs/cli.md`)

- Command table: a row `hide` / `show`: "Hide or show the analysis. While hidden, the engine keeps searching, but no scores, eval bar or candidates are shown; the status line shows only the simulation count. Use it to play against the engine without seeing its suggestions."
- Shortcut table: `ctrl+t` toggles `hide` / `show`.
- Screen description: the status line's engine state gains `hidden` (with the count). The board bullet says empty cells show `.` while analysis is paused *or hidden*. The side panel bullet says the eval bar and top N candidates are left out while analysis is paused or hidden, leaving only the move history.
- The footer line in the mock screen gains `^t Hide/show`.
- A short paragraph, "Playing against the engine": type `hide` and play your move. On the engine's turn, `show` (or `ctrl+t`), note its top candidate, hide again, *then* play that move. If you play it while the analysis is shown, the scores for your own turn appear before you can hide them. The longer you wait before revealing, the stronger the engine's move.

Commit: `docs(cli): specify hide/show in the sandbox`.

### Step 2: session (`tests/cli/sandbox/test_session.py`, then `session.py`)

Tests:

1. A new session is not hidden.
2. `hide` sets `hidden`; `show` clears it. Neither changes the position, and neither sets `position_changed`.
3. `hide` while hidden and `show` while shown return the "already" messages, with `error=False`.
4. `toggle_hidden()` flips the flag both ways.
5. `hidden` survives `play`, `undo`, `redo`, `load` and `reset` (parametrised).
6. `hidden` and `paused` are independent: setting one never changes the other.
7. `HELP` mentions `hide`, `show` and `ctrl+t`.

Commit pair: `test(cli): pin sandbox hide/show commands` / `feat(cli): add hide/show to the sandbox session`.

### Step 3: screen (`tests/cli/sandbox/test_sandbox_app.py`, then `app.py`)

Tests (headless via Pilot, placeholder engine as today). Several tests must prove that *no restart happened*. Checking `app.analysis` or the simulation count can't prove that: a restarted search emits its first snapshot at once (`last_emit = -inf`) and can refill `app.analysis` before the assertion runs. So those tests wrap `app._restart_analysis` in a counting spy (`monkeypatch.setattr` with a wrapper that increments a counter and calls the original) and assert the count is unchanged. Patch the *instance* before entering `run_test()`, and capture the baseline count only after the first snapshot has arrived, so `on_mount`'s own call is already counted.

Waiting for "a snapshot taken while hidden" can't use `app.analysis is not None`, because the snapshot from before the hide already satisfies it. Wait until `app.analysis.simulations` exceeds its value at the moment of hiding.

1. **Hiding keeps the search running.** After the first snapshot, press `ctrl+t`. The restart spy count is unchanged, and `app.analysis.simulations` grows past its value at the moment of hiding.
2. **Hiding blanks the display, and keeps it blank as snapshots arrive.** Hide, then wait until `app.analysis.simulations` exceeds its value at hiding, so at least one snapshot taken while hidden has been drawn. Only then assert: the `#board` text equals `render.board(state, None, last_move=...)` rendered to plain text, and `#side` contains the move history but not `Top 3` and not `Eval`.
3. **Hidden status line.** Before the first snapshot it is exactly `hidden…`. To make this deterministic, the test uses a small gated fake engine whose `search()` waits on a `threading.Event` before its first yield. Press `ctrl+t` straight after mount (no move needed, since the mount search is already gated), assert `hidden…`, then set the event. The wait must use a timeout (`event.wait(timeout=5)`) and the event must be set in a `finally`, so a failing assertion can't leave a worker thread blocked and hang the test run. After the first snapshot, the status contains `hidden: ` and `simulations`. Never `analysing` in either case.
4. **Showing is instant and doesn't restart, by command and by key** (parametrised over typing `show` and pressing `ctrl+t`, because `action_toggle_hidden` is a separate code path from `Session.execute`). Once a snapshot has arrived while hidden (simulations past the value at hiding), reveal. The restart spy count is unchanged, and `Top 3` is in `#side` right after the reveal, with no waiting.
5. **Hidden survives a move.** Hide, type a move, and wait for the first snapshot on the new position (`app.analysis` not `None`, and the move's cell absent from `app.analysis.scores`). Only then assert that the status says `hidden: ` and `#side` has no `Top 3`. Right after the move, `analysis` is `None` anyway, so asserting earlier would prove nothing.
6. **Pause and hide together.** Hidden plus `ctrl+p`: the status shows `paused`, the board and side panel stay blank, and `app.analysis` is `None`. `resume` restarts the search even though the analysis is hidden (`app.analysis` becomes non-`None` again) and the status returns to `hidden: … simulations`.
7. The footer lists `Hide/show`.

Implementation: add the binding and action, mirroring `action_toggle_pause`. Route `"hide"`/`"show"` through `Session.execute` as usual. Add `_shown_analysis`, and use it in `_redraw` (board) and `_side`. In `_status`, add a `hidden` branch after the paused check that reads the count from `self.analysis`. No change to `_apply`, `_restart_analysis`, `_search` or `_receive`. While hidden, `_receive` still does a full `_redraw` per snapshot, re-rendering a board that can't change. That's accepted: it is 4 redraws a second of a ≤12x12 board, the same work the shown sandbox does today. If one looks necessary, see the questions below.

Commit pair: `test(cli): pin hidden analysis in the sandbox screen` / `feat(cli): hide analysis in the sandbox`.

### Step 4: review and PR

Run `/code-review medium` on the branch (about 150–250 changed lines, one review). Fix, re-run until only nits remain. Then open a PR from `feat/sandbox-hide` to `main`.

## Definition of done

- `uv run pytest` green.
- Manual check with `uv run ox-zero sandbox --size 6`: hide, play a move, and watch the simulation count climb with no scores visible. Press `ctrl+t` and see the full analysis appear at once with a count that did not reset. Note the top move, press `ctrl+t` again, play the move, and confirm that no scores flash up for your turn.
- `docs/cli.md` describes exactly what was built.
- PR merged. This plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- Test 4 fails because the reveal triggers a restart. That means `_apply`'s engine-job check took the restart branch. Report why rather than special-casing `hide`/`show` in `_apply`.
- `ctrl+t` turns out to be swallowed by the terminal or by Textual on this machine (Terminal.app, iTerm2, or the VS Code terminal). Report it; the key is a decision in this plan, not the implementer's.
- Any design-level choice not in the Decisions table.

## Approvals (2026-10-12)

1. Commands `hide` / `show`. Shortcut first `ctrl+b`, changed to `ctrl+t` in review because `ctrl+b` is tmux's prefix key.
2. The simulation count stays visible while hidden.
3. No `--hidden` startup flag; the command is enough.
