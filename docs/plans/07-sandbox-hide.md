# Plan 07: Hide the analysis in the sandbox

| | |
|---|---|
| Status | Draft (2026-10-12). The choices below were approved by Vláďa on 2026-10-12 ([Approvals](#approvals-2026-10-12)). |
| Branches | Plan: `plan/sandbox-hide`. Implementation: `feat/sandbox-hide`. |
| Design references | [docs/cli.md](../cli.md#sandbox) (the sandbox spec this plan extends), [architecture.md](../architecture.md#sandbox-logic-separate-from-the-screen) (session/screen split) |
| Depends on | Nothing beyond `main`. Touches only `cli/sandbox/` and `docs/cli.md`. |

## Goal

Let a human play a fair game against the engine in the sandbox. The user plays their own moves and relays the engine's moves by hand: they reveal the analysis on the engine's turn, play its best move, and hide it again. While the analysis is hidden, **the search keeps running**. Only the display is suppressed, so when the user reveals it the engine has been thinking all along and nothing restarts.

This is different from `pause`, which stops the search. With `pause`, revealing on the engine's turn would mean starting the search from zero at that moment.

## Deliverables

1. `docs/cli.md`: the `hide` / `show` commands, the `ctrl+b` shortcut, the hidden status line, and a short "playing against the engine" paragraph (spec first, per CLAUDE.md).
2. `cli/sandbox/session.py`: a `hidden` flag, `hide` / `show` commands, `toggle_hidden()`, and the updated `HELP` text.
3. `cli/sandbox/app.py`: the `ctrl+b` binding, drawing with the analysis suppressed while hidden, and the hidden status text.
4. Tests in `tests/cli/sandbox/test_session.py` and `tests/cli/sandbox/test_sandbox_app.py`.

## Out of scope

- An engine opponent that plays its own moves (`--play o` or similar). The user relays moves by hand. That opponent may be a future plan; this one adds nothing that would get in its way.
- A `--hidden` startup flag (declined, see Approvals). You type `hide` once.
- Any change to the engine, `render.py`, `analyze`, `best`, or the JSON output.
- Persisting the hidden state anywhere beyond the running session.

## Decisions made in this plan

| Decision | Choice | Why |
|---|---|---|
| **Names and key** | Commands `hide` and `show`; shortcut `ctrl+b` ("blind") toggles them. Footer label `Hide/show`. | `ctrl+h` arrives as backspace in most terminals. `ctrl+a/e/d/k/w/u/x/c/v` are taken by Textual's `Input` line editing (checked against Textual 8.2.8's `Input.BINDINGS`). `ctrl+b` is free. |
| **Hidden is independent of paused** | Two separate flags on `Session`. Hiding never stops the search, and pausing never changes `hidden`. While both are set, the status line says `paused`, since nothing is being computed. | The point of the feature is that the search continues. Coupling the two flags would bring back the "restart on reveal" problem. |
| **What gets hidden** | Everything derived from the analysis: the scores and heatmap in the cells, the top-N emphasis, the eval bar, and the top-N list. The board looks exactly as it does while paused (empty cells show `.`). | Any of these would give the best move away. |
| **What stays visible** | Marks, the last-move underline, the move history, the game-over line and the winning-line highlight, and the **simulation count**: `hidden · 12,400 simulations` (`hidden` alone before the first snapshot). | None of these reveals a move. The count tells the user whether the engine has thought long enough before they reveal. |
| **Hidden persists** | `hidden` survives `play`, `undo`, `redo`, `load` and `reset`. Only `hide`, `show` and `ctrl+b` change it. | You stay blind for the whole game, not just for one move. |
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

HELP  # gains "hide · show" and "ctrl+b hide/show"

# cli/sandbox/app.py
class SandboxApp:
    BINDINGS  # + Binding("ctrl+b", "toggle_hidden", "Hide/show", priority=True)
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
- Shortcut table: `ctrl+b` toggles `hide` / `show`.
- Screen description: the status line's engine state gains `hidden` (with the count). The board bullet says empty cells show `.` while analysis is paused *or hidden*.
- The footer line in the mock screen gains `^b Hide/show`.
- A short paragraph, "Playing against the engine": type `hide`, play your move, then `show` (or `ctrl+b`) on the engine's turn. Play its top candidate, hide again. The longer you wait before revealing, the stronger its move.

Commit: `docs(cli): specify hide/show in the sandbox`.

### Step 2: session (`tests/cli/sandbox/test_session.py`, then `session.py`)

Tests:

1. A new session is not hidden.
2. `hide` sets `hidden`; `show` clears it. Neither changes the position, and neither sets `position_changed`.
3. `hide` while hidden and `show` while shown return the "already" messages, with `error=False`.
4. `toggle_hidden()` flips the flag both ways.
5. `hidden` survives `play`, `undo`, `redo`, `load` and `reset` (parametrised).
6. `hidden` and `paused` are independent: setting one never changes the other.
7. `HELP` mentions `hide`, `show` and `ctrl+b`.

Commit pair: `test(cli): pin sandbox hide/show commands` / `feat(cli): add hide/show to the sandbox session`.

### Step 3: screen (`tests/cli/sandbox/test_sandbox_app.py`, then `app.py`)

Tests (headless via Pilot, placeholder engine as today):

1. **Hiding keeps the search running.** After the first snapshot, press `ctrl+b`. Then wait until `app.analysis.simulations` grows past its value at the moment of hiding. `app.analysis` is never reset to `None`.
2. **Hiding blanks the display.** While hidden, the `#board` text equals `render.board(state, None, last_move=...)` rendered to plain text. `#side` contains the move history but not `Top 3` and not `Eval`.
3. **Hidden status line.** It contains `hidden` and `simulations` once a snapshot has arrived.
4. **Showing is instant and doesn't restart.** Once a snapshot has arrived while hidden, type `show`. Right after the command, with no waiting, `app.analysis` is not `None` and `Top 3` is in `#side`. A restart would have set `app.analysis` to `None` (`_restart_analysis` does), so this assertion is the one that catches it. A simulation-count comparison would not: a fresh search can quickly overtake the old count.
5. **Hidden survives a move.** Hide, type a move. The status still says `hidden`, `#side` still has no `Top 3`, and the search restarts on the new position (`(row, col)` of the move is absent from `app.analysis.scores` once a snapshot arrives).
6. **Pause wins the status line.** Hidden plus `ctrl+p` shows `paused`. `resume` returns to `hidden`.
7. The footer lists `Hide/show`.

Implementation: add the binding and action, mirroring `action_toggle_pause`. Route `"hide"`/`"show"` through `Session.execute` as usual. Add `_shown_analysis`, and use it in `_redraw` (board) and `_side`. In `_status`, add a `hidden` branch after the paused check that reads the count from `self.analysis`. No change to `_apply`, `_restart_analysis`, `_search` or `_receive`. If one looks necessary, see the questions below.

Commit pair: `test(cli): pin hidden analysis in the sandbox screen` / `feat(cli): hide analysis in the sandbox`.

### Step 4: review and PR

Run `/code-review medium` on the branch (about 150–250 changed lines, one review). Fix, re-run until only nits remain. Then open a PR from `feat/sandbox-hide` to `main`.

## Definition of done

- `uv run pytest` green.
- Manual check with `uv run ox-zero sandbox --size 6`: hide, play a few moves for both sides, watch the simulation count climb with no scores visible, press `ctrl+b`, and see the full analysis appear at once with a count that did not reset.
- `docs/cli.md` describes exactly what was built.
- PR merged. This plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- Test 4 fails because the reveal triggers a restart. That means `_apply`'s engine-job check took the restart branch. Report why rather than special-casing `hide`/`show` in `_apply`.
- `ctrl+b` turns out to be swallowed by the terminal or by Textual on this machine. Report it; the key is a decision in this plan, not the implementer's.
- Any design-level choice not in the Decisions table.

## Approvals (2026-10-12)

1. Commands `hide` / `show`, shortcut `ctrl+b`.
2. The simulation count stays visible while hidden.
3. No `--hidden` startup flag; the command is enough.
