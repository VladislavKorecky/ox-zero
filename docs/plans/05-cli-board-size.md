# Plan 05: `--size` flag for the CLI

| | |
|---|---|
| Status | Planned (2026-10-11) |
| Branches | Plan: `plan/cli-board-size`. Implementation: `feat/cli-board-size` (already created from `main`, empty). |
| Design references | [docs/cli.md](../cli.md) (the behaviour to keep and the spec rows to change), [cli-integration.md](../design/cli-integration.md#loading-a-model) (a checkpoint for another size is an error), [open-questions.md](../design/open-questions.md#results-2026-10-10-the-first-6x6-run) (the 6x6 run this plan makes inspectable) |
| Depends on | [Plan 03](archive/03-cli-adapter.md) merged (PR #15): `SearchEngine` size guard, `_check_engine_accepts`. [Plan 04](archive/04-training-pipeline.md) merged (PR #19): the 6x6 checkpoints under `runs/six-a/`. |

## Goal

Let the CLI analyse boards other than 12x12, so the 6x6 network from the first training run can be inspected with `analyze`, `best` and `sandbox` instead of only through metrics files. After this plan:

```bash
uv run ox-zero analyze --size 6 --model runs/six-a/gen_008.pt 2,2 3,3
uv run ox-zero sandbox --size 6 --model runs/six-a/gen_008.pt
```

Today the second line fails with `checkpoint is for a 6x6 board, position is 12x12`, because the CLI parses every position at the hard-coded default size. Everything below the parse (rules, renderer, sandbox, adapter, engine) already takes the size from the `State`; this plan threads one integer from the command line to the two `parse_position` calls and documents it.

## Deliverables

1. `--size N` on `analyze`, `best` and `sandbox` (spec change, see Decisions). Default 12.
2. `Session.from_tokens(tokens, size)` in `cli/sandbox/session.py`.
3. Tests in `tests/cli/test_app.py` and `tests/cli/sandbox/test_session.py`.
4. Documentation: `docs/cli.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `docs/design/cli-integration.md`.

## Out of scope

- Inferring the size from the loaded checkpoint when `--size` is omitted. Decided against (see Decisions); do not add it "while you're there".
- Any change to `ox_zero.game`, `ox_zero.engine`, `ox_zero.training`. `parse_position` already takes a `size`; nothing else is needed. `git diff main -- src/ox_zero/game src/ox_zero/engine src/ox_zero/training` must be empty.
- Rendering changes. The board renderer, the heatmap, the JSON shapes and the sandbox layout iterate over `state.size` and need no work. If a non-12 board renders wrongly, report it in the PR; do not fix it here unless it is a one-line bug.
- Changing the game's canonical size, the default of `BOARD_SIZE`, or the training scripts' `--size`.
- A `size` field in the JSON output. The `board` string's length already implies it, and adding a field is a spec change nobody asked for.

## Decisions made in this plan

Settled on 2026-10-11 with the author. Nothing below is open.

| Decision | Choice | Why |
|---|---|---|
| **`--size N` on all three commands, default 12** (spec change) | A shared `SizeOpt` like `TopOpt`: `typer.Option("--size", min=1, help=...)`. The value is passed to `parse_position(tokens, size)` and to `Session.from_tokens(tokens, size)`. `min=1` mirrors `initial_state`'s own check; no stricter bound. | Explicit beats inferred: positions are parsed before the engine is loaded (the sandbox needs the position first, and the usage error for a bad position must not wait for torch to import), and the no-model path has nothing to infer from. 12 stays the default because it is the game's size and every existing command line must keep working unchanged. |
| **A checkpoint of another size is an error, not an override** | Unchanged behaviour: `_check_engine_accepts` already raises on mismatch, and its message already names both sizes (`checkpoint is for a 6x6 board, position is 12x12`). `--size 6` with a 12x12 checkpoint fails the same way with the sizes swapped. No new code; one new test pins it. | The author's call. A network's input and policy head are fixed to one board, so a mismatch is never what the user meant, and silently switching the parse size would make `--size` lie. |
| **Board-string detection is unchanged** | `parse_position` already detects a board string by `len(token) == size * size` *or* by the token consisting only of `X`/`O`/`_`. With `--size 6` a 36-character string is detected by length; a 36-character string without `--size` is detected by its characters and rejected with `a board string must be 144 characters (12x12), got 36`. That message is kept as is. | The message names the expected size, which is enough of a hint. Mentioning `--size` from inside `ox_zero.game` would leak a CLI concept into the rules package. |
| **No per-size default for `--simulations`** | 800 everywhere. | A 6x6 search is faster per simulation, not in need of fewer. Keeping one default keeps the spec one line. |
| **Sandbox `reset` and `load` keep the session's size** | `reset` already builds `initial_state(self.state.size)`. `load` does not: it parses at the default size today (a bug that is invisible while every session is 12x12). It must parse at `self.state.size`. One test pins each. | The sandbox is checked against the engine once at launch (`_check_engine_accepts(engine, session.state.size)`); a `load` that changed the size would bypass that check and crash the worker thread. |

## Conventions that apply

- Test-driven: write the failing tests first, run `uv run pytest tests/cli -q` to see them fail, then implement until green. Every test in this plan runs on the placeholder engine or on `UniformEvaluator`, on CPU, in milliseconds; the one checkpoint test uses the existing `write_checkpoint` helper with `TINY`.
- Coordinates `(row, col)`, zero-based, row 0 at the top. A 6x6 win-in-one for the tests: X `0,0`, O `0,1`, X wins at `0,2` (`X O X` in row 0), same shape as the 12x12 fixture already in `test_app.py`.
- Comment thoroughly. The non-obvious bits here are small but worth a sentence each: why the size is parsed before the engine is loaded, why the sandbox checks the size once at launch, why `min=1` and not 3.
- Conventional commits, small and atomic, on `feat/cli-board-size`. Suggested sequence: `docs(cli): specify --size` / `test(cli): pin --size on analyze, best and sandbox` / `feat(cli): add --size` / `test(cli): pin the session size through load and reset` / `docs: drop the "12x12 only" notes`.
- **Code review cadence.** This is one logical chunk. Run `/code-review` at medium effort once, after step 3 and before the docs commit and the PR. Fold the findings in; if a round finds nothing design-level, stop.
- No new dependencies.

## Interfaces

```python
# cli/app.py
SizeOpt = Annotated[
    int,
    typer.Option("--size", min=1, help="Board side length. Default 12."),
]
# analyze, best, sandbox each gain `size: SizeOpt = 12`
def _parse(tokens: Sequence[str], size: int, err: Console) -> State   # passes size to parse_position

# cli/sandbox/session.py
class Session:
    @classmethod
    def from_tokens(cls, tokens: Sequence[str], size: int = BOARD_SIZE) -> Session
def _read_position(tokens: Sequence[str], size: int) -> tuple[State, list[Cell]]
```

`BOARD_SIZE` is imported from `ox_zero.game` so the sandbox and the CLI share one default; `app.py`'s `12` default on `SizeOpt` is written as `BOARD_SIZE` too. Keeping the default on `from_tokens` means `test_session.py`'s existing calls stay valid.

## Step 1: the spec (`docs/cli.md`)

Spec first, per the project rule. Changes:

- "Positions and moves": the board-string bullet becomes "*size*² characters (144 for the default 12x12 board), row 0 first, ...". Add one bullet: "The board size defaults to 12 and is set with `--size`; a move off the board or a board string of the wrong length is an error."
- Flag table: new row `--size N`, ticked for all three commands: "Board side length. Default 12. Positions are read at this size; a checkpoint trained for another size is an error (status 2)."
- `--model` row: unchanged; it already says a checkpoint for another board size is an error.
- "Why the gaps": nothing to add; `--size` applies everywhere.
- `sandbox` section: one sentence under the paragraph about `load`: "`load` and `reset` keep the size the sandbox was started with."

## Step 2: the CLI tests (`tests/cli/test_app.py`)

All on the placeholder engine (autouse `fast_engine`) unless marked `real_engine`.

1. **`analyze --size 6` renders a 6x6 board.** `run("analyze", "--size", "6", "2,2", "3,3", "--json")`: exit 0, `board` in the JSON is 36 characters, `moves` has 34 entries, every move is inside `[0, 6)`.
2. **`best --size 6` finds a 6x6 win in one.** Marked `real_engine`, with the checkpoint root patched to an empty `workdir` so the uniform search runs: `run("best", "--size", "6", "0,0", "0,1", "--simulations", "50")` prints `0,2`. The 12x12 twin of this test exists already; this one proves the search really ran on the small board (a 12x12 parse would have printed the same move, so also assert the `board` field via `--json`, or assert on the move count in stderr if `best` prints none: use `--json`).
3. **A 36-character board string parses with `--size 6`.** `run("analyze", "--size", "6", "X" + "_" * 35, "--json")` (one X, so O to move): exit 0, `to_move == "O"`. Without `--size`, the same string exits 2 and stderr contains `144 characters`.
4. **A move off the small board is a usage error.** `run("analyze", "--size", "6", "7,7")`: exit 2, stderr mentions `off the 6x6 board`, caret under the token (the existing caret tests show how to assert it).
5. **`--size 0` is rejected by Typer.** Exit 2, nothing on stdout.
6. **Size mismatch, both directions.** Extend `test_board_size_mismatch_is_a_usage_error` (parametrised over the three commands) with a second case: a 12x12 checkpoint (`write_checkpoint(..., size=12)`) and `--size 4`; stderr contains `12x12` and `4x4`, exit 2, nothing on stdout. The existing case (4x4 checkpoint, default size) stays.
7. **A matching checkpoint works.** Marked `real_engine`: `write_checkpoint(workdir / "six.pt", size=6)`, then `run("analyze", "--size", "6", "--model", str(path), "2,2", "--simulations", "10", "--json")`: exit 0, notice contains `6x6`, `board` is 36 characters. This is the command the whole plan exists for.
8. **The sandbox launches at the given size.** With the `launched` fixture: `run("sandbox", "--size", "6", "2,2")` → `session.state.size == 6`. And `run("sandbox", "--size", "6")` with no position → an empty 6x6 session.
9. **`--size` appears in `--help`** for each of the three commands. (Cheap, and catches a command that forgot the option.)

Run `uv run pytest tests/cli/test_app.py -q`; the new tests must fail on `main` with Typer's "No such option: --size".

## Step 3: implement `--size` (`cli/app.py`, `cli/sandbox/session.py`)

- Add `SizeOpt` next to `TopOpt`. Add `size: SizeOpt = BOARD_SIZE` to the three command signatures; place it right after `position` so `--help` lists it near the input it changes.
- `_parse(tokens, size, err)` passes `size` through. Update `PositionArg`'s help text: "or a board string (size² characters)".
- `sandbox` calls `Session.from_tokens(tokens, size)`.
- `session.py`: thread `size` through `from_tokens` and `_read_position`. **Latent bug, fix here:** the `load` command (`session.py:184`) calls `_read_position(tokens)` with no size, so it parses at 12 regardless of the session. On a 6x6 session, `load 1,1` would silently produce a 12x12 position, which the engine was never checked against (the size check runs once at launch) and would crash the search worker. Pass `self.state.size`. The step 4 tests pin it.
- Update the module docstring of `app.py` only if it mentions the board size; the spec reference at the top is enough.

Green: `uv run pytest tests/cli -q`.

**Review point:** `/code-review` (medium) on the branch here.

## Step 4: the session tests (`tests/cli/sandbox/test_session.py`)

1. **`from_tokens` at size 6.** `Session.from_tokens(["2,2", "3,3"], size=6)`: `state.size == 6`, history has two moves, undo twice reaches an empty 6x6 board.
2. **A 36-character board string starts a 6x6 session** with no history.
3. **`load` keeps the session's size.** Start at size 6, `load 1,1 2,2`: still 6x6. `load` with a 144-character string on a 6x6 session is an error message, not a size change.
4. **`reset` keeps the size.** Start at size 6, play a move, `reset`: `state.size == 6`, empty board.

Test 3 fails on `main` because of the `load` bug named in step 3; fix it in the same commit as the session change and say so in the commit body.

## Step 5: try it by hand

Not test-driven. Run, and paste the outputs into the PR description:

```bash
uv run ox-zero analyze --size 6 --model runs/six-a/gen_008.pt 2,2 3,3 --simulations 400
uv run ox-zero best --size 6 --model runs/six-a/gen_008.pt
uv run ox-zero sandbox --size 6 --model runs/six-a/gen_008.pt   # play a few moves, undo, reset, load, quit
```

Things to look at, and report rather than fix if wrong: the 6x6 board renders with the right column headers and no stray padding; the sandbox's mouse-to-cell mapping (it uses `state.size`) picks the right cell on a 6x6 board; `export` prints 36 characters. If the first-move scores look interesting (the 6x6 run's open question is O's dominance), note it in the PR but take it to `open-questions.md` separately, not in this PR.

## Step 6: documentation

- `README.md`: the status line's "until a 12x12 checkpoint exists under `checkpoints/`" becomes "until a checkpoint exists under `checkpoints/`; `--size` plays other boards, e.g. the 6x6 run". The quickstart gains one line: `uv run ox-zero analyze --size 6 --model runs/six-a/gen_008.pt 2,2 3,3   # inspect the 6x6 run`. Line 39: "144-character board strings" → "board strings (size² characters)".
- `docs/architecture.md`, line 87: drop "and is 12x12 only"; say that `--size` selects the board and a mismatched checkpoint is an error.
- `scripts/README.md`, the `train.py` paragraph: "the CLI auto-loads from there and is 12x12 only, so copy or symlink" → "the CLI auto-loads from there, so pass `--model runs/<name>/gen_NNN.pt --size N` to inspect a run".
- `docs/design/cli-integration.md`, "Loading a model": one dated line (2026-10-11) recording that the CLI takes `--size` and that a mismatch stays an error, linking to this plan's Decisions.
- `docs/plans/README.md`: nothing. This file moves to `archive/` on merge.

## Definition of done

- `uv run pytest` is green, including every pre-existing CLI and sandbox test.
- `uv run ox-zero best --size 6 --model runs/six-a/gen_008.pt 0,0 0,1` prints `0,2` and the loaded-model notice says `6x6`.
- `uv run ox-zero best --model runs/six-a/gen_008.pt 0,0 0,1` (no `--size`) still fails with the size message, exit 2.
- Every pre-existing command line in `docs/cli.md`'s examples works unchanged.
- `git diff main -- src/ox_zero/game src/ox_zero/engine src/ox_zero/training` is empty.
- The `/code-review` run happened and its findings were addressed or explicitly deferred in the PR description.
- A pull request from `feat/cli-board-size` to `main` with the step 5 outputs. On merge, this plan moves to `docs/plans/archive/`.
