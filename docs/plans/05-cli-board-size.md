# Plan 05: `--size` flag for the CLI

| | |
|---|---|
| Status | Planned (2026-10-11) |
| Branches | Plan: `plan/cli-board-size`. Implementation: `feat/cli-board-size` (already created from `main`, empty). |
| Design references | [docs/cli.md](../cli.md) (the behaviour to keep and the spec rows to change), [cli-integration.md](../design/cli-integration.md#loading-a-model) (a checkpoint for another size is an error), [engineering.md](../design/engineering.md) (why runs live outside `checkpoints/`), [open-questions.md](../design/open-questions.md#results-2026-10-10-the-first-6x6-run) (the 6x6 run this plan makes inspectable) |
| Depends on | [Plan 03](archive/03-cli-adapter.md) merged (PR #15): `SearchEngine` size guard, `_check_engine_accepts`, `latest_checkpoint`. [Plan 04](archive/04-training-pipeline.md) merged (PR #19): the 6x6 checkpoints under `runs/six-a/`. |

## Goal

Let the CLI analyse boards other than 12x12, so the 6x6 network from the first training run can be inspected with `analyze`, `best` and `sandbox` instead of only through metrics files. After this plan:

```bash
uv run ox-zero analyze --size 6 --model runs/six-a/gen_008.pt 2,2 3,3   # an explicit checkpoint
uv run ox-zero sandbox --size 6                                         # the newest 6x6 checkpoint under checkpoints/, or uniform search
```

Today the first line fails with `checkpoint is for a 6x6 board, position is 12x12`, because the CLI parses every position at the hard-coded default size. Everything below the parse (rules, renderer, sandbox, adapter, engine) already takes the size from the `State`; this plan threads one integer from the command line to the two `parse_position` calls, makes the checkpoint auto-load size-aware, and documents it.

## Deliverables

1. `--size N` on `analyze`, `best` and `sandbox` (spec change, see Decisions). Default 12.
2. `Session.from_tokens(tokens, size)` in `cli/sandbox/session.py`, and the `load` command parsing at the session's size.
3. `latest_checkpoint(root, size)` and `checkpoint_size(path)` in `training/checkpoint.py`; `load_engine(..., size)` in `cli/adapter.py` (spec change, see Decisions).
4. Tests in `tests/cli/test_app.py`, `tests/cli/sandbox/test_session.py`, `tests/training/test_checkpoint.py`.
5. Documentation: `docs/cli.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `docs/design/cli-integration.md`, `docs/design/engineering.md`.

## Out of scope

- Inferring the size from the loaded checkpoint when `--size` is omitted. Decided against (see Decisions); do not add it "while you're there".
- Any change to `ox_zero.game` or `ox_zero.engine`. `parse_position` already takes a `size`. `git diff main -- src/ox_zero/game src/ox_zero/engine` must be empty. In `ox_zero.training` only `checkpoint.py` changes, and only the lookup (`latest_checkpoint`, the new `checkpoint_size`); the file format, `save_checkpoint` and `load_checkpoint` are untouched.
- Rendering changes. The board renderer, the heatmap, the JSON shapes and the sandbox layout iterate over `state.size` and need no work. If a non-12 board renders wrongly, report it in the PR; do not fix it here unless it is a one-line bug.
- Changing the game's canonical size, the default of `BOARD_SIZE`, or the training scripts' `--size`.
- A `size` field in the JSON output. The `board` string's length already implies it, and adding a field is a spec change nobody asked for.

## Decisions made in this plan

Settled on 2026-10-11 with the author. Nothing below is open.

| Decision | Choice | Why |
|---|---|---|
| **`--size N` on all three commands, default 12** (spec change) | A shared `SizeOpt` like `TopOpt`: `typer.Option("--size", min=3)`; the help text does not repeat the default, Typer renders `[default: 12]` itself from `BOARD_SIZE` (as `TopOpt` does), so the text cannot drift from the constant. The value is passed to `parse_position(tokens, size)`, `Session.from_tokens(tokens, size)` and `load_engine(..., size)`. `min=3`, not `initial_state`'s `min=1`: 3 is the shortest board that can hold a line of three, and below it board-string detection misfires (`--size 2 5,11`: the token is 4 characters, `2 * 2`, so it is read as a board string and the error talks about invalid characters instead of a move off the board). | Explicit beats inferred: positions are parsed before the engine is loaded (the sandbox needs the position first, and the usage error for a bad position must not wait for torch to import), and the no-model path has nothing to infer from. 12 stays the default because it is the game's size and every existing command line must keep working unchanged. |
| **Auto-load picks the newest checkpoint *of the requested size*** (spec change) | With no `--model`, `load_engine` calls `latest_checkpoint(root, size)`, which walks the candidates in the existing order (generation, then mtime, descending) and returns the first whose stored `size` matches; `None` when none does, and the CLI then runs the uniform search with a notice naming the size. `size=None` keeps today's name-only behaviour for the one other caller (`scripts/make_checkpoint.py`). | Chosen by the author over the two alternatives (document the dead end; skip auto-load off-size). Without it, one 12x12 checkpoint in `checkpoints/` makes `--size 6` a hard error with no way to reach the uniform search, and `checkpoints/` can only ever hold one size. With it, `checkpoints/` becomes a per-size library: drop a 6x6 run in and `--size 6` finds it, `--size 12` is unaffected. The cost is reading candidates' headers until the first match, which is one small file in the common case. |
| **An explicit `--model` of another size is an error, not an override** | Unchanged behaviour: `_check_engine_accepts` already raises on mismatch, and its message already names both sizes (`checkpoint is for a 6x6 board, position is 12x12`). `--size 6 --model <12x12 file>` fails the same way with the sizes swapped. No new code; one new test pins it. | The author's call. A network's input and policy head are fixed to one board, so a mismatch is never what the user meant, and silently switching the parse size would make `--size` lie. Auto-load is different: nothing was named, so filtering is selection, not override. |
| **Reading a checkpoint's size** | `checkpoint_size(path)` does `torch.load(path, weights_only=True, map_location="cpu", mmap=True)["size"]`. `mmap=True` maps the zip archive instead of reading every tensor, so the scan stays cheap for a directory of large files. An unreadable candidate raises the same error `load_checkpoint` would (it is not skipped): a corrupt file in `checkpoints/` is something to report, not to hide, and today's behaviour already fails on it. Torch is imported inside the function, lazily, so `latest_checkpoint(root)` with `size=None` still never touches torch. | Reading contents contradicts the function's current docstring ("only names are read"), which is why the name-only path is kept for `size=None` and the docstring explains both. The matched file is loaded a second time by `load_checkpoint`; that double read is a few milliseconds with `mmap` and not worth an API that returns a half-loaded checkpoint. |
| **Board-string detection is unchanged** | `parse_position` already detects a board string by `len(token) == size * size` *or* by the token consisting only of `X`/`O`/`_`. With `--size 6` a 36-character string is detected by length; a 36-character string without `--size` is detected by its characters and rejected with `a board string must be 144 characters (12x12), got 36`. That message is kept as is. | The message names the expected size, which is enough of a hint. Mentioning `--size` from inside `ox_zero.game` would leak a CLI concept into the rules package. |
| **No per-size default for `--simulations`** | 800 everywhere. | A 6x6 search is faster per simulation, not in need of fewer. Keeping one default keeps the spec one line. |
| **Sandbox `reset` and `load` keep the session's size** | `reset` already builds `initial_state(self.state.size)`. `load` does not: it parses at the default size today (a bug that is invisible while every session is 12x12). It must parse at `self.state.size`. One test pins each. | The sandbox is checked against the engine once at launch (`_check_engine_accepts(engine, session.state.size)`); a `load` that changed the size would bypass that check and crash the worker thread. |

## Conventions that apply

- Test-driven: write the failing tests first, run `uv run pytest tests/cli tests/training -q` to see them fail, then implement until green. Every test in this plan runs on the placeholder engine or on `UniformEvaluator`, on CPU, in milliseconds; the checkpoint tests use the existing `write_checkpoint` helper with `TINY` (`tests/cli/test_app.py`) and a tiny `Network` in `tests/training/test_checkpoint.py`.
- Coordinates `(row, col)`, zero-based, row 0 at the top. A 6x6 win-in-one for the tests: X `0,0`, O `0,1`, X wins at `0,2` (`X O X` in row 0), same shape as the 12x12 fixture already in `test_app.py`.
- Comment thoroughly. The non-obvious bits are small but worth a sentence each: why the size is parsed before the engine is loaded, why the sandbox checks the size once at launch, why `min=3` (the shortest board that holds a line, and the smallest size where board-string detection cannot collide with a move token), what `mmap=True` buys and why the name-only path is kept.
- Conventional commits, small and atomic, on `feat/cli-board-size`. Suggested sequence: `docs(cli): specify --size and size-aware auto-load` / `test(cli): pin --size on analyze, best and sandbox` / `test(cli): pin the session size through load and reset` / `test(training): pin latest_checkpoint by size` / `feat(cli): add --size` / `feat(training): find the newest checkpoint of a size` / `docs: checkpoints/ holds one model per size`. Every test commit comes before the feature commits: the project rule is that no implementation is written before its tests exist, and that includes the `load` fix.
- **Code review cadence.** This is one logical chunk, so the review point is after step 3, before the docs commit and the PR. Run `/code-review` at medium effort, fold the findings in, and run it again; stop after the first round that finds nothing design-level.
- No new dependencies.

## Interfaces

```python
# cli/app.py
SizeOpt = Annotated[
    int,
    typer.Option("--size", min=3, help="Board side length."),   # Typer appends [default: 12]
]
# analyze, best, sandbox each gain `size: SizeOpt = BOARD_SIZE`   # the constant, never the literal 12
def _parse(tokens: Sequence[str], size: int, err: Console) -> State   # passes size to parse_position
def _load(model: Path | None, seed: int | None, device: Device, size: int, err: Console) -> Engine

# cli/adapter.py
def load_engine(model: Path | None, seed: int | None, device: str = "cpu",
                root: Path = Path("checkpoints"), size: int = BOARD_SIZE) -> tuple[Engine, str]
    # `model` given: load it, whatever its size (the mismatch surfaces in search(), as today).
    # Else latest_checkpoint(root, size). Else SearchEngine(UniformEvaluator()) with a notice naming `size`.
NO_MODEL_NOTICE = "No trained {size}x{size} model found: searching with uniform priors. ..."   # formatted with the size

# training/checkpoint.py
def latest_checkpoint(root: Path, size: int | None = None) -> Path | None
    # size=None: unchanged (names only). Otherwise the newest candidate whose checkpoint_size matches, or None.
def checkpoint_size(path: Path) -> int
    # The `size` stored in the file; torch imported lazily; raises like load_checkpoint on a bad file.

# cli/sandbox/session.py
class Session:
    @classmethod
    def from_tokens(cls, tokens: Sequence[str], size: int = BOARD_SIZE) -> Session
def _read_position(tokens: Sequence[str], size: int) -> tuple[State, list[Cell]]
```

`BOARD_SIZE` is imported from `ox_zero.game` everywhere a default is needed, so the CLI, the adapter and the sandbox share one constant. Keeping defaults on `from_tokens` and `load_engine` means every existing call and test stays valid.

## Step 1: the spec (`docs/cli.md`)

Spec first, per the project rule. Changes:

- "Positions and moves": the board-string bullet becomes "*size*² characters (144 for the default 12x12 board), row 0 first, ...". Add one bullet: "The board size defaults to 12 and is set with `--size`; a move off the board or a board string of the wrong length is an error."
- Flag table: new row `--size N`, ticked for all three commands: "Board side length, at least 3. Default 12. Positions are read at this size, and the default model is the newest checkpoint of this size; an explicit `--model` trained for another size is an error (status 2)."
- `--model` row: "Defaults to the newest in `checkpoints/`" becomes "Defaults to the newest checkpoint in `checkpoints/` for the board size (`--size`); checkpoints of other sizes are ignored". The rest of the row stays, including "a checkpoint for another board size is an error", which now describes the explicit `--model` case only. Make that explicit: "A path that does not exist, a file that is not a readable checkpoint, and a `--model` checkpoint for another board size are errors (status 2)."
- "Why the gaps": nothing to add; `--size` applies everywhere.
- `sandbox` section: one sentence under the paragraph about `load`: "`load` and `reset` keep the size the sandbox was started with."

## Step 2: the CLI tests (`tests/cli/test_app.py`)

All on the placeholder engine (autouse `fast_engine`) unless marked `real_engine`. The fixture's fake `load` must accept the new keyword (`def load(model, seed, device="cpu", size=BOARD_SIZE)`), or every test fails with a `TypeError` instead of the Typer error the red phase expects; change the fixture in the same commit as the tests.

1. **`analyze --size 6` renders a 6x6 board.** `run("analyze", "--size", "6", "2,2", "3,3", "--json")`: exit 0, `board` in the JSON is 36 characters, `moves` has 34 entries, every move is inside `[0, 6)`.
2. **`best --size 6` finds a 6x6 win in one.** Marked `real_engine`, with the checkpoint root patched to an empty `workdir` so the uniform search runs: `run("best", "--size", "6", "0,0", "0,1", "--simulations", "50")` prints `0,2`. The 12x12 twin of this test exists already. `best --json` carries only `move` and `score` (`report.py`, `best_json`), so this test cannot prove the parse size on its own; it proves the real search works at 6x6, while tests 1, 4 and 7 prove the size. A second, separate test (not a parametrisation: the assertions differ) does the same through `analyze --size 6 0,0 0,1 --simulations 50 --json`, asserting `top[0]["move"] == [0, 2]` and `len(board) == 36`, so both commands are exercised on the real engine.
3. **A 36-character board string parses with `--size 6`.** `run("analyze", "--size", "6", "X" + "_" * 35, "--json")` (one X, so O to move): exit 0, `to_move == "O"`. Without `--size`, the same string exits 2 and stderr contains `144 characters`.
4. **A move off the small board is a usage error.** `run("analyze", "--size", "6", "7,7")`: exit 2, stderr mentions `off the 6x6 board`, caret under the token (the existing caret tests show how to assert it).
5. **`--size 2` is rejected by Typer.** Exit 2, nothing on stdout, stderr names the minimum. (And `--size 3` is accepted: `run("analyze", "--size", "3", "--json")` exits 0 with a 9-character `board`.)
6. **Explicit `--model` mismatch, both directions.** Extend `test_board_size_mismatch_is_a_usage_error` (parametrised over the three commands) with a second case: a 12x12 checkpoint (`write_checkpoint(..., size=12)`) passed with `--model` and `--size 4`. The position must fit the small board: pass no position at all (the empty 4x4 board), not the module's `POSITION` (`5,5` is off a 4x4 board and would fail in the parser before the engine is ever loaded, so the test would exit 2 for the wrong reason). Assert the exact substring `checkpoint is for a 12x12 board, position is 4x4`, exit 2, nothing on stdout. Tighten the existing case (4x4 checkpoint, default size) to the exact substring `checkpoint is for a 4x4 board, position is 12x12` too; asserting that both sizes merely appear somewhere cannot tell the two cases apart or catch them being swapped. Keep the parametrisation over all three commands.
7. **An explicit matching checkpoint works.** Marked `real_engine`: `write_checkpoint(workdir / "six.pt", size=6)`, then `run("analyze", "--size", "6", "--model", str(path), "2,2", "--simulations", "10", "--json")`: exit 0, notice contains `6x6`, `board` is 36 characters.
8. **Auto-load picks the newest checkpoint of the requested size.** Marked `real_engine`, root patched to `workdir / "checkpoints"` as `test_newest_checkpoint_is_picked` does. Write `run/gen_002.pt` at size 12 and `six/gen_001.pt` at size 6 (different seeds). `load_engine(None, None, root=root, size=6)` returns an engine whose `size == 6` and a notice naming `six/gen_001.pt` and `generation 1`, even though generation 2 is newer; `size=12` (and the default) still picks `run/gen_002.pt`, so the existing test keeps passing. Through the CLI: `run("analyze", "--size", "6", "2,2", "--simulations", "10", "--json")` with no `--model` exits 0 with a 36-character `board` and the 6x6 notice on stderr. This is the command the auto-load decision exists for.
9. **No checkpoint of the requested size falls back to uniform search.** Same root with only the 12x12 checkpoint: `load_engine(None, None, root=root, size=6)` returns a `SearchEngine` on `UniformEvaluator`, and the notice contains both `6x6` and `uniform priors`. Through the CLI, `run("best", "--size", "6", "0,0", "0,1", "--simulations", "50")` prints `0,2` and exits 0: the 12x12 file in `checkpoints/` is not an error for a 6x6 analysis.
10. **The sandbox launches at the given size.** With the `launched` fixture: `run("sandbox", "--size", "6", "2,2")` → `session.state.size == 6`. And `run("sandbox", "--size", "6")` with no position → an empty 6x6 session. Also assert, via the `fast_engine` fixture's recorded calls (extend them to record `size`), that `sandbox --size 6` asks `load_engine` for size 6: the sandbox is the one command where the size reaches the loader through a different path than `_parse`.
11. **`--size` appears in `--help`**, parametrised over `analyze`, `best` and `sandbox`. Cheap, and together with test 6 it is the only test that touches all three commands at once, so it must stay parametrised: tests 1, 7 and 8 cover `analyze`, tests 2 and 9 cover `best`, test 10 covers `sandbox`.

Run `uv run pytest tests/cli/test_app.py -q` on the feature branch before step 3. Every test that passes `--size` on the command line fails with Typer's "No such option: --size"; the direct `load_engine(..., size=6)` calls fail with `TypeError`; test 11 fails on its assertion. The two halves that do not pass `--size` (the no-flag case of test 3 and the tightened existing case of test 6) already pass, which is expected: they pin behaviour that must not change.

## Step 2b: the session tests (`tests/cli/sandbox/test_session.py`)

1. **`from_tokens` at size 6.** `Session.from_tokens(["2,2", "3,3"], size=6)`: `state.size == 6`, history has two moves, undo twice reaches an empty 6x6 board.
2. **A 36-character board string starts a 6x6 session** with no history.
3. **`load` keeps the session's size.** Start at size 6, `load 1,1 2,2`: still 6x6. `load` with a 144-character string on a 6x6 session is an error message, not a size change.
4. **`reset` keeps the size.** Start at size 6, play a move, `reset`: `state.size == 6`, empty board.

Tests 1, 2 and 4 fail on `main` with `TypeError` (no `size` parameter). Test 3 fails for a different reason, the `load` bug described in step 3, and that is the point of writing it now: the fix lands in step 3 with a test already waiting for it. Run `uv run pytest tests/cli/sandbox -q` to see all four fail before moving on.

## Step 2c: the checkpoint lookup tests (`tests/training/test_checkpoint.py`)

The existing `latest_checkpoint` tests use empty files, because only names were read. The new ones need real files: save a tiny `Network(size, tiny_config)` with `save_checkpoint`, as the round-trip tests in the same file already do. Keep them small (1 block, 4 filters) so the whole section stays under a second.

1. **`checkpoint_size` reads the stored size.** A 4x4 and a 6x6 checkpoint in `tmp_path`: `checkpoint_size` returns 4 and 6. A file that is not a checkpoint (write a few bytes of text to `gen_001.pt`) raises; assert the exception type matches what `load_checkpoint` raises on the same file, so the two never drift apart.
2. **`latest_checkpoint(root, size)` filters by size, newest first.** `a/gen_003.pt` at 6x6, `b/gen_010.pt` at 12x12, `b/gen_002.pt` at 6x6: `size=6` returns `a/gen_003.pt` (newest 6x6, skipping the newer 12x12), `size=12` returns `b/gen_010.pt`, `size=4` returns `None`, and `size=None` returns `b/gen_010.pt` exactly as before.
3. **The name-only path never opens a file.** With `size=None`, the existing empty-file tests keep passing unchanged. Add one assertion that `latest_checkpoint(root)` on a directory of empty `gen_*.pt` files returns a path rather than raising: the proof that `size=None` did not start reading contents.
4. **Tie-breaking survives the filter.** Two 6x6 checkpoints at the same generation in different runs: the more recently modified wins, as in the existing tie test (`os.utime`).

These fail with `TypeError` (`size` unexpected) and `ImportError` (`checkpoint_size`) before step 3.

## Step 3: implement (`cli/app.py`, `cli/adapter.py`, `cli/sandbox/session.py`, `training/checkpoint.py`)

Two feature commits, in this order; the first can go green on its own if `load_engine` temporarily ignores `size`, but it is simpler to land both before running the full suite.

`training/checkpoint.py`:

- `checkpoint_size(path)`: lazy `import torch`, `torch.load(path, weights_only=True, map_location="cpu", mmap=True)["size"]`. Comment what `mmap=True` does (maps the archive, reads only the entries touched) and why `weights_only=True` is still required (never unpickle arbitrary objects from a file in a directory the user drops things into).
- `latest_checkpoint(root, size=None)`: build the candidate list as today, sort it descending by `(generation, mtime)`, then return the first candidate if `size is None`, else the first whose `checkpoint_size` equals `size`. Rewrite the docstring: names only for `size=None`; with a size, headers are read in order until the first match, which is usually the first file.

`cli/adapter.py`:

- `load_engine(..., size=BOARD_SIZE)`: `latest_checkpoint(root, size)`; the notice for the uniform fallback is formatted with the size. Update the docstring's bullet list.

`cli/app.py`:

- Add `SizeOpt` next to `TopOpt`. Add `size: SizeOpt = BOARD_SIZE` to the three command signatures; place it right after `position` so `--help` lists it near the input it changes.
- `_parse(tokens, size, err)` passes `size` through. Update `PositionArg`'s help text: "or a board string (size² characters)".
- `_load(model, seed, device, size, err)` passes `size` to `load_engine`.
- `sandbox` calls `Session.from_tokens(tokens, size)`. Extend the comment above `_check_engine_accepts` in `sandbox` ("Every position it reaches (undo included) has the start position's board size") to name `load` explicitly as the one command that could break that invariant, and point at the session test that pins it.
- Update the module docstring of `app.py` only if it mentions the board size; the spec reference at the top is enough.

`cli/sandbox/session.py`:

- Thread `size` through `from_tokens` and `_read_position`. **Latent bug, fix here:** the `load` command (`session.py:184`) calls `_read_position(tokens)` with no size, so it parses at 12 regardless of the session. On a 6x6 session, `load 1,1` would silently produce a 12x12 position, which the engine was never checked against (the size check runs once at launch) and would crash the search worker. Pass `self.state.size`. The step 2b tests pin it.

`scripts/make_checkpoint.py` keeps calling `latest_checkpoint(Path("checkpoints"))` with no size; it only reports what exists.

Green: `uv run pytest -q`.

**Review point:** `/code-review` (medium) on the branch here.

## Step 4: try it by hand

Not test-driven. Run, and paste the outputs into the PR description:

```bash
uv run ox-zero analyze --size 6 --model runs/six-a/gen_008.pt 2,2 3,3 --simulations 400
uv run ox-zero best --size 6 --model runs/six-a/gen_008.pt
mkdir -p checkpoints/six-a && cp runs/six-a/gen_008.pt checkpoints/six-a/   # then, with no --model:
uv run ox-zero analyze --size 6 2,2 3,3 --simulations 400                   # auto-loads the 6x6 file
uv run ox-zero analyze 5,5 6,6 5,6 --simulations 100                        # still uniform at 12x12, no error
uv run ox-zero sandbox --size 6   # play a few moves, undo, reset, load, quit
```

Things to look at, and report rather than fix if wrong: the 6x6 board renders with the right column headers and no stray padding; the sandbox's mouse-to-cell mapping (it uses `state.size`) picks the right cell on a 6x6 board; `export` prints 36 characters; the notices on stderr name the right file and size in each of the three loading cases. If the first-move scores look interesting (the 6x6 run's open question is O's dominance), note it in the PR but take it to `open-questions.md` separately, not in this PR.

## Step 5: documentation

- `README.md`: the status line's "until a 12x12 checkpoint exists under `checkpoints/`" becomes "until a checkpoint for the board size exists under `checkpoints/` (`--size` selects the board; 12 by default)". The quickstart gains one line with a placeholder path, never a real one, because `runs/` is gitignored: `uv run ox-zero analyze --size 6 --model runs/<name>/gen_NNN.pt 2,2 3,3   # inspect a 6x6 run from train.py --size 6`. Line 39: "144-character board strings" → "board strings (size² characters)".
- `docs/architecture.md`, line 87: "the CLI auto-loads the newest checkpoint under `checkpoints/` and is 12x12 only, so a trained model reaches it by being copied or linked there, or passed with `--model`" → "the CLI auto-loads the newest checkpoint of the requested size (`--size`, default 12) under `checkpoints/`, so a trained model of any size reaches it by being copied or linked there, or passed with `--model`".
- `scripts/README.md`, the `train.py` paragraph: "the CLI auto-loads from there and is 12x12 only, so copy or symlink a checkpoint there to use it" → "the CLI auto-loads the newest checkpoint of the requested size from there, so copy or symlink a checkpoint under `checkpoints/` to use it with `--size N`, or pass it directly with `--model`".
- `docs/design/engineering.md`, the "Why not `checkpoints/`" paragraph (line 85): the reason is no longer "12x12 only". Rewrite it: runs stay in `runs/` because a run is a working directory (buffer, metrics, every generation), while `checkpoints/` is the library of models the CLI picks from, one newest per size; a 6x6 run in it no longer breaks 12x12 commands (plan 05). Keep the sentence about releases.
- `docs/design/cli-integration.md`, "Loading a model" (line 34): "the newest checkpoint in `checkpoints/` (recursively, by generation)" gains "of the requested board size"; one dated line (2026-10-11) recording that the CLI takes `--size`, that auto-load filters by it, and that an explicit `--model` mismatch stays an error, linking to this plan's Decisions.
- `docs/plans/README.md`: nothing. This file moves to `archive/` on merge.

## Definition of done

- `uv run pytest` is green, including every pre-existing CLI, sandbox and checkpoint test.
- Author-only manual checks (the file is gitignored and exists only on the author's machine; tests 7, 8 and 9 are the reproducible equivalents): the step 4 commands run as described, and `uv run ox-zero best --size 6 --model runs/six-a/gen_008.pt 0,0 0,1` prints `0,2` with a notice saying `6x6`; the same command without `--size` still fails with the size message, exit 2.
- Every pre-existing command line in `docs/cli.md`'s examples works unchanged, with or without a 6x6 checkpoint under `checkpoints/`.
- `git diff main -- src/ox_zero/game src/ox_zero/engine` is empty; `git diff main -- src/ox_zero/training` touches only `checkpoint.py`, and only `latest_checkpoint` and the new `checkpoint_size`.
- The `/code-review` rounds happened and their findings were addressed or explicitly deferred in the PR description.
- A pull request from `feat/cli-board-size` to `main` with the step 4 outputs. On merge, this plan moves to `docs/plans/archive/`.
