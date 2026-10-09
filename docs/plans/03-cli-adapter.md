# Plan 03: Connect the engine to the CLI

| | |
|---|---|
| Status | Ready for implementation (scope and decisions settled 2026-10-09) |
| Branches | Plan: `plan/cli-adapter`. Implementation: `feat/cli-adapter`. |
| Design references | [cli-integration.md](../design/cli-integration.md) (the mapping, the port change, root expansion, loading a model), [engineering.md](../design/engineering.md) (checkpoints, devices, module layout, the measured batch-size-1 numbers), [docs/cli.md](../cli.md) (the behaviour the CLI must keep) |
| Depends on | [Plan 02](archive/02-engine-search-network.md) merged (PR #12): `ox_zero.engine` with `analyse`, `Snapshot`, `UniformEvaluator`, `NetworkEvaluator`, `select_device`, `Network`, `NetworkConfig`. |

## Goal

Make `ox-zero analyze`, `best` and `sandbox` run on the real search. After this plan the CLI loads a checkpoint when one exists, falls back to the real search with uniform priors when none does, and the `PlaceholderEngine` is test tooling only. The checkpoint format is defined here, so plan 04 (training) only has to *write* files the CLI already reads.

The proof is a CLI that finds tactics on its own with no model at all (win-in-one, loss-in-one avoided), plus the test suite.

## Deliverables

1. `cli/engine_port.py`: `Analysis` gains the engine's chosen move; `best` returns it.
2. `cli/adapter.py`: `SearchEngine`, the adapter from `engine.analyse` to the `Engine` port, and `load_engine`, the CLI's single point of obtaining an engine (moved here from `placeholder.py`).
3. `training/checkpoint.py`: save and load of a self-describing checkpoint, plus `latest_checkpoint`.
4. `--device` flag on all three commands (spec change, see Decisions).
5. `scripts/make_checkpoint.py`: writes a random-weights checkpoint so `--model` can be tried by hand before plan 04 exists.
6. Tests: `tests/cli/test_adapter.py`, `tests/training/test_checkpoint.py`, updated `tests/cli/test_engine_port.py`, `test_placeholder.py`, `test_app.py`.
7. Documentation updates in step 7; `src/ox_zero/training/README.md` deleted.

## Out of scope

- Self-play, replay buffer, trainer, tournaments, the generation loop, `scripts/train.py`. Plan 04. The checkpoint module is pulled forward from plan 04 and nothing else.
- Every search upgrade, in particular virtual loss (batched leaves in the CLI) and MCTS-Solver. Analysis stays batch size 1 and the root value stays an average.
- Any change to `ox_zero.engine`. If the adapter needs something the engine does not expose, stop and ask (see the end).
- The sandbox's UI, rendering, JSON shapes, exit codes: unchanged. Only the engine behind them changes.
- Performance work. The numbers in [engineering.md](../design/engineering.md#measured-2026-09-27) are what the CLI gets.

## Decisions made in this plan

Code-level choices the design left open, and three changes to `docs/cli.md` (marked "spec change"), all settled on 2026-10-09. Nothing below is open: an implementer follows this table without asking. Anything that turns out wrong in review gets changed here, not in code.

| Decision | Choice | Why |
|---|---|---|
| **No model: uniform-evaluator search, not the placeholder** (spec change) | With no checkpoint the CLI runs the real search with `UniformEvaluator`: plain PUCT with equal priors and value 0 at the leaves. A notice on stderr says so. `PlaceholderEngine` stays in `cli/placeholder.py` for the CLI tests, which need a fast, deterministic, time-faking engine, and is never returned by `load_engine`. | Uniform search finds short tactics (plan 02's fixtures: win in one, loss in one avoided, deep forced wins on 4x4), so the CLI becomes useful before any training. The placeholder's only purpose was to exist before the engine did. |
| **`--device auto\|cpu\|mps\|cuda`, default `cpu`** (spec change) | A new flag on `analyze`, `best`, `sandbox`. `cpu` is the default; `auto` is `select_device()`; a named device is passed to `select_device(name)`. Ignored, with no notice, when the engine is the uniform search. | Analysis runs at batch size 1, where the benchmark measured `mps` 1.7–3× slower than `cpu` ([engineering.md](../design/engineering.md#measured-2026-09-27)). `cpu` rather than `auto` was chosen deliberately: the CLI should be fastest on the machine it is developed on. The flag exists so a CUDA machine or a future batched search can be tried without a code change; revisit the default when virtual loss lands. |
| **Chosen move is a required field** | `Analysis(value, scores, simulations, chosen)`. `best` returns `(chosen, scores[chosen])`. `top(n)` still ranks by score. `PlaceholderEngine` sets `chosen` to its highest-scoring move, so its tests only change where they construct `Analysis` directly. | [cli-integration.md](../design/cli-integration.md#the-one-port-change). A default would silently hide an adapter that forgot to set it. |
| **Score mapping** | `value = (snapshot.value + 1) / 2`; `scores[a] = (snapshot.q[a] + 1) / 2`; `chosen = snapshot.best`; `simulations = snapshot.simulations`. Every snapshot is forwarded, including the first with `simulations = 0`. | The mapping table in [cli-integration.md](../design/cli-integration.md#mapping). Forwarding the `0` snapshot keeps "strictly increasing" true and gives the sandbox a full board of scores before the first simulation finishes. `engine_port.analyze` already requires `simulations >= 1`, so a capped run always ends on a later snapshot. |
| **`--seed` and the real engine** | `load_engine` still takes the seed and `SearchEngine` accepts it, but `analyse` takes no rng and needs none: in analysis mode (no root noise, most-visited choice, board-order ties) the search is deterministic. The adapter stores the seed and does nothing with it; `docs/cli.md` says so in one sentence. | Honest about what the flag does. Removing it would break the spec for no gain, and the placeholder (tests) still uses it. |
| **Checkpoint contents** | One `torch.save` dict of tensors and primitives only, loadable with `weights_only=True`: `format_version` (int, `1`), `size`, `generation`, `network_config` (a dict of the dataclass fields), `model_state`, `optimizer_state` (or `None`), `configs` (a mapping name → dict of fields, for plan 04's `SearchConfig`, `SelfPlayConfig`, `TrainConfig`), `rng` (a dict: NumPy `bit_generator.state`, torch CPU RNG state tensor; `None` when not training). | [engineering.md](../design/engineering.md#configuration-and-checkpoints): self-describing, rebuilds the network from the file alone. Plain data so loading an untrusted file never unpickles arbitrary objects. Dataclasses are reconstructed by the loader, not pickled. `configs` is a bag so plan 04 extends it without a format change. |
| **Which checkpoint is newest** | `latest_checkpoint(root)` globs `root/**/gen_*.pt`, parses the integer in `gen_(\d+)\.pt`, and returns the highest generation; ties (two runs at the same generation) go to the most recently modified file. Files that do not match the pattern are ignored. Returns `None` for no match or no `root`. | The spec says "newest in `checkpoints/` (recursively, by generation)" and the design fixes the file name. Parsing names costs nothing; loading every file to read its generation would. |
| **Board-size mismatch** | `SearchEngine.search` raises `ValueError("checkpoint is for a 6x6 board, position is 12x12")` and the CLI reports it like a position error (usage exit code, message on stderr). Checked at `search()`, not at load, because the engine is loaded before the position in `sandbox`. | [cli-integration.md](../design/cli-integration.md#loading-a-model) calls it an error. |
| **Where `load_engine` lives** | `cli/adapter.py`. `placeholder.py` keeps only `PlaceholderEngine`. `app.py` imports `load_engine` from the adapter. | The adapter is the real engine's module; the loader decides between the real engine's two evaluators. The placeholder module is test tooling and should not know about checkpoints. |
| **Notices** | Loaded a model: `Loaded <path> (generation N, SxS)`. No model: `No trained model found: searching with uniform priors. Scores reflect search alone, not a learned evaluation.` Both dim, on stderr, like today's placeholder notice. | The spec requires a notice for the no-model case; naming the loaded file is cheap and removes "which checkpoint did it pick?" from debugging. |

## Conventions that apply

- Test-driven: for each step write the failing tests first, run `uv run pytest tests/cli tests/training -q`, then implement until green. Tests run on CPU only; checkpoint and adapter tests use `NetworkConfig(blocks=1, filters=4, value_hidden=8)` on 3x3 or 4x4 so they take milliseconds.
- Comment thoroughly, connecting code to theory: why the chosen move is the most visited and not the highest `Q` (visit count integrates value and confidence; a rarely visited `Q` is noise), why `(q + 1) / 2` is a win probability only loosely (the network's `v` is a value estimate, not a calibrated probability), why `weights_only=True` matters, why BatchNorm running statistics are part of `model_state`, why the optimiser state is checkpointed (Adam's moment estimates; a resume without them is a cold restart).
- Coordinates `(row, col)`, zero-based, row 0 at the top. The adapter never reorders anything: `Snapshot.q` is already in board order and `Analysis.scores` must be too.
- Conventional commits, small and atomic, on `feat/cli-adapter`. Suggested sequence: `test(cli): pin the chosen-move field` / `feat(cli): add the engine's chosen move to Analysis`; `test(cli): add adapter tests` / `feat(cli): add the search adapter`; `test(training): add checkpoint tests` / `feat(training): add checkpoints`; `test(cli): pin engine loading` / `feat(cli): load checkpoints and add --device`; `feat(scripts): add make_checkpoint`; `docs: record the engine as connected`.
- **Code review cadence.** Run `/code-review` at medium effort twice: after step 3 (port change, adapter, checkpoints: the part with the subtle sign and shape logic) and after step 6 (loader, flag, script), before opening the PR. Fix what it finds before moving on; do not batch the whole branch into one review.
- No new dependencies.

## Interfaces

The contract the tests pin. Names and shapes are fixed; internals are free.

```python
# cli/engine_port.py
@dataclass(frozen=True)
class Analysis:
    value: float                   # win probability for the side to move, [0, 1]
    scores: Mapping[Cell, float]   # every legal move, board order, [0, 1]
    simulations: int
    chosen: Cell                   # the engine's move; most visited for the real engine
    def top(self, n: int) -> list[tuple[Cell, float]]   # unchanged: by score, ties in board order
    @property
    def best(self) -> tuple[Cell, float]                # (chosen, scores[chosen])

# cli/adapter.py
class SearchEngine:
    """Engine-port adapter around engine.search.analyse."""
    def __init__(self, evaluator: Evaluator, size: int | None = None,
                 config: SearchConfig = ANALYSIS, seed: int | None = None): ...
        # `size` is the board the evaluator was built for (None: any size, as for UniformEvaluator).
    def search(self, state: State, max_simulations: int | None = None) -> Iterator[Analysis]
        # Raises ValueError on a finished game (eagerly, like analyse) and on a size mismatch.
def load_engine(model: Path | None, seed: int | None, device: str = "cpu",
                root: Path = Path("checkpoints")) -> tuple[Engine, str]
    # Returns the engine and the notice to print. `model` given: load it (FileNotFoundError if missing).
    # Else latest_checkpoint(root). Else SearchEngine(UniformEvaluator()).
    # `device`: "auto" → select_device(), otherwise select_device(device).

# training/checkpoint.py
CHECKPOINT_FORMAT = 1
@dataclass(frozen=True)
class Checkpoint:
    network: Network               # rebuilt from size + network_config, weights loaded, on `device`
    size: int
    generation: int
    configs: Mapping[str, Mapping[str, Any]]   # plan 04's configs as field dicts; empty today
    optimizer_state: Mapping[str, Any] | None
    rng: Mapping[str, Any] | None
def save_checkpoint(path: Path, network: Network, *, generation: int,
                    optimizer: torch.optim.Optimizer | None = None,
                    configs: Mapping[str, Any] = {},        # dataclass instances; stored via dataclasses.asdict
                    rng: np.random.Generator | None = None) -> None
    # Creates parent directories. Writes CPU tensors regardless of the network's device.
def load_checkpoint(path: Path, device: torch.device | None = None) -> Checkpoint
    # torch.load(..., weights_only=True, map_location="cpu"), then .to(device or cpu).
    # Raises ValueError on an unknown format_version.
def latest_checkpoint(root: Path) -> Path | None
```

`load_engine` returning the notice (instead of printing it) keeps the adapter free of Rich and lets `test_app.py` assert the text without capturing a console.

## Step 1: the port change (`tests/cli/test_engine_port.py`, `tests/cli/test_placeholder.py`, then `engine_port.py` and `placeholder.py`)

Tests:

1. **`best` is the chosen move, not the top score.** An `Analysis` whose `chosen` is *not* the highest-scoring move: `best == (chosen, scores[chosen])`, while `top(1)` still returns the highest score. This is the test that documents why the field exists.
2. **Construction without `chosen` fails.** `Analysis(value, scores, simulations)` raises `TypeError`.
3. **The placeholder sets `chosen` to its top score.** For a few seeds and positions, every snapshot from `PlaceholderEngine.search` has `chosen == top(1)[0][0]`.
4. Every existing test that builds an `Analysis` by hand (8 sites across `tests/cli/`) gets a `chosen`; pick the top-scoring move so the expectations do not move.

Implementation: add the field, change `best`, set it in the placeholder. Update the docstring of `engine_port.py`: it currently says the only implementation is the placeholder.

## Step 2: the adapter (`tests/cli/test_adapter.py`, then `SearchEngine` in `adapter.py`)

Tests use `UniformEvaluator` and `TableEvaluator` from `ox_zero.engine`; no torch.

1. **Mapping by hand.** On the 3x3 win-in-one position `play([(0,0), (0,1)], size=3)`, take the first snapshot from `analyse` directly and the first `Analysis` from `SearchEngine(UniformEvaluator()).search`: `value == (snapshot.value + 1) / 2`, `scores[a] == (snapshot.q[a] + 1) / 2` for every legal move, `chosen == snapshot.best`, `simulations == 0`, and `scores[(0,2)] == 1.0` (the winning move has `Q = 1` from root expansion).
2. **Port contract.** With `max_simulations = 5`: `simulations` is `0, 1, 2, 3, 4, 5` and the iterator ends. `engine_port.analyze(engine, state, 5).simulations == 5`. Without a cap, `islice(..., 50)` yields 50 and dropping the iterator is clean. `scores` keys are exactly `legal_moves(state)` in board order on every snapshot.
3. **Finished game.** `engine.search(terminal_state)` raises `ValueError` without a `next()`.
4. **Tactics with no model.** Win in one: after 100 simulations `best[0] == (0,2)` and `value > 0.9`. Loss in one avoided: `from_board_string("____XXO_________", size=4)`, 200 simulations, `best[0] == (1,3)`. These are plan 02's fixtures re-run through the port: the point is that the CLI's no-model mode is a real engine.
5. **Size guard.** `SearchEngine(UniformEvaluator(), size=6).search(4x4 state)` raises `ValueError` mentioning both sizes; with `size=None` any board is accepted.
6. **Determinism and the seed.** Two engines with different seeds give identical snapshot sequences on the same position with `ANALYSIS`. Say in the test why (no noise, no sampling), so nobody later "fixes" the seed into doing something.

Implementation: a thin generator. Call `analyse(state, evaluator, config, max_simulations)`; map each snapshot. Keep the module docstring short and point at [cli-integration.md](../design/cli-integration.md).

## Step 3: checkpoints (`tests/training/test_checkpoint.py`, then `training/checkpoint.py`)

Create `tests/training/__init__.py` (or whatever `tests/engine/` does; mirror it). Delete `src/ox_zero/training/README.md` in the implementation commit, per its own note; `training/__init__.py` gets a docstring saying the package is being built (plan 04).

Tests (tiny network, CPU, `tmp_path`):

1. **Round trip restores the network exactly.** Save a `Network(4, tiny)` with random weights, load it: `size == 4`, `network.config == tiny`, every parameter and buffer equal (`torch.equal`), including BatchNorm's `running_mean`/`running_var`/`num_batches_tracked`, and the same input gives the same output in eval mode.
2. **Optimiser state round-trips.** Take one AdamW step so the state is non-empty, save with `optimizer=`, load, build a fresh AdamW on the loaded network and `load_state_dict(checkpoint.optimizer_state)`: no error, and `state_dict()` compares equal (tensors with `torch.equal`, the rest with `==`). Without `optimizer=`, `optimizer_state is None`.
3. **Configs round-trip as field dicts.** `configs={"search": SearchConfig(c_init=2.0)}` loads as `{"search": {"c_base": 19652.0, "c_init": 2.0, ...}}` and `SearchConfig(**checkpoint.configs["search"])` equals the original. (Plan 04 does the reconstruction for its own configs.)
4. **RNG state round-trips.** Save with `rng=default_rng(3)` after drawing a few numbers; a fresh generator with `bit_generator.state = checkpoint.rng["numpy"]` draws the same next number as the original. Torch: `torch.set_rng_state(checkpoint.rng["torch"])` then `torch.rand(1)` matches.
5. **Plain data only.** Load the file with `torch.load(path, weights_only=True)` directly: no error, and the top-level keys are exactly the set in Decisions. This is the test that stops anyone pickling a dataclass into the file.
6. **Format version.** A file with `format_version: 99` raises `ValueError` from `load_checkpoint`.
7. **`latest_checkpoint`.** In `tmp_path`: `a/gen_003.pt`, `b/gen_010.pt`, `b/gen_002.pt`, `notes.txt`, `b/model.pt` (empty files are fine; only names matter): returns `b/gen_010.pt`. Two runs both at `gen_005.pt`: the more recently modified wins (set `os.utime`). Empty directory and missing directory both return `None`.
8. **Device.** `load_checkpoint(path, torch.device("cpu")).network` has all parameters on CPU. (No `mps` test; the `map_location="cpu"` plus `.to(device)` path is the one `mps` would take.)

Implementation notes: `save_checkpoint` moves state dicts to CPU with `{k: v.detach().cpu() for ...}` so a checkpoint written on `mps` loads anywhere. `dataclasses.asdict` for every config. The loader rebuilds `Network(size, NetworkConfig(**network_config))` then `load_state_dict(strict=True)`. Comment why `strict=True` (an architecture mismatch must fail loudly, not load half a network).

**Review point:** run `/code-review` (medium) on the branch here.

## Step 4: engine loading and `--device` (`tests/cli/test_app.py`, then `load_engine` in `adapter.py` and `app.py`)

The existing `test_app.py` autouse fixture monkeypatches `load_engine` on `app_module` to return a `PlaceholderEngine`; it keeps doing that (the CLI tests are about output, not about loading) but must return the new `(engine, notice)` tuple. Loading tests opt out of the fixture.

Tests:

1. **No model, no checkpoints.** `load_engine(None, None, root=tmp_path / "none")` returns a `SearchEngine` whose evaluator is `UniformEvaluator`, and the notice contains "uniform priors". Through the CLI (`CliRunner`, with the checkpoint root patched to `tmp_path`): `ox-zero best 0,0 0,1 --simulations 50` prints `0,2`, the notice is on stderr, exit 0. The CLI is 12x12 only, so this is a 12x12 win-in-one: X at `0,0`, O at `0,1`, X completes `X O X` at `0,2`. Root expansion evaluates all 142 legal moves in one uniform batch, the terminal child backs up `Q = 1` on every visit, and 50 simulations are enough; the 12x12 CPU rate is about 280 simulations/s with a network and far more with uniform priors.
2. **`--model PATH` missing** still errors with the existing message and exit code (test exists; keep it).
3. **`--model PATH` present.** Write a tiny random 12x12 checkpoint with `save_checkpoint` to `tmp_path` (1 block, 4 filters: a 12x12 forward pass is still milliseconds), run `ox-zero analyze 5,5 6,6 5,6 --model <path> --simulations 20 --json`: valid JSON, 20 simulations, notice contains the path and `generation`. The engine's evaluator is a `NetworkEvaluator` on CPU.
4. **Newest checkpoint is picked.** Two checkpoints under `root`, generations 1 and 2 with different random weights; `load_engine(None, ...)` loads generation 2 (assert on the notice and on `checkpoint`-derived weights, e.g. compare one parameter).
5. **Size mismatch.** A 4x4 checkpoint and a 12x12 position: stderr says both sizes, usage exit code, nothing on stdout.
6. **`--device`.** `--device cpu` works; `--device auto` calls `select_device()` (monkeypatch it and assert); an invalid name is rejected by Typer (use an `Enum` or `Literal` for the option so the help text lists the choices). With no model the flag is accepted and ignored.

Implementation: `load_engine` as in Interfaces; `_load` in `app.py` prints the returned notice and gains the `device` argument; the three commands get `DeviceOpt`. `isinstance(engine, PlaceholderEngine)` disappears from `app.py`. Remove `load_engine` from `placeholder.py` and trim its docstring to "test tooling".

## Step 5: `scripts/make_checkpoint.py`

Not test-driven. Arguments: `--size 12`, `--blocks 4`, `--filters 64`, `--value-hidden 256`, `--seed 0`, `--out checkpoints_random/gen_000.pt`. (Changed during implementation from `--size 6` and `checkpoints/random/`: a file under `checkpoints/` is auto-loaded by every CLI run, and a 6x6 one would make every 12x12 command fail with the size error.) Seeds torch, builds the network, saves with `generation=0` and `configs={"search": ANALYSIS}`, prints the path. One usage line in `scripts/README.md` under a new "make_checkpoint.py" section, with the sentence that its weights are random and exist so that `--model` can be exercised before training.

Run it once, then `uv run ox-zero analyze 5,5 6,6 5,6 --model checkpoints_random/gen_000.pt --simulations 100` and the same without `--model` (uniform). Put both outputs in the PR description; the scores are meaningless but the plumbing is visible.

## Step 6: wire the sandbox, check by hand

No code expected: the sandbox takes any `Engine`. Run `uv run ox-zero sandbox` with no model, play a few moves, confirm the status line counts simulations and the position change cancels the search promptly (the generator yields per simulation; the thread worker drops the iterator). If the UI stutters, report the symptom and the simulation rate; do not add throttling to the adapter (the design puts throttling in the CLI, and the CLI already has it).

**Review point:** run `/code-review` (medium) again here, before the docs commit and the PR.

## Step 7: documentation

- `docs/cli.md`: the header note ("the engine behind it is a placeholder") becomes a note that the CLI runs the AlphaZero search, with uniform priors until a checkpoint exists. `--model` row: "With no checkpoint available, the search runs with uniform priors and a notice says so." New `--device` row (all three commands). `--seed` row: "The real engine's analysis is deterministic; the seed only affects engines that sample." `best` section: one sentence that the chosen move is the most visited, which can differ from the highest score shown by `analyze`.
- `README.md`: the status line becomes "the CLI runs on the engine; it plays on search alone until the training pipeline (step 4) produces a checkpoint".
- `docs/architecture.md`: `engine` row "Built (step 3), connected to the CLI"; `cli` row drops "on a placeholder engine"; `training` row "checkpoint format built (plan 03), the rest designed"; the two `PlaceholderEngine`/port notes under "Designed but not built" are resolved (the field exists; the placeholder is test tooling); `cli/adapter.py` appears in the module notes.
- `docs/design/cli-integration.md`: record the three spec changes and the device default under "Loading a model", dated 2026-10-09 and linking to this plan's Decisions; the "one port change" paragraph is marked done.
- `docs/design/engineering.md`: module layout gains `cli/adapter.py` (already listed) with `load_engine`, `training/checkpoint.py` marked built, `scripts/make_checkpoint.py` added. Note the format decision (plain data, `weights_only=True`) under "Configuration and checkpoints".
- `docs/plans/README.md`: nothing; plan 02 moves to `archive/` when PR #12 merges, which happens before this branch starts.

## Definition of done

- `uv run pytest` is green, including every pre-existing CLI test, and the new tests run on CPU in a few seconds.
- `uv run ox-zero best 0,0 0,1` prints `0,2` (the win in one) with no model present. That plus the small-board adapter tests in step 2 is the tactical evidence; 12x12 has no exact fixture.
- `uv run ox-zero analyze ... --model checkpoints_random/gen_000.pt` loads a checkpoint written by `scripts/make_checkpoint.py` and runs; the same command with a wrong-size position fails with the size message.
- `src/ox_zero/training/README.md` is gone; `ox_zero.engine` is untouched (`git diff main -- src/ox_zero/engine` is empty).
- Both `/code-review` runs happened and their findings were addressed or explicitly deferred in the PR description.
- A pull request from `feat/cli-adapter` to `main` with the two manual outputs from step 5. On merge, this plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- The adapter needs something `analyse` or `Snapshot` does not expose. Do not modify `ox_zero.engine` on this branch; report what is missing.
- The 12x12 CLI tests in step 4 are slow (more than a couple of seconds each). Report the timings before cutting budgets or sizes.
- `weights_only=True` rejects something the checkpoint needs (for example a NumPy RNG state nested too deep). Report it; do not switch to `weights_only=False`.
- The sandbox misbehaves with the real engine (stutter, a search that does not cancel, a snapshot arriving after a position change). Report the symptom; the fix may belong in the sandbox and that is a separate change.
- Any design-level choice not listed under Decisions. Stop and ask; do not decide it in code.
