# Plan 06: The experiment harness

| | |
|---|---|
| Status | Draft (2026-10-11), awaiting Vláďa's approval of the [decisions](#decisions-made-in-this-plan) and the [open items](#open-items-for-vláďa). |
| Branches | Plan: `plan/experiment-harness`. Implementation: `feat/experiment-harness`. |
| Design references | [training.md](../design/training.md#evaluation) (tournaments, Elo, logging), [upgrades.md](../design/upgrades.md) (the hypotheses this harness exists to test; "a measured Elo difference is the verdict"), [open-questions.md](../design/open-questions.md) (the first 6x6 run, the experiments to run, the provisional constants), [engineering.md](../design/engineering.md) (configuration, determinism, run directory), [search.md](../design/search.md#move-selection-self-play) (the temperature cutoff this plan makes a flag), [design README](../design/README.md#guiding-principles) (principle 3: small boards first, 12x12 paid for once). |
| Depends on | [Plan 04](archive/04-training-pipeline.md) merged (PR #19): `training.run`, `training.evaluate` (`play_match`, `elo_ratings`, `EVALUATION`), `training.metrics.read_metrics`, `training.checkpoint`, `scripts/train.py`, `scripts/evaluate.py`, the `runs/six-a/` checkpoints. [Plan 05](archive/05-cli-board-size.md) merged (PR #21). |

## Why this plan exists

The training pipeline is built and one 6x6 run exists. The next roadmap work is not a feature but a series of experiments: each row of [upgrades.md](../design/upgrades.md) is a hypothesis, and [open-questions.md](../design/open-questions.md#experiments-to-run) already lists the first one (is the O dominance a property of the game or of the exploration settings). The pipeline can run one experiment and show its curves. It cannot yet say which of two runs is better, for four reasons:

1. **No shared scale.** A run's Elo is anchored at its own generation 0, so two runs' ratings are not comparable. The one cross-run tool, `scripts/evaluate.py`, plays one pairing for 40 games, and 40 games at an even match have a standard error of about 55 Elo (score error `0.5 / √40 ≈ 0.08`, times the Elo slope at an even match, `400 / (ln 10 · 0.25) ≈ 695` Elo per unit of score). The effects we hunt are of that size.
2. **No noise floor.** One seed has run. Until the same configuration has run under two or three seeds, no difference between two runs can be told from luck.
3. **The first experiment cannot be launched.** `RunConfig.search` is hard-coded to `SELF_PLAY`: the temperature cutoff, `c_init`, `c_base`, Dirichlet `α` and `ε` have no flags, and tournament simulations are chained to self-play simulations.
4. **The measurement can saturate on parity.** If a board is a forced win for one colour and both players know it, every paired opening splits one game each, the score sits at 0.5 and the Elo curve goes flat whatever the strength. The first run's flat Elo after generation 5 is consistent with that as much as with a ceiling, and the final checkpoint's 33–7 against the uniform search says it is far from perfect play. A benchmark needs asymmetric opponents (the uniform search, early checkpoints) so that strength keeps showing.

This plan builds the apparatus that removes all four, then spends about six hours of laptop compute calibrating it: the noise floor, the cost per generation at 6x6 and 8x8, and, from those numbers, the board the experiments run on. "Save compute" in this project means not committing to the big 12x12 run before the small-board experiments have had their say; experiment runs of an hour or a few are accepted.

## Goal

After this plan, an experiment is: pick a flag or an upgrade toggle, run two or more arms under two or three seeds each (`scripts/train.py`), benchmark every final checkpoint against a frozen reference pool (`scripts/benchmark.py`), read one table (`scripts/compare_runs.py`), and record hypothesis, commands, numbers and verdict in `docs/experiments/`. Every rating is on one shared Elo scale per board size, anchored at the uniform search, with a bootstrap confidence interval, and the seed-to-seed spread of the baseline is known, so "arm B is 60 Elo better" has a meaning.

## Deliverables

1. **Search and evaluation flags in `scripts/train.py`**: `--temperature-cutoff`, `--c-init`, `--c-base`, `--dirichlet-alpha`, `--dirichlet-epsilon`, and `--eval-simulations` (decoupled from `--simulations`). Defaults reproduce today's `SELF_PLAY` and `EvalConfig` exactly, so `runs/six-a/` still resumes.
2. **`training/benchmark.py`**: the reference pool (checkpoints under `benchmarks/<S>x<S>/pool/` plus the uniform search at fixed simulation counts), player naming, the match log, the joint Bradley-Terry fit over every benchmark game of a board size anchored at `uniform@100`, and a bootstrap confidence interval. `elo_ratings` generalised from generation numbers to any sortable player key.
3. **`scripts/benchmark.py`**: plays one checkpoint (with an optional non-default search config) against the pool, appends to the match log, refits, prints the rating with its interval and the per-reference scores.
4. **`scripts/compare_runs.py`** and `training.metrics.summarise`: one row per run from `metrics.jsonl` (and the benchmark rating when it exists), one key over generations for several runs side by side.
5. **`docs/experiments/`**: the protocol (how an experiment is run and recorded) and one file per experiment. The first entry, `01-testbed-and-noise-floor.md`, holds this plan's calibration runs.
6. **Calibration** (step 6): `six-a` extended to 40 generations; an 8x8 baseline; two more seeds on the chosen board; the benchmark's own repeatability; the frozen reference pool for the chosen board.
7. Tests: `tests/training/test_benchmark.py`, additions to `test_scripts.py`, `test_metrics.py`, `test_evaluate.py`; a `build_config` test for `scripts/train.py`.
8. Documentation (step 7).

## Out of scope

- **Any upgrade itself.** No FPU, no MCTS-Solver, no global pooling, no `z` blending. This plan builds the scale they are measured on; each upgrade is its own experiment, and probably its own small plan when it touches `ox_zero.engine`.
- **Running the experiments on the list**, except the calibration one. The temperature-cutoff experiment becomes launchable here and runs after merge, as experiment 02.
- **Performance work.** If the calibration picks 8x8 and a generation costs too much for the cadence we want, the array tree and the evaluation cache ([upgrades.md](../design/upgrades.md#engineering)) become the next engineering plan. They change speed, not maths, and are verified by `scripts/bench_search.py` and bit-exact search results, not by an A/B. Not here.
- **Changes to the in-run tournament protocol.** The nearest-neighbour-plus-ladder tournament stays as the run's live progress monitor. Cross-run comparison is the benchmark's job.
- YAML/TOML experiment files (flags and the experiments log suffice until sweeps get wide), Weights & Biases, the Textual dashboard, the GUI, the 12x12 run.
- Exact-solver agreement as a metric (4x4 is solved, but the testbed is 6x6 or 8x8; see [engineering.md](../design/engineering.md#testing)).

## Decisions made in this plan

Code-level choices, plus three that touch the design and need Vláďa's approval (marked **needs approval**). An implementer follows this table; anything wrong in review changes here, not in code.

| Decision | Choice | Why |
|---|---|---|
| **Search flags build `RunConfig.search`** | `scripts/train.py` gains `--temperature-cutoff N` (default `None`: the per-size table), `--c-init`, `--c-base`, `--dirichlet-alpha` (default `None`: `11 / S²`), `--dirichlet-epsilon`; `build_config` passes them to `SearchConfig(root_noise=True, ...)`. Defaults equal `SELF_PLAY` field for field, `None` staying `None` (not resolved to the table's value), so an old checkpoint's stored `search` config compares equal and `six-a` resumes. | Decision 3 above. Resume already refuses a changed `search` field, so an experiment arm with a different cutoff is a different run name, as it should be. |
| **`--eval-simulations` decouples the tournament from self-play** | New flag, default `None` meaning "same as `--simulations`" (today's behaviour). `EvalConfig.simulations` takes it. | An experiment that varies self-play simulations must hold the measurement fixed, or it measures two things. |
| **The player is a (checkpoint, search) pair** | A benchmark player is named `<run>/gen_NNN` for a checkpoint played with the baseline evaluation search, `<run>/gen_NNN@<tag>` when played with a non-default search config (`--tag` is required then), and `uniform@<simulations>` for the uniform search. References always play the baseline `EVALUATION` search at the pool's simulation count. | Search upgrades change the engine, not only the network. Naming the pair lets one checkpoint be rated with and without an upgrade, and keeps the references fixed while candidates vary. |
| **Reference pool per board size** (**needs approval**: a new top-level directory) | `benchmarks/<S>x<S>/pool/*.pt`: copies of frozen checkpoints named `<run>-gen_NNN.pt`, gitignored (`*.pt` already is). Always also `uniform@100` and `uniform@400`. `benchmarks/README.md` is the manifest: each reference's origin (run, seed, generation, command) and the date the pool was frozen. The 6x6 pool is seeded from `six-a` generations 5, 10 and 20; the 8x8 pool from the calibration baseline (step 6). A pool is frozen once and only ever extended, never replaced: changing a reference changes the scale. | Uniform-search players need no file and exist on every machine; they anchor the scale and cover the weak end. Early-generation checkpoints spread the pool across strengths so that a strong candidate still loses some games and the fit stays informative (decision 4 above). Three checkpoints plus two uniform players are five references: enough to pin a rating, few enough to run in minutes. |
| **One match log per board size, committed** (**needs approval**: generated data in git) | `benchmarks/<S>x<S>/matches.jsonl`, one row per (candidate, reference) pairing: `{"a": name, "b": name, "wins", "draws", "losses", "as_x": [w, d, l], "as_o": [w, d, l], "openings", "simulations", "seed", "date", "a_sha256", "b_sha256"}`. `ratings.json` is derived from it (every player's rating, interval and game count) and rewritten on each benchmark. Both committed. | The log is the ground truth of every experiment's rating; the checkpoints it names are not shareable (gitignored, unreproducible across machines), the log is. A joint fit over the whole log means every new candidate's games also sharpen the references' ratings. The hashes catch a swapped pool file: a reference whose hash changed is refused. |
| **Anchor and scale** | Bradley-Terry over every row of the board's log, `uniform@100` fixed at 0 Elo. `elo_ratings` is generalised to any sortable key (`int` for the run's tournament, `str` here); the existing tests keep passing unchanged. One virtual draw per pair, as today. | The uniform search at 100 simulations is the one player every machine can reproduce exactly. Generation 0 of a run cannot anchor a cross-run scale: every run has a different one. |
| **Bootstrap interval, not an analytic error** | `bootstrap_interval(rows, player, resamples=200, rng)`: resample games within each pairing (a binomial-multinomial draw of the pairing's W/D/L counts), refit, take the 2.5th and 97.5th percentiles of the player's rating. Printed as `rating [low, high]`. | The joint fit has no closed-form error with the virtual draw and a shared anchor; the bootstrap is twenty lines and honest. It measures game-sampling noise only. Seed-to-seed training noise is the other, larger component and is measured in step 6, not computed. |
| **Benchmark game count** | `--games 50` openings per reference, each played with both colours: 100 games per reference, 500 per benchmark. | 100 games per pairing is about ±35 Elo on that link; five links pooled lands near ±20 Elo (the bootstrap reports the real figure). At the measured 0.3 s per 6x6 tournament game at 100 simulations (`six-a` generations 17–20), 400 games against the 100-simulation references cost about 2 minutes and 100 games against `uniform@400` about 2 more. 8x8 is expected to be a few times that. Cheap next to a run. |
| **Repeatable openings** | The benchmark draws its openings from `--seed` (default 0) through the same `_openings` as the tournament, one generator per pairing seeded by `(seed, reference index)`. Two candidates benchmarked with the same seed face the same openings against each reference. | Paired openings across candidates remove one source of noise from an A/B. Different seeds measure the benchmark's own repeatability (step 6). |
| **Candidate lockstep** | `scripts/benchmark.py` plays all five references' games in sequence, each pairing as one `play_match` (lockstep inside the pairing, as the tournament does). | Reuses `play_match` unchanged. Pairing-level lockstep already keeps the network's batches at 100 positions; merging pairings would need a new driver for no measurable gain at this scale. |
| **`compare_runs.py` reads files, computes nothing new** | `summarise(rows)` in `training/metrics.py` reduces a run's metrics rows to the final generation, final Elo, losses, mean game length, X win and draw rates, mean wall seconds per generation over the last five, and the mean self-play simulations per second. The script prints one row per run (Markdown table), merges `benchmarks/<S>x<S>/ratings.json` by player name when the run's final checkpoint is rated, and with `--key loss_value` prints that key per generation with one column per run. `--at N` summarises at generation `N` instead of the last. | A table that can be pasted into an experiment file as is. Design: dashboards are readers of the metrics files ([training.md](../design/training.md#logging)). |
| **`docs/experiments/` is a third kind of document** (**needs approval**: new docs directory) | `docs/experiments/README.md` holds the protocol and the index; `NN-slug.md` holds one experiment: hypothesis, what changes (flags or diff), arms and seeds, exact commands, cost, results (the `compare_runs.py` table plus benchmark ratings with intervals), verdict (adopt / reject / inconclusive), and what changed in the design docs because of it. Design docs record decisions and link to the experiment that made them; plans record how code was built; experiments record measurements. The 6x6 run's section in [open-questions.md](../design/open-questions.md#results-2026-10-10-the-first-6x6-run) stays where it is (it predates the convention) and gains a pointer. | Pasting results into open-questions.md worked for one run; a dozen experiments would bury the decisions in tables. The author's own convention separates design ("settings") from plans; experiments are neither. |
| **Testbed criteria** | The board the experiments run on is chosen in step 6 by three criteria, recorded in experiment 01: (a) *headroom*: the baseline's benchmark rating keeps rising between generation 20 and 40 by more than the seed spread; (b) *cadence*: a two-arm, three-seed experiment finishes overnight, i.e. one baseline run of the chosen length takes at most about 90 minutes; (c) *measurability*: paired-colour scores against the near-strength references are not pinned at 0.5 (per-colour results show both colours winning games). 8x8 is the expected answer if it passes (b); 6x6 is the fallback. | Vláďa's hunch (2026-10-11) is that 6x6 is too easy for upgrades to show; the data in [open-questions.md](../design/open-questions.md#results-2026-10-10-the-first-6x6-run) is ambiguous between "saturated" and "parity-pinned". Two runs settle it. Criterion (b) comes from accepting hours-long experiments but wanting them to turn around daily. |
| **8x8 calibration constants** | The provisional 6x6 constants with the board size changed: `--size 8 --generations 30 --seed 0`, everything else default (the cutoff table gives 4 moves, `α = 11/64`). `--device cpu` unless a one-generation `mps` probe is faster (as in plan 04, step 8). | Changing one thing at a time: the first 8x8 run measures cost and learnability under the known configuration. Tuning 8x8's constants is a later experiment. Thirty generations because 8x8 is expected to learn more slowly and the headroom criterion needs a curve, not a point. |

## Conventions that apply

- Test-driven for package code: failing tests first, `uv run pytest tests/training -q`, then implement. Scripts are exempt but seeded and self-describing, and their flag validation is tested in `tests/training/test_scripts.py` as today. CPU only in tests; tiny networks (`NetworkConfig(blocks=1, filters=4, value_hidden=8)`) on 3x3 or 4x4.
- `training/benchmark.py` is torch-free except for loading checkpoints, which is isolated in one function (`load_pool`) so the fit, naming and log code import without torch, like `evaluate.py`.
- Comment thoroughly, connecting code to theory ([project `CLAUDE.md`](../../CLAUDE.md)): why the anchor must be a player every machine can reproduce; why a joint fit beats per-pairing Elo differences (every game informs every rating through the graph); what the bootstrap resamples and what it does not (training noise); why references are played with the baseline search even when the candidate is not; why a flat paired-colour score can mean parity rather than equal strength; why the standard error of a 40-game match is about 55 Elo.
- Coordinates `(row, col)` zero-based; values from the side to move; policies flat in board order. Unchanged.
- Conventional commits on `feat/experiment-harness`, suggested sequence: `test(scripts): pin build_config defaults` / `feat(scripts): add search and eval-simulations flags`; `test(training): generalise elo_ratings keys` / `refactor(training): rate any sortable player key`; `test(training): add benchmark pool, log and bootstrap tests` / `feat(training): add the benchmark module`; `feat(scripts): add benchmark.py`; `test(training): add metrics summarise tests` / `feat(scripts): add compare_runs.py`; `docs(experiments): add the protocol and experiment 01`; `docs: record the experiment harness as built`.
- **Code review cadence.** `/code-review` at medium effort after step 3 (flags, fit, benchmark module and script) and after step 5 (compare script and the experiments protocol), before the calibration runs. Fix what it finds first.
- No new dependencies.

## Interfaces

The contract the tests pin. Names and shapes fixed, internals free.

```python
# scripts/train.py
def build_config(args: argparse.Namespace, name: str) -> RunConfig
    # New fields read: temperature_cutoff, c_init, c_base, dirichlet_alpha, dirichlet_epsilon,
    # eval_simulations. With every new flag at its default, the result equals the plan-04
    # configuration field for field (test: compare with RunConfig(..., search=SELF_PLAY,
    # eval=EvalConfig(simulations=args.simulations))).

# training/evaluate.py
K = TypeVar("K", int, str)                       # a sortable player key
def elo_ratings(matches: Sequence[tuple[K, K, int, int, int]], anchor: K) -> dict[K, float]
    # Unchanged semantics; `anchor` keeps its default of 0 for int keys via an overload or
    # a keyword default, so run.py and the existing tests do not change.

# training/benchmark.py                                   (torch only inside load_pool)
UNIFORM_SIMULATIONS = (100, 400)
ANCHOR = "uniform@100"
POOL_DIR = "pool"; MATCHES_FILE = "matches.jsonl"; RATINGS_FILE = "ratings.json"

def uniform_name(simulations: int) -> str              # "uniform@100"
def candidate_name(run: str, generation: int, tag: str | None) -> str
    # "six-a/gen_020", "six-a/gen_020@fpu"; `run` is the checkpoint's parent directory name
def benchmark_dir(root: Path, size: int) -> Path        # root / "6x6"

@dataclass(frozen=True)
class Reference:
    name: str
    path: Path | None            # None for uniform players
    simulations: int
    sha256: str | None

def list_pool(directory: Path) -> list[Reference]
    # The uniform references for UNIFORM_SIMULATIONS (simulations as given) followed by every
    # `<run>-gen_NNN.pt` in `directory / POOL_DIR`, sorted by name, each with its sha256.
def load_pool(references: Sequence[Reference], size: int, device) -> dict[str, Evaluator]
    # NetworkEvaluator per checkpoint (size mismatch is a ValueError naming the file),
    # UniformEvaluator for the uniform names.

@dataclass(frozen=True)
class BenchmarkRow:              # one pairing, from the candidate's perspective
    a: str; b: str
    wins: int; draws: int; losses: int
    as_x: tuple[int, int, int]; as_o: tuple[int, int, int]
    openings: int; simulations: int; seed: int; date: str
    a_sha256: str; b_sha256: str | None
    def to_json(self) -> dict[str, Any]
    @staticmethod
    def from_json(row: Mapping[str, Any]) -> BenchmarkRow

def append_rows(path: Path, rows: Sequence[BenchmarkRow]) -> None          # JSON Lines append
def read_rows(path: Path) -> list[BenchmarkRow]                           # [] when missing
def check_pool_hashes(rows: Sequence[BenchmarkRow], references: Sequence[Reference]) -> None
    # ValueError naming the reference whose sha256 in the log differs from the file's.
def as_matches(rows: Sequence[BenchmarkRow]) -> list[tuple[str, str, int, int, int]]
def rating_table(rows: Sequence[BenchmarkRow], anchor: str = ANCHOR) -> dict[str, float]
def bootstrap_interval(rows: Sequence[BenchmarkRow], player: str, rng: np.random.Generator,
                       resamples: int = 200, anchor: str = ANCHOR) -> tuple[float, float]
    # Resample each pairing's (wins, draws, losses) as a multinomial draw of its game count
    # with the observed proportions, refit, return the 2.5th and 97.5th percentiles.
def write_ratings(path: Path, rows: Sequence[BenchmarkRow], rng) -> dict[str, Any]
    # {"anchor": ANCHOR, "players": {name: {"elo", "low", "high", "games"}}}, rewritten whole.

# training/metrics.py
def summarise(rows: Sequence[Mapping[str, Any]], at: int | None = None) -> dict[str, Any]
    # Keys: generation, elo, loss_total, loss_policy, loss_value, mean_game_length,
    # x_win_rate, draw_rate, wall_seconds_mean5 (mean over the last five rows up to `at`),
    # simulations_per_second_mean. `at` picks the row with that generation (ValueError if
    # absent); None takes the last row. Empty rows: ValueError.
```

Script flags:

```
scripts/train.py        + --temperature-cutoff N  --c-init F  --c-base F  --dirichlet-alpha F
                          --dirichlet-epsilon F  --eval-simulations N
scripts/benchmark.py      --model PATH  [--root benchmarks]  [--games 50]  [--seed 0]
                          [--device cpu]  [--tag NAME  + the search flags above for the candidate]
scripts/compare_runs.py   RUN_DIR...  [--at N]  [--key NAME]  [--benchmarks benchmarks]
```

`benchmark.py` reads the board size from the checkpoint, lists the pool for that size, refuses a missing pool directory with a message that names the manifest, checks the hashes, plays the pairings, appends the rows, rewrites `ratings.json`, and prints: the candidate's name, rating and interval; one line per reference with the score and per-colour W/D/L; the pool's references and their current ratings. Negative or zero `--games`, a `--tag` without any search flag, or search flags without `--tag`, are usage errors (status 2).

## Step 1: flags (`tests/training/test_scripts.py`, a new `tests/scripts/test_train_config.py`, then `scripts/train.py`)

Tests first:

1. `build_config` with the new flags at their defaults equals the plan-04 `RunConfig` for the same other flags (`search == SELF_PLAY`, `eval.simulations == args.simulations`). This is the test that keeps `six-a` resumable.
2. Each new flag reaches its field: `--temperature-cutoff 5` gives `search.temperature_cutoff == 5`; `--eval-simulations 400` gives `eval.simulations == 400` with `selfplay.simulations` untouched.
3. Usage errors: `--temperature-cutoff -1`, `--dirichlet-epsilon 1.5`, `--eval-simulations 0` exit 2 without a traceback (the subprocess pattern of `test_scripts.py`). Add the range checks `SearchConfig` lacks (`0 ≤ ε ≤ 1`, `α > 0`, `cutoff ≥ 0`, `c_base > 0`) in `SearchConfig.__post_init__` with their own unit tests in `tests/engine/test_search.py`, so bad values are refused wherever a config is built, not only by the script.

Then implement. `scripts/train.py` imports `SearchConfig` instead of `SELF_PLAY`. Smoke check by hand: `uv run python scripts/train.py --name six-a --size 6 --generations 20 --seed 0 --device cpu` must print "Resuming" and exit at once with the run already at generation 20 (no config mismatch).

## Step 2: the fit over any key (`tests/training/test_evaluate.py`, then `evaluate.py`)

Tests: `elo_ratings` with string players and `anchor="uniform@100"` gives the same numbers as the int version under a relabelling; the existing int tests pass unchanged (default anchor 0). Implement by typing the key generically and sorting pairs with `sorted((player, opponent))` instead of `min`/`max` (identical for ints).

## Step 3: the benchmark (`tests/training/test_benchmark.py`, then `training/benchmark.py`, `scripts/benchmark.py`)

Tests, torch-free except the last two:

1. Naming: `uniform_name`, `candidate_name` with and without a tag, `benchmark_dir`.
2. `list_pool` on a temporary directory with two fake `.pt` files (any bytes) and one stray file: the two uniform references first, then the two checkpoints sorted, with correct hashes; the stray file is ignored.
3. `BenchmarkRow` JSON round trip; `append_rows` then `read_rows` returns the same rows; `read_rows` on a missing file is `[]`.
4. `check_pool_hashes`: a log row whose `b_sha256` differs from the pool's file raises naming the reference; `None` hashes (uniform) are never compared.
5. `rating_table` on hand-built rows: the anchor is 0; a candidate that beats `uniform@100` 75–25 and loses to `uniform@400` 25–75 lands between them; the result is invariant to the row order.
6. `bootstrap_interval`: contains the point estimate; narrows when every row's game count is multiplied by 4 (compare widths); a fixed `rng` makes it deterministic.
7. `write_ratings` writes every player with `elo`, `low`, `high`, `games`, and the anchor's `elo` is exactly 0.
8. `load_pool` (torch): a 3x3 checkpoint written with `save_checkpoint` loads into a `NetworkEvaluator`; a 4x4 checkpoint in a 3x3 pool raises naming the file; uniform names give `UniformEvaluator`.
9. End to end in `test_scripts.py` (torch, a few seconds): a 3x3 pool with one tiny checkpoint, `scripts/benchmark.py --model <another tiny 3x3 checkpoint> --games 1 --root <tmp>` exits 0, appends rows for every reference, writes `ratings.json` with the candidate present; a second invocation with the same flags appends again (the log is append-only; the fit uses both). Flag errors: `--games 0`, `--tag` without a search flag, a search flag without `--tag`, a missing pool directory (status 2, message names `benchmarks/README.md`).

Then implement the module, then the script. The script's search flags reuse one helper shared with `train.py` (`add_search_flags(parser)` / `search_from_args(args, root_noise)` in a new `scripts/_flags.py`, or duplicated if a shared module between scripts proves awkward; the implementer decides and says which).

**Review point:** `/code-review` (medium).

## Step 4: `compare_runs.py` (`tests/training/test_metrics.py`, then `metrics.py`, `scripts/compare_runs.py`)

Tests: `summarise` on three hand-built rows picks the last by default, honours `--at`, computes the five-row wall mean over fewer rows when fewer exist, raises on an empty list and on an absent generation. Script: in `test_scripts.py`, two temporary run directories with hand-written `metrics.jsonl` print a two-row Markdown table; `--key loss_value` prints one column per run; a directory without `metrics.jsonl` is a usage error.

## Step 5: the experiments log (`docs/experiments/README.md`, `docs/experiments/01-testbed-and-noise-floor.md`)

`README.md`: what an experiment file contains (the sections listed under Decisions), the rules (one hypothesis per experiment; arms differ in one thing; at least two seeds per arm, three when the first two disagree with the noise floor; the benchmark is the verdict, the in-run curves are the explanation; a verdict changes a design doc, which links back), the commands template, and the index table (number, title, verdict, date). `01-testbed-and-noise-floor.md`: hypothesis, arms and commands written now; results and verdict filled in step 6.

**Review point:** `/code-review` (medium) on the branch, before the compute.

## Step 6: calibration

Compute, in this order, each result recorded in experiment 01 as it lands. Times are estimates from `six-a` (26 minutes for 20 generations on CPU; the tournament is about 50 seconds per generation from generation 8 on) and the expected 2.5× per-generation cost of 8x8.

1. **Seed the 6x6 pool.** Copy `runs/six-a/gen_005.pt`, `gen_010.pt`, `gen_020.pt` to `benchmarks/6x6/pool/six-a-gen_005.pt` etc.; write their origin in `benchmarks/README.md`. Benchmark each (`scripts/benchmark.py --model`), which rates the pool against itself and the uniform players. About 15 minutes.
2. **Benchmark repeatability.** Benchmark `six-a/gen_020` again with `--seed 1`. The difference between the two ratings, against the printed intervals, is the measurement noise. About 5 minutes.
3. **6x6 headroom.** `scripts/train.py --name six-a --size 6 --generations 40 --seed 0 --device cpu` (resume). Benchmark `gen_030` and `gen_040`. About 40 minutes. Criterion (a) needs the seed spread from item 5, so the verdict waits.
4. **8x8 baseline.** One-generation `cpu` and `mps` probes (`--generations 1`, two names), then `scripts/train.py --name eight-a --size 8 --generations 30 --seed 0 --device <faster>`. Record the per-generation wall time and the throughput rows for [engineering.md](../design/engineering.md#measured-2026-10-10-self-play-throughput). About 1.5 hours, possibly more. Seed the 8x8 pool from `eight-a` generations 5, 15 and 30 (adjust if the curve says so, and say why) and benchmark them.
5. **Noise floor on the candidate board.** If 8x8 passes criterion (b) by its measured wall time, run `eight-b` (`--seed 1`) and `eight-c` (`--seed 2`) to 30 generations and benchmark their final checkpoints; otherwise run `six-b` and `six-c` to 40 generations and benchmark `gen_020` and `gen_040` of each. About 3 hours for 8x8, 1.5 hours for 6x6. The seed spread is the range of the three final ratings.
6. **Verdict.** Apply the three criteria, in the experiment file, with the numbers. The chosen board's pool is declared frozen in `benchmarks/README.md`. If 8x8 fails (b) by a modest factor, record the cost and say what speed-up would fix it rather than picking 6x6 by default: that is a decision for Vláďa.
7. **`compare_runs.py`** over every run above, pasted into the experiment file, is the results table.

If any run shows a draw rate above about 20% or any other threshold named in the design, record it; do not act on it.

## Step 7: documentation

- `scripts/README.md`: sections for `benchmark.py` and `compare_runs.py`; `train.py` lists the new flags; `evaluate.py` says when to use it (one ad-hoc pairing) and when the benchmark (a rating).
- `benchmarks/README.md`: the manifest format, the frozen pools, how to add a reference (never replace), the GitHub-release note for sharing pool checkpoints.
- `docs/experiments/README.md` and `01-testbed-and-noise-floor.md`: complete.
- `docs/design/training.md`, "Evaluation": a third item, the cross-run benchmark (shared scale per size, anchor, pool, interval), with "monitoring" versus "verdict" made explicit; link the experiments directory.
- `docs/design/engineering.md`: module layout gains `training/benchmark.py`, `scripts/benchmark.py`, `scripts/compare_runs.py`, `benchmarks/`; "Measured" gains the 8x8 throughput row and the 8x8 per-generation cost; "Configuration" notes that the search constants are flags now.
- `docs/design/open-questions.md`: the 6x6 results section gains a pointer to experiment 01; the "Experiments to run" table gains a column or note "runs as experiment NN" and the temperature-cutoff row becomes experiment 02 (to be written after merge); the testbed decision is recorded under Decisions with its date.
- `docs/design/upgrades.md`: the preamble says the verdict is the benchmark rating on the shared scale, linking training.md and the experiments directory.
- `docs/design/README.md` Files table and the root `README.md` Documentation list: add `docs/experiments/`. `docs/architecture.md`: `scripts/` sentence names the two new scripts; the documentation paragraph names the experiments log.
- `.gitignore`: `benchmarks/*/pool/` (belt and braces next to `*.pt`).
- The memory note for Claude sessions ("design vs plans") would gain the third kind; that is Vláďa's to approve separately, not part of this branch.

## Definition of done

- `uv run pytest` green; the new tests run on CPU in a few seconds.
- `scripts/train.py` with the new flags at their defaults resumes `runs/six-a/` without a config mismatch.
- `benchmarks/6x6/` and the chosen board's pool exist, are frozen in the manifest, and `ratings.json` rates every calibration checkpoint with intervals.
- Experiment 01 is complete: the noise floor, the 8x8 cost, the three criteria applied, and the testbed decision, or an explicit hand-back to Vláďa if 8x8 fails criterion (b).
- Both `/code-review` rounds happened and their findings were addressed or deferred with a reason in the PR.
- A pull request from `feat/experiment-harness` to `main`; on merge this plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- `six-a` fails to resume after step 1 with a config mismatch. Report the differing field; do not change the stored checkpoint or loosen `_check_config`.
- The 8x8 run shows no learning at all (Elo flat from generation 1, losses flat). Report the curves before changing any 8x8 constant: the plan's calibration is "the 6x6 constants on 8x8", and a tuned 8x8 is a different experiment.
- An 8x8 generation takes more than about 10 minutes. Report the per-phase seconds; the decision between a speed-up plan and 6x6 is Vláďa's.
- The bootstrap intervals are wider than the seed spread, or the two repeatability benchmarks differ by more than their intervals suggest. Report; it changes `--games`, which is a decision in this table.
- Any design-level choice not in the Decisions table.

## Open items for Vláďa

Approvals this plan needs before implementation starts:

1. A new top-level `benchmarks/` directory with a committed match log (generated data in git) and gitignored pool checkpoints.
2. A new `docs/experiments/` directory as the third kind of document, alongside design and plans.
3. The 8x8 calibration length (30 generations) and the "about 90 minutes per baseline" cadence criterion.
4. Whether the first upgrade experiment after merge is the temperature-cutoff one from open-questions.md (the plan assumes yes and numbers it 02).
