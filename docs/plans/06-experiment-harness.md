# Plan 06: The experiment harness

| | |
|---|---|
| Status | Draft (2026-10-11). The design-level decisions were approved by Vláďa on 2026-10-11 ([Approvals](#approvals-2026-10-11)); the review loop is in progress. |
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
2. **`training/benchmark.py`**: the reference pool (checkpoints under `runs/benchmarks/<S>x<S>/pool/` plus the uniform search at fixed simulation counts), player naming, the match log, the joint Bradley-Terry fit over every benchmark game of a board size anchored at `uniform@100`, and a bootstrap confidence interval. `elo_ratings` generalised from generation numbers to any sortable player key.
3. **`scripts/benchmark.py`**: plays one checkpoint (with an optional non-default search config) against the pool, appends to the match log, refits, prints the rating with its interval and the per-reference scores.
4. **`scripts/compare_runs.py`** and `training.metrics.summarise`: one row per run from `metrics.jsonl` (and the benchmark rating when it exists), one key over generations for several runs side by side.
5. **`docs/experiments/`**: the protocol (how an experiment is run and recorded) and one file per experiment. The first entry, `01-testbed-and-noise-floor.md`, holds this plan's calibration runs.
6. **Calibration** (step 6): `six-a` extended to 40 generations; an 8x8 baseline; two more seeds on the chosen board; the benchmark's own repeatability; the frozen reference pool for the chosen board.
7. Tests: `tests/training/test_benchmark.py`, additions to `test_scripts.py`, `test_metrics.py`, `test_evaluate.py`; a `build_config` test for `scripts/train.py`.
8. Documentation (step 7).

## Out of scope

- **Any upgrade itself.** No FPU, no MCTS-Solver, no global pooling, no `z` blending. This plan builds the scale they are measured on; each upgrade is its own experiment, and probably its own small plan when it touches `ox_zero.engine`.
- **Running any experiment** other than the calibration one, and **deciding which experiment comes first**. This plan builds the tools; what to test with them is Vláďa's call, made outside it.
- **Performance work.** If the calibration picks 8x8 and a generation costs too much for the cadence we want, the array tree and the evaluation cache ([upgrades.md](../design/upgrades.md#engineering)) become the next engineering plan. They change speed, not maths, and are verified by `scripts/bench_search.py` and bit-exact search results, not by an A/B. Not here.
- **Changes to the in-run tournament protocol.** The nearest-neighbour-plus-ladder tournament stays as the run's live progress monitor. Cross-run comparison is the benchmark's job.
- YAML/TOML experiment files (flags and the experiments log suffice until sweeps get wide), Weights & Biases, the Textual dashboard, the GUI, the 12x12 run.
- Exact-solver agreement as a metric (4x4 is solved, but the testbed is 6x6 or 8x8; see [engineering.md](../design/engineering.md#testing)).

## Decisions made in this plan

Code-level choices, plus three that touch the design and were approved by Vláďa on 2026-10-11 (marked **approved**; the record is in [Approvals](#approvals-2026-10-11)). An implementer follows this table; anything wrong in review changes here, not in code.

| Decision | Choice | Why |
|---|---|---|
| **Search flags build `RunConfig.search`** | `scripts/train.py` gains `--temperature-cutoff N`, `--c-init`, `--c-base`, `--dirichlet-alpha`, `--dirichlet-epsilon`, every one with argparse default `None` meaning "leave the `SearchConfig` field at its dataclass default" (`build_config` passes only the flags that were given, as keyword arguments to `SearchConfig(root_noise=True, **given)`). No numeric literal is repeated in the script, so the defaults cannot drift from `SELF_PLAY`: with no flags the result equals it field for field, `None` fields staying `None` (not resolved to the table's or the `11 / S²` value), and an old checkpoint's stored `search` config compares equal, so `six-a` resumes. `--help` shows the current dataclass value in each flag's help text. | Decision 3 above. Resume already refuses a changed `search` field, so an experiment arm with a different cutoff is a different run name, as it should be. |
| **`--eval-simulations` decouples the tournament from self-play** | New flag, default `None` meaning "same as `--simulations`" (today's behaviour). `EvalConfig.simulations` takes it. | An experiment that varies self-play simulations must hold the measurement fixed, or it measures two things. |
| **The player is a (checkpoint, search) pair** | A benchmark player is named `<run>/gen_NNN` for a checkpoint played with the baseline evaluation search, `<run>/gen_NNN@<tag>` when played with a non-default search config (`--tag` is required then), and `uniform@<simulations>` for the uniform search. The candidate always plays `CANDIDATE_SIMULATIONS = 100` per move; references always play the baseline `EVALUATION` search at their own simulation count (100 for pool checkpoints, `N` for `uniform@N`). The only candidate search flags the benchmark accepts are the ones the evaluation search consumes: `--c-init`, `--c-base`, and later each upgrade's toggle. Dirichlet `α`/`ε` do nothing without root noise and a temperature cutoff above 0 would sample moves, the thing evaluation turns off, so they are not offered. Identity has two layers. The *base name* (the name without its `@tag`) and the sha256 are bound one to one: a base name under two hashes, or one hash under two base names, is refused. The *full name* is bound to its `a_search`: every row stores the fields in which the candidate's search differs from `EVALUATION` (not from `SearchConfig()`, whose `temperature_cutoff` is `None` rather than 0), as a dict, `{}` for the baseline; a full name under two different dicts is refused, and a `--tag` whose dict would be empty (the flags given equal the defaults) is a usage error, because it would create a second player for the same engine whose only extra pairing, against its own untagged entry, is a disguised self-match. So `six-a/gen_020` and `six-a/gen_020@fpu` share a hash by design and are two players. | Search upgrades change the engine, not only the network. Naming the pair lets one checkpoint be rated with and without an upgrade, and keeps the references fixed while candidates vary. Without the identity checks, a retrained run with the same name, or two `--c-init` values under one tag, would silently merge into one Bradley-Terry player and nothing in the log could tell. |
| **`play_match` gains per-side search settings** | `play_match(a, b, size, games_per_colour, simulations, rng, search=EVALUATION, *, b_simulations=None, b_search=None)`: the two new keyword arguments default to `a`'s values, so `run.py` and `scripts/evaluate.py` are unchanged; both budgets are range-checked (`≥ 1`), not only `a`'s. `MatchGame.__init__` keeps its current signature (`simulations` and `search` are `a`'s) and gains the same two keyword arguments, so the existing tests that construct it directly and the `RecordingMatchGame` subclass in `test_evaluate.py` do not change; inside, it keeps one `(simulations, search)` per side, keyed by `Mark` like its trees, and every place that builds a tree uses the owner's config: the lazy creation in the `tree` property on a player's first turn, and the rebuild guard in `play_if_ready` (the `SearchTree(self.state, self.search, self.rng)` fallback for an unexpanded root). `play_if_ready` uses the mover's budget. | Today both sides share one `simulations` and one `search`. A `uniform@400` reference against a 100-simulation candidate, or a candidate with a non-default search against a baseline reference, needs them per side. The lazy creation and the guard are the two places where the mover's config could leak into the opponent's tree. |
| **Reference pool per board size** (**approved**: lives under `runs/`) | `runs/benchmarks/<S>x<S>/pool/*.pt`: copies of frozen checkpoints named `<run>-gen_NNN.pt`, under `runs/` with every other training artefact, so gitignored. A pool file's player name is its stem with the last `-gen_` turned into `/gen_`: `six-a-gen_005.pt` is the player `six-a/gen_005`, the same name the checkpoint has as a candidate, so one network is one player wherever it comes from. Always also `uniform@100` and `uniform@400`. `docs/experiments/pools.md` is the committed manifest: per board size, each reference's file name, origin (run, seed, generation, command), sha256 and the date the pool was frozen, so a deleted pool can be rebuilt from the same runs and a rebuilt one checked against the recorded hashes. The pool never holds a generation the calibration criteria rate: the 6x6 pool is `six-a` generations 5, 10 and 15 (generations 20 and 40 are measured), the 8x8 pool `eight-a` generations 5, 10 and 20 (15 and 30 are measured). A pool is frozen once and only ever extended, never replaced: changing a reference changes the scale. A candidate skips the reference with its own *full name* (a self-match, which the fit rejects); a tagged candidate does play its untagged pool entry, which is the one pairing that isolates the search change. So benchmarking a pool member rates it against the rest of the pool. | The pool is machine state, like a run, and the scale it defines lives and dies with it, so it sits with the runs rather than in a directory of its own; the findings it produces go to `docs/experiments/`. Uniform-search players need no file and exist on every machine; they anchor the scale and cover the weak end. Early-generation checkpoints spread the pool across strengths so that a strong candidate still loses some games and the fit stays informative (decision 4 above). Three checkpoints plus two uniform players are five references: enough to pin a rating, few enough to run in minutes. |
| **One match log per board size; the ratings file derived** (**approved**: not committed) | `runs/benchmarks/<S>x<S>/matches.jsonl`, one row per (candidate, reference) pairing: `{"a": name, "b": name, "wins", "draws", "losses", "as_x": [w, d, l], "as_o": [w, d, l], "openings", "a_simulations", "b_simulations", "a_search", "device", "date", "a_sha256", "b_sha256"}`. Append-only, gitignored with the rest of `runs/`. Two consistency checks on append, each a refusal naming the offender: the identity rules above, and a pairing already played, where a pairing is the *unordered* pair of full names. A pairing is one measurement: the openings are always every cell of the board (next row), the evaluation search is deterministic, and the log records each pair once, whichever side is called `a`. `ratings.json` (every player's rating, interval and game count) is derived from the log, rewritten by the benchmark, read by `compare_runs.py` (which refits from the log itself when the file is missing, and leaves the rating column blank when the log is missing or empty too, since `elo_ratings` raises on an anchor with no games), and gitignored. `write_ratings` on an empty log writes the anchor alone at 0 with no interval. Reading and writing reuse the run's helpers: `training.metrics.read_metrics` (JSON Lines with partial-last-line repair) and `training.atomic.atomic_write_text`. | The log is the ground truth of every rating, and a joint fit over all of it means every new candidate's games also sharpen the references' ratings. It is not committed: without the pool checkpoints, which cannot be, a log is a record of games nobody can replay, so the two stay together and are shared together (a GitHub release) if ever. What an experiment concluded is recorded in its `docs/experiments/` file with the numbers that do not change (per-reference scores) and the log size they came from. The hashes catch a swapped pool file: a reference whose hash changed is refused. The device is recorded because float32 results differ between `cpu` and `mps`; the official log is CPU-only (the script refuses another device unless `--root` is not the default), so a pairing has one reproducible answer. |
| **Anchor and scale** | Bradley-Terry over every row of the board's log, `uniform@100` fixed at 0 Elo. `elo_ratings` is generalised to any sortable key (`int` for the run's tournament, `str` here); the existing tests keep passing unchanged. One virtual draw per pair, as today. | The uniform search at 100 simulations is the one player every machine can reproduce exactly. Generation 0 of a run cannot anchor a cross-run scale: every run has a different one. |
| **Bootstrap interval, not an analytic error** | `bootstrap_intervals(rows, resamples=1000, seed=0)`: per replicate, each pairing's `n` games are redrawn as a multinomial sample with the W/D/L proportions *after* the virtual draw, `(w, d + 1, l) / (n + 1)`, and the replicate is fit by the ordinary `elo_ratings` (which adds its own virtual draw, so every replicate rating is finite); the 2.5th and 97.5th percentiles of each player's replicate ratings are its 95% interval. The generator is seeded inside from `seed`, so the same log gives the same interval on every machine. Printed as `rating [low, high]`; a pairing swept in the log is also printed as `≥` / `≤` next to its score, because its true gap is only bounded. | The joint fit has no closed-form error with the virtual draw and a shared anchor; the bootstrap is twenty lines and honest. Resampling the raw proportions would give a sweep zero variance, collapsing the interval exactly where the rating is least determined; the smoothed proportions let a swept pairing occasionally produce a draw and keep the interval open. The replicate is deliberately fit *with* the virtual draw: a replicate that happens to be a clean sweep (about a third of the replicates of a swept 72-game pairing) has no finite unregularised maximum, and a fit without the draw would run away until it overflows. The price is that the draw is applied twice to a replicate, once in the proportions and once in the fit, which pulls each replicate about half a game towards 0.5 relative to the point estimate: under 1% of the score on a 72-game pairing, a known small downward bias on swept links, stated in the code comment and preferable to a divergent fit. A thousand resamples because the tails of 200 hold five draws each and the percentiles would wobble by 10 to 15 Elo on their own; the fit over a dozen players takes milliseconds. It measures game-sampling noise only (and, with the exhaustive opening set, is a generalisation proxy rather than a sampling error). Seed-to-seed training noise is the other, larger component and is measured in step 6, not computed. |
| **Exhaustive openings, no game-count flag** | Every pairing plays every cell of the board as the first move, once with each colour: 72 games per reference and 360 per benchmark on 6x6, 128 and 640 on 8x8. There is no `--games` and no `--seed`: the opening order comes from a fixed generator and does not matter. | Openings are distinct first-move cells and the evaluation search is deterministic, so a subset of the openings is a subset of the same games and a repeat is the same games again; the only choice that adds no ambiguity is all of them, once. 72 games per pairing is about ±40 Elo at one standard error on that link, so a 95% interval of about ±80 per link and roughly ±35 to ±40 when the five links are pooled (the bootstrap reports the real figure; every figure in this plan is a 95% interval unless it says otherwise). At the measured 0.3 s per 6x6 tournament game at 100 simulations (`six-a` generations 17–20), 288 games against the 100-simulation references cost about 1.5 minutes and 72 games against `uniform@400` about the same again. 8x8 is expected to be a few times that. Cheap next to a run. |
| **Candidate lockstep, one pairing at a time** | `scripts/benchmark.py` plays the pairings `pairings(...)` returns in sequence, each as one `play_match` with the per-side settings above (lockstep inside the pairing, as the tournament does). It loads only the references it will play (`load_pool` takes that list, not the whole pool) and appends each row as its pairing finishes, so a Ctrl-C loses at most the pairing in progress and the rerun plays only what is missing. | Reuses `play_match`. Pairing-level lockstep already keeps the network's batches at 72 to 128 positions; merging pairings would need a new driver for no measurable gain at this scale. Per-pairing appends are safe because a logged pairing is skipped on rerun. |
| **`compare_runs.py` reads files, computes nothing new** | `summarise(rows, at=None)` in `training/metrics.py` reduces a run's metrics rows to the generation (`at`, or the last), its Elo, losses, mean game length, X win and draw rates, and the means of wall seconds and self-play simulations per second over the window of up to five rows ending at that generation. The script prints one row per run (Markdown table), merges the benchmark rating of exactly the summarised checkpoint (player `<run>/gen_NNN` for the chosen generation, blank when unrated) from `runs/benchmarks/<S>x<S>/ratings.json`, where `S` is read from that checkpoint with `training.checkpoint.checkpoint_size` (the metrics rows carry no size), and with `--key loss_value` prints that key per generation with one column per run. A directory without `metrics.jsonl` is skipped with a notice, so `compare_runs.py runs/*` works with `runs/benchmarks/` in the glob; it is a usage error only when no run remains. `benchmarks` is a reserved run name: `scripts/train.py --name benchmarks` is a usage error. | A table that can be pasted into an experiment file as is. Design: dashboards are readers of the metrics files ([training.md](../design/training.md#logging)). |
| **`docs/experiments/` is a third kind of document** (**approved**) | `docs/experiments/README.md` holds the protocol and the index; `NN-slug.md` holds one experiment: hypothesis, what changes (flags or diff), arms and seeds, exact commands, cost, results (the `compare_runs.py` table plus benchmark ratings with intervals), verdict (adopt / reject / inconclusive), and what changed in the design docs because of it. Design docs record decisions and link to the experiment that made them; plans record how code was built; experiments record measurements. The 6x6 run's section in [open-questions.md](../design/open-questions.md#results-2026-10-10-the-first-6x6-run) stays where it is (it predates the convention) and gains a pointer. Not to be confused with `scripts/experiments/`, which measures the *game* (random play, safe moves, exact solutions; [plan 01](archive/01-game-experiments.md)) and keeps its name and its results under `scripts/experiments/results/`; `docs/experiments/` records *training* experiments, whose runs live in `runs/`. Both READMEs say so. | Pasting results into open-questions.md worked for one run; a dozen experiments would bury the decisions in tables. The author's own convention separates design ("settings") from plans; experiments are neither. |
| **Testbed criteria** | The board the experiments run on is chosen in step 6 by three criteria, recorded in experiment 01: (a) *headroom*: the baseline's benchmark rating at its final generation exceeds its rating at the halfway generation by more than the seed spread (generations 20 and 40 on 6x6, 15 and 30 on 8x8); (b) *cadence*: a two-arm, three-seed experiment finishes overnight, i.e. one baseline run of the chosen length takes at most about 90 minutes; (c) *measurability*: paired-colour scores against the near-strength references are not pinned at 0.5 (per-colour results show both colours winning games). 8x8 is the expected answer if it passes (b); 6x6 is the fallback. | Vláďa's hunch (2026-10-11) is that 6x6 is too easy for upgrades to show; the data in [open-questions.md](../design/open-questions.md#results-2026-10-10-the-first-6x6-run) is ambiguous between "saturated" and "parity-pinned". Two runs settle it. Criterion (b) comes from accepting hours-long experiments but wanting them to turn around daily. |
| **8x8 calibration constants** | The provisional 6x6 constants with the board size changed: `--size 8 --generations 30 --seed 0`, everything else default (the cutoff table gives 4 moves, `α = 11/64`). `--device cpu` unless a one-generation `mps` probe is faster (as in plan 04, step 8). | Changing one thing at a time: the first 8x8 run measures cost and learnability under the known configuration. Tuning 8x8's constants is a later experiment. Thirty generations because 8x8 is expected to learn more slowly and the headroom criterion needs a curve, not a point. |

## Conventions that apply

- Test-driven throughout: failing tests first, `uv run pytest tests/training -q`, then implement. That includes the scripts here: their flag-validation and end-to-end tests in `tests/training/test_scripts.py` are written and seen failing before the script exists, and the scripts stay seeded and self-describing as [scripts/README.md](../../scripts/README.md) asks. CPU only in tests; tiny networks (`NetworkConfig(blocks=1, filters=4, value_hidden=8)`) on 3x3 or 4x4.
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
def play_match(a, b, size, games_per_colour, simulations, rng, search=EVALUATION,
               *, b_simulations: int | None = None, b_search: SearchConfig | None = None)
    # New keyword arguments, defaulting to a's: per-side budget and search config.
    # MatchGame stores (simulations, search) per Mark, next to its trees.

# training/benchmark.py                                   (torch only inside load_pool)
UNIFORM_SIMULATIONS = (100, 400)
CANDIDATE_SIMULATIONS = 100
ANCHOR = "uniform@100"
POOL_DIR = "pool"
# MATCHES_FILE and RATINGS_FILE are imported from training.run (same names, same roles: the
# match log and the ratings derived from it); the benchmark's ratings.json has its own schema,
# documented at write_ratings. training.metrics.repair_partial_last_line is the writer's
# `_repair_partial_last_line` made public; append_rows and run.py's own matches append (which
# lacks it today, a one-line fix with a test in test_run.py) call it before appending.

def uniform_name(simulations: int) -> str              # "uniform@100"
def candidate_name(path: Path, tag: str | None) -> str
    # runs/six-a/gen_020.pt -> "six-a/gen_020" (parent directory, stem); with tag "fpu" ->
    # "six-a/gen_020@fpu". A path inside a pool directory (pool/six-a-gen_020.pt) is named by
    # pool_name instead, so the same network has one name wherever it comes from.
def pool_name(path: Path) -> str
    # pool/six-a-gen_005.pt -> "six-a/gen_005": the stem with its last "-gen_" made "/gen_".
    # A stem without "-gen_NNN" is a ValueError naming the file.
def benchmark_dir(root: Path, size: int) -> Path        # root / "6x6"

@dataclass(frozen=True)
class Reference:
    name: str
    path: Path | None            # None for uniform players
    simulations: int
    sha256: str | None

def list_pool(directory: Path) -> list[Reference]
    # The uniform references for UNIFORM_SIMULATIONS (simulations as given) followed by every
    # `.pt` file in `directory / POOL_DIR`, sorted by name, each with its sha256. A `.pt`
    # whose stem is not `<run>-gen_NNN` is a ValueError naming the file (a mis-named pool
    # member must not silently shrink the pool); files with other suffixes are ignored.
def load_pool(references: Sequence[Reference], size: int, device) -> dict[str, Evaluator]
    # NetworkEvaluator per checkpoint (size mismatch is a ValueError naming the file),
    # UniformEvaluator for the uniform names.

@dataclass(frozen=True)
class BenchmarkRow:              # one pairing, from the candidate's perspective
    a: str; b: str
    wins: int; draws: int; losses: int
    as_x: tuple[int, int, int]; as_o: tuple[int, int, int]
    openings: int; a_simulations: int; b_simulations: int; device: str; date: str
    a_sha256: str; b_sha256: str | None
    a_search: dict[str, Any]     # fields where the candidate's search differs from EVALUATION; {} baseline
    @property
    def pairing(self) -> frozenset[str]    # {a, b}: order-free, one measurement per pair
    def to_json(self) -> dict[str, Any]
    @staticmethod
    def from_json(row: Mapping[str, Any]) -> BenchmarkRow

def base_name(name: str) -> str                          # "six-a/gen_020@fpu" -> "six-a/gen_020"
def append_rows(path: Path, rows: Sequence[BenchmarkRow]) -> None
    # JSON Lines append after `repair_partial_last_line(path)` (a crash-truncated last line
    # would otherwise glue the new row onto the fragment and break every later read) and
    # after `check_consistency` on existing + new rows. A pairing already in the file is a
    # ValueError naming the two players. The file is append-only.
def read_rows(path: Path) -> list[BenchmarkRow]
    # [] when missing; parsed through training.metrics.read_metrics (partial-line repair).
def check_consistency(rows: Sequence[BenchmarkRow], references: Sequence[Reference]) -> None
    # ValueError naming the offender when: a reference's sha256 in the log differs from the
    # pool file's; a base name appears under two sha256 values, or one sha256 under two base
    # names; a full name appears under two `a_search` dicts. `None` hashes (uniform) are
    # never compared.
def pairings(candidate: str, references: Sequence[Reference],
             logged: Sequence[BenchmarkRow]) -> list[Reference]
    # The references to play: every one whose full name differs from the candidate's and whose
    # pairing with it is not yet logged, in pool order.
def as_matches(rows: Sequence[BenchmarkRow]) -> list[tuple[str, str, int, int, int]]
def rating_table(rows: Sequence[BenchmarkRow], anchor: str = ANCHOR) -> dict[str, float]
def bootstrap_intervals(rows: Sequence[BenchmarkRow], resamples: int = 1000, seed: int = 0,
                        anchor: str = ANCHOR) -> dict[str, tuple[float, float]]
    # One resampling loop for every player at once. Per replicate, each pairing's
    # (wins, draws, losses) is a multinomial draw of its n games from the smoothed
    # proportions (w, d + 1, l) / (n + 1); the replicate is fit by `rating_table`, i.e. by
    # `elo_ratings` with its virtual draw, so every replicate is finite even when it is a
    # clean sweep. Returns each player's 2.5th and 97.5th percentiles. The generator is
    # `np.random.default_rng(seed)`: the same log gives the same intervals. Empty rows: {}.
def bootstrap_interval(rows, player, resamples=1000, seed=0, anchor=ANCHOR) -> tuple[float, float]
    # bootstrap_intervals(...)[player]; convenience for the script's one-candidate print.
def write_ratings(path: Path, rows: Sequence[BenchmarkRow]) -> dict[str, Any]
    # {"anchor": ANCHOR, "players": {name: {"elo", "low", "high", "games"}}}, rewritten whole
    # through training.atomic.atomic_write_text, values rounded to one decimal. Derived from
    # the log.

# training/metrics.py
def summarise(rows: Sequence[Mapping[str, Any]], at: int | None = None) -> dict[str, Any]
    # Keys: generation, elo, loss_total, loss_policy, loss_value, mean_game_length,
    # x_win_rate, draw_rate, wall_seconds_mean5, simulations_per_second_mean5 (both means
    # over the window of up to five rows ending at `at`). `at` picks the row with that
    # generation (ValueError if absent); None takes the last row. Empty rows: ValueError.
```

Script flags:

```
scripts/train.py        + --temperature-cutoff N  --c-init F  --c-base F  --dirichlet-alpha F
                          --dirichlet-epsilon F  --eval-simulations N
scripts/benchmark.py      --model PATH  [--root runs/benchmarks]  [--device cpu]
                          [--tag NAME  --c-init F  --c-base F]
scripts/compare_runs.py   RUN_DIR...  [--at N]  [--key NAME]  [--benchmarks runs/benchmarks]
```

`benchmark.py` reads the board size from the checkpoint, lists the pool for that size, refuses a missing pool directory with a message that names the manifest, runs the consistency checks, drops the pairing against the reference with the candidate's own full name, drops (with a notice) every pairing already in the log, so an existing candidate can be rated against a newly added reference, plays the remaining pairings with every cell as an opening, appends the rows, rewrites `ratings.json`, and prints: the candidate's name, rating and interval; one line per reference with the score and per-colour W/D/L (`≥`/`≤` on a sweep); the pool's references and their current ratings. A candidate with no pairing left to play is a usage error saying so. A `--tag` without `--c-init` or `--c-base`, either of those without `--tag`, and a `--device` other than `cpu` while `--root` is the default `runs/benchmarks`, are usage errors (status 2).

## Step 1: flags (`tests/training/test_scripts.py`, then `scripts/train.py`)

Script tests stay in `tests/training/test_scripts.py`, the one home they have today (`tests/` mirrors `src/`, and scripts have no package). The in-process `build_config` tests load `scripts/train.py` as a module with `importlib.util.spec_from_file_location` (a small helper in that test file); the flag-rejection tests keep the subprocess pattern.

Tests first:

1. `build_config` with the new flags at their defaults equals the plan-04 `RunConfig` for the same other flags (`search == SELF_PLAY`, `eval.simulations == args.simulations`). This is the test that keeps `six-a` resumable.
2. Each new flag reaches its field: `--temperature-cutoff 5` gives `search.temperature_cutoff == 5`; `--eval-simulations 400` gives `eval.simulations == 400` with `selfplay.simulations` untouched.
3. Usage errors: `--temperature-cutoff -1`, `--dirichlet-epsilon 1.5`, `--eval-simulations 0`, `--name benchmarks` exit 2 without a traceback (the subprocess pattern of `test_scripts.py`). Add the range checks `SearchConfig` lacks (`0 ≤ ε ≤ 1`, `α > 0`, `cutoff ≥ 0`, `c_base > 0`) in `SearchConfig.__post_init__` with their own unit tests in `tests/engine/test_search.py`, so bad values are refused wherever a config is built, not only by the script.

Then implement. `scripts/train.py` imports `SearchConfig` instead of `SELF_PLAY`. Smoke check by hand: `uv run python scripts/train.py --name six-a --size 6 --generations 20 --seed 0 --device cpu` must print "Resuming" and exit at once with the run already at generation 20 (no config mismatch).

## Step 2: the fit over any key, and per-side match settings (`tests/training/test_evaluate.py`, then `evaluate.py`)

Tests: `elo_ratings` with string players and `anchor="uniform@100"` gives the same numbers as the int version under a relabelling; the existing int tests pass unchanged (default anchor 0). Implement by typing the key generically and sorting pairs with `sorted((player, opponent))` instead of `min`/`max` (identical for ints).

`play_match` with `b_simulations=4` against `a`'s 1, using two `TableEvaluator`s on 3x3: `b`'s tree has at least 4 simulations when it plays and `a`'s exactly 1 (assert through a recording evaluator or by inspecting `MatchGame.trees`); with `b_search=SearchConfig(c_init=5.0, root_noise=False, temperature_cutoff=0)`, every tree `b` owns carries that config and every tree `a` owns carries the default, including a tree rebuilt by the guard in `play_if_ready` (force the guard by handing `MatchGame` a tree whose root is unexpanded, and assert the rebuilt tree's config is its owner's); `b_simulations=0` raises like `simulations=0` does; omitting the keyword arguments reproduces today's results bit for bit (compare a seeded match before and after). Implement in `MatchGame` with per-`Mark` settings.

The "one measurement per pair" rule rests on determinism, so pin it. First the numerics, in `tests/engine/test_network_evaluator.py`: on CPU, a 6x6 `Network` with seeded random weights evaluates 64 random positions in one batch and one at a time with exactly equal outputs (`torch.equal`), and the same in a shuffled batch order; `NetworkEvaluator` asserts this per-row independence in a comment today but nothing tests it on real shapes. Then the search: on 3x3 with two `TableEvaluator`s, `play_match` over all 9 openings gives the same multiset of per-opening results when the sides are swapped (with `as_x` and `as_o` exchanged) and when the opening order is shuffled. On the network path the same check runs in `test_scripts.py` test 9, on CPU, between two invocations against separate scratch roots.

## Step 3: the benchmark (`tests/training/test_benchmark.py`, then `training/benchmark.py`, `scripts/benchmark.py`)

Tests, torch-free except the last two:

1. Naming: `uniform_name`; `candidate_name` for `runs/six-a/gen_020.pt` with and without a tag, and for a pool path (gives the pool name); `pool_name` for `six-a-gen_005.pt` and for a run name that itself contains `-gen_`; a stem without `-gen_NNN` raises; `benchmark_dir`.
2. `list_pool` on a temporary directory with two fake `.pt` files (any bytes) and one stray `.txt` file: the two uniform references first, then the two checkpoints sorted, named by `pool_name`, with correct hashes; the stray file is ignored; a `gen_005.pt` without a run prefix raises naming it.
3. `BenchmarkRow` JSON round trip (including `a_search` and `device`); `append_rows` then `read_rows` returns the same rows, and a file with a truncated last line reads its complete rows; `read_rows` on a missing file is `[]`. `pairing` is the same for the two orders of a pair. Appending a row whose pairing is in the file raises naming the two players; the reversed pairing is refused too. `base_name` strips a tag and leaves an untagged name alone.
4. `check_consistency`: a log row whose `b_sha256` differs from the pool's file raises naming the reference; a base name present under two hashes raises naming it; one hash under two base names raises; a tagged and an untagged name sharing a hash is accepted; a full name present under two `a_search` dicts raises naming it; `None` hashes (uniform) are never compared. `pairings` drops the reference with the candidate's full name, keeps the one with the candidate's base name when the candidate is tagged, drops pairings already logged, and keeps the rest in pool order.
5. `rating_table` on hand-built rows: the anchor is 0; a candidate that beats `uniform@100` 75–25 and loses to `uniform@400` 25–75 lands between them; the result is invariant to the row order.
6. `bootstrap_intervals`: on a log with no sweeps, every player's interval contains that player's point estimate, and the anchor's is `(0, 0)`; intervals narrow when every row's game count is multiplied by 4 (compare widths); two calls on the same rows give the same result; on a log where a player swept every pairing, every replicate rating is finite, the interval has non-zero width (the smoothed resampling keeps it open), and its upper end is at or above the point estimate; on empty rows the result is `{}`.
7. `write_ratings` writes every player with `elo`, `low`, `high`, `games`, and the anchor's `elo` is exactly 0; writing twice gives identical bytes.
8. `load_pool` (torch): a 3x3 checkpoint written with `save_checkpoint` loads into a `NetworkEvaluator`; a 4x4 checkpoint in a 3x3 pool raises naming the file; uniform names give `UniformEvaluator`.
9. End to end in `test_scripts.py` (torch, a few seconds): a 3x3 pool with one tiny checkpoint, `scripts/benchmark.py --model <another tiny 3x3 checkpoint> --root <tmp>` exits 0, appends one row per reference with `openings == 9`, writes `ratings.json` with the candidate present; the same invocation against a second scratch root gives byte-identical rows apart from `date`, and a third scratch root in which the roles are swapped (the former candidate in the pool, the former pool checkpoint as `--model`) gives the same pairing row with `a`/`b`, `as_x`/`as_o` and wins/losses exchanged (the determinism and side-swap checks on the network path); the same invocation again on the first root exits 2 saying every pairing is already logged, and the log is unchanged; after a second pool checkpoint is added, the same invocation plays only the new pairing; `--model <the pool checkpoint itself>` plays only the uniform references and the other checkpoint; `--model <the pool checkpoint> --tag t --c-init 3.0` plays its own untagged pool entry too; the same name with different bytes is refused naming it, and so is the same bytes under a new base name. Flag errors: `--tag` without `--c-init`/`--c-base`, `--c-init` without `--tag`, `--device mps` with the default root, a missing pool directory (status 2, message names `docs/experiments/pools.md`).

Then implement the module, then the script. The two scripts share the `c_init`/`c_base` flag definitions through a small `scripts/_flags.py` (`add_puct_flags(parser)`), or duplicate them if importing between scripts proves awkward; the implementer decides and says which.

**Review point:** `/code-review` (medium).

## Step 4: `compare_runs.py` (`tests/training/test_metrics.py`, then `metrics.py`, `scripts/compare_runs.py`)

Tests: `summarise` on three hand-built rows picks the last by default, honours `--at`, computes the five-row wall mean over fewer rows when fewer exist, raises on an empty list and on an absent generation. Script: in `test_scripts.py`, two temporary run directories with hand-written `metrics.jsonl` print a two-row Markdown table; `--key loss_value` prints one column per run; a directory without `metrics.jsonl` among others is skipped with a notice on stderr, and alone is a usage error. In the `train.py` rejection table: `--name benchmarks` exits 2 naming the reserved name.

## Step 5: the experiments log (`docs/experiments/README.md`, `docs/experiments/01-testbed-and-noise-floor.md`)

`README.md`: what an experiment file contains (the sections listed under Decisions), the rules (one hypothesis per experiment; arms differ in one thing; at least two seeds per arm, three when the first two disagree with the noise floor; the benchmark is the verdict, the in-run curves are the explanation; a verdict changes a design doc, which links back; every recorded rating names the log state it came from, the row count of `runs/benchmarks/<S>x<S>/matches.jsonl` and the date, and is accompanied by the per-reference scores, which never change once logged, because a joint refit moves every rating a little as later benchmarks land), the commands template, and the index table (number, title, verdict, date). `01-testbed-and-noise-floor.md`: hypothesis, arms and commands written now; results and verdict filled in step 6.

**Review point:** `/code-review` (medium) on the branch, before the compute.

## Step 6: calibration

Compute, in this order, each result recorded in experiment 01 as it lands. Times are estimates from `six-a` (26 minutes for 20 generations on CPU; the tournament is about 50 seconds per generation from generation 8 on) and the expected 2.5× per-generation cost of 8x8.

1. **Seed the 6x6 pool.** Copy `runs/six-a/gen_005.pt`, `gen_010.pt`, `gen_015.pt` to `runs/benchmarks/6x6/pool/six-a-gen_005.pt` etc.; write their origin and hashes in `docs/experiments/pools.md`. Benchmark each pool file as a candidate (`scripts/benchmark.py --model runs/benchmarks/6x6/pool/six-a-gen_005.pt`, and so on): each skips itself and plays the other two checkpoints and the two uniform players; pairings already logged by an earlier member are skipped, so afterwards every pool member has a rating on the shared scale and no pairing was played twice. Then benchmark `runs/six-a/gen_020.pt`, the first measured point. About 15 minutes.
2. **Repeatability and side swap.** Into a scratch root (`--root /tmp/bench-check`, with the pool copied there): benchmark `six-a/gen_020` again and compare its rows with the official log's, which must be identical apart from `date`; then benchmark `six-a/gen_010` *before* `six-a/gen_005` there, so the (gen_010, gen_005) pairing is played with the sides the other way round from the official log, and compare that row with the official one with `a`/`b`, `as_x`/`as_o` and wins/losses exchanged. Both are the determinism the one-measurement-per-pair rule relies on, checked on the real 6x6 network rather than only on 3x3 in the tests. Nothing is appended to the official log. About 10 minutes.
3. **6x6 headroom.** `scripts/train.py --name six-a --size 6 --generations 40 --seed 0 --device cpu` (resume). Benchmark `gen_030` and `gen_040`. About 40 minutes. Criterion (a) needs the seed spread from item 5, so the verdict waits.
4. **8x8 baseline.** One-generation `cpu` and `mps` probes (`--generations 1`, two names), then `scripts/train.py --name eight-a --size 8 --generations 30 --seed 0 --device <faster>`. Record the per-generation wall time and the throughput rows for [engineering.md](../design/engineering.md#measured-2026-10-10-self-play-throughput). About 1.5 hours, possibly more. Seed the 8x8 pool from `eight-a` generations 5, 10 and 20 (never 15 or 30, the generations the criteria rate; adjust the others if the curve says so, and say why), benchmark them, then benchmark `eight-a/gen_015` and `gen_030`.
5. **Noise floor on the candidate board.** If 8x8 passes criterion (b) by its measured wall time, run `eight-b` (`--seed 1`) and `eight-c` (`--seed 2`) to 30 generations and benchmark their final checkpoints; otherwise run `six-b` and `six-c` to 40 generations and benchmark `gen_020` and `gen_040` of each. About 3 hours for 8x8, 1.5 hours for 6x6. The seed spread is the range of the three seeds' final ratings; none of those checkpoints is a pool member, so the three are measured alike (five pairings each).
6. **Verdict.** Apply the three criteria, in the experiment file, with the numbers. The chosen board's pool is declared frozen in `docs/experiments/pools.md`. If 8x8 fails (b) by a modest factor, record the cost and say what speed-up would fix it rather than picking 6x6 by default: that is a decision for Vláďa.
7. **`compare_runs.py`** over every run above, pasted into the experiment file, is the results table.

If any run shows a draw rate above about 20% or any other threshold named in the design, record it; do not act on it.

## Step 7: documentation

- `scripts/README.md`: sections for `benchmark.py` and `compare_runs.py`; `train.py` lists the new flags; `evaluate.py` says when to use it (one ad-hoc pairing) and when the benchmark (a rating); the `experiments/` section opens with one sentence saying these measure the game and that training experiments are recorded in `docs/experiments/`.
- `docs/experiments/pools.md`: the manifest (file, origin, hash, frozen date per board), how to add a reference (never replace), how to rebuild a lost pool and check it against the hashes, that the log and `ratings.json` live beside the pool under `runs/benchmarks/`, and the GitHub-release note for sharing a pool with its log.
- `docs/experiments/README.md` and `01-testbed-and-noise-floor.md`: complete.
- `docs/design/training.md`, "Evaluation": a third item, the cross-run benchmark (shared scale per size, anchor, pool, interval), with "monitoring" versus "verdict" made explicit; link the experiments directory.
- `docs/design/engineering.md`: module layout gains `training/benchmark.py`, `scripts/benchmark.py`, `scripts/compare_runs.py`, and `runs/benchmarks/` in the run-directory section; "Measured" gains the 8x8 throughput row and the 8x8 per-generation cost; "Configuration" notes that the search constants are flags now.
- `docs/design/open-questions.md`: the 6x6 results section gains a pointer to experiment 01; the "Experiments to run" table gains a note that experiments are now recorded in `docs/experiments/` (no experiment is assigned a number by this plan); the testbed decision is recorded under Decisions with its date.
- `docs/design/upgrades.md`: the preamble says the verdict is the benchmark rating on the shared scale, linking training.md and the experiments directory.
- `docs/design/README.md` Files table and the root `README.md` Documentation list: add `docs/experiments/`. `docs/architecture.md`: `scripts/` sentence names the two new scripts; the documentation paragraph names the experiments log.
- `docs/experiments/README.md` opens by distinguishing itself from `scripts/experiments/`.
- The memory note for Claude sessions ("design vs plans") would gain the third kind; that is Vláďa's to approve separately, not part of this branch.

## Definition of done

- `uv run pytest` green; the new tests run on CPU in a few seconds.
- `scripts/train.py` with the new flags at their defaults resumes `runs/six-a/` without a config mismatch.
- `runs/benchmarks/6x6/` and the chosen board's pool exist, are frozen in `docs/experiments/pools.md` with their hashes, and `ratings.json` rates every calibration checkpoint with intervals.
- Experiment 01 is complete: the noise floor, the 8x8 cost, the three criteria applied, and the testbed decision, or an explicit hand-back to Vláďa if 8x8 fails criterion (b).
- Both `/code-review` rounds happened and their findings were addressed or deferred with a reason in the PR.
- A pull request from `feat/experiment-harness` to `main`; on merge this plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- `six-a` fails to resume after step 1 with a config mismatch. Report the differing field; do not change the stored checkpoint or loosen `_check_config`.
- The 8x8 run shows no learning at all (Elo flat from generation 1, losses flat). Report the curves before changing any 8x8 constant: the plan's calibration is "the 6x6 constants on 8x8", and a tuned 8x8 is a different experiment.
- An 8x8 generation takes more than about 10 minutes. Report the per-phase seconds; the decision between a speed-up plan and 6x6 is Vláďa's.
- The 95% bootstrap intervals are wider than the seed spread. Report; it would change the pool, which is a decision in this table.
- The repeatability or side-swap check (step 6, item 2) finds a differing row, or the batch-invariance test below fails on this machine. The evaluation search is free of randomness (no noise, no sampling, ties broken in board order), so the only possible source is float32 inference depending on batch size or row order. Report the pairing and the differing games; do not loosen the one-measurement rule in code. The design answer, if it comes to that, is an ordered pairing key so the difference is recorded rather than suppressed, and that is a decision for this table, not for the implementer.
- Any design-level choice not in the Decisions table.

## Approvals (2026-10-11)

Vláďa's answers to the design-level questions this plan raised, recorded so the implementer does not reopen them:

1. **Pool and log location:** under `runs/benchmarks/<S>x<S>/`, gitignored with the runs. Not a top-level directory, and the match log is not committed: findings belong in `docs/experiments/`, machine state in `runs/`. The committed record of a pool is its manifest, `docs/experiments/pools.md`.
2. **`docs/experiments/`** as the third kind of document, next to design and plans: approved.
3. **Cadence:** one experiment (two arms, three seeds) must finish overnight, so a single baseline run may take about 90 minutes. This is criterion (b) in the testbed decision. The 8x8 calibration length stays at 30 generations.
4. **Which experiment runs first** is not this plan's to decide. The plan builds the tools and leaves the list in [open-questions.md](../design/open-questions.md#experiments-to-run) untouched; the experiments log starts with the calibration entry only.
