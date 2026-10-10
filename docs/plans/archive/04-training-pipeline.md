# Plan 04: The self-play training pipeline

| | |
|---|---|
| Status | Implemented and merged (PR #19, 2026-10-10). The first 6x6 run is recorded in [open-questions.md](../../design/open-questions.md#results-2026-10-10-the-first-6x6-run). |
| Branches | Plan: `plan/training-pipeline`. Implementation: `feat/training-pipeline`. |
| Design references | [training.md](../../design/training.md) (the generation loop, lockstep self-play, training data, replay buffer, evaluation, logging), [network.md](../../design/network.md) (loss, optimiser, augmentation), [engineering.md](../../design/engineering.md) (module layout, configuration and checkpoints, devices and determinism, testing), [search.md](../../design/search.md) (self-play move selection, root noise, tree reuse), [open-questions.md](../../design/open-questions.md) (the constants this plan sets provisionally) |
| Depends on | [Plan 03](03-cli-adapter.md) merged (PR #15): `training.checkpoint` (`save_checkpoint`, `load_checkpoint`, `latest_checkpoint`), `engine.search.SearchTree` with its two-phase API, `engine.network.alphazero_loss`, `engine.encoding` symmetries, `cli.adapter.load_engine`. |

Scope approved by Vláďa on 2026-10-09: the full pipeline *and* checkpoint tournaments with Elo, with a TensorBoard writer and a standalone matchup script, proven by a real 6x6 run.

## Goal

Build roadmap step 4: a run that starts from random weights and produces a sequence of checkpoints that get stronger. After this plan, `uv run python scripts/train.py --size 6` plays self-play games in lockstep, trains the network on a replay buffer with symmetry augmentation, writes one checkpoint per generation, measures each generation against recent ones with Elo, logs every number to a metrics file and TensorBoard, and survives being stopped and restarted.

The proof is a seeded 6x6 run of about 20 generations in the PR, with the loss and Elo curves. A rising Elo is the goal, not the merge gate: a flat curve gets documented in the PR and opens a tuning follow-up. The merge gate is the test suite plus a run that completes, resumes, and produces checkpoints the CLI loads.

## Deliverables

1. `training/selfplay.py`: `SelfPlayConfig`, `Examples` (the training-data record), `lockstep_step` (one simulation on many trees with one batched evaluation), `self_play` (a generation's games), `game_examples` (the `z` labelling).
2. `training/replay.py`: `ReplayBuffer` (expiry by generation, uniform sampling, save and load), `Batch`, `legal_from_planes`, `transform_batch` and `augment` (random board symmetry per example).
3. `training/trainer.py`: `TrainConfig`, `Trainer` (AdamW, one training step, one generation of steps), `StepLosses`.
4. `training/evaluate.py`: `EvalConfig`, `opponents_for` (which checkpoints a generation plays: the nearest ones plus the ladder), `play_match` (two evaluators, paired openings, lockstep), `elo_ratings` (Bradley-Terry fit over all matches so far), `elo_difference`.
5. `training/metrics.py`: `MetricsWriter`, one JSON Lines row per generation, mirrored to TensorBoard.
6. `training/run.py`: `RunConfig`, `run` (the generation loop, checkpoint per generation, tournament, resume).
7. `scripts/train.py` (start or resume a run) and `scripts/evaluate.py` (pit two checkpoints, or a checkpoint against the uniform search).
8. Tests: `tests/training/test_selfplay.py`, `test_replay.py`, `test_trainer.py`, `test_evaluate.py`, `test_metrics.py`, `test_run.py`; the `solver` fixture moves from `tests/engine/conftest.py` to `tests/conftest.py` so both packages share it.
9. The `tensorboard` dependency.
10. Documentation updates in step 9, and the real 6x6 run in step 8.

## Out of scope

- Every item in [upgrades.md](../../design/upgrades.md): no `z` blending with the search value (the example record does not even store the root value; adding it is the first step of that upgrade), no gating, no multiprocessing workers, no resignation, no playout-cap randomisation, no MCTS-Solver, no first-play urgency, no global pooling. Version 1 is the paper-faithful baseline the upgrades get measured against.
- Any change to `ox_zero.engine` or `ox_zero.cli`. The search's two-phase API was built for this plan; if it lacks something, stop and ask (see the end).
- Performance work beyond measuring. The plan records self-play throughput on `cpu` and `mps`; it does not optimise either. No `torch.compile`, no array tree, no tensorised rules.
- A 12x12 run. The 6x6 run proves the pipeline; 12x12 is a separate, budgeted run once the constants are tuned ([README.md](../../design/README.md#guiding-principles), principle 3).
- The custom Textual dashboard, Weights & Biases, YAML configs, `scripts/plot.py`.
- Hyperparameter tuning. The constants below are provisional starting points; the first run's job is to show what to tune, and tuning is a follow-up with its own numbers.

## Decisions made in this plan

Code-level choices the design left to the plan ([open-questions.md](../../design/open-questions.md#questions-for-the-plan-not-the-design)), plus four items that touch the design, marked **approved 2026-10-09**. Nothing below is open: an implementer follows this table without asking. Anything that turns out wrong in review gets changed here, not in code.

| Decision | Choice | Why |
|---|---|---|
| **Training examples store encoded planes, not `State`** | `Examples` is a struct of arrays: `planes` `uint8 [N, 3, S, S]` (the `encode` output, which is 0/1, cast down), `pi` `float32 [N, S²]`, `z` `float32 [N]`. Converted to `float32` planes when sampled. The generation is the replay buffer's key, not a column (see Buffer persistence). No stored legal mask: a cell is legal exactly when planes 0 and 1 are both zero there (recorded positions are never terminal), so `legal_from_planes` derives it where the loss needs it. | Sampling is a fancy index and augmentation is a batched array transform; storing `State` would mean re-encoding every sampled example on every step. `uint8` keeps a 12x12 example at about 1.2 KB, so 300k examples fit in 400 MB on an 8 GB machine. |
| **`z` labelling** | For each recorded position, `z = +1` if its side to move is the game's winner, `-1` if the other side won, `0` for a draw. | [training.md](../../design/training.md#training-data). The winner is the player who made the last move, so the last recorded position (where the winner is to move) gets `+1` and the signs alternate backwards. |
| **Lockstep driver with refilling** | `parallel` trees are alive at once. Every step: `select()` on every live tree, one `evaluate` on the leaves that need it (root setups included; trees whose leaf was terminal skipped the network and already counted the simulation), `expand_and_backup` on each. A tree that reaches `simulations` records `policy_target()`, plays `choose_move(move_count)`, and if the game is over hands its examples in and is replaced by a fresh game until `games` have been started. The loop ends when every started game is over. | Keeps the batch full until the tail of the generation; the alternative (one batch of `games` trees that shrinks) wastes the network at the end and needs `games` trees in memory. One batched call per step is the whole point of lockstep ([training.md](../../design/training.md#self-play-in-lockstep)). |
| **One RNG for everything** | `run` derives one `numpy.random.Generator` from the run seed and passes it to every `SearchTree`, the buffer's sampling, the augmentation, and the tournament. `torch.manual_seed(seed)` once, for the initial weights. The checkpoint stores both states (plan 03's `rng` field). | [engineering.md](../../design/engineering.md#devices-and-determinism). Single-threaded lockstep draws from the shared generator in a fixed order, so one generator gives a reproducible run and one state to checkpoint. |
| **Augmentation is per example, in 8 groups** | `augment` draws a symmetry `k` per example and applies `transform_planes` / `transform_policy` to each of the 8 groups of the batch; `z` is unchanged. | [network.md](../../design/network.md#symmetries) says per example; [training.md](../../design/training.md#augmentation) says per batch. Per example gives 8× the diversity per step for eight array calls instead of one. Step 9 aligns `training.md`. |
| **Buffer persistence** | `ReplayBuffer.save(directory)` keeps `buffer/gen_NNN.npz`, one file per held generation (atomic: temporary file then rename, written through an open file object because `np.savez` given a path without `.npz` silently appends the suffix and the rename would miss), writing only generations that have no file yet and deleting files of expired generations. `run` saves before each checkpoint and loads on resume. Missing directory on resume from a checkpoint of generation 1 or later: a warning on stderr, start with an empty buffer, continue. No warning when resuming from `gen_000.pt` alone: the buffer is first written after generation 1, so nothing was lost. The generation is the file name and the buffer's key; `Examples` carries no per-row generation column. | [engineering.md](../../design/engineering.md#devices-and-determinism): a resumed run continues rather than restarts. Without the buffer, a resumed generation would train on its own games only. One file per generation makes a save cost one generation's bytes instead of the whole buffer (about 40 MB rather than 400 MB per save at the 12x12 figure of 300k examples over ten generations), and turns "drop buffer generations above the checkpoint" into deleting files. A single generation key, not a per-row column too, means one source of truth for expiry. |
| **Metrics format** | JSON Lines, `metrics.jsonl`, one object per trained generation, fixed keys (step 6). Match results go to `matches.jsonl`, one object per pairing per generation. The current Elo table is rewritten to `ratings.json` each generation. | [training.md](../../design/training.md#logging): plain files, dashboards are readers. JSON Lines appends without rewriting, survives a crash mid-row (the reader skips a partial last line), and nests the per-opponent scores that CSV cannot. |
| **The checkpoint is the commit marker** | Order inside a generation: self-play, train, tournament, metrics row, `buffer/gen_NNN.npz`, `gen_NNN.pt` last. On resume, rows in `metrics.jsonl` and `matches.jsonl` with a generation above the latest checkpoint are dropped, buffer files above it are deleted (the buffer is written before the checkpoint, so a crash between the two leaves one generation of examples the checkpoint never saw), and the TensorBoard writer is opened with `purge_step` set to the next generation so its event file drops the same stale points. | A generation either has its checkpoint or did not happen; everything else is re-derived. Keeps resume to one rule. |
| **Resume refuses a changed config** | `run` with a name that already has checkpoints loads the latest one and compares, field by field: `size` against the checkpoint's `size`, `network` against `network_config`, and `search`, `selfplay`, `train`, `eval` and `seed` against `configs` (the checkpoint's `configs` bag gains a `run` entry: `RunIdentity(name, seed)`, a frozen dataclass in `run.py`, because `save_checkpoint` applies `dataclasses.asdict` to every entry and a plain dict would raise). Any difference raises `ValueError` naming the field. Exempt: `generations` (the target) and `tensorboard` (a reader, not part of the run); `name` is the directory and cannot differ. | A run is its config. Changing the learning rate halfway produces a curve nobody can interpret; a new config is a new run name. |
| **Generation 0 is a checkpoint** | `run` saves the random network as `gen_000.pt` before the first generation, with the optimiser and RNG state. | Elo needs an anchor (generation 0 is 0 Elo by definition), the first tournament needs an opponent, and resume from "nothing happened yet" is the same code path as any other. |
| **Training runs live in `runs/<name>/`, not `checkpoints/`** (**approved 2026-10-09**, design change) | `scripts/train.py --root` defaults to `runs/`. To use a model from the CLI, copy or symlink the checkpoint under `checkpoints/`. | `load_engine` auto-loads the newest checkpoint under `checkpoints/` by generation, and the CLI is 12x12 only. A 6x6 run in `checkpoints/` would make every `ox-zero analyze` fail with the size error until `--model` is passed, which plan 03 already hit with `make_checkpoint`. `runs/` is already gitignored (TensorBoard's default). Alternative considered: make `latest_checkpoint` size-aware; that is a CLI change and out of scope here. |
| **Tournament protocol** (**approved 2026-10-09**; [open-questions.md](../../design/open-questions.md) listed it as Open) | Each new generation `g` plays the previous `opponents = 3` checkpoints (fewer when they do not exist yet) **plus one ladder opponent**, generation `g - ladder` with `ladder = 8`, when it exists and is not already among the nearest. Per pairing: `games_per_colour = 20` first moves, drawn as `min(games_per_colour, S²)` distinct cells without replacement and cycled if more are needed, each played twice with colours swapped, so 40 games. After the first move both players search with `EVALUATION = SearchConfig(root_noise=False, temperature_cutoff=0)`: no noise, most-visited move always, `simulations` per move as in self-play. Draws score ½. | "Both as X and as O, with search but without root noise" is decided; the tie-breaking of deterministic players was open and `training.md` floats "an opening book of random first moves". Random first moves guarantee distinct games with no new search code and keep the measured play deterministic. Paired colours cancel the first-move handicap. Alternative considered: `τ = 1` sampling for the first `cutoff` moves; it reuses `choose_move` but duplicates games once the policy is sharp. **The ladder is the anti-drift measure** (added 2026-10-10): with only nearest-neighbour matches, a checkpoint's rating reaches the anchor through a chain of noisy links (about 50 Elo of noise per 40-game link), and the errors add up like a random walk, so the absolute rating of a late generation could be off by well over 100 Elo while the local curve looks fine. One long-range match per generation gives the fit a direct measurement across eight links and pins the chain. If the ladder matches come out as sweeps (score above about 95%), the ladder is too long to be informative and should be shortened in `EvalConfig`; the PR reports the ladder scores. |
| **Elo fit** | Bradley-Terry maximum likelihood over *all* matches of the run so far, by a few hundred Newton steps on the per-player ratings, generation 0 fixed at 0. One virtual draw per pairing (½ point each, +1 game) so a 40–0 sweep gives a finite rating. `elo_difference(score) = -400 · log10(1/score - 1)` for a single pairing. | Round-robin Elo from pairwise scores is what [training.md](../../design/training.md#evaluation) asks for; a joint fit uses every game, and anchoring at generation 0 keeps the scale fixed across generations. The virtual draw is the standard regulariser. |
| **TensorBoard is a regular dependency** (**approved 2026-10-09**, new dependency) | `uv add tensorboard`. `MetricsWriter` imports `torch.utils.tensorboard` lazily, inside `__init__`, only when `tensorboard=True` (the default in `scripts/train.py`; `--no-tensorboard` turns it off, tests turn it off). Event files go to `runs/<name>/tensorboard/`. | [training.md](../../design/training.md#logging): "TensorBoard from day one". An optional extra would need `uv sync --extra train` on every machine for no gain on a one-person project. Lazy import keeps `import ox_zero.training` cheap. |
| **Provisional constants for 6x6** (**approved 2026-10-09**; [open-questions.md](../../design/open-questions.md#constants-waiting-on-the-experiments) listed them as Open) | `SelfPlayConfig(games=128, parallel=64, simulations=100)`, `TrainConfig(batch_size=256, steps_per_generation=50, learning_rate=1e-3, weight_decay=1e-4, buffer_generations=10)`, `EvalConfig(opponents=3, ladder=8, games_per_colour=20, simulations=100)`, `NetworkConfig()` (4×64, value hidden 256), `SELF_PLAY` search config. | Simulations and the tower are the design's own starting points. 128 games × ~13 moves gives ~1,700 examples per generation; `K = 10` holds ~17k; 50 steps × 256 draws 12,800 per generation, so each position is seen about 7 times over its life, with 8 symmetries on top. Estimated cost per generation on the author's laptop, CPU: about a minute of self-play (~4,000 lockstep steps of up to 64 positions; `parallel` is below `games` on purpose, so the refilling keeps the batch full until the last game has started, rather than one batch of 128 that shrinks over the whole generation), seconds of training, about a minute and a half of tournament (160 games from generation 8 on, when `g - 8` first exists: three nearest opponents plus the ladder). These are starting points; the run's logged game length and loss curves re-derive them. |
| **AdamW decays every parameter** | No parameter groups: BatchNorm weights and biases get the same weight decay as the convolutions. | The paper's `c · ‖θ‖²` applies to all of θ. Excluding norms and biases is a common refinement, not the baseline; it goes on the upgrades list if the value loss plateaus. |
| **Device** | `scripts/train.py --device auto|cpu|mps|cuda`, default `auto` (`select_device()`). One `Network` object on that device is shared by the `NetworkEvaluator` (self-play and the current side of the tournament) and the `Trainer`; the evaluator sets eval mode on every call, the trainer sets train mode on every step. Opponent checkpoints are loaded onto the same device. | Self-play batches are up to `parallel` positions, where the GPU should pay off, unlike the CLI's batch size 1 ([engineering.md](../../design/engineering.md#measured-2026-09-27)). Step 8 measures it rather than assuming. |

## Conventions that apply

- Test-driven: for each step write the failing tests first, run `uv run pytest tests/training -q`, then implement until green. Tests run on CPU only. Network tests use `NetworkConfig(blocks=1, filters=4, value_hidden=8)` on 3x3 or 4x4; search-only tests use `UniformEvaluator` and never import torch (keep `selfplay.py`, `replay.py` and `evaluate.py` torch-free, like the engine's search; `trainer.py`, `metrics.py` and `run.py` are the torch side).
- Comment thoroughly, connecting code to theory ([project `CLAUDE.md`](../../../CLAUDE.md)): why `π` is the `τ = 1` distribution whatever temperature chose the move; why `z` flips sign every ply; why the network sees batches of leaves from many games (lockstep) instead of many leaves from one game (virtual loss); why old generations expire (the network they came from is weaker, so their `π` is stale); why augmentation is free supervision (the game is symmetric, the data is not); why AdamW's decoupled weight decay plays the role of the paper's L2 term without being identical to it under an adaptive optimiser (the loss module's docstring has the argument); why the trainer sets `train()` and the evaluator `eval()` on every call (BatchNorm uses batch statistics in one mode and running statistics in the other); why Elo needs an anchor and a regulariser; why evaluation games turn root noise off; why the checkpoint is saved last.
- Coordinates `(row, col)`, zero-based, row 0 at the top. Policies and masks are flat vectors of length `S²` in board order, index `row * S + col`, like `State.board` and the engine's policies.
- Values in `[-1, 1]` from the side to move's perspective, everywhere in this package.
- Conventional commits, small and atomic, on `feat/training-pipeline`. Suggested sequence: `test(training): add self-play tests` / `feat(training): add lockstep self-play`; `test(training): add replay buffer tests` / `feat(training): add the replay buffer`; the same pairs for `trainer`, `evaluate`, `metrics`, `run`; `feat(scripts): add train and evaluate`; `chore: add tensorboard`; `docs: record the training pipeline as built`.
- **Code review cadence.** Run `/code-review` at medium effort three times: after step 2 (self-play and buffer: the labelling and symmetry logic), after step 4 (trainer and tournaments: the mode switching, the Elo fit), and after step 7 (loop, resume, scripts), before the 6x6 run and the PR. Fix what it finds before moving on.
- One new dependency, `tensorboard`, and nothing else.

## Interfaces

The contract the tests pin. Names and shapes are fixed; internals are free.

```python
# training/selfplay.py                                   (torch-free)
@dataclass(frozen=True)
class SelfPlayConfig:
    games: int = 128            # games per generation
    parallel: int = 128         # trees stepped in lockstep; min(parallel, games) are alive
    simulations: int = 100      # per move; the budget after each play()

@dataclass(frozen=True)
class Examples:                 # struct of arrays, one row per recorded position
    planes: np.ndarray          # uint8   [N, 3, S, S]
    pi: np.ndarray              # float32 [N, S²]
    z: np.ndarray               # float32 [N]
    def __len__(self) -> int
    @property
    def size(self) -> int       # S, from planes.shape
    @staticmethod
    def empty(size: int) -> Examples
    @staticmethod
    def concatenate(parts: Sequence[Examples]) -> Examples

@dataclass(frozen=True)
class GameRecord:
    moves: tuple[Cell, ...]
    winner: Mark | None         # None: draw

@dataclass(frozen=True)
class SelfPlayResult:
    examples: Examples
    games: list[GameRecord]
    steps: int                  # lockstep steps taken (one evaluate call each)
    simulations: int            # full simulations over all trees, for the sims/s metric

def game_examples(states: Sequence[State], pis: Sequence[np.ndarray], final: State) -> Examples
    # states[i] is the position before move i, pis[i] its policy target; `final` is terminal.
def lockstep_step(trees: Sequence[SearchTree], evaluator: Evaluator) -> int
    # One select / batched evaluate / expand_and_backup over the trees. Returns the number
    # of simulations completed this step (terminal leaves count, root setups do not).
    # When no tree needs the network (every select hit a terminal leaf, or `trees` is empty)
    # the evaluator is not called at all: UniformEvaluator rejects an empty batch.
def self_play(evaluator: Evaluator, size: int, search: SearchConfig, config: SelfPlayConfig,
              rng: np.random.Generator) -> SelfPlayResult

# training/replay.py                                     (torch-free)
@dataclass(frozen=True)
class Batch:
    planes: np.ndarray          # float32 [B, 3, S, S]
    pi: np.ndarray              # float32 [B, S²]
    z: np.ndarray               # float32 [B]

def legal_from_planes(planes: np.ndarray) -> np.ndarray
    # bool [B, S²]: empty cells, i.e. planes 0 and 1 both zero. Equals legal_mask(state) for
    # every non-terminal position; recorded positions never are terminal.

class ReplayBuffer:
    def __init__(self, size: int, generations: int) -> None       # generations: K
    def add(self, examples: Examples, generation: int) -> None
        # Keeps the newest K generations by number: after adding generation g, anything
        # with generation <= g - K is dropped. Raises on a board-size mismatch.
    def __len__(self) -> int
    @property
    def generations(self) -> list[int]                             # held, ascending
    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch
        # Uniform over positions, with replacement, indexing the held generations in ascending
        # generation order (load sorts by number), so a resumed buffer maps the same rng draw to
        # the same example as the in-memory one. Raises ValueError when empty.
    def save(self, directory: Path) -> None
        # directory/gen_NNN.npz per held generation; writes only missing files, deletes expired ones.
    @classmethod
    def load(cls, directory: Path, size: int, generations: int, upto: int | None = None) -> ReplayBuffer
        # Reads every gen_NNN.npz, checking each against `size` (an empty directory gives an
        # empty buffer of that size); `upto` drops (and deletes) files above that generation.
    def drop_above(self, generation: int) -> None

def transform_batch(batch: Batch, ks: np.ndarray) -> Batch
    # ks: int [B] in 0..7, one symmetry per example; planes and pi transformed together.
def augment(batch: Batch, rng: np.random.Generator) -> Batch
    # transform_batch with ks drawn uniformly from 0..7.

# training/trainer.py
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    steps_per_generation: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    buffer_generations: int = 10                                   # K, read by run.py

@dataclass(frozen=True)
class StepLosses:
    total: float; policy: float; value: float

class Trainer:
    def __init__(self, network: Network, config: TrainConfig, device: torch.device,
                 optimizer_state: Mapping[str, Any] | None = None) -> None
    optimizer: torch.optim.AdamW
    def step(self, batch: Batch) -> StepLosses                     # train mode, one AdamW step
    def train_generation(self, buffer: ReplayBuffer, rng: np.random.Generator) -> StepLosses
        # steps_per_generation steps of sample -> augment -> step; returns the mean losses.

# training/evaluate.py                                   (torch-free)
EVALUATION = SearchConfig(root_noise=False, temperature_cutoff=0)

@dataclass(frozen=True)
class EvalConfig:
    opponents: int = 3          # nearest previous checkpoints
    ladder: int | None = 8      # also play generation g - ladder (anti-drift); None: off
    games_per_colour: int = 20
    simulations: int = 100

def opponents_for(generation: int, config: EvalConfig) -> list[int]
    # [g-1, ..., g-opponents] clipped at 0, then g-ladder if >= 0 and not already listed.
    # A ladder of None or <= 0 means no ladder: a generation must never be its own opponent
    # (its checkpoint is written last and does not exist during its tournament).

@dataclass(frozen=True)
class MatchResult:              # from a's perspective
    wins: int; draws: int; losses: int
    openings: tuple[Cell, ...]  # the first moves used, one per pair of games
    @property
    def games(self) -> int
    @property
    def score(self) -> float    # (wins + draws / 2) / games

class MatchGame:                # one game between two evaluators, two trees
    def __init__(self, state: State, a: Evaluator, b: Evaluator, a_plays: Mark,
                 simulations: int, search: SearchConfig, rng: np.random.Generator) -> None
    done: bool
    final: State                # valid once done
    @property
    def mover(self) -> Evaluator          # whose turn it is: a or b
    @property
    def tree(self) -> SearchTree          # the mover's tree; created from the current position
                                          # on that player's first turn
    trees: dict[Mark, SearchTree | None]  # keyed by side (X, O), never by evaluator identity: the
                                          # same evaluator object may play both sides
    def play_if_ready(self) -> None       # when tree.simulations >= simulations: choose the
                                          # most-visited move, play() it on both existing trees

def play_match(a: Evaluator, b: Evaluator, size: int, games_per_colour: int, simulations: int,
               rng: np.random.Generator, search: SearchConfig = EVALUATION) -> MatchResult
    # All games as MatchGames in lockstep: each step, lockstep_step over the trees of the games
    # where a is to move, then over those where b is to move; then play_if_ready on each.
def elo_difference(score: float) -> float   # ±inf at 0 and 1; callers wanting a finite number
                                           # apply the virtual draw first, as the script does
def elo_ratings(matches: Sequence[tuple[int, int, int, int, int]],
                anchor: int = 0) -> dict[int, float]
    # (player, opponent, wins, draws, losses) rows; players are generation numbers.

# training/metrics.py
class MetricsWriter:
    def __init__(self, directory: Path, tensorboard: bool, purge_from: int | None = None) -> None
        # purge_from: on resume, the first generation that will be re-run; passed to the
        # SummaryWriter as purge_step so stale TensorBoard points above it are discarded.
    def write(self, row: Mapping[str, Any]) -> None                # appends to metrics.jsonl
    def close(self) -> None
def read_metrics(path: Path) -> list[dict[str, Any]]              # skips a partial last line

# training/run.py
@dataclass(frozen=True)
class RunConfig:
    name: str
    size: int
    seed: int
    generations: int            # target generation; resume continues to it
    network: NetworkConfig = NetworkConfig()
    search: SearchConfig = SELF_PLAY
    selfplay: SelfPlayConfig = SelfPlayConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    tensorboard: bool = True

@dataclass(frozen=True)
class RunIdentity:              # stored in the checkpoint's configs bag as "run"
    name: str
    seed: int

def run(config: RunConfig, root: Path = Path("runs"), device: torch.device | None = None) -> Path
    # Returns the run directory. Resumes if root/name holds checkpoints.
```

Files in a run directory `runs/<name>/`: `gen_NNN.pt` (plan 03's format; `configs` holds `search`, `selfplay`, `train`, `eval` and `run` as field dicts), `buffer/gen_NNN.npz` for the held generations, `metrics.jsonl`, `matches.jsonl`, `ratings.json`, `tensorboard/`.

## Step 1: self-play (`tests/training/test_selfplay.py`, then `selfplay.py`)

Tests, all with `UniformEvaluator` or `TableEvaluator` unless stated:

1. **`z` labelling, decisive game.** The 3x3 game `0,0 0,1 0,2` (X completes `X O X`). The three positions before each move get `z = +1, -1, +1`: X is to move at positions 0 and 2 and X won. Planes equal `encode(state)` cast to `uint8`, `legal_from_planes(planes)` equals `legal_mask(state)` for each.
2. **`z` labelling, draw.** The 4x4 board `XXXXXXXXOOOOOOOO` (two rows of X over two rows of O) is full with no alternating line: `from_board_string(..., size=4)` has `winner None` and is terminal. The game that reaches it, X's cells of rows 0–1 and O's cells of rows 2–3 interleaved in board order so the turns alternate (`0,0 2,0 0,1 2,1 ...`), never produces a winner along the way, and every `z == 0`. Say in the test why that board has no line (same-mark bands two cells thick) and that this is the "wall" draw from [open-questions.md](../../design/open-questions.md#follow-up-safe-moves-over-a-game).
3. **One batched call per step.** Three trees on 3x3 positions with at most one mark (the empty board, after `1,1`, after `0,0`), so no root child is terminal and every tree needs the network on its first simulation; a counting evaluator wrapper: `lockstep_step` calls `evaluate` exactly once with exactly the three root states (the setup step), then once with three leaves; after the second step every tree has `simulations == 1`. Separately, a single tree on `X O _` (row 0, X to move: the first child in board order wins on the spot) is stepped twice: on the second step the evaluator is not called at all (the counting wrapper sees one call in total, from the setup), and the step's return value still counts the simulation. `lockstep_step([], evaluator)` returns 0 without calling the evaluator.
4. **Lockstep equals batch size 1.** With `SearchConfig()` (no noise, so no shared-RNG ordering effects), stepping three fresh trees (the positions of test 3) in lockstep until every tree has `simulations == 20` (exactly 21 steps whatever the positions: the first is the uncounted root setup, and every later step completes one simulation per tree, with or without the network, because a terminal leaf is counted too) gives the same root visit counts as `tree.simulate(evaluator, 20)` on three fresh trees. This is the test that says lockstep changes the schedule and nothing else ([training.md](../../design/training.md#self-play-in-lockstep)).
5. **A generation of games.** `self_play(UniformEvaluator(), size=4, SELF_PLAY, SelfPlayConfig(games=6, parallel=3, simulations=8), default_rng(0))`: six `GameRecord`s, each replaying (`play(record.moves, size=4)`) to a terminal state with the recorded winner; `len(examples) == sum(len(r.moves))`; every `pi` row sums to 1 and is zero where `legal_from_planes` is False; `z` of each game's last position is `+1` for a decisive game; `steps > 0` and `simulations >= 8 * len(examples)` (at least the budget per move; inherited subtree visits do not count against it, plan 02).
6. **Refilling keeps the batch bounded.** Same call with a spying evaluator: no batch is larger than `parallel`, and the total of distinct games is `games` (not `parallel`, not `games + parallel`).
7. **Seeds.** The same seed gives identical `Examples`; a different seed gives different `pi` somewhere (noise and sampling are on in `SELF_PLAY`).
8. **Temperature is honoured.** With `SearchConfig(root_noise=True, temperature_cutoff=0)` on 3x3 and a `TableEvaluator` that gives one move all the prior, every game's first move is that move (greedy from move 0); with the default cutoff, over 20 games the first moves are not all equal.
9. **Network smoke.** `NetworkEvaluator(Network(4, tiny), cpu)` plays 2 games with 4 simulations; just shape and termination checks. Marks the only torch import in this test file.

Implementation notes: a small `_Game` helper per live tree holds the `SearchTree`, the list of root states and policy targets so far; `self_play` owns the refilling loop. `lockstep_step` is the piece `evaluate.py` reuses, so it must not know about games. Record the state *before* `play` and `policy_target()` *before* `play` (the subtree reuse keeps counts, but the target is the root's distribution at decision time).

## Step 2: the replay buffer (`tests/training/test_replay.py`, then `replay.py`)

Tests:

1. **Expiry by generation.** `K = 2`: add generations 1, 2, 3 → `generations == [2, 3]` and `len` is the sum of those two. Adding generation 3 again appends (two self-play batches of one generation are fine). Adding a different board size raises `ValueError`.
2. **Uniform sampling.** Two examples that differ only in `z` (+1 and −1); 10,000 draws with a fixed seed give each between 45% and 55%. `sample` returns `float32` planes of shape `[B, 3, S, S]` and the other arrays with matching rows. Sampling from an empty buffer raises `ValueError`.
3. **Augmentation is consistent.** Build a batch of 8 copies of one 4x4 position with a one-hot `pi` on a legal move and call `transform_batch(batch, ks=arange(8))`: row `k` has `planes == transform_planes(encode(state), k)`, `pi == transform_policy(pi0, k)`, `z` unchanged, and `legal_from_planes` of the row equals `transform_policy(legal_mask(state), k)`. Property on a mixed batch through `augment` with a real rng: for every row, the cell that carries `pi`'s mass is empty in the transformed planes (`argmax(pi)` is a legal cell), which is what "permuted identically" means.
4. **`k = 0` is the identity**, and `augment` on a batch of 64 rows with a seeded rng uses more than one `k` (spy on `transform_batch`).
5. **Save and load.** `save(tmp_path / "buffer")` on a `K = 2` buffer holding generations 2 and 3 writes exactly `gen_002.npz` and `gen_003.npz`; `load` restores every array exactly and `generations` (sorted ascending whatever order the directory listing gives; test by creating the files out of order); expiry continues correctly after loading (add generation 4 → `[3, 4]`). A second `save` after that leaves `gen_003.npz`'s mtime unchanged, writes `gen_004.npz`, and deletes `gen_002.npz`; no temporary file is left behind. `load(..., upto=3)` on a directory holding 2, 3, 4 returns `[2, 3]` and deletes `gen_004.npz`.

Implementation notes: hold a `dict[int, Examples]` by generation (concatenate on `add` when a generation arrives in several batches); the dict key is the only record of an example's generation. The tests pin behaviour only. `transform_planes` and `transform_policy` already act on leading batch axes.

**Review point:** run `/code-review` (medium) on the branch here.

## Step 3: the trainer (`tests/training/test_trainer.py`, then `trainer.py`)

Tests (tiny network, CPU, `tmp_path` where files are needed):

1. **A fixed batch is memorised.** 200 steps on the same 8-example 4x4 batch (no augmentation) with `TrainConfig(learning_rate=1e-2, weight_decay=0.0)`, the settings the network's own memorise test needs (the defaults are tuned for a real run, not for overfitting 8 examples in seconds): the total loss falls to well under its starting value and the value loss to near 0; every reported loss is finite. The same test as the network's "memorise one batch", now through the trainer, so the optimiser wiring is what is under test.
2. **Mode switching.** After `step`, `network.training` is True; a `NetworkEvaluator` on the same object evaluates fine and leaves it in eval mode; a further `step` works. This pins the shared-object design.
3. **Optimiser construction.** `trainer.optimizer` is `AdamW` with `lr == config.learning_rate` and `weight_decay == config.weight_decay`; with `optimizer_state` from a checkpoint written after one step, the loaded `state_dict()` equals the saved one (the comparison from plan 03's test 2).
4. **A generation of steps.** `train_generation` on a buffer with one generation runs exactly `steps_per_generation` steps (check `optimizer.state[p]["step"]` for a parameter) and returns the mean of the step losses (compare with a hand-collected list using a seeded rng replayed).
5. **Augmentation is applied.** Monkeypatch `trainer_module.augment` with a spy: called once per step with the sampled batch and the rng.

Implementation notes: `step` builds tensors from the batch (`torch.from_numpy(...).to(device)`), calls `network.train()`, `alphazero_loss`, `zero_grad`, `backward`, `optimizer.step()`, and returns floats (`.item()` once per loss; three device syncs per step are fine at this scale). The legal mask is `legal_from_planes(batch.planes)`, passed to the loss as a bool tensor; say in a comment why it is derived rather than stored.

## Step 4: tournaments and Elo (`tests/training/test_evaluate.py`, then `evaluate.py`)

Tests:

1. **`elo_difference`.** `0.5 → 0`, `0.75 → 191 ± 1`, `0.25 → −191 ± 1`, `1.0 → +inf`, `0.0 → −inf` (no exception: `math.inf`, not a `ZeroDivisionError` or a `log10(0)` crash).
2. **`elo_ratings` on one pairing.** Player 1 beat player 0 30–0–10 over 40 games: rating of 1 is within 2 Elo of `elo_difference(30.5 / 41)` (the virtual draw is in the expectation) and player 0 is exactly 0.
3. **Chains and sweeps.** Three players, 1 over 0 by 30–0–10, 2 over 1 by 30–0–10: ratings are 0, 185 ± 2, 370 ± 2 (the virtual draw turns 75% into 30.5/41; `elo_difference(0.75)` would be 191, and the test says so). A 40–0–0 sweep gives a finite rating under 1000. Players are generation numbers; an unknown anchor raises `ValueError`. **A ladder match corrects a drifting chain:** nine players linked only by neighbour matches at 55% each, so the fit puts player 8 at 272 ± 2 Elo above 0; add one direct 8-versus-0 match of 30–0–10 (185 on its own, with the virtual draw) and player 8's rating lands at 198 ± 2, between the two. This is the test that documents why the ladder exists.
4. **Opponent selection.** `opponents_for(2, EvalConfig())` is `[1, 0]`; `opponents_for(5, EvalConfig())` is `[4, 3, 2]` (the ladder would be negative); `opponents_for(10, EvalConfig())` is `[9, 8, 7, 2]`; with `ladder=2`, generation 3 gives `[2, 1, 0]` (the ladder is already a nearest opponent, no duplicate); with `ladder=None` or `ladder=0` the ladder never appears (never `[4, 3, 2, 5]`). Say in the test what the ladder is for.
5. **Paired openings.** `play_match(UniformEvaluator(), UniformEvaluator(), size=4, games_per_colour=3, simulations=4, rng)`: `games == 6`, `openings` has 3 entries (one per pair) that are distinct cells, and game `i` and game `i + 3` start with `openings[i]` with the colours swapped. On 3x3 with 10 per colour, `openings` has 10 entries, its first 9 are distinct and the tenth repeats the first (the distinct draw is cycled).
6. **Strength shows.** A `TableEvaluator`-free "oracle" evaluator built from the session `Solver` fixture (uniform prior over `best_moves`, value from `value`; the fixture moves from `tests/engine/conftest.py` to `tests/conftest.py` first, since a conftest is visible only to its own subtree) against `UniformEvaluator` on 4x4 with 16 simulations, 4 games per colour: the oracle wins every game it plays as O (4x4 is an O win with perfect play, and every first move loses, so the random opening cannot save X). The test says why the X games are not asserted.
7. **Two evaluators, two trees, two batches.** Spies on both evaluators: on every step the states sent to `a` come only from games where `a` is to move, and the same for `b`. Then the leak test, on one `MatchGame` driven by hand and built from the **empty** 3x3 board (no opening move applied; `play_match` applies one, this test does not): `a` is an evaluator that puts all prior on the first legal cell in board order for *every* state, `b` is uniform, 1 simulation, `b` plays X and therefore moves first. Drive it in this order with `lockstep_step([game.tree], game.mover)` and `play_if_ready`: (1) `b` sets up and searches once, moves; (2) `a`'s tree is created from the new position, `a` sets up and searches once; **check now, before `a` moves**: `a`'s root (`game.trees[a_side].root`) has one child with prior 1 and the rest 0, while `b`'s root (the child `b` chose, expanded by `b`'s own simulation) has uniform priors; (3) `a` moves, both trees `play` it, `b` sets up its new root and searches once; **check again**: `b`'s root children are uniform. Checking right after `b`'s move instead would find `a`'s root unexpanded (the opponent's reply was never visited by `a`'s tree), which is correct behaviour but asserts nothing. Say in the test that a shared tree would have given `a` a root expanded with `b`'s uniform priors.
8. **Determinism.** Same seed, same `MatchResult`.
9. **Self-match.** `play_match(e, e, ...)` with one `UniformEvaluator` instance for both sides runs to completion with two distinct trees per game (`game.trees[X] is not game.trees[O]` once both exist): the trees are keyed by side, so an evaluator playing itself is fine.

Implementation notes: **two trees per game.** With one shared tree, `play` would hand player `a` the subtree that player `b` just expanded with `b`'s priors and backed up with `b`'s values, so `a`'s search would start from the opponent's opinions every ply and the measured strength would be a blend of both networks. Each player's tree is created lazily, from the current position, on that player's first turn; `SearchTree.play` raises on an unexpanded root, and the second player's tree does not exist (and must not be built and left unexpanded) until the first player has moved. From then on every move by either side is applied with `play(move)` to both trees: the mover's root was expanded by its own search, and the opponent's root is the child its own search chose last turn, which was visited and therefore expanded (a visited non-terminal leaf is always expanded; `choose_move` never picks an unvisited child). A new root that is unexpanded after `play` is set up by its owner's evaluator on the next `select`, so nothing evaluated by the other network ever enters a tree. Guard anyway: if a tree's root is unexpanded when the opponent's move arrives, rebuild that tree from the new state instead of calling `play`, and say in a comment why that should not happen. On each step, only the trees of the players to move are stepped: `a`'s go to `a` in one batch, `b`'s to `b`. The opening move is applied with `apply_move` before the `MatchGame` is built. Newton steps for the Bradley-Terry fit, on the Elo scale: `r_i += (400 / ln 10) · (s_i − E_i) / Σ_j n_ij · p_ij · (1 − p_ij)`, where `s_i` is player `i`'s actual points, `E_i = Σ_j n_ij · p_ij` its expected points and `p_ij = 1 / (1 + 10^((r_j − r_i) / 400))`. The `400 / ln 10` (about 174) is the derivative of `p` with respect to an Elo rating; without it the update is the natural-log-scale step and converges about 170 times slower. Update ratings in place as the sweep goes (Gauss–Seidel: each player's step uses the already-updated ratings of the players before it; the all-at-once Jacobi variant oscillates on chains) and stop when the largest change in a sweep is below 0.01 Elo, with a cap of 10,000 sweeps. No library needed. Expected values in the step-4 tests carry a tolerance of ±2 Elo; the 272 and 198 of test 3 were checked with a converged fit.

**Review point:** run `/code-review` (medium) on the branch here.

## Step 5: metrics (`tests/training/test_metrics.py`, then `metrics.py`)

Tests:

1. **Rows append.** Two `write` calls produce two JSON lines; `read_metrics` returns both as dicts with the same keys and values. Nested dicts (the per-opponent scores) survive.
2. **A partial last line is skipped** by `read_metrics` (truncate the file mid-row by hand).
3. **TensorBoard on.** With `tensorboard=True`, an `events.out.tfevents.*` file appears under `directory / "tensorboard"` after `write` and `close`; scalar values only, nested dicts flattened as `scores/vs_gen_003`.
4. **TensorBoard off** writes no `tensorboard/` directory.
5. **Purge on resume.** Write generations 1, 2, 3; reopen with `purge_from=3` and write generation 3 again; reading the whole `tensorboard/` directory back with `EventAccumulator(str(directory / "tensorboard")).Reload()` (the directory, not one file: the reopened writer makes a second event file, and the purge is applied by the reader when it sees the second file's session start) shows one point at step 3 for `loss_total`, not two.

## Step 6: the generation loop (`tests/training/test_run.py`, then `run.py`)

A tiny configuration for every test: 4x4, `NetworkConfig(1, 4, 8)`, `SelfPlayConfig(games=4, parallel=4, simulations=6)`, `TrainConfig(batch_size=8, steps_per_generation=2, buffer_generations=2)`, `EvalConfig(opponents=1, ladder=None, games_per_colour=1, simulations=4)`, `tensorboard=False`, `root=tmp_path`. Each `run` must finish in a few seconds on CPU; report timings if not.

Tests:

1. **End to end.** `generations=2`: the run directory holds `gen_000.pt`, `gen_001.pt`, `gen_002.pt`, `buffer/gen_001.npz`, `buffer/gen_002.npz`, `metrics.jsonl` with two rows (generations 1 and 2), `matches.jsonl` with two rows (1 vs 0, 2 vs 1), `ratings.json` with three ratings and `"0": 0.0`. Each checkpoint loads with `load_checkpoint`, its `configs` rebuild `SearchConfig(**...)`, `SelfPlayConfig(**...)`, `TrainConfig(**...)`, `EvalConfig(**...)` equal to the run's and `RunIdentity(**configs["run"])` has the run's name and seed, and `cli.adapter.load_engine(None, None, root=run_dir)` picks `gen_002.pt` and searches a 4x4 position (this is the plan 03 handshake: training writes what the CLI reads).
2. **Metrics keys.** Every row has exactly: `generation`, `games`, `examples`, `buffer_size`, `mean_game_length`, `draw_rate`, `x_win_rate`, `selfplay_seconds`, `selfplay_steps`, `simulations_per_second`, `train_seconds`, `loss_total`, `loss_policy`, `loss_value`, `eval_seconds`, `elo`, `scores` (a dict `opponent generation → score`, keys as strings after the JSON round trip), `wall_seconds`.
3. **Resume continues.** Run to `generations=2`, then call `run` again with `generations=4`: `gen_001.pt`'s mtime is unchanged, four metrics rows, five ratings, and `gen_004.pt` exists.
4. **Resume is exact.** Run 4 generations in one go with seed 7 into one directory, and 2 + 2 (two `run` calls) with seed 7 into another: `gen_004.pt` weights are `torch.equal` in both, and the two `metrics.jsonl` agree on every non-timing key. This is the test for the RNG and buffer restore together. If it fails for a reason outside the pipeline (a nondeterministic CPU kernel), report it with the differing tensor before weakening it.
5. **Commit marker.** After a 2-generation run, delete `gen_002.pt` and resume to 3: the stale rows for generation 2 are dropped, generation 2 is re-run, and the final files are consistent (three rows, four ratings). The buffer directory ends with exactly `gen_002.npz` and `gen_003.npz` (`K = 2`), and `gen_002.npz`'s row count equals that generation's `examples` metric: the stale file from the first attempt was deleted on resume, not appended to.
6. **Config mismatch.** Resume with a different `learning_rate` raises `ValueError` mentioning `learning_rate`; so does a different `seed` or `NetworkConfig`; a different `generations` or `tensorboard` is fine.
7. **Missing buffer.** Delete the `buffer/` directory after a 2-generation run, resume: a warning is printed (capsys) and the run continues. Resuming a run that was interrupted during generation 1 (only `gen_000.pt` exists, no `buffer/`) prints no warning.
8. **The ladder plays.** With `EvalConfig(opponents=1, ladder=2, games_per_colour=1, simulations=4)` and `generations=3`, `matches.jsonl` holds rows 1 vs 0, 2 vs 1, 2 vs 0, 3 vs 2, 3 vs 1, and the `scores` dict of row 3 has keys `"2"` and `"1"` (JSON object keys are strings, as in `ratings.json`).

Implementation notes: `run` builds the network on `device` (weights seeded on CPU by `torch.manual_seed` before `.to(device)`), the trainer, the evaluator, the buffer; then loops `for g in range(start, generations + 1)`. Opponents for generation `g` are `opponents_for(g, config.eval)`; each is `load_checkpoint(path, device).network` wrapped in a `NetworkEvaluator`. The current side of the match is the live network's evaluator. `elo_ratings` is fed every row of `matches.jsonl`. `wall_seconds` is the generation's total.

## Step 7: scripts (`scripts/train.py`, `scripts/evaluate.py`)

Not test-driven, but seeded and self-describing like every script ([scripts/README.md](../../../scripts/README.md)).

- `scripts/train.py --name six-a --size 6 --generations 20 --seed 0 --device auto --root runs --no-tensorboard` plus one flag per provisional constant (`--games`, `--parallel`, `--simulations`, `--batch-size`, `--steps`, `--lr`, `--weight-decay`, `--buffer-generations`, `--blocks`, `--filters`, `--eval-games`, `--eval-opponents`, `--eval-ladder`, with `--eval-ladder 0` turning the ladder off). Builds a `RunConfig`, prints it, calls `run`, prints each generation's metrics row as one line as it completes. Resume is automatic from the directory.
- `scripts/evaluate.py --a runs/six-a/gen_020.pt --b runs/six-a/gen_000.pt --games 20 --simulations 100 --seed 0 --device cpu`. `--b uniform` pits a checkpoint against the uniform search (the no-model CLI). The board size comes from checkpoint `a`; a size mismatch is an error. Prints wins, draws, losses per colour, the score, and the Elo gap as `elo_difference((points + 0.5) / (games + 1))`, the same virtual draw `elo_ratings` applies, so a 40–0 sweep against `uniform` prints a finite number (about +760) rather than crashing.
- `scripts/README.md` gains a section for each, and the "Later" paragraph goes.

**Review point:** run `/code-review` (medium) on the branch here, before the run.

## Step 8: the 6x6 run

1. **Throughput first.** Self-play rate on `cpu` and on `mps` with the provisional 6x6 configuration, one generation each (`--generations 1`, two run names): `simulations_per_second` from the metrics row, as a two-row Markdown table for `engineering.md` ("Measured") and the PR. Pick the faster device for the run.
2. **The run.** `uv run python scripts/train.py --name six-a --size 6 --generations 20 --seed 0`, with `tensorboard --logdir runs/six-a/tensorboard` open. Stop it once with Ctrl-C around generation 5 and resume with the same command; the PR notes that it continued from the right generation.
3. **In the PR description:** the throughput table; the metrics table (generation, examples, mean game length, draw rate, loss total/policy/value, Elo); a screenshot or export of the TensorBoard loss and Elo curves; `scripts/evaluate.py` output for `gen_020.pt` against `gen_000.pt` and against `uniform`; the ladder scores per generation (are they informative, or sweeps?); the wall time per generation. One paragraph of reading: did the policy loss fall, did the Elo rise, did game length grow (the wall-building prediction in [open-questions.md](../../design/open-questions.md#follow-up-safe-moves-over-a-game)), which constants look wrong.

## Step 9: documentation

- `README.md`: the status line becomes "the training pipeline (step 4) is built; the first 6x6 run is recorded in [open-questions.md](...)". Quickstart gains `uv run python scripts/train.py --size 6` with one sentence.
- `docs/architecture.md`: `training` row "Built (step 4)"; "Designed but not built" shrinks to the GUI; the dependency rule gains nothing.
- `docs/design/engineering.md`: module layout marks every `training/` module built and adds `metrics.py`, `scripts/train.py`, `scripts/evaluate.py`; drops `scripts/plot.py` (out of scope, TensorBoard reads the files); "Configuration and checkpoints" gains the run directory layout, `runs/`, and the `run` entry in `configs`; "Devices and determinism" says augmentation is drawn from the NumPy generator (it currently says torch); "Measured" gains the self-play throughput rows.
- `docs/design/training.md`: "Augmentation" says per example; "Evaluation" records the tournament protocol, the ladder opponent and the Elo fit as decided (dated, linking here), including the chain-drift argument; "Logging" names the files; "Replay buffer" notes it is persisted for resume.
- `docs/design/open-questions.md`: the tournament protocol and the generation constants move from Open to "Provisional, set in plan 04"; a new "Results" subsection holds the 6x6 run's headline numbers (game length, draw rate, Elo after 20 generations) and what they say about the temperature cutoff and `K`.
- `docs/design/upgrades.md`: add "exclude norms and biases from weight decay" under Network (low priority).
- `scripts/README.md`: as in step 7.
- `src/ox_zero/training/__init__.py`: the docstring stops saying "being built".

## Definition of done

- `uv run pytest` is green; the new tests run on CPU in well under a minute in total.
- A 6x6 run of 20 generations completed, was interrupted and resumed once, and its numbers are in the PR as listed in step 8. Elo rising is the goal, not the gate.
- `scripts/evaluate.py` runs the final checkpoint against generation 0 and against `uniform`.
- `cli.adapter.load_engine` loads a checkpoint written by the run (step 6, test 1), and `git diff main -- src/ox_zero/engine src/ox_zero/cli` is empty.
- All three `/code-review` runs happened and their findings were addressed or explicitly deferred in the PR description.
- A pull request from `feat/training-pipeline` to `main`. On merge, this plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- `SearchTree` lacks something the lockstep driver or the tournament needs (for example a way to know whether the next `select` is a root setup). Do not modify `ox_zero.engine` on this branch; report what is missing.
- The resume-exactness test (step 6, test 4) fails because of nondeterminism rather than a bug. Report the differing tensor; do not loosen the test.
- A generation on 6x6 takes more than about 5 minutes on the faster device. Report the per-phase seconds before changing any constant.
- `weights_only=True` rejects a config field this plan adds to `configs` (everything is ints, floats, bools, or `None`, so it should not). Report; do not switch to `weights_only=False`.
- TensorBoard fails to import or write on this machine. Report; do not make it optional on your own.
- Draw rate above about 20% in the 6x6 run. That is the trigger for the win/draw/loss head in [network.md](../../design/network.md#value-head); record it, do not act on it here.
- Any design-level choice not listed under Decisions. Stop and ask; do not decide it in code.
