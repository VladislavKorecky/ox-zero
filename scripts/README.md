# scripts

Runners that are not part of the package. They import `ox_zero` but nothing imports them. Run everything with `uv run python scripts/...` from the repository root.

Scripts are exempt from test-driven development, but each one must be deterministic (seeded) and print the parameters it ran with, so its results can be regenerated.

## experiments/

These measure the *game*; training experiments are recorded in [docs/experiments/](../docs/experiments/README.md). Measurements of the game itself, made before any engine code exists. They feed the open constants in [docs/design/open-questions.md](../docs/design/open-questions.md), and their results are recorded there. See [plan 01](../docs/plans/archive/01-game-experiments.md).

| Script | What it measures | Runtime (defaults) |
|---|---|---|
| `random_play.py` | Game length, outcome rates, and legal moves per decision under random and one-ply greedy play, for boards 3x3 to 12x12 | ~4 min |
| `solve_boards.py` | Exact value, optimal first moves, positions searched, one optimal line, and the length distribution of sampled optimal games for the empty 3x3 and 4x4 boards, using `ox_zero.game.Solver` | ~25 s |
| `safe_moves.py` | How many *safe* moves (not losing on the spot) remain at each move number under greedy play, for 6x6, 8x8 and 12x12 | ~4 min |

```bash
uv run python scripts/experiments/random_play.py           # all sizes, both policies, seed 0
uv run python scripts/experiments/random_play.py --sizes 6 --games 500 --policy greedy
uv run python scripts/experiments/solve_boards.py          # 3x3 and 4x4
uv run python scripts/experiments/safe_moves.py            # 6x6, 8x8, 12x12, 500 greedy games each
```

`safe_moves.py` imports the greedy policy from `random_play.py`; run it from the repository root as shown.

Pass `--help` to either script for every flag.

### results/

Committed outputs of the default runs. Each script writes two files:

- `<name>.json`: the statistics and the parameters that produced them. Rerunning with the defaults regenerates it byte-for-byte.
- `<name>.timing.json`: wall-clock times and throughput. These depend on the machine and change from run to run.

## bench_search.py

Speed of the engine's search, not a measurement of the game: simulations per second, evaluator calls per second, and the share of time spent in the evaluator, for one board size, evaluator (`network` with random weights, or `uniform`) and device. It prints a Markdown row for pasting into a PR. See [plan 02](../docs/plans/archive/02-engine-search-network.md), step 8.

```bash
uv run python scripts/bench_search.py --size 12 --device mps   # defaults: 6x6, 800 simulations, 4x64 network, 3 repeats
```

## make_checkpoint.py

Writes a checkpoint with random (seeded) weights in the real format. Its weights are random and exist so that `--model` can be exercised before training. The default output, `checkpoints_random/gen_000.pt`, is deliberately outside `checkpoints/`: the CLI loads the newest checkpoint there automatically, and a random network would replace the no-model search on every run. See [plan 03](../docs/plans/archive/03-cli-adapter.md), step 5.

```bash
uv run python scripts/make_checkpoint.py      # defaults: 12x12, 4x64 network, seed 0
uv run ox-zero analyze 5,5 6,6 5,6 --model checkpoints_random/gen_000.pt
```

## train.py

Trains a network by AlphaZero self-play: a thin wrapper around `ox_zero.training.run.run`. It prints the flags and the resulting `RunConfig`, then one line per generation as each is committed (game length, draw and X-win rates, self-play speed, losses, Elo and the tournament scores, wall time). Every default is a provisional 6x6 constant from [plan 04](../docs/plans/archive/04-training-pipeline.md) ("Provisional constants for 6x6"); `--help` lists one flag per constant. `--simulations` sets both the self-play and the tournament search; `--eval-simulations` decouples the tournament's budget from it; `--eval-ladder 0` turns the ladder opponent off. The search constants are flags: `--temperature-cutoff`, `--c-init`, `--c-base`, `--dirichlet-alpha`, `--dirichlet-epsilon` (unset means the `SearchConfig` default, so a run without them is identical to before; a changed value is a new run name). `benchmarks` is a reserved run name.

The run lives in `runs/<name>/` (default name `<size>x<size>-seed<seed>`; `--root` changes the parent): checkpoints `gen_NNN.pt`, `metrics.jsonl`, `matches.jsonl`, `ratings.json`, `buffer/` and `tensorboard/` (unless `--no-tensorboard`). Not `checkpoints/`: the CLI auto-loads the newest checkpoint of the requested size from there, so copy or symlink a checkpoint under `checkpoints/` to use it with `--size N`, or pass it directly with `--model`.

Resume is automatic. Ctrl-C at any moment leaves a valid run (the checkpoint is written last, so an interrupted generation is re-run); rerun the same command to continue, or raise `--generations` to extend a finished run. Any other changed flag is refused: a new configuration needs a new `--name`. A directory that holds an old run's `metrics.jsonl`, `matches.jsonl`, `ratings.json` or buffer files but no `gen_*.pt` is refused too (nothing is deleted): remove those files or pick another `--name`. `--games` and `--eval-opponents` must be at least 1.

```bash
uv run python scripts/train.py --size 6                     # runs/6x6-seed0, 20 generations, --device auto
uv run python scripts/train.py --name six-a --size 6 --device cpu     # 6x6: cpu is faster
uv run python scripts/train.py --name eight-a --size 8 --device mps   # 8x8: mps is 2x faster
tensorboard --logdir runs/six-a/tensorboard
```

## evaluate.py

For one ad-hoc pairing (a quick look, any two players). To *rate* a checkpoint, use `benchmark.py`. Plays checkpoint `--a` against checkpoint `--b`, or against `--b uniform` (the CLI's no-model search), with the training tournament's match: noise-free search, `--games` random opening cells each played with both colours. The board size comes from `a`; a size mismatch is an error. Prints wins, draws and losses for `a` per colour, the score, and the Elo gap `elo_difference((points + 0.5) / (games + 1))`, with the same virtual draw as the run's Elo fit, so a 40-0 sweep prints about +760 rather than infinity.

```bash
uv run python scripts/evaluate.py --a runs/six-a/gen_020.pt --b runs/six-a/gen_000.pt   # 20 openings x 2 colours, 100 simulations, cpu
uv run python scripts/evaluate.py --a runs/six-a/gen_020.pt --b uniform
```

## benchmark.py

Rates one checkpoint on the shared Elo scale of its board size: plays it against every reference of `runs/benchmarks/<S>x<S>/pool/` plus `uniform@100` and `uniform@400` (every cell of the board as the opening, once per colour, deterministic evaluation search), appends the games to `matches.jsonl`, refits one Bradley-Terry rating per player over the whole log anchored at `uniform@100` = 0, and prints the candidate's rating with a 95% bootstrap interval, the score against each reference (`≥`/`≤` marks a sweep), and the pool's ratings. A pairing already in the log is skipped, so reruns play only what is missing and Ctrl-C loses at most the pairing in progress. CPU only for the official log (`--root` for scratch work on another device). `--tag NAME` with `--c-init`/`--c-base` rates the same weights under a non-default search as a separate player `<run>/gen_NNN@NAME`. Method: `ox_zero.training.benchmark`; pools: [docs/experiments/pools.md](../docs/experiments/pools.md).

```bash
uv run python scripts/benchmark.py --model runs/six-a/gen_020.pt
uv run python scripts/benchmark.py --model runs/x/gen_020.pt --tag c3 --c-init 3.0
```

## compare_runs.py

One Markdown row per run from `metrics.jsonl` (the last generation, or `--at N`), with the benchmark rating of that exact checkpoint when it exists; or `--key loss_value` for one metric per generation, one column per run. Directories without `metrics.jsonl` are skipped with a notice, so `compare_runs.py runs/*` works.

```bash
uv run python scripts/compare_runs.py runs/six-a runs/six-b
uv run python scripts/compare_runs.py runs/six-a runs/six-b --key loss_value
```
