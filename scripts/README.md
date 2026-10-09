# scripts

Runners that are not part of the package. They import `ox_zero` but nothing imports them. Run everything with `uv run python scripts/...` from the repository root.

Scripts are exempt from test-driven development, but each one must be deterministic (seeded) and print the parameters it ran with, so its results can be regenerated.

## experiments/

Measurements of the game itself, made before any engine code exists. They feed the open constants in [docs/design/open-questions.md](../docs/design/open-questions.md), and their results are recorded there. See [plan 01](../docs/plans/archive/01-game-experiments.md).

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

Writes a checkpoint with random (seeded) weights in the real format. Its weights are random and exist so that `--model` can be exercised before training. The default output, `checkpoints_random/gen_000.pt`, is deliberately outside `checkpoints/`: the CLI loads the newest checkpoint there automatically, and a random network would replace the no-model search on every run. See [plan 03](../docs/plans/03-cli-adapter.md), step 5.

```bash
uv run python scripts/make_checkpoint.py      # defaults: 12x12, 4x64 network, seed 0
uv run ox-zero analyze 5,5 6,6 5,6 --model checkpoints_random/gen_000.pt
```

## Later

Launching training, evaluating checkpoints against each other, and similar runners will live here once the training pipeline exists.
