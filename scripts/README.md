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

Writes a checkpoint with random (seeded) weights in the real format. Its weights are random and exist so that `--model` can be exercised before training. The default output, `checkpoints_random/gen_000.pt`, is deliberately outside `checkpoints/`: the CLI loads the newest checkpoint there automatically, and a random network would replace the no-model search on every run. See [plan 03](../docs/plans/archive/03-cli-adapter.md), step 5.

```bash
uv run python scripts/make_checkpoint.py      # defaults: 12x12, 4x64 network, seed 0
uv run ox-zero analyze 5,5 6,6 5,6 --model checkpoints_random/gen_000.pt
```

## train.py

Trains a network by AlphaZero self-play: a thin wrapper around `ox_zero.training.run.run`. It prints the flags and the resulting `RunConfig`, then one line per generation as each is committed (game length, draw and X-win rates, self-play speed, losses, Elo and the tournament scores, wall time). Every default is a provisional 6x6 constant from [plan 04](../docs/plans/04-training-pipeline.md) ("Provisional constants for 6x6"); `--help` lists one flag per constant. `--simulations` sets both the self-play and the tournament search; `--eval-ladder 0` turns the ladder opponent off.

The run lives in `runs/<name>/` (default name `<size>x<size>-seed<seed>`; `--root` changes the parent): checkpoints `gen_NNN.pt`, `metrics.jsonl`, `matches.jsonl`, `ratings.json`, `buffer/` and `tensorboard/` (unless `--no-tensorboard`). Not `checkpoints/`: the CLI auto-loads from there and is 12x12 only, so copy or symlink a checkpoint there to use it.

Resume is automatic. Ctrl-C at any moment leaves a valid run (the checkpoint is written last, so an interrupted generation is re-run); rerun the same command to continue, or raise `--generations` to extend a finished run. Any other changed flag is refused: a new configuration needs a new `--name`.

```bash
uv run python scripts/train.py --size 6                     # runs/6x6-seed0, 20 generations, --device auto
uv run python scripts/train.py --name six-a --device mps    # same constants, named run
tensorboard --logdir runs/six-a/tensorboard
```

## evaluate.py

Plays checkpoint `--a` against checkpoint `--b`, or against `--b uniform` (the CLI's no-model search), with the training tournament's match: noise-free search, `--games` random opening cells each played with both colours. The board size comes from `a`; a size mismatch is an error. Prints wins, draws and losses for `a` per colour, the score, and the Elo gap `elo_difference((points + 0.5) / (games + 1))`, with the same virtual draw as the run's Elo fit, so a 40-0 sweep prints about +760 rather than infinity.

```bash
uv run python scripts/evaluate.py --a runs/six-a/gen_020.pt --b runs/six-a/gen_000.pt   # 20 openings x 2 colours, 100 simulations, cpu
uv run python scripts/evaluate.py --a runs/six-a/gen_020.pt --b uniform
```
