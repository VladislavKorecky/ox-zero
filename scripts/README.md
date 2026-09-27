# scripts

Runners that are not part of the package. They import `ox_zero` but nothing imports them. Run everything with `uv run python scripts/...` from the repository root.

Scripts are exempt from test-driven development, but each one must be deterministic (seeded) and print the parameters it ran with, so its results can be regenerated.

## experiments/

Measurements of the game itself, made before any engine code exists. They feed the open constants in [docs/design/open-questions.md](../docs/design/open-questions.md), and their results are recorded there. See [plan 01](../docs/plans/archive/01-game-experiments.md).

| Script | What it measures | Runtime (defaults) |
|---|---|---|
| `random_play.py` | Game length, outcome rates, and legal moves per decision under random and one-ply greedy play, for boards 3x3 to 12x12 | ~4 min |
| `solve_boards.py` | Exact value, optimal first moves, positions searched, and one optimal line for the empty 3x3 and 4x4 boards, using `ox_zero.game.Solver` | ~3 s |

```bash
uv run python scripts/experiments/random_play.py           # all sizes, both policies, seed 0
uv run python scripts/experiments/random_play.py --sizes 6 --games 500 --policy greedy
uv run python scripts/experiments/solve_boards.py          # 3x3 and 4x4
```

Pass `--help` to either script for every flag.

### results/

Committed outputs of the default runs. Each script writes two files:

- `<name>.json`: the statistics and the parameters that produced them. Rerunning with the defaults regenerates it byte-for-byte.
- `<name>.timing.json`: wall-clock times and throughput. These depend on the machine and change from run to run.

## Later

Launching training, evaluating checkpoints, and similar runners will live here once the engine and training pipeline exist.
