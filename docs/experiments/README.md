# Training experiments

The log of **training experiments**: runs of `scripts/train.py` that test a hypothesis, measured on the shared benchmark scale. Not to be confused with [`scripts/experiments/`](../../scripts/experiments/), which measures the *game* itself (random play, safe moves, exact solutions; [plan 01](../plans/archive/01-game-experiments.md)) and keeps its results under `scripts/experiments/results/`. Here the runs live in `runs/` and only their findings are committed.

Three kinds of document, kept apart: [design](../design/README.md) records decisions (the settings), [plans](../plans/README.md) record how code is built, experiments record measurements and verdicts. A verdict that changes a decision edits the design doc, which links back to the experiment.

## Protocol

1. **One hypothesis per experiment.** Arms differ in exactly one thing: a flag of `scripts/train.py` or an upgrade toggle.
2. **At least two seeds per arm**, three when the first two disagree by more than about 60 Elo on 8x8 (the noise floor of [experiment 01](01-testbed-and-noise-floor.md#verdict): seeds spread about 50 Elo, and game sampling alone puts about ±62 on the difference of two ratings). One seed cannot tell a difference from luck. An arm difference under about 60 Elo is not resolvable with three seeds per arm on the current pool.
3. **Run** each arm and seed under its own name (`<arm>-<seed>`); a changed flag is a new run, resume refuses it.
4. **Rate** every final checkpoint with `scripts/benchmark.py`, and with `--tag` and the search flags when the arm changes the search. The benchmark is the verdict; the in-run curves (`scripts/compare_runs.py`, TensorBoard) are the explanation.
5. **Tabulate** with `scripts/compare_runs.py runs/<arm>-*`, and paste the table.
6. **Record** the experiment in `NN-slug.md` (below), add it to the index, and change the design docs the verdict touches.

Every recorded rating names the log state it came from (the row count of `runs/benchmarks/<S>x<S>/matches.jsonl` and the date) and is accompanied by the per-reference scores, which never change once logged: a joint refit moves every rating a little as later benchmarks land, the scores do not. Every figure is a 95% interval unless it says otherwise. The bootstrap interval measures game-sampling noise only; the seed-to-seed spread of the baseline (experiment 01) is the other, larger component.

## An experiment file

`NN-slug.md`, with these sections:

- **Hypothesis**: what is expected to change, and why.
- **What changes**: the flags or the diff.
- **Arms and seeds**: names, seeds, board size.
- **Commands**: exact, copy-pasteable.
- **Cost**: wall time per run, device.
- **Results**: the `compare_runs.py` table; benchmark ratings with intervals; per-reference scores; log row count and date.
- **Verdict**: adopt / reject / inconclusive, with the numbers that decide it.
- **Design changes**: which design docs changed because of it.

Commands template:

```bash
uv run python scripts/train.py --name <arm>-s0 --size <S> --generations <N> --seed 0 --device <cpu on 6x6, mps on 8x8> <flag>
uv run python scripts/benchmark.py --model runs/<arm>-s0/gen_<N>.pt
uv run python scripts/compare_runs.py runs/<arm>-s*
```

## Index

| # | Experiment | Verdict | Date |
|---|---|---|---|
| [01](01-testbed-and-noise-floor.md) | Testbed and noise floor | 8x8 is the testbed; noise floor about 50 Elo | 2026-10-12 |
| [02](02-benchmark-openings.md) | How informative is each benchmark opening | Edges and corners most informative; trimming openings cannot shrink the interval | 2026-10-12 |

The benchmark's reference pools are recorded in [pools.md](pools.md).
