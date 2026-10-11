# Experiment 01: testbed and noise floor

| | |
|---|---|
| Status | Arms and commands written; **results and verdict pending** (the calibration compute of [plan 06](../plans/06-experiment-harness.md), step 6, has not run). |
| Plan | [Plan 06](../plans/06-experiment-harness.md) |

## Hypothesis

Calibration, not a test of an upgrade. It answers four questions the later experiments depend on:

1. **Noise floor.** How far apart do the final benchmark ratings of the *same configuration* land under different seeds? Any later difference smaller than this is luck.
2. **Benchmark repeatability.** Does the same pairing give the same games twice, and with the sides swapped? (The one-measurement-per-pairing rule assumes so.)
3. **Cost.** Seconds per generation at 6x6 and 8x8, and so whether a two-arm, three-seed experiment fits overnight.
4. **Testbed.** 6x6 or 8x8: the board on which upgrades can show.

## What changes

Nothing about training: every run uses the provisional constants of [open-questions.md](../design/open-questions.md) (the cutoff table, `α = 11 / S²`, 128 games, 100 simulations, 4x64 network); only the board size and seed vary.

## Arms and seeds

| Run | Board | Seed | Generations | Role |
|---|---|---|---|---|
| `six-a` | 6x6 | 0 | 40 | 6x6 reference run; supplies the 6x6 pool (generations 5, 10, 15) |
| `eight-a` | 8x8 | 0 | 30 | 8x8 reference run; supplies the 8x8 pool (generations 5, 10, 20) |
| `eight-b`, `eight-c`, `eight-d` | 8x8 | 1, 2, 3 | 30 | measured seeds, if 8x8 passes criterion (b) |
| `six-b`, `six-c`, `six-d` | 6x6 | 1, 2, 3 | 40 | measured seeds, otherwise |

The reference run's own curve is reported, but the criteria and the seed spread are computed over three *other* seeds: a network measured against its own earlier generations carries a lineage bias.

## Criteria

The testbed is the board that passes all three, with 8x8 preferred:

- (a) **Headroom**: each measured seed's benchmark rating at its final generation exceeds its rating at the halfway generation by more than the seed spread (generations 20 and 40 on 6x6, 15 and 30 on 8x8).
- (b) **Cadence**: one baseline run takes at most about 90 minutes.
- (c) **Measurability**: paired-colour scores against the near-strength references are not pinned at 0.5.

## Commands

See [plan 06, step 6](../plans/06-experiment-harness.md#step-6-calibration) for the sequence. The shapes:

```bash
uv run python scripts/train.py --name six-a --size 6 --generations 40 --seed 0 --device cpu
uv run python scripts/train.py --name eight-a --size 8 --generations 30 --seed 0 --device <cpu|mps>
uv run python scripts/benchmark.py --model runs/six-a/gen_020.pt
uv run python scripts/compare_runs.py runs/six-a runs/eight-a
```

## Cost, results, verdict

*Pending.*

## Design changes

*Pending.*
