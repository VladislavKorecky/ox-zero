# Experiment 01: testbed and noise floor

| | |
|---|---|
| Status | **Complete** (2026-10-12): calibration compute of [plan 06](../plans/06-experiment-harness.md), step 6. |
| Plan | [Plan 06](../plans/06-experiment-harness.md) |
| Verdict | **8x8 is the testbed**; noise floor about 50 Elo between seeds at generation 30. |

## Hypothesis

Calibration, not a test of an upgrade. It answers four questions the later experiments depend on:

1. **Noise floor.** How far apart do the final benchmark ratings of the *same configuration* land under different seeds? Any later difference smaller than this is luck.
2. **Benchmark repeatability.** Does the same pairing give the same games twice, and with the sides swapped? (The one-measurement-per-pairing rule assumes so.)
3. **Cost.** Seconds per generation at 6x6 and 8x8, and so whether a two-arm, three-seed experiment fits overnight.
4. **Testbed.** 6x6 or 8x8: the board on which upgrades can show.

## What changes

Nothing about training: every run uses the provisional constants of [open-questions.md](../design/open-questions.md) (the cutoff table, `α = 11 / S²`, 128 games, 100 simulations, 4x64 network); only the board size, seed and device vary.

## Arms and seeds

| Run | Board | Seed | Generations | Device | Role |
|---|---|---|---|---|---|
| `six-a` | 6x6 | 0 | 40 | cpu | 6x6 reference run; supplies the 6x6 pool (generations 5, 10, 15) |
| `eight-a` | 8x8 | 0 | 30 | mps | 8x8 reference run; supplies the 8x8 pool (generations 5, 10, 20) |
| `eight-b`, `eight-c`, `eight-d` | 8x8 | 1, 2, 3 | 30 | mps | measured seeds (8x8 passed criterion (b)) |
| `probe8-cpu`, `probe8-mps` | 8x8 | 0 | 1 | cpu, mps | one-generation device probes |

The reference run's own curve is reported, but the criteria and the seed spread are computed over the three *other* seeds: a network measured against its own earlier generations carries a lineage bias.

## Criteria

The testbed is the board that passes all three, with 8x8 preferred:

- (a) **Headroom**: each measured seed's benchmark rating at its final generation exceeds its rating at the halfway generation by more than the seed spread (generations 20 and 40 on 6x6, 15 and 30 on 8x8).
- (b) **Cadence**: one baseline run takes at most about 90 minutes.
- (c) **Measurability**: paired-colour scores against the near-strength references are not pinned at 0.5.

## Commands

As run, in this order (2026-10-11, the author's 8 GB Apple Silicon laptop, 4 performance + 4 efficiency cores):

```bash
# 1. 6x6 pool: copy six-a generations 5, 10, 15 to runs/benchmarks/6x6/pool/six-a-gen_NNN.pt, then
uv run python scripts/benchmark.py --model runs/benchmarks/6x6/pool/six-a-gen_005.pt   # then _010, _015
uv run python scripts/benchmark.py --model runs/six-a/gen_020.pt
# 2. repeatability and side swap, in a scratch root holding a copy of the pool
uv run python scripts/benchmark.py --root <scratch> --model runs/six-a/gen_020.pt
uv run python scripts/benchmark.py --root <scratch> --model runs/benchmarks/6x6/pool/six-a-gen_010.pt
# 3. six-a to 40
uv run python scripts/train.py --name six-a --size 6 --generations 40 --seed 0 --device cpu
uv run python scripts/benchmark.py --model runs/six-a/gen_030.pt                       # and gen_040
# 4. 8x8 probes, reference run, pool
uv run python scripts/train.py --name probe8-cpu --size 8 --generations 1 --seed 0 --device cpu --no-tensorboard
uv run python scripts/train.py --name probe8-mps --size 8 --generations 1 --seed 0 --device mps --no-tensorboard
uv run python scripts/train.py --name eight-a --size 8 --generations 30 --seed 0 --device mps
# copy eight-a generations 5, 10, 20 to runs/benchmarks/8x8/pool/eight-a-gen_NNN.pt, benchmark each, then
uv run python scripts/benchmark.py --model runs/eight-a/gen_015.pt                     # and gen_030
# 5. measured seeds, the three in parallel
uv run python scripts/train.py --name eight-b --size 8 --generations 30 --seed 1 --device mps   # c: seed 2, d: seed 3
uv run python scripts/benchmark.py --model runs/eight-b/gen_015.pt                     # and gen_030, for b, c, d
```

Deviations from the plan's sequence, to fit the night: `six-a`'s generations 21–40 trained while the step 1 and step 2 benchmarks played (so its late per-generation times carry some contention, about +10%); the three measured seeds trained in parallel with each other and with the 8x8 benchmarks (their wall times are not cadence measurements). `eight-a`, the run criterion (b) is decided from, ran alone.

## Cost

**Device probe (8x8, generation 1).** The GPU wins on 8x8, unlike 6x6 ([engineering.md](../design/engineering.md#measured-2026-10-10-self-play-throughput)):

| Device | Simulations/s | Self-play s | Train s | Eval s | Wall s |
|---|---|---|---|---|---|
| cpu | 2,435 | 34.4 | 9.2 | 16.1 | 59.7 |
| mps | 5,032 | 16.7 | 2.2 | 12.0 | 30.8 |

**Per run.**

| Run | Device | Generations | Wall time | Mean s/gen from gen 8 | Slowest gen | Phases from gen 8 (self-play / train / tournament s) |
|---|---|---|---|---|---|---|
| `six-a` | cpu | 40 | 51 min (26 for 1–20; 25 for 21–40, partly contended) | 84 | 110 | 28 / 6 / 50 |
| `eight-a` | mps | 30 | **84.7 min** | 202 | 244 | 68 / 1 / 133 |
| `eight-b/c/d` | mps, 3 in parallel | 30 | 116–126 min each, three at once and sharing the CPU with the 8x8 benchmarks (not a cadence measurement) | | | |

- **The tournament is two thirds of an 8x8 generation** (133 of 202 s). The run's own tournament is monitoring, not the verdict, since the benchmark rates the result; it is the obvious lever if the cadence ever needs to shrink (fewer `--eval-opponents` or `--eval-games`). Recorded, not acted on.
- **Benchmark cost.** A 6x6 benchmark (5 pairings, 72 games each) takes about 1.5 minutes on `cpu`; an 8x8 one (5 pairings, 128 games each) about 14 minutes while training shares the machine, about 8 minutes with the machine to itself (the three `gen_030` benchmarks: 476–512 s).
- **Running three seeds in parallel** on one `mps` device took 116–126 minutes each against 85 for `eight-a` alone: about twice the throughput of running them one after another. With that, a two-arm, three-seed 8x8 experiment (six runs) is two parallel batches of about 2 hours plus six final benchmarks of about 8 minutes (twelve if the halfway generations are rated too), roughly 5 hours: it fits a night.

## Results

### Benchmark repeatability (step 2)

Into a scratch root with a copy of the 6x6 pool: `six-a/gen_020` benchmarked again gave all five rows identical to the official log apart from `date`; `six-a/gen_010` benchmarked there before `gen_005` played the (gen_010, gen_005) pairing with the sides the other way round, and the row equals the official one with `a`/`b`, wins/losses and `as_x`/`as_o` exchanged; its other three rows were identical too. **Nine of nine rows reproduced.** This held while `six-a` trained on the same CPU at the same time, so it is not sensitive to load. The one-measurement-per-pairing rule stands; the official log was untouched (14 rows before and after).

### 6x6

Log state: `runs/benchmarks/6x6/matches.jsonl`, 24 rows, 2026-10-11.

| Player | Elo [95%] |
|---|---|
| `uniform@400` | −30 [−83, +25] |
| `uniform@100` | 0 (anchor) |
| `six-a/gen_005` (pool) | +60 [+16, +106] |
| `six-a/gen_010` (pool) | +140 [+95, +180] |
| `six-a/gen_015` (pool) | +155 [+109, +199] |
| `six-a/gen_020` | +156 [+105, +204] |
| `six-a/gen_030` | +238 [+186, +287] |
| `six-a/gen_040` | +282 [+226, +335] |

Per-reference scores of the measured `six-a` generations (score, then the candidate's W–D–L as X and as O):

| Candidate | uniform@100 | uniform@400 | gen_005 | gen_010 | gen_015 |
|---|---|---|---|---|---|
| `six-a/gen_020` | 0.736 (18–0–18, 35–0–1) | 0.750 (22–0–14, 32–0–4) | 0.625 (9–0–27, 36–0–0) | 0.514 (2–0–34, 35–0–1) | 0.500 (3–0–33, 33–0–3) |
| `six-a/gen_030` | 0.833 (22–4–10, 36–0–0) | 0.819 (23–0–13, 36–0–0) | 0.736 (17–0–19, 36–0–0) | 0.597 (10–0–26, 33–0–3) | 0.639 (12–0–24, 34–0–2) |
| `six-a/gen_040` | 0.722 (16–0–20, 36–0–0) | 0.778 (20–0–16, 36–0–0) | 0.778 (20–0–16, 36–0–0) | 0.778 (21–0–15, 35–0–1) | 0.806 (22–0–14, 36–0–0) |

`six-a` kept learning past generation 20 (+156 → +282), but **near-strength 6x6 pairings are pinned at 0.5**: between networks of similar strength O wins almost every game, so a paired-colour score is X-wins-for-one-side against X-wins-for-the-other and sits at 0.500–0.514 (`gen_020` against `gen_010` and `gen_015`). Only a clearly stronger network wins as X. On 6x6 the score against a near-strength reference therefore measures little, which is criterion (c).

### 8x8

Log state: `runs/benchmarks/8x8/matches.jsonl`, 49 rows, 2026-10-12.

| Player | Elo [95%] |
|---|---|
| `eight-a/gen_005` (pool) | −204 [−231, −172] |
| `eight-a/gen_010` (pool) | −123 [−146, −95] |
| `uniform@100` (anchor) | 0 |
| `eight-a/gen_015` | +25 [−5, +52] |
| `eight-b/gen_015` | +49 [+17, +80] |
| `eight-c/gen_015` | +63 [+31, +94] |
| `uniform@400` | +71 [+42, +98] |
| `eight-d/gen_015` | +92 [+56, +123] |
| `eight-a/gen_020` (pool) | +168 [+142, +191] |
| `eight-d/gen_030` | +308 [+264, +347] |
| `eight-a/gen_030` | +331 [+286, +373] |
| `eight-c/gen_030` | +334 [+289, +376] |
| `eight-b/gen_030` | +360 [+312, +401] |

Per-reference scores of every non-pool candidate (score, then the candidate's W–D–L as X and as O):

| Candidate | uniform@100 | uniform@400 | gen_005 | gen_010 | gen_020 |
|---|---|---|---|---|---|
| `eight-a/gen_015` | 0.445 (14–0–50, 43–0–21) | 0.328 (12–0–52, 30–0–34) | 0.930 (55–0–9, 64–0–0) | 0.898 (57–0–7, 58–0–6) | 0.164 (2–0–62, 19–0–45) |
| `eight-a/gen_030` | 0.820 (41–0–23, 64–0–0) | 0.816 (42–1–21, 62–0–2) | 0.977 (61–0–3, 64–0–0) | 0.906 (52–0–12, 64–0–0) | 0.789 (40–0–24, 61–0–3) |
| `eight-b/gen_015` | 0.578 (18–0–46, 56–0–8) | 0.523 (12–0–52, 55–0–9) | 0.820 (48–0–16, 57–0–7) | 0.758 (39–0–25, 58–0–6) | 0.234 (1–0–63, 29–0–35) |
| `eight-b/gen_030` | 0.898 (51–0–13, 64–0–0) | 0.828 (42–0–22, 64–0–0) | 0.953 (59–0–5, 63–0–1) | 0.914 (54–0–10, 63–0–1) | 0.805 (43–0–21, 60–0–4) |
| `eight-c/gen_015` | 0.852 (45–0–19, 64–0–0) | 0.609 (19–0–45, 59–0–5) | 0.625 (28–0–36, 52–0–12) | 0.492 (18–0–46, 45–0–19) | 0.422 (20–0–44, 34–0–30) |
| `eight-c/gen_030` | 0.828 (41–2–21, 64–0–0) | 0.793 (40–1–23, 61–0–3) | 0.969 (60–0–4, 64–0–0) | 0.891 (52–0–12, 62–0–2) | 0.836 (48–0–16, 59–0–5) |
| `eight-d/gen_015` | 0.742 (35–0–29, 60–0–4) | 0.453 (6–0–58, 52–0–12) | 0.766 (37–0–27, 61–0–3) | 0.758 (40–0–24, 57–0–7) | 0.461 (19–0–45, 40–0–24) |
| `eight-d/gen_030` | 0.773 (35–4–25, 62–0–2) | 0.766 (37–0–27, 61–0–3) | 0.977 (61–0–3, 64–0–0) | 0.969 (60–0–4, 64–0–0) | 0.742 (38–0–26, 57–0–7) |


Observations:

- **Every seed learned, and kept learning.** Generation 15 → 30: `eight-b` +49 → +360, `eight-c` +63 → +334, `eight-d` +92 → +308. The reference run `eight-a` (+25 → +331) lands inside the seeds' range, so no lineage bias shows at this resolution.
- **Seed spread (the noise floor): 52 Elo** at generation 30 (+308 to +360) and 43 Elo at generation 15 (+49 to +92), the range of three seeds. A range of three is itself a rough estimate; read "about 50 Elo".
- **The bootstrap intervals are wider than the seed spread**: about 85–90 Elo wide (±44) at generation 30, against a 52 Elo range. Plan 06 lists this as a point to report rather than act on, because the remedy would change the pool. What it means: with five references and 128 games per pairing, game-sampling noise is about as large as the seed-to-seed difference it is supposed to resolve, so the measured spread is mostly benchmark noise and the true seed component may be smaller. A difference between two arms is believable only beyond both, roughly 50–90 Elo; finer resolution needs more references near the candidates' strength (a pool extension, see the next point) or more seeds per arm. **Decision for Vláďa**; the pool is unchanged.
- **The pool is thin at the top.** The references are −204, −123, 0, +71 and +168; every final candidate is 140–192 Elo above the strongest, scores 0.74–0.84 against it and 0.89–0.98 against `gen_005` and `gen_010`, which therefore add little information. Later experiments, whose candidates will be stronger still, will push this further. The rules allow extending a pool (never replacing it): a later `eight-a` generation that no criterion rates (`gen_025`) would be the natural candidate. Not done here, because it is a choice about the scale.
- **The scale is not quite transitive.** `eight-c/gen_015` beats `uniform@100` 0.852 but only breaks even with `eight-a/gen_010` (0.492), which rates −123; `eight-a/gen_015` beats `gen_005` and `gen_010` 0.93 and 0.90 but loses to both uniform players. Early networks and the uniform search are good at different things, and one number per player averages over that. Another reason the near-strength references matter most.
- **Measurable.** Unlike 6x6, both colours win games between near-equal 8x8 networks: `eight-c/gen_015` against `gen_020` won 20 of 64 as X and 34 of 64 as O, score 0.422. Scores against each candidate's nearest-rated reference spread from 0.445 to 0.836.
- **Early 8x8 networks are weaker than the uniform search.** Generations 5 and 10 rate −204 and −123, and the run's own Elo at generation 10 (+435 over its random generation 0) says nothing about strength against a non-network player. 8x8 needs about 15 generations to reach the uniform search's level.

### compare_runs.py

Final generations (`compare_runs.py runs/six-a runs/eight-a runs/eight-b runs/eight-c runs/eight-d`; `elo` is each run's own tournament scale and not comparable across runs, `bench` is the shared one; `s/gen` of `eight-b/c/d` is contended):

| run | gen | elo | loss | policy | value | len | X win | draw | s/gen | sim/s | bench |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| six-a | 40 | +1143 | 2.000 | 1.679 | 0.321 | 18.0 | 2% | 1% | 91 | 8439 | +282 [+226, +335] |
| eight-a | 30 | +1258 | 3.102 | 2.339 | 0.763 | 38.7 | 39% | 1% | 234 | 5991 | +331 [+286, +373] |
| eight-b | 30 | +1257 | 3.122 | 2.411 | 0.711 | 34.0 | 50% | 0% | 259 | 4702 | +360 [+312, +401] |
| eight-c | 30 | +1198 | 3.114 | 2.283 | 0.831 | 34.1 | 38% | 0% | 268 | 4385 | +334 [+289, +376] |
| eight-d | 30 | +1062 | 2.965 | 2.198 | 0.767 | 39.0 | 47% | 2% | 320 | 4316 | +308 [+264, +347] |

The halfway generations (`--at 15` for 8x8, `--at 20` for 6x6):

| run | gen | elo | loss | policy | value | len | X win | draw | s/gen | sim/s | bench |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eight-a | 15 | +801 | 3.416 | 2.660 | 0.755 | 25.2 | 43% | 0% | 184 | 4925 | +25 [-5, +52] |
| eight-b | 15 | +717 | 3.581 | 2.817 | 0.764 | 26.9 | 49% | 0% | 293 | 3211 | +49 [+17, +80] |
| eight-c | 15 | +736 | 3.433 | 2.710 | 0.723 | 23.7 | 60% | 0% | 282 | 3190 | +63 [+31, +94] |
| eight-d | 15 | +645 | 3.873 | 3.078 | 0.795 | 20.7 | 59% | 0% | 216 | 3245 | +92 [+56, +123] |

| run | gen | elo | loss | policy | value | len | X win | draw | s/gen | sim/s | bench |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| six-a | 20 | +818 | 1.874 | 1.558 | 0.316 | 16.6 | 1% | 0% | 92 | 7783 | +156 [+105, +204] |

### Draw rates

The highest self-play draw rate of any generation was 5.5% on 6x6 (`six-a`) and 2.3% on 8x8 (`eight-d`), all below the 20% trigger for a win/draw/loss value head ([network.md](../design/network.md#value-head)).

## Verdict

Applied to the measured seeds `eight-b`, `eight-c`, `eight-d` (8x8) and to `six-a` (6x6, which had no measured seeds because 8x8 passed (b)):

| Criterion | 8x8 | 6x6 |
|---|---|---|
| (a) Headroom: final − halfway > seed spread (52) | **Pass**: +311, +271, +216 | `six-a` only: +126 (generation 20 → 40); no measured seeds |
| (b) Cadence: one baseline run ≤ about 90 min | **Pass**: `eight-a` 84.7 min on `mps`, slowest generation 244 s | Pass: `six-a` 51 min for 40 generations on `cpu` |
| (c) Measurability: near-strength scores not pinned at 0.5 | **Pass**: 0.45–0.84 against the nearest-rated reference, both colours win | **Fail**: 0.500 and 0.514 for `gen_020` against `gen_015` and `gen_010`; O wins almost every game between near equals |

**Verdict: 8x8 is the testbed.** It passes all three criteria; 6x6 fails measurability, because between networks of similar strength the second player wins nearly every game and a paired-colour score collapses to 0.5.

**Noise floor: about 50 Elo** between seeds of the same configuration at generation 30 (52 measured, range of three), with 95% bootstrap intervals of about ±44 Elo per rating. Since the intervals are as wide as the spread, most of the 52 may be benchmark noise. Working rule: an arm difference under 50 Elo is noise, 50–90 Elo is a grey zone that needs the third seed (protocol rule 2), and only a difference beyond about 90 Elo stands on two seeds.

**Repeatability:** confirmed, nine of nine rows bit-identical including a side-swapped pairing.

**Cost:** an 8x8 baseline run is 85 minutes on `mps` with the machine to itself; three in parallel take about 2 hours; an 8x8 benchmark is about 8 minutes on `cpu`. A two-arm, three-seed experiment is roughly 5 hours (six final benchmarks).

**Pools:** the 8x8 pool (`eight-a` generations 5, 10, 20) is frozen as the testbed's scale ([pools.md](pools.md)). The 6x6 pool stays recorded for completeness.

**Open for Vláďa** (reported, not acted on, as plan 06 asks): the bootstrap intervals are wider than the seed spread, and the pool is thin above +170 Elo; extending the 8x8 pool with `eight-a/gen_025` is the candidate remedy. Separately, two thirds of an 8x8 generation is the run's own tournament, the cheapest place to cut if cadence ever matters.

## Design changes

- [open-questions.md](../design/open-questions.md): the testbed decision (8x8) and the noise floor, under Decisions; pointers to this experiment.
- [engineering.md](../design/engineering.md): the 8x8 throughput and per-generation cost under "Measured"; `mps` for 8x8 training.
- [pools.md](pools.md): both pools in the manifest; the 8x8 pool frozen as the testbed's scale.
