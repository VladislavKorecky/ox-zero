# Experiment 02: how informative is each benchmark opening

| | |
|---|---|
| Status | **Complete** (2026-10-12). A measurement of the benchmark, not a training experiment: no arms, no new runs. |
| Prompted by | Vláďa's opening tip and the [forced-openings result](../design/open-questions.md#results-2026-10-12-forced-openings): the whole second ring is lost for X by force |
| Verdict | Edges and corners are the most informative openings, the proven losses still carry some information, and trimming the opening set cannot shrink the benchmark's interval |

## Hypothesis

The benchmark plays every cell as X's first move, once per colour ([experiment 01](01-testbed-and-noise-floor.md) found its 95% interval as wide as the seed spread). If an opening decides the game before the players matter, its pair of games splits 1-1 and measures nothing. The second ring is proven lost for X, so its pairs should be uninformative; edges and corners, where X is not attacked, should be the most informative.

A pair is **informative** when one player wins both of its games (as X and as O). When the same colour wins both, the pair splits 1-1 whatever the players' strengths.

## What was run

No training. The four generation-30 networks of experiment 01, near-equal in strength (benchmark +308 to +360): all six pairings, every one of the 64 openings with both colours, 384 pairs, the deterministic evaluation search at 100 simulations, on `mps` (outside the official, `cpu`-only benchmark log).

```bash
uv run python scripts/experiments/opening_outcomes.py --device mps \
    runs/eight-a/gen_030.pt runs/eight-b/gen_030.pt runs/eight-c/gen_030.pt runs/eight-d/gen_030.pt
```

Output: `scripts/experiments/results/opening_outcomes.json` (every pair). About 9 minutes.

## Results

| Opening class | Cells | X wins | Informative pairs | O won both | X won both |
|---|---|---|---|---|---|
| Corner | 4 | 47.9% | 54.2% | 25.0% | 20.8% |
| Edge (not corner) | 24 | 41.3% | 51.4% | 32.6% | 15.3% |
| Second ring (proven lost) | 20 | 12.9% | 25.8% | 74.2% | 0.0% |
| Interior | 16 | 19.3% | 32.3% | 64.6% | 3.1% |
| All | 64 | 27.3% | 38.8% | 53.1% | 7.8% |

One draw in 768 games.

The networks' own view of the empty board agrees (`ox-zero analyze --size 8 --simulations 3200` with `eight-b` and `eight-c`): edge and corner first moves score 0.40–0.56 for X, every interior and second-ring cell 0.04–0.41.

## Verdict

- **Edges and corners are balanced openings** (X wins 41–48%) and the most informative per game (about half their pairs).
- **The interior is bad for X but not proven lost** (X wins 19%), and only a third of its pairs are informative.
- **The proven losses are not pure noise.** A quarter of second-ring pairs are informative: generation-30 networks miss the forced win often enough (X wins 13% of those games) that these openings measure tactics.
- **Trimming the opening set cannot shrink the interval.** Per pairing, all 64 openings give about 25 informative pairs, the 44 without the second ring about 20, the 28 edges and corners about 15: dropping openings makes the benchmark cheaper per unit of information, never more precise. More resolution needs *more* balanced openings, for example two-move openings that start on an edge.

## Design changes

None: the benchmark keeps every cell as an opening, on Vláďa's decision of 2026-10-16. The options this experiment raised, richer balanced openings (more resolution) or the 44 openings without the second ring (cheaper, slightly less information), were both declined:

- **The full set is cheap enough.** An 8x8 benchmark is about 8 minutes on `cpu` ([experiment 01](01-testbed-and-noise-floor.md#cost)), and trimming only buys cost, never precision (verdict above).
- **The second ring stays in the Elo.** Its informative pairs are exactly the games where a network misses the forced win, so a tactical regression already costs rating. This is why no separate tactics score is kept (below).
- **The scale stays.** Any change to the opening set would need a new 8x8 match log.

Reconsider when the benchmark moves to a larger board: 12x12 has 144 openings and longer games, so a benchmark there costs several times more, and the trade between cost and information is worth measuring again. Recorded in [upgrades.md](../design/upgrades.md#engineering).

## Follow-ups dropped (2026-10-16)

Both follow-ups recorded after this experiment were dropped by Vláďa:

- **Tactics score** (a separate score for converting the proven-lost openings against a perfect-defender oracle). Not built: the benchmark reports Elo only. The proven-lost openings stay in the Elo set, where missing the forced win already shows up as lost rating, and a second number would need the forced-win search moved into the package for little extra information.
- **Are the inner rings lost too?** (an unbounded forced-win search on 8x8, then an exact proof-number solver on 5x5 and 6x6). Not run: the target board is 12x12, whose centre is out of reach for any such search, so a small-board answer says nothing about whether edges and corners are the only safe openings there. If they are, self-play should discover it. `scripts/experiments/forced_openings.py` stays as the tool that proved the second ring.
