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

None. Changing the benchmark's opening set changes the scale and needs a new 8x8 match log; that is a decision for Vláďa, recorded here as an option: richer balanced openings (more resolution) or the 44 openings without the second ring (cheaper, slightly less information).

## Follow-ups (TODO)

- **TODO: tactics score for the benchmark.** Keep the proven-lost openings in the Elo benchmark (they matter exactly when a network does *not* know the forced win, or forgets it during training), and also report them separately. Concept, chosen by Vláďa on 2026-10-14 over scoring the pool games: the candidate plays O from each proven-lost opening (20 on 8x8) against a *perfect defender*, an X oracle built from the forced-win search that always plays the longest defence. The tactics score is the share of openings the candidate converts within the proven horizon (4 O moves); a game counts as failed the moment O's move leaves no forced win within the remaining moves. Measures tactics alone, independent of the pool, deterministic and cheap. Reported next to the Elo, never inside it, so the rating scale is unchanged. Needs the search moved from `scripts/experiments/forced_openings.py` into the package. To be planned in its own session.
- **TODO: are the inner rings lost too?** Vláďa's hypothesis: O might also force a win against interior openings by starting from the corners and pushing X with setup moves. The searches so far limited O to moves within distance 2 of a mark, which excludes exactly those corner setups, so they do not test it. To run overnight (no distance limit; 4 O moves; may take hours):

  ```bash
  uv run python scripts/experiments/forced_openings.py --sizes 8 --moves 4 --radius 0 \
      --out scripts/experiments/results/forced_openings.8x8-moves4-radius0.json
  ```

  The script maps every representative, so the edge and second-ring cells are rechecked too (the ring is already proven; edges are cheap). An exact solver for 5x5 and 6x6 (proof-number search) is the stronger follow-up if this is inconclusive.
