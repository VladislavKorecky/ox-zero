# Open questions

Decisions that wait on data. The first implementation step, before any engine code, is a set of throwaway experiments on the game itself (scratch scripts, not package code). Their results get recorded here and the affected constants in the other files change from **Open** to **Decided**.

**Status (2026-09-27):** the game experiments have run ([Results](#results-2026-09-27)) and the constants they could settle are settled ([Decisions](#decisions-2026-09-27)). What is still open waits on training runs, not on more game experiments.

**Status (2026-10-10):** the training pipeline is built ([plan 04](../plans/archive/04-training-pipeline.md#decisions-made-in-this-plan)) and set the training constants provisionally for 6x6; the first 6x6 run is in [Results (2026-10-10)](#results-2026-10-10-the-first-6x6-run). Provisional means "a starting point the run's numbers re-derive", not decided.

## Experiments to run

| Experiment | Method | What it decides |
|---|---|---|
| **Game length and draw rate** per board size (4, 5, 6, 8, 12) | Random play, and later play with the first trained models | Value head (scalar vs win/draw/loss), temperature cutoff fraction, expected examples per game, buffer size `K`, whether 12×12 is sane to start with |
| **First-player advantage and parity** | Random play win rates by side; exact values on small boards | Whether an empty-count/parity input plane or global pooling is needed from the start |
| **Branching factor over a game** | Average legal moves per move number, per board size | Dirichlet `α = 10 / avg legal moves`, and how to define "average" (over positions in typical games, not over the empty board) |
| **Exact solutions of 3×3, 4×4, maybe 5×5** | Minimax with memoisation on the immutable `State` | The solver fixture for search tests; also tells us who wins small boards, a sanity check on the rules and on the engine's first results |
| **Safe moves over a game** (added after the first results) | Greedy play, counting the moves that do not lose on the spot at each move number | Whether the game is decided by running out of safe moves (a counting problem: global pooling, MCTS-Solver), and a better guide to trained-game length than random play |

## Constants waiting on the experiments

| Constant | File | Placeholder before the experiments | Status |
|---|---|---|---|
| Dirichlet `α` per board size | [search.md](search.md) | `10 / avg legal moves` | **Decided:** `α = 11 / S²` |
| Temperature cutoff fraction | [search.md](search.md) | 20% of `S²` | **Decided:** a quarter of the measured mean game length: 2, 3, 4, 7 moves on 4x4, 6x6, 8x8, 12x12 |
| Value head type | [network.md](network.md) | Scalar `tanh`; revisit if draws are common | **Decided:** scalar; draws are absent from 5x5 up |
| Extra input plane | [network.md](network.md) | None; revisit if parity matters | **Decided:** none in version 1; parity matters but the plane is the wrong fix |
| Tower size per board size | [network.md](network.md) | 4×64 on 6×6, ~6×128 on 12×12 | **Provisional, set in plan 04:** 4×64 on 6x6; 12x12 open |
| Learning rate, weight decay | [network.md](network.md) | `1e-3`, `1e-4` | **Provisional, set in plan 04:** `1e-3`, `1e-4` (AdamW); the first run's loss was still falling at generation 20 |
| Simulations per self-play move | [search.md](search.md) | Paper: 800; likely far fewer on small boards | **Provisional, set in plan 04:** 100 on 6x6, in self-play and in the tournament |
| Games per generation, training steps per generation, buffer generations `K` | [training.md](training.md) | `K ≈ 10–20`; the rest budget-driven | **Provisional, set in plan 04:** 128 games (64 in lockstep), 50 steps of batch 256, `K = 10` on 6x6 |
| Tournament protocol (games, opponents, simulations, tie-breaking of deterministic players) | [training.md](training.md#tournament-protocol) | Undecided | **Provisional, set in plan 04:** 3 nearest opponents plus a ladder opponent `g − 8`; 20 random openings × both colours; noise-free search at 100 simulations |
| Root-expansion evaluations and the `simulations` count | [cli-integration.md](cli-integration.md) | Not counted | **Decided in plan 02:** not counted; the first snapshot reports `simulations = 0` |

## Questions for the plan, not the design

These are code-level and belong in the implementation plan: whether training examples store `State` objects or pre-encoded planes; the exact `Evaluator` return types (NumPy arrays vs tensors); how the lockstep loop handles games finishing at different times; the metrics file format (CSV vs JSON Lines) and the TensorBoard writer's placement.

## Results (2026-09-27)

Measured by [plan 01](../plans/archive/01-game-experiments.md) with `scripts/experiments/random_play.py` (seed 0) and `scripts/experiments/solve_boards.py`. Raw numbers are in `scripts/experiments/results/`: `*.json` regenerates byte-for-byte, and `*.timing.json` holds the machine-dependent wall-clock numbers. All runs are pure Python 3.14, single-threaded, on an 8 GB Apple Silicon laptop.

Average legal moves are pooled over every decision in every game. At a non-terminal position every empty cell is legal, so a decision's legal-move count is `S²` minus the move number, and the average is fixed by how long games last. The first and last quarters are the decisions with index `i < L/4` and `i ≥ 3L/4` in a game of length `L`.

### Random play (uniformly random legal move)

| Size | Games | Mean length | Median | P10 | P90 | X win % | O win % | Draw % | Avg legal moves (all / first ¼ / last ¼) |
|---|---|---|---|---|---|---|---|---|---|
| 3x3 | 2000 | 6.5 | 6 | 4 | 9 | 43.1 | 50.4 | 6.5 | 6.0 / 8.4 / 3.2 |
| 4x4 | 2000 | 8.7 | 9 | 5 | 12 | 45.5 | 54.4 | 0.1 | 11.8 / 15.1 / 7.9 |
| 5x5 | 2000 | 10.9 | 11 | 6 | 15 | 49.2 | 50.7 | 0.0 | 19.5 / 23.8 / 14.6 |
| 6x6 | 2000 | 12.7 | 13 | 7 | 19 | 49.9 | 50.1 | 0.0 | 29.4 / 34.5 / 23.7 |
| 8x8 | 2000 | 17.6 | 17 | 10 | 25 | 46.9 | 53.0 | 0.0 | 54.7 / 61.9 / 46.9 |
| 12x12 | 2000 | 28.6 | 28 | 15 | 42 | 48.1 | 51.9 | 0.0 | 128.5 / 140.3 / 116.1 |

### Greedy play (one-ply lookahead: take a win, avoid giving one, else random)

A second, less silly data point, not a model of strong play. Greedy 12x12 uses 500 games instead of 2000, because the lookahead costs about 0.25 s per 12x12 game.

| Size | Games | Mean length | Median | P10 | P90 | X win % | O win % | Draw % | Avg legal moves (all / first ¼ / last ¼) |
|---|---|---|---|---|---|---|---|---|---|
| 3x3 | 2000 | 8.6 | 9 | 6 | 9 | 3.0 | 12.3 | 84.7 | 5.1 / 8.1 / 1.7 |
| 4x4 | 2000 | 10.2 | 10 | 6 | 14 | 31.0 | 66.0 | 2.9 | 11.0 / 15.0 / 6.6 |
| 5x5 | 2000 | 11.7 | 11 | 7 | 18 | 43.8 | 56.2 | 0.0 | 19.0 / 23.7 / 13.7 |
| 6x6 | 2000 | 13.4 | 13 | 8 | 19 | 44.3 | 55.7 | 0.0 | 29.1 / 34.5 / 23.2 |
| 8x8 | 2000 | 17.3 | 17 | 11 | 25 | 45.6 | 54.4 | 0.0 | 55.0 / 61.9 / 47.5 |
| 12x12 | 500 | 27.0 | 27 | 17 | 37 | 49.8 | 50.2 | 0.0 | 129.8 / 140.7 / 118.5 |

### Exact solutions

Negamax with a transposition table (`ox_zero.game.Solver`). The value is for X, who moves first. The optimal-play length comes from one game in which both sides always play the first optimal move in board order.

| Size | Value | Best first moves for X | Positions solved | Time | Optimal-play length |
|---|---|---|---|---|---|
| 3x3 | Draw | Every cell except the centre `1,1` (the centre loses) | 3,784 | 0.03 s | 9 (full board, draw) |
| 4x4 | **O wins** | None: all 16 first moves lose | 64,057 | 0.53 s | 8 (O completes `O X O` on the anti-diagonal `0,2 1,1 2,0`) |
| 5x5 | Unknown | n/a | n/a | Did not finish in 10 min | n/a |

- **4x4 timing and canonicalisation.** 4x4 solves in 0.53 s over 64,057 cached positions, far under the 2-minute target, so symmetry canonicalisation was not needed and `symmetry.py` was not built. The count is small because of the `+1` short-circuit. X's nodes are all losses, so every move there is searched, but at O's nodes the search stops at the first winning reply, and most of O's alternatives are never explored. An independent cross-check without the short-circuit also gives O wins, but it takes about 2.5 min and visits 6.2 million positions.
- **5x5 attempt.** It was run once with a 10-minute wall-clock limit and killed at the limit without a result. The cache grew by roughly 1 GB in the first minute, so on this machine memory would run out before time would.

### Throughput of the pure-Python rules

- `apply_move` alone, measured over the greedy policy's lookahead calls: about 310k positions/s on 3x3, 210k on 6x6, and 170k on 12x12.
- A full random-play step (`legal_moves`, a random choice, `apply_move`): about 230k positions/s on 3x3 and 65k on 12x12.

### What this suggests

- **Dirichlet `α`.** The 12x12 average is 128.5 legal moves under random play and 129.8 under greedy play, giving `α ≈ 0.078`. For 8x8, 54.7 gives `α ≈ 0.18`. For 6x6, 29.4 gives `α ≈ 0.34`. These match the rough expectations in [search.md](search.md) (0.07 and 0.3). Averaging over the first or last quarter only would move 12x12 between 0.071 and 0.086, because games end with most of the board still empty.
- **Temperature cutoff.** Games are short compared with the board: 12x12 random games average 28.6 moves (P90 42), and greedy games 27.0. The placeholder cutoff of 20% of `S²` is 29 moves on 12x12, so it would cover an entire typical random-strength game. On 6x6 it is 7 moves, about half of the 12.7-move mean.
- **Value head.** From 5x5 up, no game out of 2000 (500 for greedy 12x12) ended in a draw, under either policy. Draws are common only on 3x3, which is also a draw with perfect play. Stronger play does add draws, though: greedy play raised the draw rate on 3x3 from 6.5% to 84.7% and on 4x4 from 0.1% to 2.9%. The data says nothing yet about draws between trained agents on large boards.
- **Parity and first-player advantage.** The second player is favoured on small boards. 4x4 is an O win with perfect play. O wins at least half of random games at every size (50.1–54.4%), and more under greedy play on 4x4 to 8x8 (54.4–66.0%). On 12x12 the edge is gone: 51.9% of 2000 random games is within sampling noise of even, and an earlier, much larger simulation by the author (millions of random 12x12 games, in Go) came out almost exactly 50/50. Random-play balance says nothing about the game's true value in either direction: 3x3 is close to even at random and a draw when solved, 4x4 is 54% O at random and an O win when solved. Which side is to move matters, at least on small boards. That bears on the parity or empty-count input plane question.
- **Examples per game.** At random or greedy strength, one game yields about 27–29 positions on 12x12, about 13 on 6x6, and about 17 on 8x8. That is the starting figure for buffer size in games and for examples per generation, and it will change once trained agents play longer games.
- **Solver reach.** 3x3 and 4x4 are available as exact ground truth for search tests. 4x4 is the more useful one: a forced win for O, and the first optimal line ends in 8 plies. 5x5 is out of reach for the plain solver on this machine.

### Follow-up: safe moves over a game

Added the same day, after reading the numbers above. Measured with `scripts/experiments/safe_moves.py` (500 greedy games per size, seed 0). A move is *safe* if it does not lose on the spot: after it, the opponent has no move that completes an alternating line. The greedy policy never plays an unsafe move while a safe one exists, so under it every decisive game ends by a **forced loss**, a position where the side to move has no safe move at all.

Move numbers are 0-based, so `S² − k` cells are empty at move `k`. "Forced loss" and "win available" are shares of the decisions at that move number.

| Board | Move | Games alive | Legal | Safe: mean | Forced loss % | Win available % |
|---|---|---|---|---|---|---|
| 6x6 | 4 | 500 | 32 | 17.6 | 2.8 | 0.2 |
| 6x6 | 8 | 434 | 28 | 8.5 | 9.9 | 5.3 |
| 6x6 | 12 | 283 | 24 | 5.1 | 15.9 | 15.9 |
| 6x6 | 16 | 122 | 20 | 2.7 | 22.1 | 21.3 |
| 8x8 | 6 | 497 | 58 | 32.4 | 2.2 | 0.8 |
| 8x8 | 12 | 394 | 52 | 16.7 | 8.4 | 8.4 |
| 8x8 | 18 | 184 | 46 | 9.0 | 12.5 | 14.1 |
| 8x8 | 24 | 56 | 40 | 6.1 | 16.1 | 30.4 |
| 12x12 | 8 | 500 | 136 | 93.3 | 0.4 | 0.0 |
| 12x12 | 16 | 461 | 128 | 57.7 | 3.2 | 2.8 |
| 12x12 | 24 | 308 | 120 | 35.4 | 10.4 | 7.1 |
| 12x12 | 32 | 129 | 112 | 22.5 | 11.6 | 13.2 |

The full per-move series is in `scripts/experiments/results/safe_moves.json`. The minimum safe count is 0 from move 4 (6x6, 8x8) or 8 (12x12) on: a forced loss can arrive that early.

**Length of optimal play, sampled.** `solve_boards.py` now also plays 500 games per solved size in which both sides pick uniformly among their optimal moves. On 4x4 the mean is 8.4 moves (P10 6, P90 12, max 14), which is 52% of the board and no longer than random play (8.7). Caveat: in a lost position every move is optimal, so the losing side plays randomly rather than stubbornly; the figure is a floor for games between two strong players, not an estimate. 3x3 always runs the full 9 moves to a draw.

**What this adds:**

- **OXOX is an avoidance game.** Under greedy play on 12x12, half the legal moves lose on the spot by move 16 and three quarters by move 28; on 6x6 only about 5 safe moves remain by move 12. Games end when the side to move runs out of safe moves. Tactics are local (a threat spans three cells) but the outcome is global: who has more safe moves left, which is a parity-like count. That is a job for global pooling, not for convolutions, and it is why O wins 4x4 and leads every size under greedy play. A naive player runs out of safe moves; a skilled one manages the supply, see the next point.
- **Trained games will be longer than these; how much longer is unknown.** Every length figure here comes from play that never builds structure: scattered marks poison their own neighbourhoods. Experience from human play (the author's) says the counter is the **wall**: same-mark blocks at least two cells thick contain no alternating triple and can be extended safely, and two cooperating walls fill the board to a draw (the two-row stripes mentioned under the value head are exactly this). The tactical layer on top is **islands**: a few well-placed marks that cut into the opponent's wall and its supply of safe moves. Games between players who know this run to 50–60 moves or more on 12x12. The 4x4 optimal-play sample cannot show any of it, because the board is too small for walls. The temperature cutoff and the examples-per-game figures are therefore provisional and get re-derived from the logged average game length once trained agents exist.
- **Proof propagation should pay.** With most legal moves losing on the spot mid-game, an MCTS that averages terminal values instead of proving them spends its simulations relearning the same one-ply facts. MCTS-Solver moves to the front of the search upgrades.
- **The empty 4x4 board is a poor solver fixture.** All 16 first moves lose, so "the search picks a proven-best move" is vacuous there. Fixtures need positions whose optimal moves are a strict, small subset of the legal ones. The empty 3x3 is one: eight moves draw and the centre loses.

## Results (2026-10-10): the first 6x6 run

Measured by [plan 04](../plans/archive/04-training-pipeline.md), step 8: `uv run python scripts/train.py --name six-a --size 6 --generations 20 --seed 0 --device cpu`, every other constant at its provisional value (table above). The run was interrupted with Ctrl-C during generation 6 and resumed with the same command; it continued from generation 6 and completed generation 20. About 26 minutes of wall time on the author's 8 GB Apple Silicon laptop. Throughput is in [engineering.md](engineering.md#measured-2026-10-10-self-play-throughput).

"Elo" is the final Bradley-Terry fit over all of the run's matches, generation 0 at 0. "Ladder" is generation `g`'s score against `g − 8`. Mean length is in moves; the rates are over the generation's 128 self-play games.

| Gen | Examples | Mean length | Draw rate | X win rate | Loss (total / policy / value) | Elo | Ladder | Wall s |
|---|---|---|---|---|---|---|---|---|
| 1 | 696 | 5.4 | 0.00 | 0.53 | 4.471 / 3.494 / 0.976 | 177 | | 17.4 |
| 2 | 1022 | 8.0 | 0.00 | 0.48 | 4.316 / 3.400 / 0.916 | 398 | | 26.7 |
| 3 | 1027 | 8.0 | 0.00 | 0.45 | 4.183 / 3.317 / 0.866 | 535 | | 35.2 |
| 4 | 1626 | 12.7 | 0.00 | 0.33 | 3.975 / 3.181 / 0.794 | 662 | | 47.6 |
| 5 | 2256 | 17.6 | 0.00 | 0.39 | 3.666 / 2.949 / 0.717 | 697 | | 55.4 |
| 6 | 1767 | 13.8 | 0.00 | 0.21 | 3.393 / 2.733 / 0.660 | 691 | | 54.5 |
| 7 | 2437 | 19.0 | 0.00 | 0.15 | 3.132 / 2.537 / 0.595 | 726 | | 60.9 |
| 8 | 2086 | 16.3 | 0.00 | 0.12 | 2.985 / 2.390 / 0.595 | 730 | 1.00 | 59.1 |
| 9 | 2441 | 19.1 | 0.00 | 0.26 | 2.871 / 2.282 / 0.589 | 762 | 1.00 | 69.8 |
| 10 | 2286 | 17.9 | 0.00 | 0.20 | 2.782 / 2.226 / 0.557 | 767 | 0.85 | 66.0 |
| 11 | 1752 | 13.7 | 0.00 | 0.17 | 2.669 / 2.126 / 0.543 | 756 | 0.78 | 67.2 |
| 12 | 2257 | 17.6 | 0.00 | 0.20 | 2.535 / 2.014 / 0.521 | 768 | 0.70 | 69.3 |
| 13 | 2318 | 18.1 | 0.00 | 0.14 | 2.416 / 1.925 / 0.491 | 782 | 0.65 | 72.0 |
| 14 | 2405 | 18.8 | 0.00 | 0.30 | 2.347 / 1.811 / 0.536 | 766 | 0.60 | 73.8 |
| 15 | 2175 | 17.0 | 0.00 | 0.02 | 2.208 / 1.735 / 0.474 | 808 | 0.55 | 72.7 |
| 16 | 2314 | 18.1 | 0.00 | 0.06 | 2.154 / 1.673 / 0.481 | 794 | 0.55 | 110.3 |
| 17 | 2390 | 18.7 | 0.00 | 0.09 | 2.093 / 1.641 / 0.452 | 797 | 0.57 | 96.3 |
| 18 | 2591 | 20.2 | 0.00 | 0.04 | 2.052 / 1.626 / 0.426 | 811 | 0.55 | 84.7 |
| 19 | 2429 | 19.0 | 0.00 | 0.04 | 1.971 / 1.609 / 0.362 | 813 | 0.57 | 83.1 |
| 20 | 2119 | 16.6 | 0.00 | 0.01 | 1.874 / 1.558 / 0.316 | 818 | 0.60 | 83.3 |

`scripts/evaluate.py`, 20 openings per colour, 100 simulations: `gen_020` against `gen_000` won 40–0–0 (+763 Elo with the virtual draw); against the uniform search (the CLI's no-model engine) it won 33–0–7, score 0.825, +260 Elo: 14–0–6 as X and 19–0–1 as O.

### What this shows

Observations from one seeded run, not conclusions beyond it.

- **The losses fell throughout.** Policy loss 3.49 → 1.56 and value loss 0.98 → 0.32 over 20 generations, and both were still falling at the end.
- **Games got longer, as the wall-building prediction expected.** The mean self-play game grew from 5.4 moves at generation 1 to about 17–20 from generation 5 on, against 12.7 for random and 13.4 for greedy play ([random play](#random-play-uniformly-random-legal-move), [greedy play](#greedy-play-one-ply-lookahead-take-a-win-avoid-giving-one-else-random)). That fits "trained games will be longer than these" ([follow-up](#follow-up-safe-moves-over-a-game)). The other half of that prediction, two cooperating walls filling the board to a draw, did not appear: see the next point.
- **No draws.** The self-play draw rate was 0.00 in every generation, far below the 20% trigger for a win/draw/loss value head ([network.md](network.md#value-head)). The scalar head stands.
- **Self-play converges on O winning.** X's self-play win rate fell from about 0.5 to 0.01–0.04 by generations 15–20: O wins almost every game. 4x4 is a proven O win ([exact solutions](#exact-solutions)); this suggests 6x6 may be one too, but self-play at this strength is not a proof.
- **Most of the Elo gain is early.** Elo rose steeply to about 700 by generation 5 and then slowly to 818 at generation 20; later generations differ by small margins (scores against the previous generation 0.45–0.62 from generation 5 on).
- **The ladder is informative at distance 8.** Ladder scores fell from sweeps (1.00 at generations 8 and 9) to 0.55–0.60 at generations 15–20, so at `ladder = 8` it measures something rather than reporting sweeps; no reason to shorten it.
- **Strength against the uniform search is moderate.** `gen_020` sweeps the random network but beats the uniform search only 33–7, with 6 of its 7 losses as X: the side self-play has learned to lose with.

**Constants to look at next** (suggestions for the tuning follow-up, not decisions):

- **Temperature cutoff and simulations per move**, given that X almost never wins in self-play: with sampling only for the first 3 moves on 6x6 and deterministic play after, self-play may be exploring too little of X's options, and 100 simulations may be too few to find X's defences. The cutoff was derived from random-play lengths (a quarter of 12.7); a quarter of the measured ~18 would be 4–5 moves.
- **`K = 10` seems fine:** nothing in the run points at stale data or too little of it.
- **Steps per generation could rise**, since the loss was still falling at generation 20.

## Decisions (2026-09-27)

Recorded in the files they belong to; summarised here so the trail from number to decision is in one place.

| Decision | Data | Where |
|---|---|---|
| `α = 11 / S²` (0.31 on 6x6, 0.17 on 8x8, 0.076 on 12x12) | Measured average legal moves are 29.4, 54.7, 128.5, about `0.9 · S²` at every size because games end with most of the board empty; `10 / (0.9 · S²) ≈ 11 / S²` | [search.md](search.md#root-exploration-noise-self-play-only) |
| Temperature cutoff: a quarter of the measured mean game length, per size. **Provisional** | 20% of `S²` is 29 moves on 12x12, longer than a whole game at current strength (28.6), so every move would be sampled and `z` labels would carry sampling noise through the tactical phase. Opening diversity is already large (up to 144 first moves plus root noise). The lengths behind the numbers are random-play lengths; wall-building play is expected to be longer, so the values get re-derived from logged game lengths. | [search.md](search.md#move-selection-self-play) |
| Scalar value head stays | Zero draws in 12,500 games from 5x5 up; 4x4 is decisive under perfect play. But that is random-strength play. A line-free full board exists (two-thick same-mark bands), and two players who both build walls reach it, so self-play could converge on draws the way tic-tac-toe does. Scalar first; the trigger to revisit is the logged self-play draw rate, not the random-play one. | [network.md](network.md#value-head) |
| No fourth input plane in version 1 | Parity matters (O wins 4x4; safe-move counting decides games), but an empty-count plane counts empties, not safe cells. Both heads already have a fully connected layer that can count. Global pooling is the real fix and is promoted in the upgrades list. | [network.md](network.md#input-encoding), [upgrades.md](upgrades.md#network) |
| MCTS-Solver is the first search upgrade to test | Safe-move table above | [upgrades.md](upgrades.md#search) |
| 5x5 is out of scope for the solver fixture; fixtures use positions with a strict subset of optimal moves | 5x5 grew the cache by about 1 GB per minute and did not finish in 10 minutes; the empty 4x4 has no strict subset | [engineering.md](engineering.md#testing) |
| Start on 6x6 with 100–200 simulations per move and 4×64; 12x12 after the pipeline works | Unchanged from the guiding principles; the data confirms 12x12 random-strength games touch only a fifth of the board, so nothing about 12x12 is learnable before the small boards are | [README.md](README.md#guiding-principles) |
