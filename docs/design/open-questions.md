# Open questions

Decisions that wait on data. The first implementation step, before any engine code, is a set of throwaway experiments on the game itself (scratch scripts, not package code). Their results get recorded here and the affected constants in the other files change from **Open** to **Decided**.

## Experiments to run

| Experiment | Method | What it decides |
|---|---|---|
| **Game length and draw rate** per board size (4, 5, 6, 8, 12) | Random play, and later play with the first trained models | Value head (scalar vs win/draw/loss), temperature cutoff fraction, expected examples per game, buffer size `K`, whether 12×12 is sane to start with |
| **First-player advantage and parity** | Random play win rates by side; exact values on small boards | Whether an empty-count/parity input plane or global pooling is needed from the start |
| **Branching factor over a game** | Average legal moves per move number, per board size | Dirichlet `α = 10 / avg legal moves`, and how to define "average" (over positions in typical games, not over the empty board) |
| **Exact solutions of 3×3, 4×4, maybe 5×5** | Minimax with memoisation on the immutable `State` | The solver fixture for search tests; also tells us who wins small boards, a sanity check on the rules and on the engine's first results |

## Constants waiting on the experiments

| Constant | File | Current placeholder |
|---|---|---|
| Dirichlet `α` per board size | [search.md](search.md) | `10 / avg legal moves` |
| Temperature cutoff fraction | [search.md](search.md) | 20% of `S²` |
| Value head type | [network.md](network.md) | Scalar `tanh`; revisit if draws are common |
| Extra input plane | [network.md](network.md) | None; revisit if parity matters |
| Tower size per board size | [network.md](network.md) | 4×64 on 6×6, ~6×128 on 12×12 |
| Learning rate, weight decay | [network.md](network.md) | `1e-3`, `1e-4` |
| Simulations per self-play move | [search.md](search.md) | Paper: 800; likely far fewer on small boards |
| Games per generation, training steps per generation, buffer generations `K` | [training.md](training.md) | `K ≈ 10–20`; the rest budget-driven |
| Tournament protocol (games, opponents, simulations, tie-breaking of deterministic players) | [training.md](training.md) | Undecided |
| Root-expansion evaluations and the `simulations` count | [cli-integration.md](cli-integration.md) | Not counted |

## Questions for the plan, not the design

These are code-level and belong in the implementation plan: whether training examples store `State` objects or pre-encoded planes; the exact `Evaluator` return types (NumPy arrays vs tensors); how the lockstep loop handles games finishing at different times; the metrics file format (CSV vs JSON Lines) and the TensorBoard writer's placement.

## Results (2026-09-27)

Measured by [plan 01](../plans/01-game-experiments.md) with `scripts/experiments/random_play.py` (seed 0) and `scripts/experiments/solve_boards.py`. Raw numbers are in `scripts/experiments/results/`: `*.json` regenerates byte-for-byte, and `*.timing.json` holds the machine-dependent wall-clock numbers. All runs are pure Python 3.14, single-threaded, on an 8 GB Apple Silicon laptop.

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
- **Parity and first-player advantage.** The second player is favoured. 4x4 is an O win with perfect play. O wins at least half of random games at every size (50.1–54.4%), and more under greedy play on 4x4 to 8x8 (54.4–66.0%). The edge fades on 12x12 (51.9% random, 50.2% greedy). Which side is to move matters, at least on small boards. That bears on the parity or empty-count input plane question.
- **Examples per game.** At random or greedy strength, one game yields about 27–29 positions on 12x12, about 13 on 6x6, and about 17 on 8x8. That is the starting figure for buffer size in games and for examples per generation, and it will change once trained agents play longer games.
- **Solver reach.** 3x3 and 4x4 are available as exact ground truth for search tests. 4x4 is the more useful one: a forced win for O, and the first optimal line ends in 8 plies. 5x5 is out of reach for the plain solver on this machine.
