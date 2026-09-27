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
