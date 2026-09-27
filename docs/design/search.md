# Search

Monte Carlo tree search (MCTS) guided by the network, as in AlphaZero (2018). One search answers one question: given a position, how good is each move? Self-play and the CLI both use the same search with different settings (see [Modes](#modes)).

## Node statistics

The tree is a plain tree of positions. Each node holds its `State` and, for every legal move `a`, the edge statistics from the paper:

| Symbol | Meaning |
|--------|---------|
| `N(s,a)` | Visit count. |
| `W(s,a)` | Total value backed up through the edge. |
| `Q(s,a) = W / N` | Mean value, from the perspective of the player who plays `a` at `s`. `0` when `N = 0`. |
| `P(s,a)` | Prior probability from the network's policy head, after masking illegal moves and renormalising. |

`N(s) = Σ_a N(s,a)` is the parent's visit count. In code it is the parent node's own visit count, incremented on every node of a backed-up path as in the paper's pseudocode: exactly `Σ_a N(s,a)` at the root (expanding the root is setup and adds no visit), and one more at an expanded inner node, the visit that expanded it (decided in [plan 02](../plans/02-engine-search-network.md#decisions-made-in-this-plan)).

**Decided:** a plain tree, no transposition sharing. Two move orders reaching the same position get two nodes. This wastes some evaluations but keeps the visit-count maths exactly as in the paper. Deferred: [evaluation cache, transposition DAG](upgrades.md#search).

## One simulation

A simulation walks from the root to a leaf, evaluates the leaf, and backs the value up. Repeated `n` times, it produces the visit distribution that the search's answer is read from.

### 1. Selection

From the root, repeatedly pick the move maximising

```
score(s,a) = Q(s,a) + U(s,a)
U(s,a)     = C(s) · P(s,a) · √N(s) / (1 + N(s,a))
C(s)       = log((1 + N(s) + c_base) / c_base) + c_init
```

with `c_base = 19652` and `c_init = 1.25`. This is PUCT (predictor + upper confidence bound for trees) with the 2018 paper's slowly growing exploration constant; the 2017 paper used a fixed `c_puct`. `U` is large for moves the network likes (`P`) and shrinks as a move is visited (`1 + N(s,a)`), while `√N(s)` keeps exploration alive as the parent accumulates visits.

**Unvisited children:** `Q = 0`, as in the pseudocode. In `[-1, 1]` value space this means "assume a draw". Known to over-explore in wide positions; deferred alternatives in [upgrades.md](upgrades.md#search).

Selection stops at a node that has not been expanded yet, or at a terminal position.

### 2. Expansion and evaluation

The leaf state is evaluated by the network: `(p, v) = f_θ(s)`. Every legal move gets a child edge with `N = 0`, `W = 0`, and `P(s,a) = p_a / Σ_{legal b} p_b`. Illegal cells are masked out before the softmax (see [network.md](network.md#policy-head)).

If the leaf is **terminal**, the network is not asked. Its value is exact:

```
v = -1  if the side to move has lost (the previous move completed a line)
v =  0  if the board is full with no winner
```

A terminal node has no children. The side to move can never have *won* at a terminal position, because whoever completed the line was the previous mover, so `v = +1` never occurs at a leaf.

**Decided:** terminal values enter the averaging like any other backup, with no proof propagation. Deferred: [MCTS-Solver](upgrades.md#search).

### 3. Backup

`v` is from the perspective of the side to move at the leaf. Walking back up the path, each edge `(s,a)` belongs to the player who moved at `s`, and players alternate, so the sign flips at every step:

```
for each edge (s,a) on the path, from leaf to root:
    v      = -v            # the parent's mover is the leaf's opponent, and so on
    N(s,a) += 1
    W(s,a) += v
```

(Equivalently: the edge into the leaf receives `-v_leaf`, its parent's edge `+v_leaf`, and so on.) The first flip is applied before the leaf's own edge is updated because `Q(s,a)` is defined for the player who played `a`, and that player is the leaf's opponent.

## Root exploration noise (self-play only)

Before the first simulation from a self-play root, the priors are mixed with Dirichlet noise so that a move the network dislikes still gets tried:

```
P'(s,a) = (1 - ε) · P(s,a) + ε · η_a,     η ~ Dir(α)
```

with `ε = 0.25` and `α` scaled to the branching factor. The paper's values are 0.03 for Go (361 cells) and 0.3 for chess (about 35 legal moves), which both follow the rule of thumb `α ≈ 10 / (average number of legal moves)`.

**Decided:** `α = 11 / S²`. The game experiments ([open-questions.md](open-questions.md#results-2026-09-27)) measured the average number of legal moves per decision at 29.4 on 6x6, 54.7 on 8x8 and 128.5 on 12x12, which is about `0.9 · S²` at every size because games end with most of the board still empty. So `10 / (average legal moves) ≈ 10 / (0.9 · S²) ≈ 11 / S²`, and the closed form replaces a per-size table: `α ≈ 0.31` on 6x6, `0.17` on 8x8, `0.076` on 12x12, within 10% of the measured values and in line with the paper's 0.03 for Go and 0.3 for chess.

Noise is applied to the root only, and never in analysis.

## Move selection (self-play)

After `n` simulations the root's visit counts define the policy target and the move distribution:

```
π(a) = N(s_0, a)^(1/τ) / Σ_b N(s_0, b)^(1/τ)
```

- `τ = 1` for the opening moves: the move is sampled proportionally to visits, so games are diverse.
- `τ → 0` afterwards: the most visited move is played.

The stored policy target is always the `τ = 1` distribution, regardless of how the move was chosen.

**Decided:** the cutoff is a per-size constant in `SearchConfig`, set to **a quarter of the measured mean game length**: 2 moves on 4x4, 3 on 6x6, 4 on 8x8, 7 on 12x12. Neither the paper's fixed 30 nor the earlier placeholder of 20% of cells survives the data: games are short compared with the board (28.6 moves on 12x12 at random strength, [open-questions.md](open-questions.md#results-2026-09-27)), so 20% of cells would sample every move of every game, and sampled blunders in the tactical phase would feed noise into the `z` labels. Opening diversity does not need the help: there are up to 144 first moves and root noise on top. These values are provisional: the lengths behind them are random-play lengths, and games between players who build walls are expected to be longer ([open-questions.md](open-questions.md#follow-up-safe-moves-over-a-game)). Re-derive them from the logged average game length once trained agents exist.

The **best move** of a search, in every mode, is the most visited child. Not the highest `Q`: a child visited three times can have a wildly wrong `Q`, and the visit count already integrates both `Q` and the network's confidence.

## Tree reuse

After a move is played, the chosen child's subtree is kept and becomes the new root; the rest of the tree is dropped. Its visit counts are real simulations and are not thrown away. In self-play the Dirichlet noise is re-applied to the new root's priors. This matches the paper and costs some bookkeeping in the tree code.

## Modes

The same search runs in two configurations:

| | Self-play | Analysis (CLI) |
|---|---|---|
| Root noise | Yes | No |
| Temperature | Fraction of the board, then greedy | Not applicable; the CLI reads `Q` per move and the most visited move |
| Root expansion | Normal: children are created when the root is expanded, visited as PUCT chooses | **All root children are evaluated up front** in one batched network call before the loop starts, so every legal move has a real visit and a real `Q` from the first snapshot (see [cli-integration.md](cli-integration.md)) |
| Batching | Across many games in lockstep, one simulation per tree per step ([training.md](training.md)) | One simulation at a time, batch size 1. Deferred: [virtual loss](upgrades.md#search) |
| Snapshots | None | The search is a generator that yields after every simulation batch |

Root expansion in analysis adds exactly one visit to every root child. It is a feature, not a distortion: analysis wants an honest number for every move, and PUCT would otherwise never look at moves the network's prior rules out.

## Constants

| Constant | Value | Source |
|----------|-------|--------|
| `c_base` | 19652 | AlphaZero 2018 pseudocode |
| `c_init` | 1.25 | AlphaZero 2018 pseudocode |
| `ε` (noise weight) | 0.25 | AlphaZero 2018 |
| `α` (Dirichlet) | `11 / S²` | `10 / avg legal moves` with the measured averages ([open-questions.md](open-questions.md#decisions-2026-09-27)) |
| Temperature cutoff | 2, 3, 4, 7 moves on 4x4, 6x6, 8x8, 12x12 | A quarter of the measured mean game length ([open-questions.md](open-questions.md#decisions-2026-09-27)) |
| Simulations per move, self-play | **Open**; the paper used 800. Start at 100–200 on 6x6 | Budget-dependent |
| Simulations, CLI default | 800 | Fixed by `docs/cli.md` |
