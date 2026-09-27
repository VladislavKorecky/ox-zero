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

`N(s) = Σ_a N(s,a)` is the parent's visit count.

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

**Decided:** `α = 10 / (average legal moves)`, computed per board size. **Open:** the averages come from the game experiments ([open-questions.md](open-questions.md)); rough expectations are `α ≈ 0.07` on 12x12 and `α ≈ 0.3` on 6x6.

Noise is applied to the root only, and never in analysis.

## Move selection (self-play)

After `n` simulations the root's visit counts define the policy target and the move distribution:

```
π(a) = N(s_0, a)^(1/τ) / Σ_b N(s_0, b)^(1/τ)
```

- `τ = 1` for the opening moves: the move is sampled proportionally to visits, so games are diverse.
- `τ → 0` afterwards: the most visited move is played.

The stored policy target is always the `τ = 1` distribution, regardless of how the move was chosen.

**Decided:** the cutoff is a fraction of the board area rather than the paper's fixed 30 moves (a Go-sized number that would cover most of a 6x6 game). Initial fraction: 20% of cells, so about 30 moves on 12x12 and 7 on 6x6. **Open:** tune after the game-length experiment.

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
| `α` (Dirichlet) | `10 / avg legal moves`, per board size | Rule of thumb consistent with the paper's Go/chess/shogi values. **Open** |
| Temperature cutoff | 20% of `S²` moves | Board-relative analogue of the paper's 30. **Open** |
| Simulations per move, self-play | **Open**; the paper used 800 | Budget-dependent |
| Simulations, CLI default | 800 | Fixed by `docs/cli.md` |
