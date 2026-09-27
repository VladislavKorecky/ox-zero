# Network

One network `f_θ(s) = (p, v)`: a residual convolutional tower with a policy head and a value head, as in AlphaZero. The shape is configuration, so the same code serves a 4x4 test board and the 12x12 game.

## Input encoding

OXOX has two properties that simplify the encoding compared to chess or Go.

1. **The game is Markov.** Board plus side to move is the complete state: there is no ko, no repetition rule, no castling rights. AlphaZero's stack of the last 8 positions exists for those rules. We use **no history planes**.
2. **Marks are interchangeable.** Swapping every X for O and every O for X leaves every alternating line alternating, and the side to move is unchanged (it is determined by the number of marks). So a position and its swapped twin are the same game. Encoding marks *relative to the side to move* is exactly the canonical form that removes this redundancy, and the paper's "colour to play" plane carries no information here. We omit it.

**Decided:** three planes of shape `S × S`, float32:

| Plane | Content | Why |
|-------|---------|-----|
| 0 | 1 where the **side to move** has a mark | Relative encoding; see above |
| 1 | 1 where the **other side** has a mark | |
| 2 | All ones | With zero padding, the convolution sees where the board ends. AlphaZero's chess encoding has the same constant plane. |

The encoding is a pure function `State -> tensor[3, S, S]` and lives on the network side of the evaluator seam ([engineering.md](engineering.md#the-evaluator-seam)).

**Open:** whether a fourth plane (fraction of empty cells, broadcast) helps. Tempo and parity may matter in OXOX and convolutions cannot count; the experiments decide ([open-questions.md](open-questions.md)). Deferred to [upgrades.md](upgrades.md#network) either way.

## Body: the residual tower

```
input  (3 planes, S×S)
  │
stem:  conv 3×3 (3→F) → BatchNorm → ReLU
  │
block 1..N, each:
  x ──► conv 3×3 (F→F) → BN → ReLU → conv 3×3 (F→F) → BN ──► (+) ──► ReLU ──► out
  └──────────────────── skip connection ──────────────────────┘
  │
  ├── policy head
  └── value head
```

- Every convolution uses padding 1, so the trunk is a stack of `F` feature maps of size `S × S` throughout.
- A **residual block** computes a correction `F(x)` and outputs `x + F(x)`. "Do nothing" is trivially learnable and gradients flow straight through the additions, which is what allows deep stacks to train (He et al., 2015).
- **Batch normalisation** after every convolution, as in the paper. The network is in training mode during optimisation and evaluation mode during search, so batch statistics are frozen at inference.
- `N` (blocks) and `F` (filters) come from `NetworkConfig`. Paper: 19 or 39 blocks, 256 filters. Us: **Open**, with starting points of 4 blocks × 64 filters on 6x6 and something like 6 × 128 on 12x12.

### Kernel size

**Decided:** 3×3 everywhere. Stacked 3×3 convolutions grow the receptive field by 2 per layer:

| 3×3 convs stacked | Each output cell sees |
|---|---|
| 1 (stem) | 3×3 |
| 2 | 5×5 |
| 3 | 7×7 |
| 9 (stem + 4 blocks) | 19×19, the whole 12×12 board |

So a 4-cell tactical pattern such as `O _ X _` is visible to the second layer, and the reach of a pattern is governed by **depth**, not kernel size. Two 3×3 layers cost `18·F²` weights against `25·F²` for one 5×5 and add a nonlinearity in between. What stacked 3×3 layers are bad at is global information (tempo, parity); a bigger kernel does not fix that either. Deferred: [5×5 stem, global pooling](upgrades.md#network).

## Heads

### Policy head

```
conv 1×1 (F→2) → BN → ReLU → flatten (2·S²) → linear → S² logits
```

One logit per cell. Illegal cells (occupied, or all cells when the game is over) are set to `-∞` before the softmax, so they receive exactly zero probability and contribute nothing to the loss. Masking happens in the evaluator, not in the search, so the search never sees an illegal prior.

### Value head

```
conv 1×1 (F→1) → BN → ReLU → flatten (S²) → linear → 256 → ReLU → linear → 1 → tanh
```

**Decided:** a single scalar `v ∈ [-1, 1]`, from the perspective of the side to move: `+1` certain win, `0` draw, `-1` certain loss. The CLI displays `(v + 1) / 2`. Deferred: [win/draw/loss head](upgrades.md#network), to be revisited if the draw rate turns out high.

## Loss

For a training example `(s, π, z)` (position, search visit distribution, game result from `s`'s side to move):

```
L = (z - v)²  -  π^T log p  +  c · ||θ||²
```

- `(z - v)²`: mean squared error of the value head against the game outcome.
- `-π^T log p`: cross-entropy between the search's visit distribution and the network's policy. Over legal moves only; the masked logits are `-∞` and drop out.
- `c · ||θ||²`: L2 regularisation, realised as weight decay in the optimiser (see below).

This is the AlphaZero objective and is not a tuning choice. What the network is asked to predict is fixed; how the weights move is a separate decision.

## Optimiser

**Decided:** **AdamW** (Loshchilov & Hutter, 2019), a deviation from the paper's SGD with momentum 0.9.

- Why: far less sensitive to the learning rate, no schedule needed to get a first working pipeline, and it adapts per parameter. The paper's optimiser and stepped schedule go on the [upgrades list](upgrades.md#network).
- Why AdamW and not Adam plus an L2 term: Adam rescales gradients per parameter, so an L2 term added to the loss is also rescaled and stops acting as uniform weight decay. AdamW applies decay directly to the weights, which is what the paper's `c · ||θ||²` intends.
- Initial values: learning rate `1e-3`, weight decay `1e-4`, no schedule. **Open:** adjust once loss curves exist.

## Symmetries

The square board has the 8 symmetries of the dihedral group D4 (4 rotations × optional reflection). Combined with the mark swap described above, every position has **16** equivalent views.

**Decided:** used for **training augmentation only**. Each sampled example is transformed by a random symmetry before the gradient step; the policy target `π` is permuted the same way, `z` is unchanged. The mark swap needs no work in our encoding: the relative planes are already swap-invariant, so only the 8 spatial symmetries are actually applied. This multiplies the effective size of a small dataset for free. AlphaZero dropped symmetries for generality across games; we have no such constraint.

Not used during search (the paper's AlphaGo Zero predecessor evaluated leaves under a random symmetry). Deferred: [randomised-symmetry inference](upgrades.md#network).

## Numerics and devices

- float32 everywhere. `mps` has no float64.
- The device is chosen at runtime: `cuda` if available, else `mps`, else `cpu`. No module hardcodes one.
- Inference in search runs under `torch.no_grad()` / `inference_mode` with the model in eval mode.
