"""The policy/value network `f_θ(s) = (p, v)` and the AlphaZero loss.

Design: docs/design/network.md. Reference: Silver et al. 2018 (AlphaZero),
He et al. 2015 ("Deep residual learning for image recognition").

Architecture
------------
    input  [B, 3, S, S]               (encoding.py: mine, theirs, ones)
      │
    stem   conv 3×3 (3→F) → BN → ReLU
      │
    N residual blocks, each:
      x ─► conv 3×3 → BN → ReLU → conv 3×3 → BN ─► (+x) ─► ReLU
      │
      ├─ policy head: conv 1×1 (F→2) → BN → ReLU → flatten → linear → S² logits
      └─ value head:  conv 1×1 (F→1) → BN → ReLU → flatten → linear → H → ReLU
                      → linear → 1 → tanh

`F` (filters), `N` (blocks) and `H` (value hidden size) come from
`NetworkConfig`; the board size `S` is a constructor argument. Every 3×3
convolution uses padding 1, so the trunk keeps `F` feature maps of exactly
`S × S` throughout, and the same code serves a 3x3 test board and the 12x12
game.

The policy head outputs raw *logits*, not probabilities. Turning them into a
distribution needs the legal-move mask (illegal cells set to `-inf`, then
softmax), and that belongs to whoever knows the position: the evaluator at
inference, `alphazero_loss` in training.

Dtype: float32 only. Apple's `mps` backend has no float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from ox_zero.engine.encoding import NUM_PLANES


@dataclass(frozen=True)
class NetworkConfig:
    """The architecture's size. Stored in checkpoints so a network can be rebuilt.

    Attributes:
        blocks: Number of residual blocks `N` (paper: 19 or 39).
        filters: Feature maps `F` per convolution (paper: 256).
        value_hidden: Width `H` of the value head's hidden layer (paper: 256).
    """

    blocks: int = 4
    filters: int = 64
    value_hidden: int = 256


def _conv_bn(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
    """A convolution followed by batch normalisation (no activation).

    Batch normalisation (Ioffe & Szegedy, 2015) rescales every channel to zero
    mean and unit variance, then applies a learned scale `γ` and shift `β`.
    It keeps activations in a stable range as the network trains, which
    allows higher learning rates and deeper stacks.

    It has two modes, and mixing them up is a classic bug:

    - `train()`: normalise with the mean and variance of the *current batch*
      (taken over the batch and every board cell), and update running
      averages of both.
    - `eval()`: normalise with those frozen running averages. The output for
      a position then depends only on that position, not on its batch mates.
      Search must always run in eval mode.

    The convolution has no bias: BN subtracts the per-channel mean right
    after, which would cancel any constant bias, and `β` plays its role.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResidualBlock(nn.Module):
    """`out = ReLU(x + F(x))` with `F` = conv → BN → ReLU → conv → BN.

    The block learns a *correction* `F(x)` to its input rather than a whole
    new representation. "Change nothing" is just `F = 0`, which is easy to
    learn, and in backprop the `+ x` path passes the gradient straight
    through every block unchanged. That identity path is what lets deep
    stacks train at all (He et al., 2015).
    """

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.first = _conv_bn(filters, filters, 3)
        self.second = _conv_bn(filters, filters, 3)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.second(F.relu(self.first(x)))
        return F.relu(x + residual)


class Network(nn.Module):
    """The AlphaZero residual tower with a policy head and a value head.

    Attributes:
        size: Board side length `S` the network was built for.
        config: The architecture, for rebuilding from a checkpoint.
    """

    def __init__(self, size: int, config: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        self.size = size
        self.config = config
        cells = size * size
        f = config.filters

        # Receptive field: each stacked 3×3 convolution lets an output cell
        # see one more cell in every direction. The stem plus two per block
        # is 1 + 2N convolutions, so with the default N = 4 an output cell
        # sees a (1 + 2·9) = 19 x 19 window: all of a 12x12 board.
        self.stem = nn.Sequential(_conv_bn(NUM_PLANES, f, 3), nn.ReLU())
        self.tower = nn.Sequential(*(ResidualBlock(f) for _ in range(config.blocks)))

        # Policy head. The 1×1 convolution squeezes F feature maps into 2
        # per cell; the linear layer then sees the whole board at once
        # (2·S² inputs), so each cell's logit can depend on distant cells.
        self.policy_head = nn.Sequential(
            _conv_bn(f, 2, 1),
            nn.ReLU(),
            nn.Flatten(),  # [B, 2, S, S] -> [B, 2·S²]
            nn.Linear(2 * cells, cells),
        )

        # Value head. Same idea, 1 map per cell, then a small fully connected
        # network down to one number. `tanh` squashes it into (-1, 1), the
        # range of game results for the side to move (loss, draw, win).
        self.value_head = nn.Sequential(
            _conv_bn(f, 1, 1),
            nn.ReLU(),
            nn.Flatten(),  # [B, 1, S, S] -> [B, S²]
            nn.Linear(cells, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, planes: Tensor) -> tuple[Tensor, Tensor]:
        """`[B, 3, S, S]` planes -> (`[B, S²]` policy logits, `[B]` values in [-1, 1])."""
        if planes.shape[-3:] != (NUM_PLANES, self.size, self.size):
            raise ValueError(
                f"expected planes of shape [B, {NUM_PLANES}, {self.size}, {self.size}],"
                f" got {list(planes.shape)}"
            )
        trunk = self.tower(self.stem(planes))
        logits = self.policy_head(trunk)
        # The value head ends in [B, 1]; drop the trailing axis to get [B].
        values = self.value_head(trunk).squeeze(-1)
        return logits, values


def alphazero_loss(
    logits: Tensor, values: Tensor, pi: Tensor, z: Tensor, legal: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """The AlphaZero objective: `(z - v)² - π^T log p`, averaged over the batch.

    Args:
        logits: `[B, S²]` raw policy logits from the network.
        values: `[B]` value predictions `v`.
        pi: `[B, S²]` search visit distributions (the policy targets); zero on
            illegal cells.
        z: `[B]` game results from each position's side to move: +1, 0, -1.
        legal: `[B, S²]` bool, True on legal moves.

    Returns:
        `(total, policy_loss, value_loss)`, each a scalar batch mean.

    - **Value loss**, `(z - v)²`: mean squared error between the predicted and
      the actual game result.
    - **Policy loss**, `-Σ_a π_a log p_a`: cross-entropy between the search's
      visit distribution and the network's policy. Minimising it pulls the
      raw network towards what the (stronger) search found. Its minimum is
      not 0 but the entropy of `π`, `H(π) = -Σ π log π`, because
      cross-entropy = `H(π) + KL(π || p)` and only the KL term depends on the
      network. It reaches 0 only for one-hot targets.

    No regularisation term: the paper's `c·||θ||²` is applied as weight decay
    by the AdamW optimiser (docs/design/network.md, "Optimiser"). Adding it
    here as well would count it twice, and with Adam an L2 term in the loss
    gets rescaled per parameter and stops acting as uniform decay anyway.
    """
    # Illegal cells get logit -inf, so after the softmax their probability is
    # exactly 0 and the legal cells' probabilities sum to 1 on their own.
    masked = logits.masked_fill(~legal, float("-inf"))
    log_p = F.log_softmax(masked, dim=-1)
    # The masked log-probabilities are -inf, and π is 0 there. IEEE says
    # 0 · -inf = NaN, and one NaN in the loss turns every weight into NaN on
    # the next step. Replace them with 0 first: those cells contribute
    # nothing to the loss either way, and now their gradient is 0, not NaN.
    log_p = log_p.masked_fill(~legal, 0.0)
    policy_loss = -(pi * log_p).sum(dim=-1).mean()
    value_loss = (z - values).pow(2).mean()
    return policy_loss + value_loss, policy_loss, value_loss
