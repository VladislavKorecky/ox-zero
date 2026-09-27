"""Position encoding and board symmetries: the network's view of a `State`.

Design: docs/design/network.md ("Input encoding", "Symmetries").

Input planes
------------
A neural network cannot read a `State`; it reads a stack of `S x S` grids of
numbers, one grid per *feature plane*. We use three:

    plane 0   1 where the side to move has a mark, else 0
    plane 1   1 where the other side has a mark, else 0
    plane 2   all ones

Why these and nothing else:

- **No history planes.** AlphaZero stacks the last 8 positions because Go
  (ko) and chess (repetition, castling) have rules that depend on the past.
  OXOX is *Markov*: board plus side to move is the whole state.
- **Relative, not absolute.** Swapping every X for O leaves every
  alternating line alternating, so a position and its colour-swapped twin
  are the same game. Encoding "mine / theirs" instead of "X / O" maps both
  twins to the same input: the network never has to learn the same thing
  twice. (The side to move is implied by the mark count, so the paper's
  "colour to play" plane carries no information here and is omitted.)
- **The ones plane.** Convolutions pad the board with zeros. Without a
  plane that is 1 on the board and (implicitly) 0 in the padding, an empty
  edge cell looks exactly like "off the board". AlphaZero's chess encoding
  has the same constant plane.

Symmetries: the dihedral group D4
---------------------------------
The square has 8 symmetries: 4 rotations (0, 90, 180, 270 degrees) and those
same 4 rotations applied after a mirror flip. Together they form the group
D4. Every OXOX rule is symmetric under all of them (rows, columns and both
diagonals map onto rows, columns and diagonals), so a rotated or mirrored
position is equally good for the same side and its best move is the
rotated/mirrored best move. Training uses this to multiply the data 8-fold
("augmentation"): transform the planes and the policy target `pi` by the
same symmetry, keep the value target `z` unchanged.

Combined with the mark swap above there are 16 equivalent views of every
position, but the relative encoding already absorbs the mark swap, so only
the 8 spatial symmetries need code.

Indexing: `k` in `0..7`. If `k >= 4`, flip left-right first; then rotate by
`k % 4` quarter turns (counter-clockwise, NumPy's `rot90` convention).

- `k = 0` is the identity.
- Inverses: a rotation by `k` quarter turns is undone by `-k mod 4` turns.
  Every *reflection* (`k >= 4`) is its own inverse, since mirroring twice
  across the same axis restores the board. So `inverse(k) = k` for
  `k >= 4` and `(-k) % 4` otherwise.

Everything here is NumPy, not torch: the search stays torch-free (see
docs/design/engineering.md, "The evaluator seam"). `NetworkEvaluator` is the
one place that converts to tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np

from ox_zero.game.rules import Cell, State, is_terminal

NUM_PLANES = 3
NUM_SYMMETRIES = 8


def encode(state: State) -> np.ndarray:
    """The input planes for one position: `float32 [3, S, S]`.

    Plane 0 is the side to move's marks, plane 1 the other side's, plane 2
    all ones. See the module docstring for why.
    """
    size = state.size
    me, them = state.to_move, state.to_move.opponent
    planes = np.zeros((NUM_PLANES, size * size), dtype=np.float32)
    # Build the planes flat (one entry per cell, board order) and reshape at
    # the end: `board` is already a flat tuple in the same order.
    planes[0] = [mark is me for mark in state.board]
    planes[1] = [mark is them for mark in state.board]
    planes[2] = 1.0
    # [3, S*S] -> [3, S, S]: row `r` of the grid is flat entries r*S .. r*S+S-1,
    # which is exactly the row-major layout of `State.board`.
    return planes.reshape(NUM_PLANES, size, size)


def encode_batch(states: Sequence[State]) -> np.ndarray:
    """The planes for several positions of the same size: `float32 [B, 3, S, S]`."""
    return np.stack([encode(state) for state in states])


def legal_mask(state: State) -> np.ndarray:
    """`bool [S²]`: True on every legal move. All False once the game is over.

    The network outputs a logit for every cell, occupied or not. This mask is
    what the evaluator uses to set illegal logits to `-inf` before the
    softmax, so illegal moves get exactly zero probability.
    """
    if is_terminal(state):
        return np.zeros(state.size * state.size, dtype=np.bool_)
    return np.array([mark is None for mark in state.board], dtype=np.bool_)


def transform_planes(planes: np.ndarray, k: int) -> np.ndarray:
    """Apply symmetry `k` to the last two (spatial) axes. Returns a contiguous copy.

    Works on a single `[C, S, S]` stack, a batch `[B, C, S, S]`, or a bare
    `[S, S]` grid, because it only touches the last two axes.
    """
    _check_symmetry(k)
    out = planes
    if k >= 4:
        # Mirror left-right: reverse the column axis.
        out = np.flip(out, axis=-1)
    # `k % 4` counter-clockwise quarter turns in the (row, col) plane.
    out = np.rot90(out, k % 4, axes=(-2, -1))
    # `flip` and `rot90` return *views* that walk memory backwards (negative
    # strides) instead of copying. `torch.from_numpy` rejects those, so
    # materialise a normal C-ordered copy here once rather than everywhere.
    return np.ascontiguousarray(out)


def transform_policy(policy: np.ndarray, k: int) -> np.ndarray:
    """Apply symmetry `k` to a policy over cells (the last axis, length `S²`).

    A policy is one number per cell, i.e. a scalar field over the board, so
    it moves exactly like a single plane: unflatten the last axis to
    `[S, S]`, reuse `transform_planes`, flatten back.
    """
    cells = policy.shape[-1]
    size = _side(cells)
    grid = policy.reshape(*policy.shape[:-1], size, size)
    return transform_planes(grid, k).reshape(policy.shape)


def transform_cell(cell: Cell, k: int, size: int) -> Cell:
    """Where `cell` ends up under symmetry `k` on an `S x S` board."""
    row, col = cell
    return divmod(int(_cell_maps(size)[k, row * size + col]), size)


def inverse_symmetry(k: int) -> int:
    """The symmetry that undoes `k` (see the module docstring)."""
    _check_symmetry(k)
    return k if k >= 4 else (-k) % 4


@cache
def _cell_maps(size: int) -> np.ndarray:
    """`int [8, S²]`: entry `[k, i]` is where flat cell `i` lands under symmetry `k`.

    How this works: rather than derive eight coordinate formulas by hand (and
    risk one disagreeing with `transform_planes`), transform a grid whose
    entries are their own flat indices, `np.arange(S²).reshape(S, S)`. After
    the transform, the value `i` sits at the position where cell `i` moved to.
    `argsort` of the flattened result inverts that "position -> old index"
    table into "old index -> position". The array operation is the single
    source of truth for all three transforms.
    """
    grid = np.arange(size * size).reshape(size, size)
    maps = np.empty((NUM_SYMMETRIES, size * size), dtype=np.int64)
    for k in range(NUM_SYMMETRIES):
        maps[k] = np.argsort(transform_planes(grid, k).ravel())
    maps.flags.writeable = False  # cached and shared: guard against mutation
    return maps


def _side(cells: int) -> int:
    size = round(cells**0.5)
    if size * size != cells:
        raise ValueError(f"a policy must have S² entries, got {cells}")
    return size


def _check_symmetry(k: int) -> None:
    if not 0 <= k < NUM_SYMMETRIES:
        raise ValueError(f"symmetry index must be in 0..{NUM_SYMMETRIES - 1}, got {k}")
