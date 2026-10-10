"""The replay buffer: the training set, and the symmetry augmentation applied to it.

Design: docs/design/training.md ("Replay buffer", "Augmentation") and
docs/design/network.md ("Symmetries").
Plan: docs/plans/04-training-pipeline.md, step 2.

What the buffer is
------------------
AlphaZero trains on the self-play examples `(s, π, z)` of recent generations,
sampled uniformly into mini-batches. Training on a *window* of generations
instead of only the newest one smooths learning: consecutive mini-batches mix
positions from many games, so the gradient is not dominated by one
generation's quirks, and the network does not forget positions the newest
games happened not to visit.

Why old generations expire
--------------------------
`π` is the search's visit distribution, and the search was guided by the
network of its generation. A generation-`g - 20` example records what a much
weaker network's search thought; its `π` (and, to a lesser degree, its `z`,
since weak play produces different results) is stale. Training on it would
pull the network back towards its own past. So the buffer keeps only the
newest `K` generations by number: after adding generation `g`, everything with
generation `<= g - K` is dropped. `K` trades freshness against data volume.

The generation is the key, not a column
---------------------------------------
Examples are held in a `dict[int, Examples]` keyed by generation. The key is
the only record of an example's generation: expiry drops whole keys, saving
writes one file per key, and resume deletes keys (files) above the checkpoint.
One source of truth, no per-row column to keep in sync.

Augmentation: free supervision
------------------------------
OXOX's rules are invariant under the 8 symmetries of the square (4 rotations,
each optionally mirrored): an alternating line stays an alternating line. So
if `(s, π, z)` is a correct example, so is `(T s, T π, z)` for every symmetry
`T`; the value does not change, the policy moves with the board. The *game*
is symmetric but the *data* is not (self-play visits some orientations more
than others), so applying a random `T` per example teaches the network the
symmetry for free and multiplies the effective data by up to 8. The one thing
that must never happen is moving the planes without moving `π` identically:
the network would then learn to play the wrong cells.

This module is torch-free: it produces NumPy batches, and the trainer turns
them into tensors.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ox_zero.engine.encoding import NUM_SYMMETRIES, transform_planes, transform_policy
from ox_zero.training.selfplay import Examples

# `gen_007.npz`: the generation number is the file name. Three digits is the
# padding (so a directory listing sorts sensibly by eye); `load` still parses
# the number and sorts numerically, so generation 1000 would work too.
_FILE_PATTERN = re.compile(r"^gen_(\d+)\.npz$")


def _file_name(generation: int) -> str:
    return f"gen_{generation:03d}.npz"


@dataclass(frozen=True)
class Batch:
    """One training mini-batch, ready to become tensors.

    Attributes:
        planes: `float32 [B, 3, S, S]`, the network input.
        pi: `float32 [B, S²]`, the policy targets (rows sum to 1).
        z: `float32 [B]`, the value targets for the side to move.
    """

    planes: np.ndarray
    pi: np.ndarray
    z: np.ndarray


def legal_from_planes(planes: np.ndarray) -> np.ndarray:
    """`bool [B, S²]`: the empty cells, i.e. planes 0 and 1 both zero.

    The loss needs a legal mask (illegal logits are excluded from the policy
    softmax). Rather than store one per example, derive it: plane 0 holds the
    side to move's marks and plane 1 the opponent's, so a cell is empty when
    both are zero, and in a non-terminal position every empty cell is legal.
    Recorded positions are never terminal (a move was searched and played from
    each), so this equals `legal_mask(state)` for every example. Works on
    `uint8` or `float32` planes, and after any symmetry, since the mask is
    computed from the (already transformed) planes themselves.
    """
    batch = planes.shape[0]
    # [B, S, S] & [B, S, S] -> [B, S, S], then flatten the grid to board order
    # (row-major, index row * S + col), matching policies.
    return ((planes[:, 0] == 0) & (planes[:, 1] == 0)).reshape(batch, -1)


class ReplayBuffer:
    """The examples of the newest `K` generations, sampled uniformly.

    Args:
        size: Board side `S`; every added example must match it.
        generations: `K`, how many generations (by number) to keep.
    """

    def __init__(self, size: int, generations: int) -> None:
        if generations < 1:
            raise ValueError(f"generations must be at least 1, got {generations}")
        self.size = size
        self.capacity = generations
        # Generation -> its examples. Plain dict; `generations` sorts the keys.
        self._by_generation: dict[int, Examples] = {}

    def add(self, examples: Examples, generation: int) -> None:
        """Add one batch of examples for `generation`, then expire old generations.

        A generation may arrive in several batches (several self-play calls);
        they are concatenated. After the add, every generation `<= g - K`,
        where `g` is the newest held, is dropped: its `π` came from a network
        `K` or more generations weaker.

        Raises:
            ValueError: the examples are for another board size.
        """
        if examples.size != self.size:
            raise ValueError(
                f"examples are for a {examples.size}x{examples.size} board, "
                f"the buffer holds {self.size}x{self.size}"
            )
        held = self._by_generation.get(generation)
        self._by_generation[generation] = (
            examples if held is None else Examples.concatenate([held, examples])
        )
        newest = max(self._by_generation)
        for old in [g for g in self._by_generation if g <= newest - self.capacity]:
            del self._by_generation[old]

    def __len__(self) -> int:
        return sum(len(examples) for examples in self._by_generation.values())

    @property
    def generations(self) -> list[int]:
        """The held generation numbers, ascending."""
        return sorted(self._by_generation)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """`batch_size` examples drawn uniformly over positions, with replacement.

        Uniform over *positions*, not games or generations: each held example
        is equally likely, as in AlphaZero. With replacement because the
        trainer takes many mini-batches per generation and independence
        between draws is what makes each mini-batch gradient an unbiased
        estimate of the full-buffer gradient.

        The rows are indexed over the held generations concatenated in
        ascending generation order. A buffer restored by `load` therefore
        lays out its rows exactly like the in-memory one, so the same rng
        state draws the same examples, which is what makes a resumed run
        reproduce an uninterrupted one.

        Raises:
            ValueError: the buffer is empty.
        """
        total = len(self)
        if total == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = rng.integers(0, total, size=batch_size)
        # Concatenating on every call costs one copy of the buffer per step.
        # At the 6x6 scale (tens of thousands of rows) that is cheap next to
        # a training step; caching the concatenation is an easy upgrade if a
        # 12x12 run shows it in a profile.
        held = Examples.concatenate([self._by_generation[g] for g in self.generations])
        return Batch(
            # uint8 0/1 -> float32 0.0/1.0: the network's input dtype.
            planes=held.planes[indices].astype(np.float32),
            pi=held.pi[indices],
            z=held.z[indices],
        )

    def save(self, directory: Path) -> None:
        """Persist to `directory/gen_NNN.npz`, one file per held generation.

        Incremental: a generation's examples never change once its self-play
        is done, so only generations with no file yet are written, and the
        files of expired generations (not held, older than the newest held)
        are deleted. A save therefore costs one generation's bytes, not the
        whole buffer's. Files above the newest held generation are left
        alone; resume removes those through `load(..., upto=...)`.

        Each write is atomic: the data goes to a temporary file in the same
        directory, which is then renamed over the final name (a rename within
        one filesystem is atomic), so a crash mid-write never leaves a
        truncated `gen_NNN.npz` that `load` would trip over.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        on_disk = _generation_files(directory)
        held = self.generations

        for generation in held:
            if generation in on_disk:
                continue
            examples = self._by_generation[generation]
            fd, temporary = tempfile.mkstemp(dir=directory, prefix=".gen_", suffix=".tmp")
            try:
                # Write through the open file object, not the path: given a
                # path that does not end in `.npz`, `np.savez` silently
                # appends the suffix, and the rename below would then miss
                # the file it actually wrote.
                with os.fdopen(fd, "wb") as handle:
                    np.savez(handle, planes=examples.planes, pi=examples.pi, z=examples.z)
                os.replace(temporary, directory / _file_name(generation))
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise

        if held:
            newest = held[-1]
            for generation, path in on_disk.items():
                if generation not in self._by_generation and generation < newest:
                    path.unlink()

    @classmethod
    def load(
        cls, directory: Path, size: int, generations: int, upto: int | None = None
    ) -> ReplayBuffer:
        """Restore a buffer saved by `save`.

        Generations are read in ascending numeric order (not directory-listing
        order), so `sample` lays out rows as the saved buffer did. `upto`, the
        generation of the checkpoint being resumed, drops and deletes files
        above it: the buffer is written before the checkpoint, so a crash
        between the two leaves a generation of examples that the checkpoint's
        network never trained on and that the resumed run will play again.

        An empty directory gives an empty buffer of `size`.

        Raises:
            ValueError: a file holds examples for another board size.
        """
        directory = Path(directory)
        buffer = cls(size, generations)
        for generation, path in sorted(_generation_files(directory).items()):
            if upto is not None and generation > upto:
                path.unlink()
                continue
            with np.load(path) as data:
                examples = Examples(planes=data["planes"], pi=data["pi"], z=data["z"])
            if examples.size != size:
                raise ValueError(
                    f"{path.name} holds {examples.size}x{examples.size} examples, "
                    f"expected {size}x{size}"
                )
            buffer.add(examples, generation)
        return buffer

    def drop_above(self, generation: int) -> None:
        """Forget every held generation numbered above `generation`."""
        for g in [g for g in self._by_generation if g > generation]:
            del self._by_generation[g]


def _generation_files(directory: Path) -> dict[int, Path]:
    """`{generation: path}` for every `gen_NNN.npz` in `directory` (none if it is missing)."""
    if not directory.is_dir():
        return {}
    files: dict[int, Path] = {}
    for path in directory.iterdir():
        match = _FILE_PATTERN.match(path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def transform_batch(batch: Batch, ks: np.ndarray) -> Batch:
    """Apply symmetry `ks[i]` to example `i`: planes and `π` together, `z` unchanged.

    How this works: `transform_planes` / `transform_policy` take one `k` but
    act on any leading batch axes. So rather than loop over examples, group
    the rows by their symmetry and make (at most) 8 batched calls, writing
    each group's result back into its rows. The same `k` goes to a row's
    planes and its `π`, which is what keeps the policy target pointing at the
    same (moved) cells; `z` is a scalar about the position and no symmetry
    changes who is winning.
    """
    ks = np.asarray(ks)
    if ks.shape != batch.z.shape:
        raise ValueError(f"need one symmetry per example: ks {ks.shape}, batch {batch.z.shape}")
    planes = np.empty_like(batch.planes)
    pi = np.empty_like(batch.pi)
    for k in range(NUM_SYMMETRIES):
        rows = ks == k
        if rows.any():
            planes[rows] = transform_planes(batch.planes[rows], k)
            pi[rows] = transform_policy(batch.pi[rows], k)
    return Batch(planes=planes, pi=pi, z=batch.z.copy())


def augment(batch: Batch, rng: np.random.Generator) -> Batch:
    """A uniformly random symmetry per example (see the module docstring).

    Per example rather than one per batch: every mini-batch then shows the
    network all eight orientations at once, for the same eight array calls
    (plan 04, "Augmentation is per example, in 8 groups"). Drawn from the
    run's one NumPy generator so a seeded run is reproducible.
    """
    ks = rng.integers(0, NUM_SYMMETRIES, size=batch.z.shape[0])
    return transform_batch(batch, ks)
