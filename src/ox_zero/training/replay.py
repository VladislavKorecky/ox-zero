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

# The temporary files `save` writes before renaming them into place. A process
# killed mid-save (SIGKILL, power loss) can leave one behind; it is never a
# valid generation file, so `load` and `save` delete any they find.
_TEMP_PREFIX = ".gen_"
_TEMP_SUFFIX = ".tmp"


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
        # Dirty tracking for incremental saves. `_clean` holds the generations
        # whose in-memory examples are byte-for-byte what `_clean_directory`
        # holds on disk (set by `save` after writing, and by `load` after
        # reading). Anything else held is "dirty" and `save` rewrites it:
        # a generation that grew by a second `add` batch, or one re-played
        # after `drop_above`. Why it matters: resume must reproduce an
        # uninterrupted run, which needs the files to match memory exactly;
        # a skipped rewrite would silently lose rows. Unchanged generations
        # stay clean, so a save still writes only what changed.
        self._clean: set[int] = set()
        self._clean_directory: Path | None = None
        # `sample` needs the held generations as one array set (ascending
        # generation order). Building it copies the whole buffer, so it is
        # cached here and invalidated by every mutation (`_changed`).
        self._concatenated: Examples | None = None

    def _changed(self, generation: int) -> None:
        """Bookkeeping after `generation`'s examples were modified or removed:
        it no longer matches its file, and the sampling cache is stale."""
        self._clean.discard(generation)
        self._concatenated = None

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
        self._changed(generation)
        newest = max(self._by_generation)
        for old in [g for g in self._by_generation if g <= newest - self.capacity]:
            del self._by_generation[old]
            self._changed(old)

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
        # Concatenating costs one copy of the whole buffer, and the trainer
        # samples many mini-batches between mutations, so the result is
        # cached and rebuilt only after `add` / `drop_above` change the
        # buffer. The layout (ascending generations) is the same either way,
        # so caching never changes which rows an rng draws.
        if self._concatenated is None:
            self._concatenated = Examples.concatenate(
                [self._by_generation[g] for g in self.generations]
            )
        held = self._concatenated
        return Batch(
            # uint8 0/1 -> float32 0.0/1.0: the network's input dtype.
            planes=held.planes[indices].astype(np.float32),
            pi=held.pi[indices],
            z=held.z[indices],
        )

    def save(self, directory: Path) -> None:
        """Persist to `directory/gen_NNN.npz`, one file per held generation.

        Incremental: a generation is written only if it has no file yet or
        has changed since its file was written or read (a second `add` batch
        for it, or a re-play after `drop_above`; see `_clean` in `__init__`).
        An unchanged generation's file is not touched, so a save normally
        costs one generation's bytes, not the whole buffer's. The files of
        expired generations (not held, older than the newest held) are
        deleted. Files above the newest held generation are left alone;
        resume removes those through `load(..., upto=...)`, and a re-played
        generation overwrites its stale file because it is dirty.

        Each write is atomic and durable: the data goes to a temporary file
        in the same directory, is `fsync`ed to the disk, and is then renamed
        over the final name (a rename within one filesystem is atomic). So a
        crash never leaves a truncated `gen_NNN.npz` that `load` would trip
        over, nor (after power loss) a renamed file whose data never reached
        the disk. Stale temporary files from a killed save are removed.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _remove_stale_temporaries(directory)
        on_disk = _generation_files(directory)
        held = self.generations
        # "Clean" means "matches the files in the directory it was synced
        # with"; against any other directory nothing is known to match.
        same_directory = (
            self._clean_directory is not None
            and self._clean_directory.resolve() == directory.resolve()
        )
        clean = self._clean if same_directory else set()

        for generation in held:
            if generation in on_disk and generation in clean:
                continue  # unchanged since its file was written: leave it be
            examples = self._by_generation[generation]
            fd, temporary = tempfile.mkstemp(
                dir=directory, prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX
            )
            try:
                # Write through the open file object, not the path: given a
                # path that does not end in `.npz`, `np.savez` silently
                # appends the suffix, and the rename below would then miss
                # the file it actually wrote.
                with os.fdopen(fd, "wb") as handle:
                    np.savez(handle, planes=examples.planes, pi=examples.pi, z=examples.z)
                    # Flush Python's buffer to the OS, then force the OS to
                    # put the bytes on disk *before* the rename. Without the
                    # fsync a power loss can persist the rename (metadata)
                    # but not the data, leaving an empty or garbage file
                    # under the final name.
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, directory / _file_name(generation))
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise

        # Every held generation now matches this directory.
        self._clean = set(held)
        self._clean_directory = directory

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
        if directory.is_dir():
            _remove_stale_temporaries(directory)
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
        # What was just read is exactly what is on disk, so the next `save`
        # need not rewrite it (expiry inside `add` may have dropped some
        # generations; only the held ones are marked).
        buffer._clean = set(buffer._by_generation)
        buffer._clean_directory = directory
        return buffer

    def drop_above(self, generation: int) -> None:
        """Forget every held generation numbered above `generation`.

        Memory only: the files stay. That is safe because a generation
        dropped here and later re-added is dirty (see `_clean`), so the next
        `save` overwrites its stale file; and a dropped generation that is
        never re-added is either deleted by `save` once a newer one is held,
        or removed by `load(..., upto=...)` on resume.
        """
        for g in [g for g in self._by_generation if g > generation]:
            del self._by_generation[g]
            self._changed(g)


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


def _remove_stale_temporaries(directory: Path) -> None:
    """Delete `.gen_*.tmp` files left by a save that was killed mid-write.

    Safe because only one process saves a run's buffer, and `save` removes
    its own temporary on any exception, so any temporary seen here belongs to
    a dead process and holds an incomplete write.
    """
    for path in directory.glob(f"{_TEMP_PREFIX}*{_TEMP_SUFFIX}"):
        path.unlink(missing_ok=True)


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
    # The loop below fills only rows whose k is in 0..7; a row with any other
    # k would keep `np.empty_like`'s uninitialised memory and silently become
    # a garbage training example. So reject such ks up front.
    if ks.size and (ks.min() < 0 or ks.max() >= NUM_SYMMETRIES):
        raise ValueError(f"symmetry indices must be in 0..{NUM_SYMMETRIES - 1}, got {ks}")
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
