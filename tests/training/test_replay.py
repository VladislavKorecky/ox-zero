"""Tests for the replay buffer: expiry, uniform sampling, augmentation, persistence.

The replay buffer is the training set of AlphaZero: the examples of the last
`K` generations of self-play, sampled uniformly into mini-batches. These tests
pin the four things it must get right: which generations it keeps (old ones
expire because their `π` came from a weaker network), that sampling is
uniform, that a board symmetry moves the planes and the policy target
together (otherwise augmentation would teach the network wrong moves), and
that a saved buffer comes back exactly, in the same order, so a resumed run
draws the same examples.

Plan: docs/plans/04-training-pipeline.md, step 2. Torch-free.
"""

import os
import stat

import numpy as np
import pytest

import ox_zero.training.replay as replay_module
from ox_zero.engine.encoding import encode, legal_mask, transform_planes, transform_policy
from ox_zero.game.rules import State, play
from ox_zero.training.replay import (
    Batch,
    ReplayBuffer,
    augment,
    legal_from_planes,
    transform_batch,
)
from ox_zero.training.selfplay import Examples


def examples_of(states: list[State], z: list[float] | None = None) -> Examples:
    """Examples for `states`: real planes, a uniform legal `π`, and the given `z`."""
    planes = np.stack([encode(s) for s in states]).astype(np.uint8)
    masks = np.stack([legal_mask(s) for s in states]).astype(np.float32)
    pi = masks / masks.sum(axis=1, keepdims=True)
    zs = np.zeros(len(states), dtype=np.float32) if z is None else np.array(z, dtype=np.float32)
    return Examples(planes=planes, pi=pi.astype(np.float32), z=zs)


def tagged(size: int, n: int, tag: float) -> Examples:
    """`n` examples of the empty `size` board whose `z` is `tag` (to tell batches apart)."""
    return examples_of([play([], size=size)] * n, [tag] * n)


# --- 1. Expiry by generation -------------------------------------------------


def test_expiry_keeps_newest_k_generations() -> None:
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 2, 1.0), generation=1)
    buffer.add(tagged(3, 3, 2.0), generation=2)
    buffer.add(tagged(3, 4, 3.0), generation=3)
    assert buffer.generations == [2, 3]
    assert len(buffer) == 3 + 4

    # A second batch of an existing generation appends to it.
    buffer.add(tagged(3, 5, 3.0), generation=3)
    assert buffer.generations == [2, 3]
    assert len(buffer) == 3 + 4 + 5


def test_add_rejects_another_board_size() -> None:
    buffer = ReplayBuffer(size=3, generations=2)
    with pytest.raises(ValueError):
        buffer.add(tagged(4, 1, 0.0), generation=1)


# --- 2. Uniform sampling ------------------------------------------------------


def test_sampling_is_uniform_and_typed() -> None:
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 1, 1.0), generation=1)
    buffer.add(tagged(3, 1, -1.0), generation=2)

    batch = buffer.sample(10_000, np.random.default_rng(0))
    share = float(np.mean(batch.z == 1.0))
    assert 0.45 <= share <= 0.55

    assert batch.planes.dtype == np.float32
    assert batch.planes.shape == (10_000, 3, 3, 3)
    assert batch.pi.dtype == np.float32
    assert batch.pi.shape == (10_000, 9)
    assert batch.z.dtype == np.float32
    assert batch.z.shape == (10_000,)


def test_sampling_empty_buffer_raises() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(size=3, generations=2).sample(4, np.random.default_rng(0))


# --- 3. Augmentation is consistent -------------------------------------------


def test_transform_batch_matches_engine_transforms() -> None:
    # An asymmetric 4x4 position, so the eight symmetries all differ.
    state = play([(0, 0), (0, 1), (2, 3)], size=4)
    pi0 = np.zeros(16, dtype=np.float32)
    pi0[1 * 4 + 2] = 1.0  # one-hot on the legal cell (1, 2)
    assert legal_mask(state)[1 * 4 + 2]

    planes0 = encode(state)
    batch = Batch(
        planes=np.repeat(planes0[None], 8, axis=0),
        pi=np.repeat(pi0[None], 8, axis=0),
        z=np.full(8, 0.5, dtype=np.float32),
    )
    out = transform_batch(batch, ks=np.arange(8))
    for k in range(8):
        np.testing.assert_array_equal(out.planes[k], transform_planes(planes0, k))
        np.testing.assert_array_equal(out.pi[k], transform_policy(pi0, k))
        assert out.z[k] == 0.5
        np.testing.assert_array_equal(
            legal_from_planes(out.planes[k : k + 1])[0],
            transform_policy(legal_mask(state), k),
        )


def test_augment_moves_policy_mass_with_the_board() -> None:
    # A mixed batch: several positions, each with π one-hot on a legal cell.
    rng = np.random.default_rng(1)
    states = [
        play([], size=4),
        play([(0, 0)], size=4),
        play([(0, 0), (3, 2)], size=4),
        play([(1, 1), (0, 3), (2, 0)], size=4),
    ] * 8
    planes = np.stack([encode(s) for s in states])
    pi = np.zeros((len(states), 16), dtype=np.float32)
    for i, s in enumerate(states):
        legal = np.flatnonzero(legal_mask(s))
        pi[i, rng.choice(legal)] = 1.0
    batch = Batch(planes=planes, pi=pi, z=np.zeros(len(states), dtype=np.float32))

    out = augment(batch, np.random.default_rng(2))
    legal = legal_from_planes(out.planes)
    rows = np.arange(len(states))
    assert np.all(legal[rows, np.argmax(out.pi, axis=1)])


# --- 4. k = 0 is the identity; augment draws several k ------------------------


def test_k_zero_is_identity() -> None:
    state = play([(0, 0), (0, 1), (2, 3)], size=4)
    batch = Batch(
        planes=encode(state)[None],
        pi=(legal_mask(state) / legal_mask(state).sum()).astype(np.float32)[None],
        z=np.array([1.0], dtype=np.float32),
    )
    out = transform_batch(batch, ks=np.zeros(1, dtype=np.int64))
    np.testing.assert_array_equal(out.planes, batch.planes)
    np.testing.assert_array_equal(out.pi, batch.pi)
    np.testing.assert_array_equal(out.z, batch.z)


@pytest.mark.parametrize("bad", [-1, 8])
def test_transform_batch_rejects_out_of_range_symmetries(bad: int) -> None:
    state = play([], size=4)
    batch = Batch(
        planes=np.repeat(encode(state)[None], 2, axis=0),
        pi=np.full((2, 16), 1 / 16, dtype=np.float32),
        z=np.zeros(2, dtype=np.float32),
    )
    with pytest.raises(ValueError):
        transform_batch(batch, ks=np.array([0, bad]))


def test_augment_uses_several_symmetries(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[np.ndarray] = []
    real = replay_module.transform_batch

    def spy(batch: Batch, ks: np.ndarray) -> Batch:
        seen.append(np.asarray(ks))
        return real(batch, ks)

    monkeypatch.setattr(replay_module, "transform_batch", spy)
    state = play([], size=4)
    batch = Batch(
        planes=np.repeat(encode(state)[None], 64, axis=0),
        pi=np.full((64, 16), 1 / 16, dtype=np.float32),
        z=np.zeros(64, dtype=np.float32),
    )
    augment(batch, np.random.default_rng(0))
    assert len(seen) == 1
    assert seen[0].shape == (64,)
    assert len(np.unique(seen[0])) > 1
    assert np.all((seen[0] >= 0) & (seen[0] < 8))


# --- 5. Save and load ---------------------------------------------------------


def distinct(n: int, seed: int) -> Examples:
    """`n` 3x3 examples whose rows all differ: varied positions, random `π` and `z`."""
    rng = np.random.default_rng(seed)
    games = [[], [(0, 0)], [(0, 0), (1, 1)], [(2, 2), (0, 1)], [(1, 0)]]
    states = [play(games[i % len(games)], size=3) for i in range(n)]
    planes = np.stack([encode(s) for s in states]).astype(np.uint8)
    pi = rng.random((n, 9)).astype(np.float32)
    pi /= pi.sum(axis=1, keepdims=True)
    return Examples(planes=planes, pi=pi, z=rng.uniform(-1, 1, n).astype(np.float32))


def test_save_and_load_round_trip(tmp_path) -> None:
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(distinct(2, seed=1), generation=1)
    buffer.add(distinct(3, seed=2), generation=2)
    buffer.add(distinct(4, seed=3), generation=3)
    buffer.save(directory)
    assert sorted(p.name for p in directory.iterdir()) == ["gen_002.npz", "gen_003.npz"]

    loaded = ReplayBuffer.load(directory, size=3, generations=2)
    assert loaded.generations == [2, 3]
    assert len(loaded) == len(buffer)
    # Same order as the in-memory buffer: the same rng draws the same examples.
    # 500 draws over 7 distinct rows hit every row, so this also shows every
    # array came back exactly.
    a = buffer.sample(500, np.random.default_rng(7))
    b = loaded.sample(500, np.random.default_rng(7))
    assert len(np.unique(a.z)) == 7
    np.testing.assert_array_equal(a.planes, b.planes)
    np.testing.assert_array_equal(a.pi, b.pi)
    np.testing.assert_array_equal(a.z, b.z)

    # Expiry continues after loading.
    loaded.add(distinct(4, seed=4), generation=4)
    assert loaded.generations == [3, 4]

    # A second save: gen_003 untouched, gen_004 written, gen_002 deleted.
    # Backdate gen_003 first so a rewrite would show even on a coarse clock.
    path3 = directory / "gen_003.npz"
    old = path3.stat().st_mtime_ns - 10**9
    os.utime(path3, ns=(old, old))
    loaded.save(directory)
    assert path3.stat().st_mtime_ns == old
    assert sorted(p.name for p in directory.iterdir()) == ["gen_003.npz", "gen_004.npz"]


def test_load_sorts_by_generation_number(tmp_path) -> None:
    directory = tmp_path / "buffer"
    # Write the files out of order (3 before 2), each from its own buffer.
    # Each buffer saves into its own scratch directory and the file is moved
    # over: `save` deletes files of generations it does not hold, so saving
    # the second buffer into `directory` directly would delete the first's.
    directory.mkdir()
    for generation, tag in ((3, 3.0), (2, 2.0)):
        single = ReplayBuffer(size=3, generations=5)
        single.add(tagged(3, generation, tag), generation=generation)
        scratch = tmp_path / f"scratch_{generation}"
        single.save(scratch)
        name = f"gen_{generation:03d}.npz"
        os.replace(scratch / name, directory / name)
    loaded = ReplayBuffer.load(directory, size=3, generations=5)
    assert loaded.generations == [2, 3]
    assert len(loaded) == 2 + 3

    reference = ReplayBuffer(size=3, generations=5)
    reference.add(tagged(3, 2, 2.0), generation=2)
    reference.add(tagged(3, 3, 3.0), generation=3)
    a = reference.sample(100, np.random.default_rng(3))
    b = loaded.sample(100, np.random.default_rng(3))
    np.testing.assert_array_equal(a.z, b.z)


def test_load_empty_directory_and_size_check(tmp_path) -> None:
    directory = tmp_path / "buffer"
    directory.mkdir()
    empty = ReplayBuffer.load(directory, size=4, generations=2)
    assert len(empty) == 0 and empty.generations == []

    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 1, 0.0), generation=1)
    buffer.save(directory)
    with pytest.raises(ValueError):
        ReplayBuffer.load(directory, size=4, generations=2)


def test_save_leaves_no_temporary_file(tmp_path) -> None:
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 2, 0.0), generation=1)
    buffer.save(directory)
    buffer.add(tagged(3, 2, 0.0), generation=2)
    buffer.save(directory)
    assert sorted(p.name for p in directory.iterdir()) == ["gen_001.npz", "gen_002.npz"]


def test_load_upto_drops_and_deletes_later_files(tmp_path) -> None:
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=3)
    for generation in (2, 3, 4):
        buffer.add(tagged(3, 1, float(generation)), generation=generation)
    buffer.save(directory)

    loaded = ReplayBuffer.load(directory, size=3, generations=3, upto=3)
    assert loaded.generations == [2, 3]
    assert sorted(p.name for p in directory.iterdir()) == ["gen_002.npz", "gen_003.npz"]


def test_drop_above() -> None:
    buffer = ReplayBuffer(size=3, generations=3)
    for generation in (1, 2, 3):
        buffer.add(tagged(3, 1, 0.0), generation=generation)
    buffer.drop_above(1)
    assert buffer.generations == [1]
    assert len(buffer) == 1


def test_drop_above_keeps_the_boundary_and_handles_no_op() -> None:
    buffer = ReplayBuffer(size=3, generations=5)
    for generation in (2, 3, 4):
        buffer.add(tagged(3, generation, float(generation)), generation=generation)
    buffer.drop_above(3)  # 3 itself stays: "above" is strict
    assert buffer.generations == [2, 3]
    buffer.drop_above(10)  # nothing above 10: no change
    assert buffer.generations == [2, 3]
    # Sampling sees only what is held after the drop (no stale cached rows).
    batch = buffer.sample(200, np.random.default_rng(0))
    assert set(np.unique(batch.z)) == {2.0, 3.0}
    buffer.drop_above(0)
    assert buffer.generations == [] and len(buffer) == 0


def test_drop_above_then_re_added_generation_is_saved(tmp_path) -> None:
    # Resume in memory: generation 6 is forgotten and played again. The
    # re-run's examples must replace the stale gen_006.npz on the next save.
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=5)
    buffer.add(tagged(3, 2, 5.0), generation=5)
    buffer.add(tagged(3, 2, 6.0), generation=6)
    buffer.save(directory)

    buffer.drop_above(5)
    buffer.add(tagged(3, 3, 66.0), generation=6)
    buffer.save(directory)

    loaded = ReplayBuffer.load(directory, size=3, generations=5)
    assert loaded.generations == [5, 6]
    assert len(loaded) == 2 + 3
    batch = loaded.sample(300, np.random.default_rng(0))
    assert set(np.unique(batch.z)) == {5.0, 66.0}


# --- Saving generations that changed after their file was written -------------


def test_save_rewrites_a_generation_that_grew(tmp_path) -> None:
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 5, 1.0), generation=5)
    buffer.save(directory)
    buffer.add(tagged(3, 5, 2.0), generation=5)  # a second batch of the same generation
    buffer.save(directory)

    loaded = ReplayBuffer.load(directory, size=3, generations=2)
    assert len(loaded) == 10
    a = buffer.sample(300, np.random.default_rng(4))
    b = loaded.sample(300, np.random.default_rng(4))
    np.testing.assert_array_equal(a.z, b.z)


def test_save_after_load_then_add_rewrites_the_generation(tmp_path) -> None:
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 2, 1.0), generation=1)
    buffer.add(tagged(3, 2, 2.0), generation=2)
    buffer.save(directory)

    loaded = ReplayBuffer.load(directory, size=3, generations=2)
    # Loading must not count as a change: an unchanged generation is not rewritten.
    path1 = directory / "gen_001.npz"
    old = path1.stat().st_mtime_ns - 10**9
    os.utime(path1, ns=(old, old))
    loaded.add(tagged(3, 3, 22.0), generation=2)
    loaded.save(directory)
    assert path1.stat().st_mtime_ns == old

    again = ReplayBuffer.load(directory, size=3, generations=2)
    assert len(again) == 2 + 2 + 3
    assert set(np.unique(again.sample(300, np.random.default_rng(0)).z)) == {1.0, 2.0, 22.0}


def test_save_to_a_second_directory_writes_everything(tmp_path) -> None:
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 2, 1.0), generation=1)
    buffer.save(tmp_path / "a")
    buffer.save(tmp_path / "b")
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == ["gen_001.npz"]


# --- Durability: fsync, stale temporary files ---------------------------------


def test_save_fsyncs_before_rename(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def replace(src, dst) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(replay_module.os, "fsync", fsync)
    monkeypatch.setattr(replay_module.os, "replace", replace)
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 1, 0.0), generation=1)
    buffer.save(tmp_path)
    assert "replace" in events
    assert "fsync" in events[: events.index("replace")]


def test_save_fsyncs_the_directory_after_the_renames(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A rename is a change to the *directory*; until the directory itself is
    # fsynced, a power loss can forget it. So the last fsync must be on a
    # directory descriptor, after every replace and unlink.
    events: list[str] = []
    real_fsync, real_replace, real_unlink = os.fsync, os.replace, replay_module.Path.unlink

    def fsync(fd: int) -> None:
        events.append("fsync-dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync-file")
        real_fsync(fd)

    def replace(src, dst) -> None:
        events.append("replace")
        real_replace(src, dst)

    def unlink(self, missing_ok: bool = False) -> None:
        events.append("unlink")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(replay_module.os, "fsync", fsync)
    monkeypatch.setattr(replay_module.os, "replace", replace)
    monkeypatch.setattr(replay_module.Path, "unlink", unlink)
    buffer = ReplayBuffer(size=3, generations=1)
    buffer.add(tagged(3, 1, 0.0), generation=1)
    buffer.save(tmp_path)
    buffer.add(tagged(3, 1, 0.0), generation=2)  # expires generation 1
    events.clear()
    buffer.save(tmp_path)
    assert "replace" in events and "unlink" in events
    assert events[-1] == "fsync-dir"


def test_save_deletes_files_of_generations_not_held(tmp_path) -> None:
    # `drop_above` is memory-only; a dropped generation that is never re-added
    # must not survive on disk, or a later `load` would bring it back.
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=3)
    for generation in (1, 2, 3):
        buffer.add(tagged(3, 1, float(generation)), generation=generation)
    buffer.save(directory)
    buffer.drop_above(1)
    buffer.save(directory)
    assert sorted(p.name for p in directory.iterdir()) == ["gen_001.npz"]
    assert ReplayBuffer.load(directory, size=3, generations=3).generations == [1]


def test_saved_files_follow_the_umask(tmp_path) -> None:
    # `mkstemp` creates owner-only (0600) files and the rename keeps the
    # mode; saved files should get the ordinary 0666 & ~umask, like any file
    # `open()` creates.
    old_umask = os.umask(0o022)
    try:
        buffer = ReplayBuffer(size=3, generations=2)
        buffer.add(tagged(3, 1, 0.0), generation=1)
        buffer.save(tmp_path)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE((tmp_path / "gen_001.npz").stat().st_mode) == 0o644


def test_stale_temporary_files_are_ignored_and_removed(tmp_path) -> None:
    directory = tmp_path / "buffer"
    buffer = ReplayBuffer(size=3, generations=2)
    buffer.add(tagged(3, 1, 0.0), generation=1)
    buffer.save(directory)

    # What a SIGKILL mid-save leaves behind.
    (directory / ".gen_abc123.tmp").write_bytes(b"half a file")
    loaded = ReplayBuffer.load(directory, size=3, generations=2)
    assert loaded.generations == [1]
    assert sorted(p.name for p in directory.iterdir()) == ["gen_001.npz"]

    (directory / ".gen_def456.tmp").write_bytes(b"half a file")
    loaded.save(directory)
    assert sorted(p.name for p in directory.iterdir()) == ["gen_001.npz"]


# --- Sampling caches the concatenation ----------------------------------------


def test_sample_caches_the_concatenation(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = ReplayBuffer(size=3, generations=3)
    buffer.add(tagged(3, 2, 1.0), generation=1)
    buffer.add(tagged(3, 2, 2.0), generation=2)
    reference = buffer.sample(50, np.random.default_rng(9))

    calls = 0
    real = Examples.concatenate

    def counting(parts):
        nonlocal calls
        calls += 1
        return real(parts)

    monkeypatch.setattr(Examples, "concatenate", staticmethod(counting))
    first = buffer.sample(50, np.random.default_rng(9))
    second = buffer.sample(50, np.random.default_rng(9))
    assert calls == 0  # served from the cache built by the first sample
    np.testing.assert_array_equal(first.z, reference.z)
    np.testing.assert_array_equal(second.z, reference.z)

    # Mutations invalidate the cache: new rows show up in the next sample.
    buffer.add(tagged(3, 2, 3.0), generation=3)
    assert 3.0 in buffer.sample(300, np.random.default_rng(0)).z
    buffer.drop_above(2)
    assert 3.0 not in buffer.sample(300, np.random.default_rng(0)).z


# --- legal_from_planes --------------------------------------------------------


def test_legal_from_planes_matches_legal_mask() -> None:
    states = [play([], size=4), play([(0, 0), (1, 1), (3, 3)], size=4)]
    planes = np.stack([encode(s) for s in states])
    expected = np.stack([legal_mask(s) for s in states])
    np.testing.assert_array_equal(legal_from_planes(planes), expected)
    np.testing.assert_array_equal(legal_from_planes(planes.astype(np.uint8)), expected)
    assert legal_from_planes(planes).dtype == np.bool_
