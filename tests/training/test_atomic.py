"""Tests for `atomic_write`, the one atomic-and-durable file write of the training package.

A run's files (checkpoints, buffer generations, metrics, ratings) must never
be seen half-written: a crash mid-write would otherwise leave a truncated
file under the final name. The helper writes a uniquely named temporary file
in the same directory, fsyncs it, renames it over the target (atomic within
one filesystem) and fsyncs the directory. These tests pin that contract:
the content, the order of the syscalls, the permissions, and that a failed
write leaves neither a temporary nor a damaged target behind.
"""

import os
import stat

import pytest

import ox_zero.training.atomic as atomic_module
from ox_zero.training.atomic import atomic_write


def test_writes_the_content(tmp_path):
    path = tmp_path / "out.bin"
    atomic_write(path, lambda file: file.write(b"hello"))
    assert path.read_bytes() == b"hello"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


def test_replaces_an_existing_file(tmp_path):
    path = tmp_path / "out.bin"
    path.write_bytes(b"old")
    atomic_write(path, lambda file: file.write(b"new"))
    assert path.read_bytes() == b"new"


def test_fsyncs_the_file_then_renames_then_fsyncs_the_directory(tmp_path, monkeypatch):
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("fsync-dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync-file")
        real_fsync(fd)

    def replace_(src, dst):
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(atomic_module.os, "fsync", fsync)
    monkeypatch.setattr(atomic_module.os, "replace", replace_)
    atomic_write(tmp_path / "out.bin", lambda file: file.write(b"x"))
    assert events == ["fsync-file", "replace", "fsync-dir"]


def test_a_failing_write_cleans_up_and_keeps_the_old_file(tmp_path):
    path = tmp_path / "out.bin"
    path.write_bytes(b"old")

    def explode(file):
        file.write(b"half")
        raise RuntimeError("disk on fire")

    with pytest.raises(RuntimeError, match="disk on fire"):
        atomic_write(path, explode)
    # No temporary left behind, and the target is untouched.
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]
    assert path.read_bytes() == b"old"


def test_a_keyboard_interrupt_also_cleans_up(tmp_path):
    def interrupt(file):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        atomic_write(tmp_path / "out.bin", interrupt)
    assert list(tmp_path.iterdir()) == []


def test_the_file_follows_the_umask(tmp_path):
    previous = os.umask(0o022)
    try:
        path = tmp_path / "out.bin"
        atomic_write(path, lambda file: file.write(b"x"))
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_temporaries_are_hidden_and_unique(tmp_path):
    """The temporary starts with a dot and ends in `.tmp` (so no reader
    mistakes it for a finished file), and two writes never share one."""
    real_open = os.open
    names: list[str] = []

    def spy_open(path, *args, **kwargs):
        names.append(os.path.basename(path))
        return real_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(atomic_module.os, "open", spy_open)
        atomic_write(tmp_path / "out.bin", lambda file: file.write(b"x"))
        atomic_write(tmp_path / "out.bin", lambda file: file.write(b"x"))
    # The directory fsync opens the directory too; keep only the temporaries.
    temporaries = [name for name in names if name != tmp_path.name]
    assert len(temporaries) == 2
    assert temporaries[0] != temporaries[1]
    for name in temporaries:
        assert name.startswith(".out.bin.") and name.endswith(".tmp")
