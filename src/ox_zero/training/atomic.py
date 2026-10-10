"""Atomic, durable file writes: the one helper behind every file a run writes.

Plan: docs/plans/archive/04-training-pipeline.md, the Decisions rows "The checkpoint
is the commit marker" and "Buffer persistence".

Why
---
A training run can be killed at any instant. If it dies while overwriting
`gen_005.pt`, `buffer/gen_005.npz` or `ratings.json` in place, the file is
left half-written under its real name, and the next reader (resume, the
CLI, TensorBoard) trips over it. The classic fix has four steps:

1. write the new bytes to a *temporary* file in the same directory;
2. `fsync` it, so the bytes are on the disk and not only in the OS's page
   cache (otherwise a power loss can persist step 3 but not the data,
   leaving an empty or garbage file under the final name);
3. `os.replace` it over the final name: a rename within one filesystem is
   atomic, so a reader sees the old file or the new one, never a mix;
4. `fsync` the directory: on POSIX a file's *name* lives in its directory's
   entries, so until the directory is synced the rename itself may still
   only be in memory.

If anything fails (an exception, Ctrl-C) the temporary is deleted and the
old file is untouched. Checkpoints, buffer files and the JSON logs used to
carry three copies of this pattern that had drifted apart; this module is
the single copy.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

TEMP_SUFFIX = ".tmp"


def atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Atomically replace `path` with what `write` writes to a binary file.

    `write` receives an open binary file and writes the whole content to it
    (its return value is ignored). The parent directory must exist.

    The temporary is `.<name>.<16 random hex digits>.tmp` next to `path`:
    the leading dot hides it from globs like `gen_*.pt` (no reader mistakes
    a half-written file for a finished one), the random part means two
    writes never share a temporary, and `O_EXCL` makes creation fail rather
    than reuse an existing file. Created with mode 0o666, which the kernel
    reduces by the umask, so the final file gets the same permissions a
    plain `open(path, "w")` would give it (`tempfile.mkstemp` would make it
    owner-only 0o600, and the rename would keep that).
    """
    path = Path(path)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}{TEMP_SUFFIX}"
    # O_BINARY exists only on Windows, where a descriptor is text mode by
    # default and would mangle newline bytes; elsewhere it is 0.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(temporary, flags, 0o666)
    try:
        with os.fdopen(fd, "wb") as file:
            write(file)
            file.flush()  # Python's buffer -> the OS
            os.fsync(file.fileno())  # the OS's page cache -> the disk (step 2)
        os.replace(temporary, path)  # step 3
    except BaseException:
        # BaseException, not Exception: clean up on Ctrl-C too, then re-raise.
        # A failing cleanup must not replace the error that caused it.
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    fsync_directory(path.parent)  # step 4


def atomic_write_text(path: Path, text: str) -> None:
    """`atomic_write` for a UTF-8 text file."""
    data = text.encode("utf-8")
    atomic_write(path, lambda file: file.write(data))


def fsync_directory(directory: Path) -> None:
    """`fsync` the directory itself, making renames and unlinks in it durable.

    On POSIX a file's name lives in its directory's entries, so after
    `os.replace` the file's *data* is on disk (it was fsynced) but the new
    *name* may still only be in the page cache until the directory is
    fsynced. Opening a directory as a file is POSIX-only (Windows raises),
    and some filesystems refuse `fsync` on a directory; either way there is
    nothing better to do, so the error is ignored.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
