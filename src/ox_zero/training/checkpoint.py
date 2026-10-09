"""Checkpoints: one self-describing file per network generation.

Design: docs/design/engineering.md ("Configuration and checkpoints"); format
decided in docs/plans/03-cli-adapter.md ("Decisions").

A checkpoint rebuilds the network from the file alone: the board size and
`NetworkConfig` are stored next to the weights, so loading needs no
code-side configuration that could drift out of sync with the file. It also
carries what a training run needs to resume where it stopped (optimiser
state, the configs it ran with, RNG state), which plan 04 writes.

Plain data only
---------------
The file is one `torch.save` dict of tensors, numbers, strings, lists, dicts
and `None`. `torch.save` uses Python's pickle, and unpickling can run
arbitrary code: a pickled object names a class and pickle imports and calls
it. `torch.load(..., weights_only=True)` uses a restricted unpickler that
only accepts those primitive types, so a downloaded checkpoint cannot
execute anything. That only works if we never put anything else in the file:
dataclasses are stored as field dicts (`dataclasses.asdict`) and rebuilt by
the loader, never pickled as objects.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# torch (and the network, which needs it) is imported inside the functions
# that use it, not here. `latest_checkpoint` only looks at file names, and the
# CLI calls it on every run to decide *whether* a network is needed at all;
# importing torch costs about half a second, which the no-model path should
# not pay. Under TYPE_CHECKING the names still exist for annotations.
if TYPE_CHECKING:
    import numpy as np
    import torch

    from ox_zero.engine.network import Network

CHECKPOINT_FORMAT = 1

# The file name fixed by the design: `gen_<generation>.pt`, e.g. gen_042.pt.
_NAME = re.compile(r"gen_(\d+)\.pt")


@dataclass(frozen=True)
class Checkpoint:
    """A loaded checkpoint.

    Attributes:
        network: Rebuilt from `size` and the stored `NetworkConfig`, weights
            loaded, on the requested device.
        size: Board side length the network plays.
        generation: Which training generation produced it (0: untrained).
        configs: The run's configs (plan 04: search, self-play, training) as
            field dicts; the caller rebuilds the dataclasses it knows.
        optimizer_state: `optimizer.state_dict()`, or `None` if not saved.
        rng: `{"numpy": bit_generator.state, "torch": CPU RNG state}`, or
            `None` if not saved.
    """

    network: Network
    size: int
    generation: int
    configs: Mapping[str, Mapping[str, Any]]
    optimizer_state: Mapping[str, Any] | None
    rng: Mapping[str, Any] | None


def save_checkpoint(
    path: Path,
    network: Network,
    *,
    generation: int,
    optimizer: torch.optim.Optimizer | None = None,
    configs: Mapping[str, Any] = {},
    rng: np.random.Generator | None = None,
) -> None:
    """Write a checkpoint to `path`, creating parent directories.

    Args:
        network: Saved via `state_dict()`, which holds the learned parameters
            *and* the buffers. BatchNorm's running mean and variance are
            buffers, not parameters, but eval-mode output depends on them:
            a checkpoint without them would play differently.
        generation: The generation number (also the file name's, by
            convention; not checked here).
        optimizer: Its state is saved so training can resume exactly. For
            Adam(W) that state is the per-parameter moment estimates (running
            averages of the gradient and its square) and the step count;
            resuming without them restarts the optimiser cold, with a burst
            of oversized early steps until the averages warm up again.
        configs: Dataclass instances by name, stored with `dataclasses.asdict`.
        rng: The NumPy generator to record. The torch *CPU* generator's
            state is recorded with it. Randomness drawn on a GPU (`mps`,
            `cuda`) has its own generator, which is not saved, so only
            CPU-side random sequences resume exactly.

    The file is written atomically: to a temporary file in the same
    directory, then renamed over `path`. A rename within one file system
    either happens completely or not at all, so a save interrupted by
    Ctrl-C or a crash never leaves a truncated `gen_N.pt` behind for
    `latest_checkpoint` to pick as the newest.
    """
    import torch

    data = {
        "format_version": CHECKPOINT_FORMAT,
        "size": network.size,
        "generation": generation,
        "network_config": dataclasses.asdict(network.config),
        # Tensors go to CPU first, so a checkpoint written on `mps` or `cuda`
        # loads on any machine. `detach` drops any autograd link.
        "model_state": {k: v.detach().cpu() for k, v in network.state_dict().items()},
        "optimizer_state": _to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "configs": {name: dataclasses.asdict(config) for name, config in configs.items()},
        "rng": (
            {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state()}
            if rng is not None
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temporary name starts with a dot and does not end in `.pt`, so
    # `latest_checkpoint` never mistakes a half-written file for a checkpoint;
    # the random part keeps two concurrent saves from colliding. Mode 0o666
    # is what a plain `open` asks for: the kernel subtracts the umask, so the
    # checkpoint gets the usual permissions (`tempfile.mkstemp` would make
    # it owner-only, and the rename would keep that).
    temporary = path.parent / f".{path.name}.{secrets.token_hex(4)}.tmp"
    # O_EXCL: fail rather than reuse an existing file. O_BINARY exists only on
    # Windows, where a descriptor is text mode by default and would mangle
    # every newline byte in the zip; elsewhere it is 0 and changes nothing.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(temporary, flags, 0o666)
    try:
        with os.fdopen(fd, "wb") as file:
            torch.save(data, file)
            # Push the bytes to disk before the rename. Without this, after a
            # power cut the rename can be on disk while the data is not,
            # leaving exactly the truncated file the rename exists to avoid.
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        # BaseException, not Exception: clean up on Ctrl-C too, then re-raise.
        # A failing cleanup must not replace the error that caused it.
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def load_checkpoint(path: Path, device: torch.device | None = None) -> Checkpoint:
    """Read a checkpoint and rebuild its network on `device` (CPU by default).

    Raises:
        ValueError: The file is not a checkpoint this code can read: an
            unknown `format_version`, missing keys, weights that do not fit
            the stored architecture, not a checkpoint at all, or a `device`
            this machine does not have. Whatever
            the cause, one exception type with the path in the message, so
            the CLI can report it cleanly.
    """
    import torch

    from ox_zero.engine.network import Network, NetworkConfig

    try:
        # `weights_only=True`: plain data only, see the module docstring.
        # `map_location="cpu"`: tensors land on CPU whatever device saved
        # them; the network is moved afterwards, in one place.
        data = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:  # corrupt zip, refused pickle, ...
        raise ValueError(f"{path}: not a readable checkpoint ({error})") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a checkpoint (contains {type(data).__name__})")
    version = data.get("format_version")
    if version != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: checkpoint format {version} is not supported (expected {CHECKPOINT_FORMAT})"
        )

    try:
        network = Network(data["size"], NetworkConfig(**data["network_config"]))
        # strict=True: every key in the file must match a key in the network
        # and vice versa, with matching shapes. An architecture mismatch must
        # fail loudly here instead of loading half the weights and leaving
        # the rest at random initialisation.
        network.load_state_dict(data["model_state"], strict=True)
        checkpoint = Checkpoint(
            network=network,
            size=data["size"],
            generation=data["generation"],
            configs=data["configs"],
            optimizer_state=data["optimizer_state"],
            rng=data["rng"],
        )
    except (KeyError, TypeError, RuntimeError) as error:
        # KeyError: a missing top-level key. TypeError: `network_config` has
        # unknown or missing fields. RuntimeError: `load_state_dict` found
        # missing, unexpected, or wrongly shaped weights.
        raise ValueError(f"{path}: malformed checkpoint ({error})") from error

    target = device if device is not None else torch.device("cpu")
    try:
        network.to(target)
    except torch.OutOfMemoryError:
        raise  # the device exists but is full: not a "missing device" error
    except (RuntimeError, AssertionError) as error:
        # A device this machine does not have. torch raises AssertionError
        # ("Torch not compiled with CUDA enabled") or RuntimeError depending
        # on the backend.
        raise ValueError(f"cannot load {path} onto device {target}: {error}") from error
    network.eval()
    return checkpoint


def latest_checkpoint(root: Path) -> Path | None:
    """The newest `gen_N.pt` anywhere under `root`, by generation number.

    Ties (two runs at the same generation) go to the most recently modified
    file. Other files are ignored. `None` when nothing matches or `root` does
    not exist. Only names are read, never file contents: loading every
    checkpoint to find its generation would cost seconds.
    """
    if not root.is_dir():
        return None
    candidates = []
    for path in root.rglob("gen_*.pt"):
        match = _NAME.fullmatch(path.name)
        # `is_file` follows symlinks: a directory with a checkpoint's name, or
        # a link whose target was deleted, is skipped instead of crashing
        # `stat` (or being returned and failing later in `torch.load`).
        if match and path.is_file():
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue  # deleted between the listing and now (a run pruning old files)
            candidates.append((int(match.group(1)), modified, path))
    if not candidates:
        return None
    # Tuples compare element by element: generation first, then mtime.
    return max(candidates, key=lambda c: (c[0], c[1]))[2]


def _fsync_directory(directory: Path) -> None:
    """Make a rename inside `directory` durable (the rename is directory metadata).

    POSIX only: Windows cannot open a directory this way, and its rename
    durability works differently, so there it is skipped.
    """
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _to_cpu(value: Any) -> Any:
    """Copy every tensor in a nested state dict to CPU, leaving the rest as is."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    # Exact types only: a subclass (a namedtuple, say) may not accept a
    # single iterable in its constructor. Optimiser state dicts hold plain
    # dicts, lists and tuples; anything else is left untouched.
    if type(value) in (list, tuple):
        return type(value)(_to_cpu(v) for v in value)
    return value
