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

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ox_zero.engine.network import Network, NetworkConfig

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
        rng: The NumPy generator to record; the torch CPU RNG state is
            recorded with it, so a resumed run continues the same random
            sequences.
    """
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
    torch.save(data, path)


def load_checkpoint(path: Path, device: torch.device | None = None) -> Checkpoint:
    """Read a checkpoint and rebuild its network on `device` (CPU by default).

    Raises:
        ValueError: The file's `format_version` is not one this code reads.
    """
    # `weights_only=True`: plain data only, see the module docstring.
    # `map_location="cpu"`: tensors land on CPU whatever device saved them;
    # the network is moved afterwards, in one place.
    data = torch.load(path, weights_only=True, map_location="cpu")
    version = data.get("format_version")
    if version != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: checkpoint format {version} is not supported (expected {CHECKPOINT_FORMAT})"
        )

    network = Network(data["size"], NetworkConfig(**data["network_config"]))
    # strict=True: every key in the file must match a key in the network and
    # vice versa. An architecture mismatch must fail loudly here instead of
    # loading half the weights and leaving the rest at random initialisation.
    network.load_state_dict(data["model_state"], strict=True)
    network.to(device if device is not None else torch.device("cpu"))
    network.eval()

    return Checkpoint(
        network=network,
        size=data["size"],
        generation=data["generation"],
        configs=data["configs"],
        optimizer_state=data["optimizer_state"],
        rng=data["rng"],
    )


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
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        return None
    # Tuples compare element by element: generation first, then mtime.
    return max(candidates, key=lambda c: (c[0], c[1]))[2]


def _to_cpu(value: Any) -> Any:
    """Copy every tensor in a nested state dict to CPU, leaving the rest as is."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value
