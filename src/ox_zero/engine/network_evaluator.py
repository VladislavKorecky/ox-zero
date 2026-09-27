"""The network behind the evaluator seam: `NetworkEvaluator` and `select_device`.

Design: docs/design/engineering.md ("The evaluator seam", "Devices and
determinism") and docs/design/network.md ("Policy head", "Numerics and
devices").

This is the only module where the search's NumPy world meets torch. It is
kept apart from `evaluator.py` so that the search, which imports only the
protocol, never pays the half second `import torch` costs.

One evaluation, step by step:

    states ─► encode_batch ─► torch.from_numpy ─► .to(device)
           ─► net.eval(), inference_mode ─► (logits, values)
           ─► logits[illegal] = -inf ─► softmax ─► .cpu().numpy()
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ox_zero.engine.encoding import encode_batch, legal_mask
from ox_zero.engine.evaluator import Policies, Values
from ox_zero.engine.network import Network
from ox_zero.game.rules import State


def select_device(preference: str | None = None) -> torch.device:
    """The device to run the network on: the named one, or the best available.

    With no preference (or `"auto"`): `cuda` if an NVIDIA GPU is available,
    else `mps` (Apple Silicon's GPU), else `cpu`. Never hardcode a device
    elsewhere; tests pass `"cpu"` explicitly.
    """
    if preference is not None and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NetworkEvaluator:
    """Scores positions with a `Network`, satisfying the `Evaluator` protocol.

    The network is moved to `device` once, here. Evaluation always runs it in
    eval mode (frozen BatchNorm statistics, so a position's answer does not
    depend on its batch mates) and under `torch.inference_mode` (no autograd
    bookkeeping: faster, less memory).
    """

    def __init__(self, network: Network, device: torch.device | None = None) -> None:
        self.device = device if device is not None else select_device()
        self.network = network.to(self.device)

    def evaluate(self, states: Sequence[State]) -> tuple[Policies, Values]:
        size = self.network.size
        if any(state.size != size for state in states):
            raise ValueError(f"this network plays {size}x{size}; got a different board size")
        if not states:
            return np.zeros((0, size * size), dtype=np.float32), np.zeros(0, dtype=np.float32)

        planes = torch.from_numpy(encode_batch(states)).to(self.device)
        legal = torch.from_numpy(np.stack([legal_mask(state) for state in states])).to(
            self.device
        )

        # Set eval mode on every call, not once: a trainer sharing this
        # network object may have switched it to train mode in between.
        self.network.eval()
        with torch.inference_mode():
            logits, values = self.network(planes)
            # Illegal cells get logit -inf, so softmax gives them exactly 0
            # and renormalises over the legal cells alone. The search never
            # sees an illegal prior (docs/design/network.md, "Policy head").
            policies = torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)

        # Back to host memory and NumPy for the torch-free search. Everything
        # stays float32 (never `.double()`: MPS has no float64).
        return policies.cpu().numpy(), values.cpu().numpy()
