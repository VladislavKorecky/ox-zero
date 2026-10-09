"""The adapter between the AlphaZero search and the CLI's engine port.

Design: docs/design/cli-integration.md (the mapping, root expansion, loading
a model). The port (`engine_port.py`) is what the CLI wants to show; the
search (`ox_zero.engine.search.analyse`) is what the engine computes. This
module translates one into the other and adds nothing of its own.

It is also where the CLI gets its engine (`load_engine`): a trained network
when a checkpoint exists, the same search with uniform priors otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ox_zero.cli.engine_port import Analysis, Engine
from ox_zero.engine.evaluator import Evaluator, UniformEvaluator
from ox_zero.engine.search import ANALYSIS, SearchConfig, Snapshot, analyse
from ox_zero.game import State


class SearchEngine:
    """Engine-port adapter around `engine.search.analyse`.

    Args:
        evaluator: Where the search gets its priors and values: a
            `NetworkEvaluator` for a trained model, `UniformEvaluator` when
            there is none.
        size: The board size the evaluator was built for. A network's input
            and policy head are fixed to one size, so searching another
            board is an error. `None` accepts any size (as `UniformEvaluator`
            does).
        config: Search constants. The default `ANALYSIS` expands every root
            child up front, so each legal move has a score from the first
            snapshot on.
        seed: Accepted for the port's `--seed` flag and stored, but unused:
            analysis mode has no noise and no sampling, so the search is
            deterministic and there is nothing for a seed to drive.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        size: int | None = None,
        config: SearchConfig = ANALYSIS,
        seed: int | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.size = size
        self.config = config
        self.seed = seed

    def search(self, state: State, max_simulations: int | None = None) -> Iterator[Analysis]:
        if self.size is not None and state.size != self.size:
            size, other = self.size, state.size
            raise ValueError(
                f"checkpoint is for a {size}x{size} board, position is {other}x{other}"
            )
        # `analyse` validates eagerly (a finished game raises here, not on the
        # first `next()`), so wrapping its iterator in a generator expression
        # keeps that: only the mapping is lazy.
        snapshots = analyse(state, self.evaluator, self.config, max_simulations)
        return (_to_analysis(snapshot) for snapshot in snapshots)


def _to_analysis(snapshot: Snapshot) -> Analysis:
    """Map a search snapshot onto what the CLI shows.

    Values: the search works in `[-1, 1]` from the side to move's point of
    view (-1 loss, 0 draw, +1 win), as the value head `v` and terminal
    results do. The CLI shows `[0, 1]`, read as a win probability, so both the
    root value and every move's `Q(root, a)` go through `(x + 1) / 2`. That
    reading is loose: a draw maps to 0.5 just as a coin flip would, and the
    network's `v` is a value estimate trained by squared error, not a
    calibrated probability.

    Move scores: `Q(root, a)` is stored from the side to move at the root's
    point of view already (the search negates values on the way up), so it
    is the score of "play `a`" for the player choosing. No sign flip here.

    The choice: `snapshot.best`, the most visited move, not the highest `Q`.
    PUCT spends visits on moves that keep looking good, so a visit count
    combines value with confidence; a high `Q` from three visits is noise.
    That is the move AlphaZero plays, and the one `best` reports.

    Every snapshot is forwarded, including the first one with
    `simulations = 0` (after root setup): it already has a score for every
    move, so the sandbox can draw a full board before the first simulation.
    """
    return Analysis(
        value=(snapshot.value + 1) / 2,
        # `snapshot.q` is in board order; a dict comprehension keeps it.
        scores={move: (q + 1) / 2 for move, q in snapshot.q.items()},
        simulations=snapshot.simulations,
        chosen=snapshot.best,
    )


NO_MODEL_NOTICE = (
    "No trained model found: searching with uniform priors."
    " Scores reflect search alone, not a learned evaluation."
)


def load_engine(
    model: Path | None,
    seed: int | None,
    device: str = "cpu",
    root: Path = Path("checkpoints"),
) -> tuple[Engine, str]:
    """The engine the CLI should use, and a one-line notice saying which.

    - `model` given: load that checkpoint.
    - Otherwise the newest checkpoint under `root` (by generation).
    - Otherwise no model at all: the real search with `UniformEvaluator`,
      i.e. plain PUCT with equal priors and value 0 at every leaf. It still
      finds short tactics, because finished games inside the tree have exact
      values; it just has no positional judgement.

    `device` is `"auto"` (best available, `select_device()`) or a torch
    device name. It only matters for a network; the uniform search never
    touches torch and ignores it. The default is `cpu` because the CLI runs
    the search at batch size 1, where the GPU's per-call overhead dominates
    (docs/design/engineering.md, "Measured").

    The notice is returned rather than printed so this module stays free of
    any output concerns; the CLI prints it.

    Raises:
        FileNotFoundError: `model` was given but does not exist.
    """
    if model is not None and not model.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model}")

    # Imported here, not at the top: these pull in torch (about half a
    # second), which the no-model path never needs.
    from ox_zero.training.checkpoint import latest_checkpoint

    path = model if model is not None else latest_checkpoint(root)
    if path is None:
        return SearchEngine(UniformEvaluator(), seed=seed), NO_MODEL_NOTICE

    from ox_zero.engine import network_evaluator
    from ox_zero.training.checkpoint import load_checkpoint

    # Called through the module so tests can substitute `select_device`.
    torch_device = network_evaluator.select_device(None if device == "auto" else device)
    checkpoint = load_checkpoint(path, torch_device)
    evaluator = network_evaluator.NetworkEvaluator(checkpoint.network, torch_device)
    size = checkpoint.size
    notice = f"Loaded {path} (generation {checkpoint.generation}, {size}x{size})"
    return SearchEngine(evaluator, size=size, seed=seed), notice
