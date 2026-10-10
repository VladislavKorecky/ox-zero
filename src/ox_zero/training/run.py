"""The generation loop: self-play, train, evaluate, log, checkpoint; and resume.

Design: docs/design/training.md ("The generation loop"), docs/design/engineering.md
("Configuration and checkpoints", "Devices and determinism").
Plan: docs/plans/04-training-pipeline.md, step 6, and the Decisions rows
"One RNG for everything", "Buffer persistence", "Metrics format", "The
checkpoint is the commit marker", "Resume refuses a changed config",
"Generation 0 is a checkpoint", "Tournament protocol" and "Device".

One generation
--------------
AlphaZero's training is a loop of two halves. *Policy evaluation and
improvement by search*: the current network plays games against itself,
and at every move MCTS turns the network's raw opinion into a stronger one
(the visit distribution `π`). *Learning*: the network is trained to predict
`π` and the game result `z` from the positions it saw, so the next
generation's raw opinion is closer to what search found. Each turn of the
loop is one generation `g`:

1. self-play with the live network (the examples go into the replay buffer);
2. train on the buffer (`steps_per_generation` AdamW steps);
3. tournament: the freshly trained network against a few earlier
   checkpoints, for the Elo curve;
4. one row in `metrics.jsonl`, the pairings in `matches.jsonl`, the refitted
   Elo table in `ratings.json`;
5. the buffer's new generation in `buffer/gen_NNN.npz`;
6. `gen_NNN.pt`, last.

Why the checkpoint is written last
----------------------------------
A run can be killed at any moment (Ctrl-C, a crash, a laptop lid). Instead
of making every file transactional, one file is the commit marker: a
generation happened if and only if its checkpoint exists. Everything else
written during generation `g` (metrics, matches, buffer) is written before
`gen_g.pt`, so after a crash it may exist *without* the checkpoint; resume
then drops those leftovers (rows above the latest checkpoint, buffer files
above it, TensorBoard points via `purge_step`) and re-runs generation `g`
from the checkpoint's state. One rule, no partial generations.

Why one RNG
-----------
Every random choice in the run (Dirichlet noise and move sampling in every
search tree, the buffer's sampling, the augmentation symmetries, the
tournament openings) is drawn from one `numpy.random.Generator` seeded
from the run seed. The loop is single-threaded, so those draws happen in a
fixed order and the whole run is a deterministic function of the seed.
One generator also means one state to save: the checkpoint stores it (with
torch's CPU RNG state, which only the initial weights draw from), and a
resumed run continues the *same* random sequence. That is what makes a run
stopped after generation 2 and resumed bit-for-bit equal to one that ran
straight through (step 6, test 4).

Why resume refuses a changed config
-----------------------------------
A run *is* its configuration. If the learning rate or the simulation count
changed halfway, the loss and Elo curves would mix two experiments and
nobody could tell which part of the curve came from which. So resuming
compares the requested config with the one stored in the latest checkpoint,
field by field, and raises on any difference; a new config is a new run
name. Two fields are exempt: `generations` (the target, which resume exists
to extend) and `tensorboard` (a reader of the metrics, not part of the run).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.network_evaluator import NetworkEvaluator, select_device
from ox_zero.engine.search import SELF_PLAY, SearchConfig
from ox_zero.game.rules import Mark
from ox_zero.training.atomic import atomic_write_text, fsync_directory
from ox_zero.training.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from ox_zero.training.evaluate import (
    EVALUATION,
    EvalConfig,
    elo_ratings,
    opponents_for,
    play_match,
)
from ox_zero.training.metrics import METRICS_FILE, MetricsWriter, read_metrics
from ox_zero.training.replay import ReplayBuffer
from ox_zero.training.selfplay import SelfPlayConfig, self_play
from ox_zero.training.trainer import Trainer, TrainConfig

MATCHES_FILE = "matches.jsonl"
RATINGS_FILE = "ratings.json"
BUFFER_DIR = "buffer"
# The smallest board a run accepts: the length of a winning line (OXO / XOX).
MIN_SIZE = 3


class ConfigMismatchError(ValueError):
    """Resuming with a config that differs from the one the run was trained
    with (see "Why resume refuses a changed config" above).

    A `ValueError` subclass, so a caller can tell this expected, user-facing
    refusal apart from a `ValueError` raised by a genuine bug, and print it
    as a message instead of a traceback.
    """


class RunDirectoryError(ValueError):
    """The run directory holds a previous run's files but no checkpoint.

    Starting fresh there would append the new run's rows to the old run's
    `metrics.jsonl` / `matches.jsonl` and mix two experiments in one curve.
    The run never deletes them itself (they may be the only record of an
    experiment); the user removes them or picks another name.
    """


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a training run.

    Attributes:
        name: The run's directory under `root` (`runs/<name>/`).
        size: Board side length.
        seed: Seeds torch (initial weights) and the run's one NumPy generator.
        generations: Target generation. A resumed run continues up to it;
            the only field (with `tensorboard`) that may change on resume.
        network: Architecture of the network being trained.
        search: Self-play search constants (root noise on, temperature cutoff).
        selfplay: Games, lockstep width and simulations per generation.
        train: Batch size, steps, optimiser constants, buffer length `K`.
        eval: Tournament opponents, games and simulations.
        tensorboard: Mirror the metrics to TensorBoard event files.
    """

    name: str
    size: int
    seed: int
    generations: int
    network: NetworkConfig = NetworkConfig()
    search: SearchConfig = SELF_PLAY
    selfplay: SelfPlayConfig = SelfPlayConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    tensorboard: bool = True

    def __post_init__(self) -> None:
        # Reject nonsense at construction, like the sub-configs do, so a bad
        # flag in scripts/train.py becomes a usage error instead of a run
        # that fails later (or worse, does something silently pointless).
        #
        # Size: the game itself accepts any positive size, but a winning
        # line is three cells long, so on a board smaller than 3x3 nobody
        # can ever win. Every game would be a draw, every value target
        # `z` would be 0, and the run would learn nothing about the game.
        if self.size < MIN_SIZE:
            raise ValueError(
                f"size must be at least {MIN_SIZE} (a winning line is {MIN_SIZE} long), "
                f"got {self.size}"
            )
        # 0 is allowed: it writes only gen_000.pt, the random-network anchor.
        if self.generations < 0:
            raise ValueError(f"generations must be at least 0, got {self.generations}")
        # The name is one directory under `root`. A separator would nest it
        # (or, with "..", escape `root`); "." would be `root` itself.
        separators = [sep for sep in (os.sep, os.altsep, "/") if sep]
        if (
            not self.name
            or self.name in (".", "..")
            or any(sep in self.name for sep in separators)
        ):
            raise ValueError(
                f"name must be a single non-empty directory name, got {self.name!r}"
            )


@dataclass(frozen=True)
class RunIdentity:
    """The run's name and seed, stored in each checkpoint's `configs` as `"run"`.

    A dataclass rather than a plain dict because `save_checkpoint` applies
    `dataclasses.asdict` to every entry of `configs`.
    """

    name: str
    seed: int


def run(config: RunConfig, root: Path = Path("runs"), device: torch.device | None = None) -> Path:
    """Train `config` up to `config.generations`; returns the run directory.

    If `root / config.name` already holds checkpoints, the run resumes from
    the newest one (after checking that the config has not changed) instead
    of starting over.

    Args:
        root: Parent of the run directory (`runs/` by default, not
            `checkpoints/`: the CLI auto-loads from there and is 12x12 only).
        device: Where the network lives; `None` picks the best available
            (`select_device()`). Opponent checkpoints load onto the same one.

    Raises:
        ConfigMismatchError: Resuming with a config that differs from the
            stored one (the message names the field).
        RunDirectoryError: The directory has no checkpoint but holds a
            previous run's logs or buffer (nothing is deleted).
    """
    device = device if device is not None else select_device()
    run_dir = Path(root) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir = run_dir / BUFFER_DIR

    # The run's one NumPy generator (see "Why one RNG" in the module docstring).
    rng = np.random.default_rng(config.seed)

    latest = latest_generation(run_dir)
    if latest is None:
        _check_no_leftovers(run_dir)
        # A fresh run. Seed torch once, for the initial weights, and build
        # them on the CPU before moving: CPU initialisation is reproducible on
        # every machine, while GPU generators differ by backend.
        torch.manual_seed(config.seed)
        network = Network(config.size, config.network)
        trainer = Trainer(network, config.train, device)
        buffer = ReplayBuffer(config.size, config.train.buffer_generations)
        # Generation 0 is a checkpoint: the random network is the Elo anchor
        # (0 by definition) and generation 1's first opponent, and "resume
        # before anything happened" becomes the same code path as any resume.
        _save(run_dir, 0, network, trainer, config, rng)
        latest = 0
    else:
        checkpoint = _load(run_dir / checkpoint_name(latest), device)
        _check_config(config, checkpoint)
        network = checkpoint.network
        trainer = Trainer(network, config.train, device, checkpoint.optimizer_state)
        # Restore the random streams exactly where the checkpointed run left
        # them. The torch state is set *after* the network was rebuilt, since
        # building a Network draws initial weights from torch's generator.
        assert checkpoint.rng is not None, "run checkpoints always store the RNG state"
        rng.bit_generator.state = checkpoint.rng["numpy"]
        torch.set_rng_state(checkpoint.rng["torch"])

        # The commit marker rule: anything written for a generation above the
        # latest checkpoint is a leftover of an interrupted generation.
        _truncate_rows(run_dir / METRICS_FILE, latest)
        _truncate_rows(run_dir / MATCHES_FILE, latest)
        # ratings.json is derived from matches.jsonl, so refit it from the
        # rows just kept: an interrupted generation may already have written
        # its own rating, for a generation that (by the rule) never happened.
        _write_ratings(run_dir)
        if buffer_dir.is_dir():
            # `upto` also deletes buffer files above the checkpoint: the buffer
            # is saved just before the checkpoint, so a crash between the two
            # leaves examples the checkpoint's network never trained on.
            buffer = ReplayBuffer.load(
                buffer_dir, config.size, config.train.buffer_generations, upto=latest
            )
        else:
            if latest >= 1:
                # Generation 1 onwards saved a buffer before its checkpoint, so
                # its absence means it was deleted. Continue (the run is still
                # valid), but the next generations train on fewer examples.
                print(
                    f"warning: {buffer_dir} is missing; resuming generation {latest + 1} "
                    "with an empty replay buffer",
                    file=sys.stderr,
                )
            # latest == 0: interrupted during generation 1, before the first
            # buffer save. Nothing was lost; no warning.
            buffer = ReplayBuffer(config.size, config.train.buffer_generations)

    if latest >= config.generations:
        return run_dir

    # The live network's evaluator. It shares the network object with the
    # trainer (no copying of weights between self-play and training); the
    # evaluator sets eval mode on each call, the trainer train mode on each step.
    evaluator = NetworkEvaluator(network, device)
    # On resume, `purge_from` makes TensorBoard drop the stale points of the
    # generation being re-run, mirroring the row truncation above.
    writer = MetricsWriter(run_dir, config.tensorboard, purge_from=latest + 1)
    try:
        for generation in range(latest + 1, config.generations + 1):
            _generation(
                generation, config, run_dir, network, evaluator, trainer, buffer, writer,
                rng, device,
            )
    finally:
        writer.close()
    return run_dir


def _generation(
    generation: int,
    config: RunConfig,
    run_dir: Path,
    network: Network,
    evaluator: NetworkEvaluator,
    trainer: Trainer,
    buffer: ReplayBuffer,
    writer: MetricsWriter,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    """One generation, in the commit-marker order (checkpoint last)."""
    started = time.perf_counter()

    # 1. Self-play: the live network generates this generation's examples.
    clock = time.perf_counter()
    result = self_play(evaluator, config.size, config.search, config.selfplay, rng)
    selfplay_seconds = time.perf_counter() - clock
    buffer.add(result.examples, generation)

    # 2. Training on the whole buffer (the newest K generations).
    clock = time.perf_counter()
    losses = trainer.train_generation(buffer, rng)
    train_seconds = time.perf_counter() - clock

    # 3. Tournament: the trained network (the current side, the live
    #    evaluator) against earlier checkpoints, with noise-free search.
    clock = time.perf_counter()
    match_rows: list[dict[str, Any]] = []
    for opponent in opponents_for(generation, config.eval):
        opponent_net = _load(run_dir / checkpoint_name(opponent), device).network
        match = play_match(
            evaluator,
            NetworkEvaluator(opponent_net, device),
            config.size,
            config.eval.games_per_colour,
            config.eval.simulations,
            rng,
            EVALUATION,
        )
        match_rows.append(
            {
                "generation": generation,
                "opponent": opponent,
                "wins": match.wins,
                "draws": match.draws,
                "losses": match.losses,
                "score": match.score,
            }
        )
    eval_seconds = time.perf_counter() - clock

    # 4. Logs: matches first (the Elo fit reads them), then ratings, then
    #    the metrics row that quotes this generation's rating.
    matches_path = run_dir / MATCHES_FILE
    with matches_path.open("a", encoding="utf-8") as file:
        for row in match_rows:
            file.write(json.dumps(row) + "\n")
    table = _write_ratings(run_dir)

    games = result.games
    lengths = [len(record.moves) for record in games]
    count = max(len(games), 1)
    writer.write(
        {
            "generation": generation,
            "games": len(games),
            "examples": len(result.examples),
            "buffer_size": len(buffer),
            "mean_game_length": float(np.mean(lengths)) if lengths else 0.0,
            "draw_rate": sum(record.winner is None for record in games) / count,
            "x_win_rate": sum(record.winner is Mark.X for record in games) / count,
            "selfplay_seconds": selfplay_seconds,
            "selfplay_steps": result.steps,
            "simulations_per_second": result.simulations / max(selfplay_seconds, 1e-9),
            "train_seconds": train_seconds,
            "loss_total": losses.total,
            "loss_policy": losses.policy,
            "loss_value": losses.value,
            "eval_seconds": eval_seconds,
            "elo": table[generation],
            "scores": {row["opponent"]: row["score"] for row in match_rows},
            # Everything up to the checkpoint; the two saves below are not timed.
            "wall_seconds": time.perf_counter() - started,
        }
    )

    # 5. The buffer, before the checkpoint (see `ReplayBuffer.load(upto=...)`).
    buffer.save(run_dir / BUFFER_DIR)

    # 6. The commit marker.
    _save(run_dir, generation, network, trainer, config, rng)


def _write_ratings(run_dir: Path) -> dict[int, float]:
    """Refit the Elo table from `matches.jsonl`, write `ratings.json`, return it.

    Bradley-Terry over *every* match of the run so far, anchored at
    generation 0, so earlier generations' ratings are refined too. With no
    matches (only `gen_000.pt` committed) there is nothing to fit, and a
    fresh run has no `ratings.json` at that point either (generation 1
    writes the first), so any existing one is removed.
    """
    matches_path = run_dir / MATCHES_FILE
    ratings_path = run_dir / RATINGS_FILE
    rows = read_metrics(matches_path) if matches_path.exists() else []
    if not rows:
        if ratings_path.exists():
            ratings_path.unlink()
            fsync_directory(run_dir)  # make the deletion durable too
        return {}
    table = elo_ratings(
        [(row["generation"], row["opponent"], row["wins"], row["draws"], row["losses"])
         for row in rows],
        anchor=0,
    )
    atomic_write_text(
        ratings_path, json.dumps({str(g): table[g] for g in sorted(table)}, indent=1) + "\n"
    )
    return table


def checkpoint_name(generation: int) -> str:
    """The checkpoint file name of `generation`: `gen_NNN.pt` (at least 3 digits)."""
    return f"gen_{generation:03d}.pt"


def latest_generation(run_dir: Path) -> int | None:
    """The highest generation with a `gen_NNN.pt` file in `run_dir`, or `None`.

    Not recursive, and only regular files count (a directory or a dangling
    link with a checkpoint's name is skipped). Unlike
    `checkpoint.latest_checkpoint`, which searches a whole tree of runs for
    the CLI, this looks at one run directory: its checkpoints are the run's
    commit markers, so the answer is "how far has this run got".
    """
    generations = []
    for path in run_dir.glob("gen_*.pt"):
        digits = path.stem.removeprefix("gen_")
        if digits.isdigit() and path.is_file():
            generations.append(int(digits))
    return max(generations, default=None)


def _check_no_leftovers(run_dir: Path) -> None:
    """Raise `RunDirectoryError` if a checkpoint-less `run_dir` holds old run files.

    Called only when `run_dir` has no `gen_*.pt`. A fresh run creates every
    one of these files itself, so finding one means a previous run (say, one
    whose checkpoints were deleted, or one killed before its generation-0
    checkpoint by a much older version) left it. Appending to it would mix
    two runs. An empty `buffer/` holds no examples, so it does not count.
    """
    buffer_dir = run_dir / BUFFER_DIR
    found = [name for name in (METRICS_FILE, MATCHES_FILE, RATINGS_FILE) if (run_dir / name).exists()]
    if buffer_dir.is_dir() and any(buffer_dir.iterdir()):
        found.append(f"{BUFFER_DIR}/")
    if found:
        raise RunDirectoryError(
            f"{run_dir} holds a previous run's files ({', '.join(found)}) but no "
            "gen_*.pt checkpoint to resume from; remove them or use a new run name"
        )


def _save(
    run_dir: Path,
    generation: int,
    network: Network,
    trainer: Trainer,
    config: RunConfig,
    rng: np.random.Generator,
) -> None:
    """Write `gen_NNN.pt` with weights, optimiser, configs and RNG state."""
    save_checkpoint(
        run_dir / checkpoint_name(generation),
        network,
        generation=generation,
        optimizer=trainer.optimizer,
        configs={
            "search": config.search,
            "selfplay": config.selfplay,
            "train": config.train,
            "eval": config.eval,
            "run": RunIdentity(config.name, config.seed),
        },
        rng=rng,
    )


def _load(path: Path, device: torch.device) -> Checkpoint:
    """`load_checkpoint` without disturbing torch's random state.

    Rebuilding a `Network` initialises fresh random weights (then overwritten
    by the stored ones), which advances torch's global generator. Loading an
    opponent must not change the run's random streams, so the draw happens
    inside `fork_rng`, which restores the CPU generator on exit.
    """
    with torch.random.fork_rng(devices=[]):
        return load_checkpoint(path, device)


def _check_config(config: RunConfig, checkpoint: Checkpoint) -> None:
    """Raise `ConfigMismatchError` naming the first field where `config` differs from
    what the checkpoint was trained with (see the module docstring)."""
    stored = checkpoint.configs
    if config.size != checkpoint.size:
        _mismatch("size", checkpoint.size, config.size)
    _compare("network", dataclasses.asdict(checkpoint.network.config), dataclasses.asdict(config.network))
    run_entry = stored.get("run", {})
    if run_entry.get("seed") != config.seed:
        _mismatch("seed", run_entry.get("seed"), config.seed)
    for name in ("search", "selfplay", "train", "eval"):
        _compare(name, stored.get(name, {}), dataclasses.asdict(getattr(config, name)))


def _compare(prefix: str, stored: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for field in sorted(set(stored) | set(current)):
        if stored.get(field) != current.get(field):
            _mismatch(f"{prefix}.{field}", stored.get(field), current.get(field))


def _mismatch(field: str, stored: Any, current: Any) -> None:
    raise ConfigMismatchError(
        f"cannot resume: {field} is {current!r} but the run was trained with {stored!r}; "
        "a changed configuration needs a new run name"
    )


def _truncate_rows(path: Path, latest: int) -> None:
    """Drop JSON Lines rows whose `generation` is above `latest`; rewrite the file.

    `read_metrics` also skips a partial last line left by a crash mid-write.
    """
    if not path.exists():
        return
    rows = [row for row in read_metrics(path) if row["generation"] <= latest]
    atomic_write_text(path, "".join(json.dumps(row) + "\n" for row in rows))
