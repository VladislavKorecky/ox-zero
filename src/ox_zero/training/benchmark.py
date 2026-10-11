"""The cross-run benchmark: one Elo scale per board size, anchored at the uniform search.

Design: docs/design/training.md ("Evaluation"). Plan: docs/plans/06-experiment-harness.md,
step 3. Protocol: docs/experiments/README.md.

Why a second kind of rating
---------------------------
A run's own tournament (`training.evaluate`, `training.run`) anchors Elo at
that run's generation 0, a random network that differs from run to run, so two
runs' ratings cannot be compared. To say "arm B is 60 Elo better than arm A"
every checkpoint must be rated on *one* scale. This module builds it:

- **Anchor.** `uniform@100`, the uniform-prior zero-value search at 100
  simulations, is fixed at 0 Elo. It is the one player every machine can
  reproduce exactly (no weights), so the scale is the same wherever it is fit.
- **Reference pool.** A few frozen checkpoints (`runs/benchmarks/<S>x<S>/pool/`)
  plus the uniform search at two budgets. A candidate plays each reference
  once; references spread across strengths, so a strong candidate still loses
  some games and the fit stays informative. (Against a single equal opponent
  on a board that is a forced win for one colour, every opening pair splits
  one game each: the score sits at 0.5 whatever the strength, "parity
  saturation". Asymmetric references avoid that.)
- **Joint Bradley-Terry fit.** Every game of every pairing in the board's log
  enters one maximum-likelihood fit (`evaluate.elo_ratings`). A new candidate's
  games therefore also sharpen the references' ratings: information flows
  through the match graph, which a per-pairing Elo difference cannot do.
- **Bootstrap interval.** The fit has no closed-form error once a virtual draw
  and a shared anchor are involved, so `bootstrap_intervals` resamples the
  games. It measures *game-sampling* noise only; the larger seed-to-seed
  training noise is measured by running seeds, not computed.

Players are (checkpoint, search) pairs: `<run>/gen_NNN` is a checkpoint played
with the baseline evaluation search, `<run>/gen_NNN@<tag>` the same weights
with a non-default search (an upgrade under test), `uniform@N` the uniform
search with N simulations per move.

This module is torch-free except `load_pool`, which loads checkpoints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ox_zero.engine.evaluator import Evaluator, UniformEvaluator
from ox_zero.engine.search import SearchConfig
from ox_zero.training.atomic import atomic_write_text
from ox_zero.training.evaluate import EVALUATION, elo_ratings
from ox_zero.training.metrics import read_metrics, repair_partial_last_line

# Simulations per move of the uniform references, and of every candidate.
UNIFORM_SIMULATIONS = (100, 400)
CANDIDATE_SIMULATIONS = 100
ANCHOR = "uniform@100"
POOL_DIR = "pool"

# `<run>-gen_NNN`: a pool file's stem. Greedy `.+` makes the *last* "-gen_"
# the separator, so a run name may itself contain one.
_POOL_STEM = re.compile(r"^(?P<run>.+)-gen_(?P<generation>\d+)$")


# --- Naming ------------------------------------------------------------------------


def uniform_name(simulations: int) -> str:
    return f"uniform@{simulations}"


def pool_name(path: Path) -> str:
    """`pool/six-a-gen_005.pt` -> `six-a/gen_005`, the name the checkpoint has as a candidate."""
    match = _POOL_STEM.match(Path(path).stem)
    if match is None:
        raise ValueError(f"{path}: a pool file must be named <run>-gen_NNN.pt")
    return f"{match['run']}/gen_{match['generation']}"


def candidate_name(path: Path, tag: str | None) -> str:
    """The player name of the checkpoint at `path`, with `@tag` for a non-default search.

    The path is resolved first, so every spelling of one file gives one name.
    A pool file is recognised by its stem shape (`<run>-gen_NNN`), not by its
    directory, so one network has one name wherever it comes from.
    """
    path = Path(path).resolve()
    if _POOL_STEM.match(path.stem):
        name = pool_name(path)
    else:
        name = f"{path.parent.name}/{path.stem}"
    return f"{name}@{tag}" if tag else name


def base_name(name: str) -> str:
    """The name without its `@tag`: the checkpoint, whatever search plays it."""
    # Only an '@' after the last '/' starts a tag: a run name may itself contain one.
    head, slash, tail = name.rpartition("/")
    return head + slash + tail.split("@", 1)[0]


def benchmark_dir(root: Path, size: int) -> Path:
    return Path(root) / f"{size}x{size}"


# --- The reference pool ------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """One opponent every candidate plays: a pool checkpoint or a uniform search."""

    name: str
    path: Path | None  # None for uniform players (no file, no hash)
    simulations: int
    sha256: str | None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def list_pool(directory: Path) -> list[Reference]:
    """The uniform references, then every `.pt` in `directory/pool`, sorted by name.

    A `.pt` whose stem is not `<run>-gen_NNN` is an error rather than being
    skipped: a mis-named pool member must not silently shrink the pool (and
    with it, change the scale).
    """
    references = [
        Reference(uniform_name(n), None, n, None) for n in UNIFORM_SIMULATIONS
    ]
    pool = Path(directory) / POOL_DIR
    members: list[Reference] = []
    if pool.is_dir():
        for path in pool.glob("*.pt"):
            members.append(
                Reference(pool_name(path), path, CANDIDATE_SIMULATIONS, file_sha256(path))
            )
    return references + sorted(members, key=lambda r: r.name)


def load_pool(references: Sequence[Reference], size: int, device: Any) -> dict[str, Evaluator]:
    """An evaluator per reference. The only place this module touches torch."""
    from ox_zero.engine.network_evaluator import NetworkEvaluator
    from ox_zero.training.checkpoint import load_checkpoint

    loaded: dict[str, Evaluator] = {}
    for reference in references:
        if reference.path is None:
            loaded[reference.name] = UniformEvaluator()
            continue
        checkpoint = load_checkpoint(reference.path, device)
        if checkpoint.size != size:
            raise ValueError(
                f"{reference.path} is {checkpoint.size}x{checkpoint.size}, "
                f"but this pool is {size}x{size}"
            )
        loaded[reference.name] = NetworkEvaluator(checkpoint.network, device)
    return loaded


# --- The match log -----------------------------------------------------------------


def search_delta(search: SearchConfig) -> dict[str, Any]:
    """The fields in which `search` differs from `EVALUATION`; `{}` for the baseline.

    Compared with `EVALUATION`, not `SearchConfig()`: the latter's
    `temperature_cutoff` is `None` ("per size"), the evaluation search's is 0.
    """
    return {
        field.name: getattr(search, field.name)
        for field in dataclasses.fields(search)
        if getattr(search, field.name) != getattr(EVALUATION, field.name)
    }


@dataclass(frozen=True)
class BenchmarkRow:
    """One pairing, from the candidate's (`a`'s) perspective."""

    a: str
    b: str
    wins: int
    draws: int
    losses: int
    as_x: tuple[int, int, int]
    as_o: tuple[int, int, int]
    openings: int
    a_simulations: int
    b_simulations: int
    device: str
    date: str
    a_sha256: str
    b_sha256: str | None
    a_search: dict[str, Any]

    @property
    def pairing(self) -> frozenset[str]:
        """The unordered pair of players: a pairing is one measurement, whoever is `a`."""
        return frozenset((self.a, self.b))

    def to_json(self) -> dict[str, Any]:
        return {
            "a": self.a, "b": self.b,
            "wins": self.wins, "draws": self.draws, "losses": self.losses,
            "as_x": list(self.as_x), "as_o": list(self.as_o),
            "openings": self.openings,
            "a_simulations": self.a_simulations, "b_simulations": self.b_simulations,
            "a_search": self.a_search, "device": self.device, "date": self.date,
            "a_sha256": self.a_sha256, "b_sha256": self.b_sha256,
        }

    @staticmethod
    def from_json(row: Mapping[str, Any]) -> BenchmarkRow:
        return BenchmarkRow(
            a=row["a"], b=row["b"],
            wins=row["wins"], draws=row["draws"], losses=row["losses"],
            as_x=tuple(row["as_x"]), as_o=tuple(row["as_o"]),
            openings=row["openings"],
            a_simulations=row["a_simulations"], b_simulations=row["b_simulations"],
            device=row["device"], date=row["date"],
            a_sha256=row["a_sha256"], b_sha256=row["b_sha256"],
            a_search=dict(row["a_search"]),
        )


def read_rows(path: Path) -> list[BenchmarkRow]:
    """Every complete row of the log; `[]` if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    return [BenchmarkRow.from_json(row) for row in read_metrics(path)]


def append_rows(path: Path, rows: Sequence[BenchmarkRow]) -> None:
    """Append rows to the JSON Lines log. Append-only.

    A pairing already in the file (in either order) is a `ValueError` naming
    the two players: each pairing is one measurement. A crash-truncated last
    line is repaired first, or the new row would be glued onto the fragment.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    repair_partial_last_line(path)
    seen = {row.pairing for row in read_rows(path)}
    for row in rows:
        if row.pairing in seen:
            raise ValueError(f"pairing {row.a} vs {row.b} is already in {path}")
        seen.add(row.pairing)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row.to_json()) + "\n")


def check_consistency(rows: Sequence[BenchmarkRow], references: Sequence[Reference]) -> None:
    """Refuse a log in which one name means two different things.

    Identity has two layers. The *base name* (without `@tag`) and the sha256
    are bound one to one: a retrained run reusing a name, or one file under
    two names, would silently merge two networks into one Bradley-Terry
    player. The *full name* is bound to its `a_search`: two `--c-init` values
    under one tag would merge two engines. A reference whose hash differs from
    the pool file's was swapped. `None` hashes (uniform) are never compared.

    Raises:
        ValueError: naming the offender.
    """
    pool_hash = {r.name: r.sha256 for r in references if r.sha256 is not None}
    hashes_of: dict[str, set[str]] = defaultdict(set)  # base name -> hashes
    names_of: dict[str, set[str]] = defaultdict(set)  # hash -> base names
    searches: dict[str, dict[str, Any]] = {}

    for row in rows:
        for name, sha in ((row.a, row.a_sha256), (row.b, row.b_sha256)):
            if sha is None:
                continue
            base = base_name(name)
            if base in pool_hash and pool_hash[base] != sha:
                raise ValueError(
                    f"{base}: the log has hash {sha[:12]} but the pool file has "
                    f"{pool_hash[base][:12]}; a pool file must never change"
                )
            hashes_of[base].add(sha)
            names_of[sha].add(base)
        if row.a in searches and searches[row.a] != row.a_search:
            raise ValueError(
                f"{row.a} appears with two search configurations: "
                f"{searches[row.a]} and {row.a_search}"
            )
        searches[row.a] = row.a_search

    for base, shas in hashes_of.items():
        if len(shas) > 1:
            raise ValueError(f"{base} appears under {len(shas)} different checkpoint hashes")
    for sha, bases in names_of.items():
        if len(bases) > 1:
            raise ValueError(f"hash {sha[:12]} appears under two names: {sorted(bases)}")


def pairings(
    candidate: str, references: Sequence[Reference], logged: Sequence[BenchmarkRow]
) -> list[Reference]:
    """The references `candidate` still has to play, in pool order.

    Not its own full name (a self-match, which the fit rejects), and not a
    pairing already logged. A *tagged* candidate does play its untagged pool
    entry: that is the one pairing which isolates the search change.
    """
    done = {row.pairing for row in logged}
    return [
        r for r in references
        if r.name != candidate and frozenset((candidate, r.name)) not in done
    ]


# --- Ratings -----------------------------------------------------------------------


def as_matches(rows: Sequence[BenchmarkRow]) -> list[tuple[str, str, int, int, int]]:
    return [(r.a, r.b, r.wins, r.draws, r.losses) for r in rows]


def rating_table(rows: Sequence[BenchmarkRow], anchor: str = ANCHOR) -> dict[str, float]:
    """Joint Bradley-Terry Elo of every player in `rows`, `anchor` at 0.

    Raises `ValueError` (from `elo_ratings`) if the anchor has played no game
    or some player is not connected to it.
    """
    return elo_ratings(as_matches(rows), anchor)


def bootstrap_intervals(
    rows: Sequence[BenchmarkRow], resamples: int = 1000, seed: int = 0, anchor: str = ANCHOR
) -> dict[str, tuple[float, float]]:
    """95% bootstrap interval of every player's rating (2.5th and 97.5th percentile).

    Per replicate, each pairing's `n` games are redrawn as a multinomial
    sample with the W/D/L proportions *after* the virtual draw,
    `(w, d + 1, l) / (n + 1)`, and the replicate is fit by the ordinary
    `elo_ratings`.

    Why smoothed proportions: resampling the raw proportions would give a
    clean sweep zero variance, collapsing the interval exactly where the rating
    is least determined. The smoothed ones let a swept pairing occasionally
    produce a draw, keeping the interval open.

    Why the fit also adds its virtual draw: a replicate that happens to be a
    clean sweep has no finite unregularised maximum, and a fit without the
    draw would run away. The price is that the draw is applied twice, which
    pulls each replicate about half a game towards 0.5 (under 1% of the score
    on 72 games): a known small bias on swept links, preferable to a
    divergent fit.

    What this is not: it measures game-sampling noise only. Seed-to-seed
    training noise is the larger component and is measured by running seeds.
    The generator is seeded, so the same log gives the same intervals.
    """
    if not rows:
        return {}
    rng = np.random.default_rng(seed)
    counts = np.array([[r.wins, r.draws, r.losses] for r in rows], dtype=float)
    n = counts.sum(axis=1)
    smoothed = counts + np.array([0.0, 1.0, 0.0])
    smoothed /= (n + 1)[:, None]

    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        replicate = [
            (r.a, r.b, *(int(x) for x in rng.multinomial(int(n[i]), smoothed[i])))
            for i, r in enumerate(rows)
        ]
        for player, elo in elo_ratings(replicate, anchor).items():
            samples[player].append(elo)
    return {
        player: (float(np.percentile(elos, 2.5)), float(np.percentile(elos, 97.5)))
        for player, elos in samples.items()
    }


def bootstrap_interval(
    rows: Sequence[BenchmarkRow], player: str, resamples: int = 1000, seed: int = 0,
    anchor: str = ANCHOR,
) -> tuple[float, float]:
    """`bootstrap_intervals(...)[player]`, for printing one candidate."""
    return bootstrap_intervals(rows, resamples, seed, anchor)[player]


def write_ratings(
    path: Path, rows: Sequence[BenchmarkRow], resamples: int = 1000
) -> dict[str, Any]:
    """Rewrite `ratings.json`, derived from the log: `{"anchor", "players": {name: {...}}}`.

    Each player has `elo`, `low`, `high` (the bootstrap interval) and `games`,
    all rounded to one decimal. Written atomically; writing the same log twice
    gives identical bytes.

    Raises:
        ValueError: `rows` is empty (there is no anchor to rate against).
    """
    if not rows:
        raise ValueError("cannot rate an empty match log")
    table = rating_table(rows)
    intervals = bootstrap_intervals(rows, resamples)
    games: dict[str, int] = defaultdict(int)
    for row in rows:
        n = row.wins + row.draws + row.losses
        games[row.a] += n
        games[row.b] += n
    players = {
        name: {
            "elo": round(table[name], 1),
            "low": round(intervals[name][0], 1),
            "high": round(intervals[name][1], 1),
            "games": games[name],
        }
        for name in sorted(table)
    }
    result = {"anchor": ANCHOR, "players": players}
    atomic_write_text(Path(path), json.dumps(result, indent=2) + "\n")
    return result
