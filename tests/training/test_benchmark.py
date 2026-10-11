"""Tests for the cross-run benchmark (plan 06, step 3): naming, the reference
pool, the append-only match log, the joint Bradley-Terry fit and its bootstrap.

Torch-free except the two `load_pool` tests at the end, like `benchmark.py`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ox_zero.training.benchmark import (
    ANCHOR,
    BenchmarkRow,
    Reference,
    append_rows,
    as_matches,
    base_name,
    benchmark_dir,
    bootstrap_interval,
    bootstrap_intervals,
    candidate_name,
    check_consistency,
    list_pool,
    pairings,
    pool_name,
    rating_table,
    read_rows,
    uniform_name,
    write_ratings,
)

# --- Helpers ---------------------------------------------------------------------


def row(a: str, b: str, wins: int, draws: int, losses: int, **overrides) -> BenchmarkRow:
    """A hand-built pairing. The colour split is arbitrary but consistent."""
    fields = dict(
        a=a, b=b, wins=wins, draws=draws, losses=losses,
        as_x=(wins, draws, losses), as_o=(0, 0, 0),
        openings=36, a_simulations=100, b_simulations=100, device="cpu",
        date="2026-10-11T00:00:00", a_sha256="aa", b_sha256="bb", a_search={},
    )
    fields.update(overrides)
    return BenchmarkRow(**fields)


def ref(name: str, sha: str | None = "bb", simulations: int = 100) -> Reference:
    return Reference(name=name, path=None, simulations=simulations, sha256=sha)


# --- 1. Naming ---------------------------------------------------------------------


def test_uniform_name() -> None:
    assert uniform_name(100) == "uniform@100"
    assert ANCHOR == "uniform@100"


def test_candidate_name_from_a_run_directory(tmp_path: Path) -> None:
    path = tmp_path / "six-a" / "gen_020.pt"
    assert candidate_name(path, None) == "six-a/gen_020"
    assert candidate_name(path, "fpu") == "six-a/gen_020@fpu"


def test_candidate_name_is_independent_of_how_the_path_is_spelled(tmp_path: Path) -> None:
    (tmp_path / "six-a").mkdir()
    direct = tmp_path / "six-a" / "gen_020.pt"
    roundabout = tmp_path / "six-a" / ".." / "six-a" / "gen_020.pt"
    assert candidate_name(roundabout, None) == candidate_name(direct, None)


def test_a_pool_path_gets_the_pool_name(tmp_path: Path) -> None:
    path = tmp_path / "pool" / "six-a-gen_005.pt"
    assert candidate_name(path, None) == "six-a/gen_005"
    assert candidate_name(path, "t") == "six-a/gen_005@t"
    # A run that happens to be called `pool` still names by its parent.
    assert candidate_name(tmp_path / "pool" / "gen_005.pt", None) == "pool/gen_005"


def test_pool_name() -> None:
    assert pool_name(Path("pool/six-a-gen_005.pt")) == "six-a/gen_005"
    # The last "-gen_" splits, so a run name may contain one itself.
    assert pool_name(Path("pool/odd-gen_run-gen_010.pt")) == "odd-gen_run/gen_010"


@pytest.mark.parametrize("stem", ["gen_005", "six-a", "six-a-gen_x", "six-a-gen_"])
def test_pool_name_rejects_other_stems(stem: str) -> None:
    with pytest.raises(ValueError, match=stem):
        pool_name(Path(f"pool/{stem}.pt"))


def test_benchmark_dir() -> None:
    assert benchmark_dir(Path("runs/benchmarks"), 6) == Path("runs/benchmarks/6x6")


def test_base_name() -> None:
    assert base_name("six-a/gen_020@fpu") == "six-a/gen_020"
    assert base_name("six-a/gen_020") == "six-a/gen_020"
    assert base_name("x@y/gen_001@t") == "x@y/gen_001"


# --- 2. list_pool ------------------------------------------------------------------


def test_list_pool(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "b-gen_010.pt").write_bytes(b"two")
    (pool / "a-gen_005.pt").write_bytes(b"one")
    (pool / "notes.txt").write_text("ignored")
    refs = list_pool(tmp_path)
    assert [r.name for r in refs] == ["uniform@100", "uniform@400", "a/gen_005", "b/gen_010"]
    assert [r.simulations for r in refs] == [100, 400, 100, 100]
    assert refs[0].path is None and refs[0].sha256 is None
    import hashlib

    assert refs[2].sha256 == hashlib.sha256(b"one").hexdigest()
    assert refs[2].path == pool / "a-gen_005.pt"


def test_list_pool_rejects_a_misnamed_member(tmp_path: Path) -> None:
    (tmp_path / "pool").mkdir()
    (tmp_path / "pool" / "gen_005.pt").write_bytes(b"x")
    with pytest.raises(ValueError, match="gen_005.pt"):
        list_pool(tmp_path)


# --- 3. The row and the log --------------------------------------------------------


def test_row_json_round_trip() -> None:
    original = row("x/gen_1@t", "uniform@100", 30, 4, 38, a_search={"c_init": 3.0}, b_sha256=None)
    assert BenchmarkRow.from_json(original.to_json()) == original


def test_append_then_read(tmp_path: Path) -> None:
    path = tmp_path / "matches.jsonl"
    assert read_rows(path) == []
    rows = [row("a", "b", 10, 0, 5), row("a", "c", 3, 3, 3)]
    append_rows(path, rows[:1])
    append_rows(path, rows[1:])
    assert read_rows(path) == rows


def test_a_truncated_last_line_is_dropped_and_repaired_on_append(tmp_path: Path) -> None:
    path = tmp_path / "matches.jsonl"
    append_rows(path, [row("a", "b", 10, 0, 5)])
    with path.open("a") as file:
        file.write('{"a": "a", "b": "c", "wi')  # a crash mid-line
    assert len(read_rows(path)) == 1
    append_rows(path, [row("a", "d", 1, 1, 1)])
    assert [r.b for r in read_rows(path)] == ["b", "d"]


def test_pairing_ignores_order() -> None:
    assert row("a", "b", 1, 0, 0).pairing == row("b", "a", 0, 0, 1).pairing


def test_a_pairing_already_logged_is_refused_in_either_order(tmp_path: Path) -> None:
    path = tmp_path / "matches.jsonl"
    append_rows(path, [row("a", "b", 10, 0, 5)])
    with pytest.raises(ValueError, match="a.*b"):
        append_rows(path, [row("a", "b", 1, 0, 0)])
    with pytest.raises(ValueError, match="a.*b"):
        append_rows(path, [row("b", "a", 1, 0, 0)])
    assert len(read_rows(path)) == 1


# --- 4. Identity checks and pairings -----------------------------------------------


def test_a_reference_whose_hash_changed_is_refused() -> None:
    rows = [row("c", "pool/gen_5", 5, 0, 5, b_sha256="old")]
    with pytest.raises(ValueError, match="pool/gen_5"):
        check_consistency(rows, [ref("pool/gen_5", sha="new")])


def test_a_base_name_under_two_hashes_is_refused() -> None:
    rows = [row("r/gen_1", "u", 1, 0, 1, a_sha256="h1"), row("r/gen_1", "v", 1, 0, 1, a_sha256="h2")]
    with pytest.raises(ValueError, match="r/gen_1"):
        check_consistency(rows, [])


def test_one_hash_under_two_base_names_is_refused() -> None:
    rows = [row("r/gen_1", "u", 1, 0, 1, a_sha256="h"), row("s/gen_1", "v", 1, 0, 1, a_sha256="h")]
    with pytest.raises(ValueError, match="h"):
        check_consistency(rows, [])


def test_a_tag_shares_the_hash_of_its_base_name() -> None:
    rows = [
        row("r/gen_1", "u", 1, 0, 1, a_sha256="h"),
        row("r/gen_1@t", "u", 1, 0, 1, a_sha256="h", a_search={"c_init": 3.0}),
    ]
    check_consistency(rows, [])


def test_a_full_name_under_two_search_dicts_is_refused() -> None:
    rows = [
        row("r/gen_1@t", "u", 1, 0, 1, a_sha256="h", a_search={"c_init": 3.0}),
        row("r/gen_1@t", "v", 1, 0, 1, a_sha256="h", a_search={"c_init": 4.0}),
    ]
    with pytest.raises(ValueError, match="r/gen_1@t"):
        check_consistency(rows, [])


def test_uniform_hashes_are_never_compared() -> None:
    rows = [
        row("c", "uniform@100", 1, 0, 1, a_sha256="h1", b_sha256=None),
        row("d", "uniform@100", 1, 0, 1, a_sha256="h2", b_sha256=None),
    ]
    check_consistency(rows, [ref("uniform@100", sha=None)])


def test_pairings() -> None:
    refs = [ref("u@100"), ref("r/gen_5"), ref("r/gen_10"), ref("r/gen_15")]
    logged = [row("c", "r/gen_10", 1, 0, 1)]
    # Own full name dropped, logged pairing dropped, the rest in pool order.
    assert [r.name for r in pairings("r/gen_5", refs, [])] == ["u@100", "r/gen_10", "r/gen_15"]
    assert [r.name for r in pairings("c", refs, logged)] == ["u@100", "r/gen_5", "r/gen_15"]
    # A tagged candidate still plays its own untagged entry.
    assert "r/gen_5" in [r.name for r in pairings("r/gen_5@t", refs, [])]


# --- 5. Ratings --------------------------------------------------------------------


def test_rating_table_anchors_and_orders() -> None:
    rows = [
        row("c", "uniform@100", 75, 0, 25),
        row("c", "uniform@400", 25, 0, 75),
        row("uniform@400", "uniform@100", 90, 0, 10),
    ]
    table = rating_table(rows)
    assert table[ANCHOR] == 0.0
    assert table["uniform@100"] < table["c"] < table["uniform@400"]
    assert rating_table(list(reversed(rows))) == pytest.approx(table)


def test_as_matches() -> None:
    assert as_matches([row("a", "b", 3, 2, 1)]) == [("a", "b", 3, 2, 1)]


# --- 6. Bootstrap ------------------------------------------------------------------


def balanced_log(scale: int = 1) -> list[BenchmarkRow]:
    return [
        row("c", "uniform@100", 30 * scale, 4 * scale, 10 * scale),
        row("c", "uniform@400", 14 * scale, 4 * scale, 26 * scale),
        row("uniform@400", "uniform@100", 36 * scale, 4 * scale, 4 * scale),
    ]


def test_intervals_contain_the_point_estimate_and_pin_the_anchor() -> None:
    log = balanced_log()
    table = rating_table(log)
    intervals = bootstrap_intervals(log, resamples=50)
    assert intervals[ANCHOR] == (0.0, 0.0)
    for player, (low, high) in intervals.items():
        assert low <= table[player] <= high


def test_more_games_narrow_the_interval() -> None:
    narrow = bootstrap_intervals(balanced_log(4), resamples=50)["c"]
    wide = bootstrap_intervals(balanced_log(1), resamples=50)["c"]
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_the_bootstrap_is_reproducible() -> None:
    log = balanced_log()
    assert bootstrap_intervals(log, resamples=50) == bootstrap_intervals(log, resamples=50)
    assert bootstrap_interval(log, "c", resamples=50) == bootstrap_intervals(log, resamples=50)["c"]


def test_a_swept_player_has_a_finite_open_interval() -> None:
    log = [row("c", "uniform@100", 72, 0, 0), row("c", "uniform@400", 72, 0, 0),
           row("uniform@400", "uniform@100", 50, 0, 22)]
    point = rating_table(log)["c"]
    low, high = bootstrap_intervals(log, resamples=50)["c"]
    assert math.isfinite(low) and math.isfinite(high)
    assert high > low
    assert high >= point


def test_bootstrap_of_nothing() -> None:
    assert bootstrap_intervals([], resamples=5) == {}


# --- 7. ratings.json ---------------------------------------------------------------


def test_write_ratings(tmp_path: Path) -> None:
    path = tmp_path / "ratings.json"
    log = balanced_log()
    written = write_ratings(path, log, resamples=20)
    assert set(written["players"]) == {"c", "uniform@100", "uniform@400"}
    assert written["anchor"] == ANCHOR
    assert written["players"][ANCHOR]["elo"] == 0.0
    assert set(written["players"]["c"]) == {"elo", "low", "high", "games"}
    assert written["players"]["c"]["games"] == 88
    first = path.read_bytes()
    write_ratings(path, log, resamples=20)
    assert path.read_bytes() == first


def test_write_ratings_refuses_an_empty_log(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_ratings(tmp_path / "ratings.json", [])


# --- 8. load_pool (torch) ----------------------------------------------------------


def _checkpoint(path: Path, size: int) -> None:
    import torch

    from ox_zero.engine.network import Network, NetworkConfig
    from ox_zero.training.checkpoint import save_checkpoint

    torch.manual_seed(0)
    save_checkpoint(
        path, Network(size, NetworkConfig(blocks=1, filters=4, value_hidden=8)), generation=0
    )


def test_load_pool(tmp_path: Path) -> None:
    from ox_zero.engine.evaluator import UniformEvaluator
    from ox_zero.engine.network_evaluator import NetworkEvaluator
    from ox_zero.training.benchmark import load_pool

    _checkpoint(tmp_path / "pool" / "t-gen_001.pt", 3)
    refs = list_pool(tmp_path)
    loaded = load_pool(refs, 3, None)
    assert isinstance(loaded["t/gen_001"], NetworkEvaluator)
    assert isinstance(loaded["uniform@100"], UniformEvaluator)
    assert isinstance(loaded["uniform@400"], UniformEvaluator)


def test_load_pool_rejects_a_wrong_board_size(tmp_path: Path) -> None:
    from ox_zero.training.benchmark import load_pool

    _checkpoint(tmp_path / "pool" / "t-gen_001.pt", 4)
    with pytest.raises(ValueError, match="t-gen_001.pt"):
        load_pool(list_pool(tmp_path), 3, None)
