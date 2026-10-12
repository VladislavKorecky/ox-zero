"""Tests for scripts/experiments/forced_openings.py: the bounded forced-win search.

The search answers "can the side to move force a win within k of its own
moves, against every defence?". Its ground truth is the exact solver
(`ox_zero.game.solver`): on 3x3 and 4x4, with a horizon long enough to cover
the whole game and every cell as a candidate, the two must agree on every
opening. The other tests pin the pieces: immediate wins, double threats, the
defender's escape by winning first, and the known 8x8 results.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ox_zero.game import apply_move, initial_state, play
from ox_zero.game.solver import Solver

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "forced_openings.py"


@pytest.fixture(scope="module")
def fo():
    """Import the script as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location("forced_openings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["forced_openings"] = module
    spec.loader.exec_module(module)
    return module


# --- winning_cells ---------------------------------------------------------------


def test_winning_cells_empty_board_has_none(fo):
    assert fo.winning_cells(initial_state(6)) == []


def test_winning_cells_finds_the_end_of_o_x_gap(fo):
    # X 1,1  O 0,0  X 5,5: O to move; the diagonal O(0,0) X(1,1) _(2,2) is
    # completed by O at 2,2 (OXO).
    state = play([(1, 1), (0, 0), (5, 5)], 6)
    assert (2, 2) in fo.winning_cells(state)


def test_winning_cells_finds_the_middle_of_x_gap_x(fo):
    # X 0,0  O 5,5  X 0,2: O to move wins in the middle, X O X.
    state = play([(0, 0), (5, 5), (0, 2)], 6)
    assert fo.winning_cells(state) == [(0, 1)]


# --- forces_win: the pieces ------------------------------------------------------


def test_immediate_win_is_a_forced_win_in_one(fo):
    state = play([(0, 0), (5, 5), (0, 2)], 6)
    assert fo.forces_win(state, 1)


def test_no_win_in_one_on_a_quiet_board(fo):
    assert not fo.forces_win(apply_move(initial_state(6), (0, 0)), 1)


def test_double_threat_wins_in_two(fo):
    # The proven 1,1 line on 8x8: O's edge marks force X's blocks, and at this
    # point O 1,0 makes two winning cells, so O to move wins within 2 moves
    # but not within 1.
    state = play([(1, 1), (0, 0), (2, 2), (0, 1), (2, 1), (0, 2), (2, 0)], 8)
    assert fo.forces_win(state, 2)
    assert not fo.forces_win(state, 1)


def test_a_single_threat_is_not_a_win_in_two(fo):
    # X 1,1 then O 0,0 threatens 2,2, but X just blocks: O has no forced win
    # within 2 of its moves from the position after X 1,1 (it needs 3).
    state = apply_move(initial_state(8), (1, 1))
    assert not fo.forces_win(state, 2)


# --- forces_win against the exact solver -----------------------------------------


@pytest.mark.parametrize("size", [3, 4])
def test_agrees_with_the_exact_solver_on_every_opening(fo, size):
    """With every cell a candidate and a horizon covering the whole game, the
    bounded search is exact: O forces a win after X's opening iff the solver
    says the position (O to move) is worth +1."""
    solver = Solver()
    horizon = (size * size) // 2 + 1
    for row in range(size):
        for col in range(size):
            state = apply_move(initial_state(size), (row, col))
            expected = solver.value(state) == 1
            assert fo.forces_win(state, horizon, radius=None) == expected, (row, col)


# --- the 8x8 results recorded in open-questions.md -------------------------------


@pytest.mark.parametrize("cell", [(1, 1), (1, 2), (1, 3)])
def test_second_ring_openings_lose_on_8x8(fo, cell):
    assert fo.forces_win(apply_move(initial_state(8), cell), 4)


@pytest.mark.parametrize("cell", [(0, 0), (0, 3)])
def test_edge_openings_have_no_short_forced_loss_on_8x8(fo, cell):
    assert not fo.forces_win(apply_move(initial_state(8), cell), 3)


# --- symmetry --------------------------------------------------------------------


@pytest.mark.parametrize("size", [3, 4, 8, 12])
def test_representatives_cover_every_cell_once_by_symmetry(fo, size):
    """Every cell is the image of exactly one representative under the
    board's eight symmetries (rotations and reflections)."""
    reps = fo.representatives(size)
    covered = {}
    for rep in reps:
        for image in fo.symmetric_images(rep, size):
            covered.setdefault(image, set()).add(rep)
    assert set(covered) == {(r, c) for r in range(size) for c in range(size)}
    assert all(len(owners) == 1 for owners in covered.values())


def test_opening_map_is_symmetric_and_marks_the_second_ring(fo):
    grid = fo.opening_map(8, 3)
    assert grid[(1, 1)] and grid[(6, 6)] and grid[(1, 6)] and grid[(6, 1)]
    for (r, c), lost in grid.items():
        assert lost == grid[(c, r)] == grid[(7 - r, c)] == grid[(r, 7 - c)]


# --- witness lines ---------------------------------------------------------------


@pytest.mark.parametrize("cell", [(1, 1), (1, 2), (1, 3)])
def test_witness_line_replays_to_an_o_win(fo, cell):
    """`witness` returns a principal line of the proof; replayed from the
    opening it ends with the attacker completing a line."""
    start = apply_move(initial_state(8), cell)
    line = fo.witness(start, 4)
    state = start
    for move in line:
        state = apply_move(state, move)
    assert state.winner is not None and state.winner.name == "O"


def test_witness_is_none_without_a_forced_win(fo):
    assert fo.witness(apply_move(initial_state(8), (0, 0)), 3) is None


@pytest.mark.parametrize("cell", [(1, 1), (1, 2), (1, 3)])
def test_witness_is_the_longest_defence_against_the_fastest_attack(fo, cell):
    """The witness is a principal line: the attacker always picks a move that
    wins in the fewest moves, the defender a reply that holds out longest. So
    its number of attacker moves is the position's exact distance to a forced
    win, the smallest `k` for which `forces_win` holds."""
    start = apply_move(initial_state(8), cell)
    distance = next(k for k in range(1, 5) if fo.forces_win(start, k))
    line = fo.witness(start, 4)
    attacker_moves = len(line[0::2])  # attacker, defender, attacker, ...
    assert attacker_moves == distance


# --- the attacker under threat ---------------------------------------------------


def test_attacker_facing_a_double_threat_cannot_force_a_win(fo):
    # O to move, no immediate win, and X threatens 2,2 and 3,4 at once: O
    # blocks one, X takes the other, at every horizon.
    state = play([(0, 2), (1, 2), (5, 4), (4, 4), (4, 1)], 6)
    assert fo.winning_cells(state) == []
    for moves in (1, 2, 3):
        assert not fo.forces_win(state, moves)


def test_clear_caches_empties_the_memo(fo):
    fo.forces_win(apply_move(initial_state(8), (1, 1)), 3)
    assert fo.cache_size() > 0
    fo.clear_caches()
    assert fo.cache_size() == 0


@pytest.mark.parametrize("size", [3, 4])
def test_radius_two_agrees_with_the_exact_solver_on_every_opening(fo, size):
    """The published configuration (attacker within distance 2 of a mark,
    defender everywhere) agrees with the solver on small boards too."""
    solver = Solver()
    horizon = (size * size) // 2 + 1
    for row in range(size):
        for col in range(size):
            state = apply_move(initial_state(size), (row, col))
            assert fo.forces_win(state, horizon, radius=2) == (solver.value(state) == 1), (row, col)


def test_witness_moves_stay_within_the_search_radius(fo):
    """Every attacker move of a witness is one the search itself considered:
    within distance 2 of a mark already on the board."""
    start = apply_move(initial_state(12), (1, 4))
    state = start
    for i, move in enumerate(fo.witness(start, 4)):
        if i % 2 == 0:  # attacker moves
            marks = [divmod(j, 12) for j, v in enumerate(state.board) if v is not None]
            assert min(max(abs(move[0] - a), abs(move[1] - b)) for a, b in marks) <= 2
        state = apply_move(state, move)


def test_radius_flag_is_recorded(fo, tmp_path):
    import json, subprocess
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(SCRIPT), "--sizes", "4", "--moves", "2", "--radius", "3", "--out", str(out)],
                   check=True, capture_output=True)
    assert json.loads(out.read_text())["parameters"]["radius"] == 3
