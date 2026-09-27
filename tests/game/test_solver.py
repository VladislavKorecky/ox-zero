"""Tests for the exact minimax (negamax) solver on small boards.

The solver is the project's only source of ground truth: for boards small
enough to search exhaustively it returns the *exact* game value, which later
serves as a fixture for testing MCTS and the trained network.

Value convention (same as the search design, docs/design/search.md): values
are from the perspective of the side to move. `+1` means the side to move
wins with perfect play, `0` a draw, `-1` the side to move loses.

Coordinates are `(row, col)`, zero-based, row 0 at the top.
"""

import random
import time

from ox_zero.game import (
    Mark,
    State,
    apply_move,
    initial_state,
    is_terminal,
    legal_moves,
    play,
)
from ox_zero.game.solver import Solver


# --- Terminal positions ------------------------------------------------------


def test_won_position_is_a_loss_for_the_side_to_move():
    # X O X in row 0: X's third move completed the line, so X has won and it
    # is O (the side to move) who has lost.
    state = play([(0, 0), (0, 1), (0, 2)], size=3)
    assert state.winner is Mark.X

    solver = Solver()
    assert solver.value(state) == -1
    assert solver.best_moves(state) == []


def test_full_1x1_board_is_a_draw():
    state = play([(0, 0)], size=1)
    assert is_terminal(state)

    solver = Solver()
    assert solver.value(state) == 0
    assert solver.best_moves(state) == []


def test_full_2x2_board_is_a_draw():
    # No line of three fits on a 2x2 board, so a full one is always a draw.
    state = play([(0, 0), (0, 1), (1, 0), (1, 1)], size=2)
    assert is_terminal(state)

    solver = Solver()
    assert solver.value(state) == 0
    assert solver.best_moves(state) == []


# --- Short tactics -----------------------------------------------------------


def test_win_in_one():
    # X O _
    # _ _ _
    # _ _ _      X to move; (0,2) completes X O X.
    state = play([(0, 0), (0, 1)], size=3)

    solver = Solver()
    assert solver.value(state) == 1
    assert solver.best_moves(state) == [(0, 2)]


def test_win_in_one_with_two_winning_moves_listed_in_board_order():
    # X O _
    # O X _
    # _ _ _      X to move; (0,2) completes row 0 and (2,0) completes
    #            column 0, both as X O X.
    state = play([(0, 0), (0, 1), (1, 1), (1, 0)], size=3)

    solver = Solver()
    assert solver.value(state) == 1
    assert solver.best_moves(state) == [(0, 2), (2, 0)]


def test_forced_loss_lists_every_move():
    # X . O
    # X O O
    # X X .      O to move, no line complete yet.
    #
    # O's two moves both lose: after O (0,1), X (2,2) completes the diagonal
    # X O X; after O (2,2), X (0,1) completes column 1 as X O X. Every legal
    # move achieves the value -1, so both are "best".
    state = play([(0, 0), (1, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)], size=3)
    assert state.winner is None
    assert state.to_move is Mark.O

    solver = Solver()
    assert solver.value(state) == -1
    assert solver.best_moves(state) == [(0, 1), (2, 2)]


# --- Negamax consistency (property test) --------------------------------------


def _random_positions(size: int, count: int, rng: random.Random) -> list[State]:
    """`count` reachable positions: random play stopped at a random depth."""
    positions = []
    for _ in range(count):
        state = initial_state(size)
        for _ in range(rng.randrange(size * size)):
            if is_terminal(state):
                break
            state = apply_move(state, rng.choice(legal_moves(state)))
        positions.append(state)
    return positions


def test_value_satisfies_the_negamax_recurrence():
    # The defining property of negamax: a position is worth the best of its
    # children, each negated because the child is scored for the opponent.
    #     value(s) = max over a of -value(apply_move(s, a))
    # `best_moves` must be exactly the moves reaching that maximum.
    rng = random.Random(0)
    solver = Solver()
    positions = _random_positions(3, 100, rng) + _random_positions(4, 100, rng)

    for state in positions:
        if is_terminal(state):
            continue
        child_values = {
            move: -solver.value(apply_move(state, move)) for move in legal_moves(state)
        }
        best = max(child_values.values())
        assert solver.value(state) == best
        assert solver.best_moves(state) == [
            move for move in legal_moves(state) if child_values[move] == best
        ]


# --- Cache -------------------------------------------------------------------


def test_cache_is_filled_once_and_per_instance():
    solver = Solver()
    assert solver.positions_solved == 0

    solver.value(initial_state(3))
    solved = solver.positions_solved
    assert solved > 0

    # A second query of the same position is a pure cache hit.
    solver.value(initial_state(3))
    assert solver.positions_solved == solved

    assert Solver().positions_solved == 0


# --- Board sizes ---------------------------------------------------------------


def test_1x1_board_is_a_draw():
    # One move fills the board, and no line of three fits.
    assert Solver().value(initial_state(1)) == 0


def test_2x2_board_is_a_draw():
    assert Solver().value(initial_state(2)) == 0


# --- Speed guard ---------------------------------------------------------------


def test_3x3_solves_in_well_under_a_second():
    start = time.perf_counter()
    Solver().value(initial_state(3))
    assert time.perf_counter() - start < 1.0


def test_4x4_solves():
    # No timing assertion: this guards against the search blowing up so badly
    # that the test suite stops finishing at all.
    assert Solver().value(initial_state(4)) in (-1, 0, 1)
