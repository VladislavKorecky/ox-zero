"""Tests for the position encoding and the board symmetries.

The encoding turns a `State` into the network's input: three `S x S` planes
(side to move, other side, all ones), see docs/design/network.md, "Input
encoding". The symmetries are the 8 elements of the dihedral group D4 acting
on the square board; they are used for training augmentation, so planes,
policies and single cells must all be transformed the same way or the
augmented examples would teach the network wrong moves.

Coordinates are `(row, col)`, zero-based, row 0 at the top. Policies are flat
vectors in board order, index `row * S + col`.
"""

import random

import numpy as np
import pytest

from ox_zero.engine.encoding import (
    NUM_PLANES,
    NUM_SYMMETRIES,
    encode,
    encode_batch,
    inverse_symmetry,
    legal_mask,
    transform_cell,
    transform_planes,
    transform_policy,
)
from ox_zero.game import Mark, apply_move, initial_state, is_terminal, legal_moves, play


def _cells_of(plane: np.ndarray) -> set[tuple[int, int]]:
    """The `(row, col)` cells where a plane is 1."""
    return {(int(r), int(c)) for r, c in np.argwhere(plane == 1.0)}


# --- Plane contents -----------------------------------------------------------


def test_planes_are_side_to_move_other_side_and_ones():
    # X: (0,0) (2,3); O: (1,1) (3,0). Four marks, so X is to move.
    state = play([(0, 0), (1, 1), (2, 3), (3, 0)], size=4)
    assert state.to_move is Mark.X

    planes = encode(state)
    assert _cells_of(planes[0]) == {(0, 0), (2, 3)}  # side to move = X
    assert _cells_of(planes[1]) == {(1, 1), (3, 0)}  # other side = O
    assert np.all(planes[2] == 1.0)
    # Nothing but zeros and ones.
    assert set(np.unique(planes[:2])) <= {0.0, 1.0}


def test_encoding_is_relative_to_the_side_to_move():
    # One more X mark hands the move to O. Plane 0 must now be O's marks: the
    # encoding says "mine / theirs", not "X / O".
    state = play([(0, 0), (1, 1), (2, 3), (3, 0), (0, 3)], size=4)
    assert state.to_move is Mark.O

    planes = encode(state)
    assert _cells_of(planes[0]) == {(1, 1), (3, 0)}
    assert _cells_of(planes[1]) == {(0, 0), (2, 3), (0, 3)}


@pytest.mark.parametrize("size", [3, 4, 6, 12])
def test_shape_and_dtype(size):
    planes = encode(initial_state(size))
    assert NUM_PLANES == 3
    assert planes.shape == (3, size, size)
    assert planes.dtype == np.float32


def test_encode_batch_stacks():
    states = [play([(0, 0)], size=6), play([(1, 1), (2, 2)], size=6), initial_state(6)]
    batch = encode_batch(states)
    assert batch.shape == (3, 3, 6, 6)
    assert batch.dtype == np.float32
    for i, state in enumerate(states):
        np.testing.assert_array_equal(batch[i], encode(state))


# --- Legal-move mask ------------------------------------------------------------


def test_legal_mask_matches_legal_moves():
    state = play([(0, 0), (1, 1), (2, 3)], size=4)
    mask = legal_mask(state)
    assert mask.shape == (16,)
    assert mask.dtype == np.bool_
    expected = {r * 4 + c for r, c in legal_moves(state)}
    assert set(np.flatnonzero(mask).tolist()) == expected


def test_legal_mask_is_all_false_when_the_game_is_over():
    # X O X in row 0: X has won, empty cells remain but none is legal.
    state = play([(0, 0), (0, 1), (0, 2)], size=3)
    assert is_terminal(state)
    assert not legal_mask(state).any()


# --- Symmetries -----------------------------------------------------------------


def _asymmetric_planes() -> np.ndarray:
    # Three marks placed so that no rotation or reflection maps the pattern
    # onto itself, hence every symmetry produces a different board.
    return encode(play([(0, 1), (2, 2), (3, 0)], size=4))


def test_there_are_eight_distinct_symmetries():
    assert NUM_SYMMETRIES == 8
    planes = _asymmetric_planes()
    views = [transform_planes(planes, k).tobytes() for k in range(NUM_SYMMETRIES)]
    assert len(set(views)) == 8


def test_symmetry_zero_is_the_identity():
    planes = _asymmetric_planes()
    np.testing.assert_array_equal(transform_planes(planes, 0), planes)


@pytest.mark.parametrize("k", range(8))
def test_inverse_undoes_the_symmetry(k):
    planes = _asymmetric_planes()
    back = transform_planes(transform_planes(planes, k), inverse_symmetry(k))
    np.testing.assert_array_equal(back, planes)
    assert inverse_symmetry(inverse_symmetry(k)) == k


@pytest.mark.parametrize("k", range(8))
def test_policies_and_cells_transform_like_planes(k):
    size = 4
    for row in range(size):
        for col in range(size):
            onehot = np.zeros(size * size, dtype=np.float32)
            onehot[row * size + col] = 1.0
            moved = transform_policy(onehot, k)
            new_row, new_col = transform_cell((row, col), k, size)
            expected = np.zeros_like(onehot)
            expected[new_row * size + new_col] = 1.0
            np.testing.assert_array_equal(moved, expected)


def test_transform_policy_acts_on_the_last_axis_of_a_batch():
    rng = np.random.default_rng(0)
    policies = rng.random((5, 16)).astype(np.float32)
    batch = transform_policy(policies, 3)
    assert batch.shape == (5, 16)
    for i in range(5):
        np.testing.assert_array_equal(batch[i], transform_policy(policies[i], 3))


@pytest.mark.parametrize("k", range(8))
def test_playing_transformed_moves_gives_transformed_planes(k):
    # The symmetry is a symmetry of the *game*: playing the mirrored moves
    # reaches the mirrored position, so the encodings must agree.
    size = 5
    rng = random.Random(k)
    state = initial_state(size)
    moves = []
    while len(moves) < 8 and not is_terminal(state):
        move = rng.choice(legal_moves(state))
        moves.append(move)
        state = apply_move(state, move)

    mirrored = play([transform_cell(m, k, size) for m in moves], size=size)
    np.testing.assert_array_equal(encode(mirrored), transform_planes(encode(state), k))


@pytest.mark.parametrize("k", range(8))
def test_transformed_planes_are_contiguous(k):
    # `np.flip` and `np.rot90` return views with negative strides, which
    # `torch.from_numpy` refuses. The transform must hand back a real copy.
    out = transform_planes(_asymmetric_planes(), k)
    assert out.flags["C_CONTIGUOUS"]
