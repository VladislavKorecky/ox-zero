"""Tests for lockstep self-play: labelling, batching, refilling, seeds, temperature.

Self-play is where AlphaZero's training data comes from: the current network
plays itself, the search at every move produces a policy target `π` (the
root's visit distribution), and the game's result labels every position with
a value target `z`. These tests pin the three things that can silently go
wrong: the labels (`z` signs, `π` rows), the batching (one network call per
lockstep step, never more than `parallel` positions), and the randomness
(seeded, and honouring the temperature schedule).

Plan: docs/plans/archive/04-training-pipeline.md, step 1. Torch is imported only by
the network smoke test, locally, so the rest of the file stays fast.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from ox_zero.engine.encoding import encode, legal_mask
from ox_zero.engine.evaluator import Evaluator, TableEvaluator, UniformEvaluator
from ox_zero.engine.search import SELF_PLAY, SearchConfig, SearchTree
from ox_zero.game.notation import from_board_string
from ox_zero.game.rules import (
    Mark,
    State,
    apply_move,
    initial_state,
    is_terminal,
    play,
)
from ox_zero.training.replay import legal_from_planes
from ox_zero.training.selfplay import (
    Examples,
    GameRecord,
    SelfPlayConfig,
    SelfPlayResult,
    game_examples,
    lockstep_step,
    self_play,
)


def uniform_pi(state: State) -> np.ndarray:
    """A valid `float32` policy target: equal mass on every legal move."""
    mask = legal_mask(state).astype(np.float32)
    return mask / np.float32(mask.sum())


class CountingEvaluator:
    """Wraps an evaluator and remembers every batch it was asked to score."""

    def __init__(self, inner: Evaluator | None = None) -> None:
        self.inner = inner if inner is not None else UniformEvaluator()
        self.calls: list[list[State]] = []

    def evaluate(self, states: Sequence[State]):
        self.calls.append(list(states))
        return self.inner.evaluate(states)


def three_positions() -> list[State]:
    """3x3 positions with at most one mark: no root child can be a finished game."""
    empty = initial_state(3)
    return [empty, apply_move(empty, (1, 1)), apply_move(empty, (0, 0))]


# --- Test 1-2: game_examples labels positions ---------------------------------


def test_z_labelling_decisive_game() -> None:
    # X plays 0,0, O plays 0,1, X plays 0,2: row 0 reads X O X, an
    # alternating line, so X wins on move 3.
    moves = [(0, 0), (0, 1), (0, 2)]
    states = [play(moves[:i], size=3) for i in range(3)]
    final = play(moves, size=3)
    assert final.winner is Mark.X
    pis = [uniform_pi(s) for s in states]

    examples = game_examples(states, pis, final)

    assert len(examples) == 3
    assert examples.size == 3
    # z is from the side to move's perspective: X is to move before moves 0
    # and 2 (and won, so +1), O before move 1 (and lost, so -1).
    np.testing.assert_array_equal(examples.z, np.array([1, -1, 1], dtype=np.float32))
    assert examples.planes.dtype == np.uint8
    assert examples.pi.dtype == np.float32
    assert examples.z.dtype == np.float32
    for i, state in enumerate(states):
        np.testing.assert_array_equal(examples.planes[i], encode(state).astype(np.uint8))
        np.testing.assert_array_equal(examples.pi[i], pis[i])
    np.testing.assert_array_equal(
        legal_from_planes(examples.planes), np.stack([legal_mask(s) for s in states])
    )


def test_z_labelling_draw() -> None:
    # The "wall" draw (docs/design/open-questions.md, "Follow-up: safe moves
    # over a game"): two rows of X over two rows of O. Every line of three
    # crosses at most one X/O boundary, and the bands are two cells thick, so
    # a line always contains two equal marks next to each other (X X O,
    # X O O, X X X, ...), never the alternating X O X or O X O.
    board = "XXXXXXXXOOOOOOOO"
    final_from_string = from_board_string(board, size=4)
    assert final_from_string.winner is None
    assert is_terminal(final_from_string)

    # Interleave X's cells (rows 0-1) and O's cells (rows 2-3) in board order
    # so the turns alternate: 0,0 2,0 0,1 2,1 ...
    x_cells = [(r, c) for r in (0, 1) for c in range(4)]
    o_cells = [(r, c) for r in (2, 3) for c in range(4)]
    moves = [cell for pair in zip(x_cells, o_cells) for cell in pair]
    states = [play(moves[:i], size=4) for i in range(len(moves))]
    final = play(moves, size=4)
    assert final.board == final_from_string.board
    assert final.winner is None and is_terminal(final)
    # No intermediate position was already decided (the game reached the full board).
    assert all(not is_terminal(s) for s in states)
    pis = [uniform_pi(s) for s in states]

    examples = game_examples(states, pis, final)

    assert len(examples) == 16
    np.testing.assert_array_equal(examples.z, np.zeros(16, dtype=np.float32))


def test_examples_empty_and_concatenate() -> None:
    empty = Examples.empty(4)
    assert len(empty) == 0
    assert empty.size == 4
    assert empty.planes.shape == (0, 3, 4, 4) and empty.planes.dtype == np.uint8
    assert empty.pi.shape == (0, 16) and empty.pi.dtype == np.float32
    assert empty.z.shape == (0,) and empty.z.dtype == np.float32

    moves = [(0, 0), (0, 1), (0, 2)]
    states = [play(moves[:i], size=3) for i in range(3)]
    pis = [uniform_pi(s) for s in states]
    one = game_examples(states, pis, play(moves, size=3))
    joined = Examples.concatenate([one, Examples.empty(3), one])
    assert len(joined) == 6
    np.testing.assert_array_equal(joined.z, np.concatenate([one.z, one.z]))
    with pytest.raises(ValueError):
        Examples.concatenate([one, Examples.empty(4)])


# --- Test 3-4: lockstep_step --------------------------------------------------


def test_one_batched_call_per_step() -> None:
    positions = three_positions()
    trees = [SearchTree(state, SearchConfig()) for state in positions]
    evaluator = CountingEvaluator()

    # Step 1 is the root setup of every tree: one call with exactly the three
    # root states, and no simulation counted (setup is not a simulation).
    assert lockstep_step(trees, evaluator) == 0
    assert evaluator.calls == [positions]

    # Step 2: one call with one leaf per tree, one simulation each.
    assert lockstep_step(trees, evaluator) == 3
    assert len(evaluator.calls) == 2
    assert len(evaluator.calls[1]) == 3
    assert all(tree.simulations == 1 for tree in trees)


def test_terminal_leaf_skips_the_network_but_counts() -> None:
    # Row 0 is X O _, X to move: the first child in board order, 0,2,
    # completes X O X. With uniform priors PUCT picks it first, and it is a
    # finished game: its value is exact, so no network call is needed.
    state = from_board_string("XO_______", size=3)
    assert state.to_move is Mark.X
    tree = SearchTree(state, SearchConfig())
    evaluator = CountingEvaluator()

    assert lockstep_step([tree], evaluator) == 0  # root setup
    assert lockstep_step([tree], evaluator) == 1  # terminal leaf, still counted
    assert len(evaluator.calls) == 1  # only the setup reached the network
    assert tree.simulations == 1


def test_empty_step_does_not_call_the_evaluator() -> None:
    evaluator = CountingEvaluator()
    assert lockstep_step([], evaluator) == 0
    assert evaluator.calls == []


def test_lockstep_equals_batch_size_one() -> None:
    # Lockstep changes the schedule of the network calls, nothing else: each
    # tree still sees exactly the evaluations it would have asked for alone.
    # No root noise, so no shared-RNG ordering can differ between the two.
    config = SearchConfig()
    evaluator = UniformEvaluator()
    lockstep = [SearchTree(state, config) for state in three_positions()]
    steps = 0
    while any(tree.simulations < 20 for tree in lockstep):
        lockstep_step(lockstep, evaluator)
        steps += 1
    # One uncounted setup step, then one simulation per tree per step,
    # whether the leaf needed the network or was a finished game.
    assert steps == 21

    for tree, state in zip(lockstep, three_positions()):
        alone = SearchTree(state, config)
        alone.simulate(evaluator, 20)
        assert {m: c.visit_count for m, c in tree.root.children.items()} == {
            m: c.visit_count for m, c in alone.root.children.items()
        }


# --- Test 5-8: self_play ------------------------------------------------------


SMALL = SelfPlayConfig(games=6, parallel=3, simulations=8)


def test_config_defaults() -> None:
    assert SelfPlayConfig() == SelfPlayConfig(games=128, parallel=64, simulations=100)


def test_config_needs_at_least_one_game() -> None:
    # Zero games would give an empty generation, and training on an empty
    # buffer (generation 1 has nothing older) has nothing to sample.
    with pytest.raises(ValueError, match="games"):
        SelfPlayConfig(games=0)
    with pytest.raises(ValueError, match="games"):
        SelfPlayConfig(games=-1)


def test_a_generation_of_games() -> None:
    result = self_play(UniformEvaluator(), 4, SELF_PLAY, SMALL, np.random.default_rng(0))

    assert isinstance(result, SelfPlayResult)
    assert len(result.games) == 6
    for record in result.games:
        assert isinstance(record, GameRecord)
        final = play(record.moves, size=4)
        assert is_terminal(final)
        assert final.winner is record.winner

    examples = result.examples
    assert examples.size == 4
    assert len(examples) == sum(len(r.moves) for r in result.games)
    np.testing.assert_allclose(examples.pi.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all(examples.pi[~legal_from_planes(examples.planes)] == 0)

    # Each game's last recorded position is the winner to move (the winner
    # made the last move), so its z is +1 in a decisive game.
    end = 0
    for record in result.games:
        end += len(record.moves)
        if record.winner is not None:
            assert examples.z[end - 1] == 1.0
        else:
            assert examples.z[end - 1] == 0.0

    assert result.steps > 0
    # At least the per-move budget; inherited subtree visits are extra.
    assert result.simulations >= 8 * len(examples)


def test_refilling_keeps_the_batch_bounded() -> None:
    evaluator = CountingEvaluator()
    result = self_play(evaluator, 4, SELF_PLAY, SMALL, np.random.default_rng(1))

    assert max(len(batch) for batch in evaluator.calls) <= SMALL.parallel
    # At most one network call per lockstep step (none when every leaf was terminal).
    assert len(evaluator.calls) <= result.steps
    # Every game starts from the empty board, so the number of distinct games
    # is the number of times the empty board was set up as a root.
    empty = initial_state(4)
    assert sum(batch.count(empty) for batch in evaluator.calls) == SMALL.games
    assert len(result.games) == SMALL.games


def test_seeds() -> None:
    def run(seed: int) -> Examples:
        return self_play(
            UniformEvaluator(), 4, SELF_PLAY, SMALL, np.random.default_rng(seed)
        ).examples

    a, b, c = run(7), run(7), run(8)
    np.testing.assert_array_equal(a.planes, b.planes)
    np.testing.assert_array_equal(a.pi, b.pi)
    np.testing.assert_array_equal(a.z, b.z)
    # Noise and sampling are on in SELF_PLAY, so another seed plays other games.
    assert a.pi.shape != c.pi.shape or not np.array_equal(a.pi, c.pi)


def _favourite_evaluator() -> TableEvaluator:
    """All of the empty 3x3 board's prior on the centre, 1,1."""
    prior = np.zeros(9, dtype=np.float32)
    prior[4] = 1.0
    return TableEvaluator({initial_state(3): (prior, 0.0)})


def test_temperature_zero_is_greedy_from_move_0() -> None:
    search = SearchConfig(root_noise=True, temperature_cutoff=0)
    config = SelfPlayConfig(games=10, parallel=5, simulations=10)
    result = self_play(_favourite_evaluator(), 3, search, config, np.random.default_rng(0))
    assert all(record.moves[0] == (1, 1) for record in result.games)


def test_default_cutoff_samples_the_opening() -> None:
    search = SearchConfig(root_noise=True)
    config = SelfPlayConfig(games=20, parallel=10, simulations=10)
    result = self_play(_favourite_evaluator(), 3, search, config, np.random.default_rng(0))
    assert len({record.moves[0] for record in result.games}) > 1


# --- Test 9: the real network -------------------------------------------------


def test_network_smoke() -> None:
    import torch  # the only torch import in this file: selfplay itself is torch-free

    from ox_zero.engine.network import Network, NetworkConfig
    from ox_zero.engine.network_evaluator import NetworkEvaluator

    torch.manual_seed(0)
    network = Network(4, NetworkConfig(blocks=1, filters=4, value_hidden=8))
    evaluator = NetworkEvaluator(network, torch.device("cpu"))
    config = SelfPlayConfig(games=2, parallel=2, simulations=4)

    result = self_play(evaluator, 4, SELF_PLAY, config, np.random.default_rng(0))

    assert len(result.games) == 2
    n = sum(len(r.moves) for r in result.games)
    assert result.examples.planes.shape == (n, 3, 4, 4)
    assert result.examples.pi.shape == (n, 16)
    assert result.examples.z.shape == (n,)
    for record in result.games:
        assert is_terminal(play(record.moves, size=4))


def test_self_play_rejects_expand_root() -> None:
    # `self_play` drives trees through `lockstep_step`, which never runs the
    # analysis-only up-front child evaluation, so an `expand_root=True`
    # config would silently search differently from `SearchTree.simulate`.
    assert SELF_PLAY.expand_root is False
    with pytest.raises(ValueError):
        self_play(
            UniformEvaluator(), 4, SearchConfig(expand_root=True), SMALL, np.random.default_rng(0)
        )
