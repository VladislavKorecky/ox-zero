"""Exact minimax solver for small OXOX boards.

Why an exact solver
-------------------
AlphaZero never computes the true value of a position; it estimates it, with a
network trained on its own self-play. On boards small enough to search every
line of play (3x3, 4x4, maybe 5x5) we can compute the *exact* answer instead.
That makes the solver the project's only ground truth: a fixture for testing
MCTS and the network ("does search find the known best move?") and a sanity
check on the rules themselves. See docs/design/engineering.md, "Testing".

Negamax
-------
Minimax says: the player to move picks the move that is best for them,
assuming the opponent then does the same. OXOX is *zero-sum* (one side's win
is the other's loss) and the players *alternate*, so we can score every
position from the point of view of whoever is to move and collapse "max for
me, min for you" into a single rule:

    value(s) = max over legal a of  -value(apply_move(s, a))

The child `apply_move(s, a)` is scored for the *opponent*, so its value is
negated to bring it back to our perspective. This formulation is called
*negamax* (Knuth & Moore, 1975, "An analysis of alpha-beta pruning").

Values are `+1` (the side to move wins with perfect play), `0` (draw), `-1`
(the side to move loses). This is the same convention as the network's value
head `v` and the MCTS backups in docs/design/search.md, which is exactly what
lets the solver's numbers be compared with the engine's later.

At a terminal position the side to move can never have won: the game ends on
the move that completes a line, and that move was made by the *previous*
player. So terminal values are only `-1` (somebody won, i.e. the side to move
lost) or `0` (full board, no line).

Memoisation (transposition table)
---------------------------------
Many move orders reach the same position: X(0,0) O(1,1) X(2,2) and
X(2,2) O(1,1) X(0,0) leave identical boards. Such repeats are called
*transpositions*. Without a cache the solver re-searches each one from
scratch, and the work grows like the number of move *orders* (16! for 4x4,
about 2e13). With a cache it grows like the number of distinct *positions*
(at most 3^16, about 4e7, for 4x4, and far fewer in practice because games end
early). `State` is frozen and hashable precisely so it can be a dictionary key
here (see the module docstring of `rules.py`).
"""

from __future__ import annotations

from ox_zero.game.rules import Cell, State, apply_move, is_terminal, legal_moves


class Solver:
    """Exact game values by negamax with a transposition table.

    Each instance owns its own cache, so a fresh `Solver()` starts empty and
    reusing one instance across queries shares all the work already done.
    """

    def __init__(self) -> None:
        # position -> exact value for the side to move at that position.
        self._cache: dict[State, int] = {}

    @property
    def positions_solved(self) -> int:
        """Number of distinct positions whose exact value is cached."""
        return len(self._cache)

    def value(self, state: State) -> int:
        """The exact game value for the side to move: `+1`, `0`, or `-1`.

        Recursion depth is at most the number of cells (16 on 4x4, 25 on
        5x5), far below Python's default recursion limit of 1000.
        """
        cached = self._cache.get(state)
        if cached is not None:
            return cached

        if is_terminal(state):
            value = -1 if state.winner is not None else 0
        else:
            # Start from the worst possible outcome and improve on it.
            value = -1
            for move in legal_moves(state):
                value = max(value, -self.value(apply_move(state, move)))
                if value == 1:
                    # Nothing beats a forced win, so the remaining moves
                    # cannot change the answer. This is still an *exact*
                    # value, not a bound: we only skip moves that could at
                    # most tie. (Alpha-beta pruning would go further and
                    # cache bounds like "at least 0", which would make
                    # `best_moves` and the cache unreliable; don't add it.)
                    break

        self._cache[state] = value
        return value

    def best_moves(self, state: State) -> list[Cell]:
        """Every legal move that achieves `value(state)`, in board order.

        Unlike `value`, this cannot stop at the first winning move: it has to
        score every child to know which of them are optimal. Returns `[]` for
        a terminal position, which has no moves.
        """
        target = self.value(state)
        return [
            move
            for move in legal_moves(state)
            if -self.value(apply_move(state, move)) == target
        ]
