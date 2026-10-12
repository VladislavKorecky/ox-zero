"""Which first moves lose by force: a bounded forced-win search over every opening.

Prompted by Vláďa's observation (2026-10-12) that the cell diagonally next to a
corner (1,1) is "a guaranteed loss" for the first player. The search below
proves that and more: on 8x8 and 12x12 every cell of the *second ring* (one
step in from the edge) loses by force for X. Results are recorded in
docs/design/open-questions.md.

The question
------------
After X's first move, can O force a win within `k` of its own moves, against
*every* defence? This is a bounded version of the exact game value the solver
(`ox_zero.game.solver`) computes on small boards: an AND/OR tree search.

- At an OR node (the attacker, here O, to move) it is enough that *one* move
  leads to a win.
- At an AND node (the defender, X, to move) *every* reply must lose.

`True` is a proof: a winning strategy exists and the search found it. `False`
only says "no forced win within `k` moves"; a longer one may exist.

Why the tree stays small
------------------------
Two rules of OXOX cut the defender's options without losing exactness:

- If the defender can win at once, it does, and the attacker's line fails.
- If the attacker has a winning cell (it would complete a line next move),
  the defender must occupy that cell: nothing else stops it. Two or more
  winning cells cannot all be blocked with one move (a *double threat*), so
  the defender is lost. Placing a mark elsewhere never removes a win, because
  a winning cell is a fact about marks already on the board.

Only when the attacker threatens nothing is the defender free, and then every
empty cell is tried.

The attacker's moves are limited to cells within `radius` (2 by default) of
an existing mark. A line of three spans two steps, so a farther move cannot
form a line with any existing mark; it could still be a slow setup move, which
is why a `False` under the radius is not a proof of safety. `radius=None`
tries every cell (exact within the horizon; used by the tests on 3x3 and 4x4,
where it agrees with the solver on every opening).

Memoisation: positions repeat across move orders (transpositions), so results
are cached per (position, moves left, radius).

Symmetry: the board has eight symmetries (four rotations, each optionally
mirrored), and an opening's value is the same for all its images, so only one
representative per orbit is searched (10 of 64 cells on 8x8, 21 of 144 on 12x12).

Output: a grid per size on stdout (`L` = proven loss for X within the horizon),
one witness line per lost representative, and a JSON file. The JSON is
deterministic; wall-clock times go to a separate `*.timing.json`.

Usage:
    uv run python scripts/experiments/forced_openings.py                 # 8x8 and 12x12, 4 O moves
    uv run python scripts/experiments/forced_openings.py --sizes 8 --moves 5 \
        --out scripts/experiments/results/forced_openings.8x8-moves5.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from ox_zero.engine.encoding import NUM_SYMMETRIES, transform_cell
from ox_zero.game import apply_move, initial_state, legal_moves
from ox_zero.game.rules import State

Cell = tuple[int, int]

DEFAULT_OUT = Path(__file__).parent / "results" / "forced_openings.json"

# Caches shared by every search in the process. `State` is a frozen dataclass,
# so it hashes by value. `main` clears them between board sizes.
_ATTACKER: dict[tuple[State, int, int | None], bool] = {}
_DEFENDER: dict[tuple[State, int, int | None], bool] = {}
_WINS: dict[State, list[Cell]] = {}


def clear_caches() -> None:
    """Forget every memoised result (they are only valid per position anyway)."""
    _ATTACKER.clear()
    _DEFENDER.clear()
    _WINS.clear()


def cache_size() -> int:
    """Entries across all caches."""
    return len(_ATTACKER) + len(_DEFENDER) + len(_WINS)


def _near(state: State, radius: int | None) -> list[Cell]:
    """Empty cells within `radius` (Chebyshev distance) of a mark, board order.

    `radius=None` means every empty cell. On the empty board there is no mark
    to be near, so every cell is returned.
    """
    if radius is None or state.move_count == 0:
        return legal_moves(state)
    n = state.size
    found: set[Cell] = set()
    for i, mark in enumerate(state.board):
        if mark is None:
            continue
        row, col = divmod(i, n)
        for r in range(max(0, row - radius), min(n, row + radius + 1)):
            for c in range(max(0, col - radius), min(n, col + radius + 1)):
                if state.board[r * n + c] is None:
                    found.add((r, c))
    return sorted(found)


def winning_cells(state: State) -> list[Cell]:
    """Cells where the side to move completes an alternating line at once.

    A completing cell is always next to (or one gap from) existing marks, so
    looking within distance 2 of the marks finds all of them.
    """
    if state.winner is not None:
        return []
    if state not in _WINS:
        _WINS[state] = [m for m in _near(state, 2) if apply_move(state, m).winner is not None]
    return _WINS[state]


def _pass(state: State) -> State:
    """The same board with the other side to move: "what would they win at?"."""
    return dataclasses.replace(state, to_move=state.to_move.opponent)


def forces_win(state: State, moves: int, radius: int | None = 2) -> bool:
    """True if the side to move forces a win within `moves` of its own moves."""
    key = (state, moves, radius)
    if key in _ATTACKER:
        return _ATTACKER[key]
    if winning_cells(state):
        result = True
    elif moves <= 1:
        result = False
    else:
        # The same forcing rule as for the defender: if the opponent already
        # threatens a win, the attacker must block it (one threat) or is lost
        # (two or more), so only the block is worth trying.
        candidates = _defender_replies(state)
        if candidates is None:
            result = False
        else:
            near = set(_near(state, radius))
            forced = len(winning_cells(_pass(state))) == 1
            result = any(
                _defender_lost(apply_move(state, m), moves - 1, radius)
                for m in candidates
                if forced or m in near
            )
    _ATTACKER[key] = result
    return result


def _defender_replies(state: State) -> list[Cell] | None:
    """The defender's moves worth trying, or None if it is already lost.

    One forced block when the attacker has one winning cell, every empty cell
    when it has none (empty only on a full board).
    """
    threats = winning_cells(_pass(state))
    if len(threats) >= 2:
        return None  # a double threat: one move cannot block both
    if threats:
        return threats  # the one forced block
    return legal_moves(state)


def _defender_lost(state: State, moves: int, radius: int | None) -> bool:
    """Defender to move; the attacker then has `moves` moves left."""
    key = (state, moves, radius)
    if key in _DEFENDER:
        return _DEFENDER[key]
    if winning_cells(state):
        result = False  # the defender wins first
    else:
        replies = _defender_replies(state)
        if replies is None:
            result = True
        elif not replies:
            result = False  # full board: a draw is not a win for the attacker
        else:
            result = all(forces_win(apply_move(state, r), moves, radius) for r in replies)
    _DEFENDER[key] = result
    return result


def distance(state: State, moves: int, radius: int | None = 2) -> int | None:
    """The fewest own moves in which the side to move forces a win (at most
    `moves`), or None if it cannot within `moves`."""
    return next((k for k in range(1, moves + 1) if forces_win(state, k, radius)), None)


def witness(state: State, moves: int, radius: int | None = 2) -> list[Cell] | None:
    """A principal line of the winning strategy, ending in the winning move.

    Minimax over distance to the win: the attacker always plays a move that
    wins in the fewest moves, and the defender, where it has a choice, the
    reply that holds out longest (ties go to the first in board order). Every
    other reply loses too; this line is the most resistant one, so it is the
    one worth reading. None if there is no forced win within `moves`.
    """
    k = distance(state, moves, radius)
    if k is None:
        return None
    line: list[Cell] = []
    while True:
        wins = winning_cells(state)
        if wins:
            return line + [wins[0]]
        # An attacker move that keeps the win within k: after it, the defender
        # is lost with k - 1 attacker moves to come. `distance` makes k minimal,
        # so this is a fastest move.
        move = next(
            m for m in legal_moves(state) if _defender_lost(apply_move(state, m), k - 1, radius)
        )
        state = apply_move(state, move)
        replies = _defender_replies(state)
        if replies is None:  # double threat: block one, the other wins
            replies = winning_cells(_pass(state))[:1]
        # The defender's most resistant reply: the one after which the attacker
        # needs the most moves (at most k - 1, since every reply loses).
        reply = max(replies, key=lambda r: distance(apply_move(state, r), k - 1, radius) or 0)
        line += [move, reply]
        state = apply_move(state, reply)
        k = distance(state, k - 1, radius)


# --- symmetry ----------------------------------------------------------------------


def symmetric_images(cell: Cell, size: int) -> set[Cell]:
    """The cell's images under the board's eight symmetries (the dihedral group
    D4: the identity, three rotations, four reflections), from the same
    `transform_cell` the network's augmentation uses."""
    return {transform_cell(cell, k, size) for k in range(NUM_SYMMETRIES)}


def representatives(size: int) -> list[Cell]:
    """One cell per symmetry orbit: the smallest in board order."""
    return [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) == min(symmetric_images((r, c), size))
    ]


def opening_map(size: int, moves: int, radius: int | None = 2) -> dict[Cell, bool]:
    """For every first move of X: does O force a win within `moves` O moves?"""
    grid: dict[Cell, bool] = {}
    for rep in representatives(size):
        lost = forces_win(apply_move(initial_state(size), rep), moves, radius)
        for image in symmetric_images(rep, size):
            grid[image] = lost
    return grid


def _format(cell: Cell) -> str:
    return f"{cell[0]},{cell[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--moves", type=int, default=4, help="O's moves to win within")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    parameters = {"sizes": args.sizes, "moves": args.moves, "radius": 2}
    print(f"Parameters: {json.dumps(parameters)}\n")

    results, timings = [], []
    for size in args.sizes:
        clear_caches()  # positions of one size never recur in another
        start = time.perf_counter()
        grid = opening_map(size, args.moves)
        seconds = round(time.perf_counter() - start, 1)
        lost = sorted(cell for cell, value in grid.items() if value)
        lines = {
            _format(rep): [_format(m) for m in witness(apply_move(initial_state(size), rep), args.moves)]
            for rep in representatives(size)
            if grid[rep]
        }
        results.append({"size": size, "lost": [_format(c) for c in lost], "witness": lines})
        timings.append({"size": size, "seconds": seconds})

        print(f"## {size}x{size}: {len(lost)} of {size * size} openings lose within {args.moves} O moves\n")
        print("```")
        for r in range(size):
            print(" ".join("L" if grid[(r, c)] else "." for c in range(size)))
        print("```\n")
        for rep, line in lines.items():
            print(f"- X {rep}: " + " ".join(line))
        print()

    print("## Wall time (machine-dependent)\n")
    for t in timings:
        print(f"- {t['size']}x{t['size']}: {t['seconds']} s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"parameters": parameters, "results": results}, indent=2) + "\n")
    timing_out = args.out.with_suffix(".timing.json")
    timing_out.write_text(json.dumps({"parameters": parameters, "timing": timings}, indent=2) + "\n")
    print(f"\nWrote {args.out} and {timing_out}")


if __name__ == "__main__":
    main()
