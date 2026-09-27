# Plan 01: Game experiments and the exact solver

| | |
|---|---|
| Status | Ready for implementation |
| Branches | Plan: `plan/game-experiments`. Implementation: `feat/game-experiments`. |
| Design references | [open-questions.md](../design/open-questions.md) (what to measure and why), [engineering.md](../design/engineering.md#testing) (the solver as a test fixture), [search.md](../design/search.md#constants) (the constants waiting on the data) |
| Depends on | Nothing. Only `ox_zero.game`, which is done. |

## Goal

Produce the measurements that ten design constants are waiting on, and build the one piece of ground truth the project will ever have: an exact minimax solver for small boards. The solver is real package code under test and stays; the experiment scripts are reproducible scratch work.

## Deliverables

1. `src/ox_zero/game/solver.py`: an exact negamax solver over `State`, with tests in `tests/game/test_solver.py`.
2. `scripts/experiments/random_play.py`: game statistics under random and greedy play for several board sizes, printed as Markdown and saved as JSON.
3. `scripts/experiments/solve_boards.py`: solves the empty board for small sizes and reports value, best first moves, positions searched, and time.
4. `scripts/experiments/results/`: the JSON outputs, committed.
5. A **Results** section appended to `docs/design/open-questions.md` with the measured numbers, and nothing else in `docs/design/` changed.

## Out of scope

- Any decision about the design constants. The plan records numbers; the design is updated afterwards in a separate discussion. Do not edit `search.md`, `network.md`, or `training.md`.
- Any engine, network, or training code.
- Symmetry handling in the `game` package, unless step 3 forces it (see there).
- The CLI.

## Conventions that apply

- Test-driven: for the solver, write the failing tests first, run them (`uv run pytest tests/game/test_solver.py`), then implement until green. Scripts under `scripts/` are exempt from TDD but must be deterministic (seeded) and print the parameters they ran with.
- Comment thoroughly, connecting code to theory (negamax, memoisation, transpositions, why the value is from the side to move's perspective). See the project `CLAUDE.md`.
- Coordinates are `(row, col)`, zero-based, row 0 at the top. Board order is left to right, top to bottom.
- Conventional commits, small and atomic, on `feat/game-experiments`. Suggested sequence: `test(game): add solver tests`, `feat(game): add exact minimax solver`, `feat(scripts): add random-play experiment`, `feat(scripts): add board solving experiment`, `docs(design): record game experiment results`.
- Python 3.14, run everything with `uv run ...`. Do not add dependencies; the standard library plus the existing ones (NumPy is available) are enough.

## Step 1: solver tests (`tests/game/test_solver.py`)

Write these first. They define the interface below and must fail with `ImportError` before step 2.

Interface under test:

```python
from ox_zero.game.solver import Solver

solver = Solver()
solver.value(state)          # -> int in {-1, 0, +1}: exact game value for the side to move
solver.best_moves(state)     # -> list[Cell]: every legal move that achieves value(state), in board order; [] if terminal
solver.positions_solved      # -> int: number of distinct positions whose value is cached
```

Value convention, identical to the search design: `+1` the side to move wins with perfect play, `0` draw, `-1` the side to move loses. At a terminal position the side to move can never have won (the previous mover completed the line), so terminal values are `-1` (someone won) or `0` (full board).

Tests to write, each a plain function using `initial_state`, `play`, and `apply_move` from `ox_zero.game`:

1. **Terminal values.** A position where the previous move completed `XOX` has value `-1`. A full board without a line (use a 1x1 board after one move, and a 2x2 full board, which cannot contain a line of three) has value `0`. `best_moves` is `[]` for both.
2. **Win in one.** On 3x3, `play([(0,0), (0,1)])` leaves `X O _` in row 0 with X to move. `value == +1`, and `best_moves == [(0,2)]`.
3. **Win in one, several winning moves.** Construct a position with two distinct completing cells; `best_moves` lists both, in board order.
4. **Forced loss.** Use this 3x3 position, O to move, reached by `play([(0,0), (1,1), (1,0), (0,2), (2,0), (1,2), (2,1)], size=3)`:

   ```
   X . O
   X O O
   X X .
   ```

   No line is complete yet. O's only moves are `(0,1)` and `(2,2)`, and neither completes a line for O. After O plays `(0,1)`, X plays `(2,2)` and completes the diagonal `X O X`. After O plays `(2,2)`, X plays `(0,1)` and completes column 1 as `X O X`. Assert `value == -1` and `best_moves == [(0,1), (2,2)]` (both moves lose; every legal move achieves the value, so all are listed). Verify the move sequence with the rules before relying on it; if `play` raises, the sequence is wrong, not the rules.
5. **Negamax consistency (property).** For 200 random reachable positions on 3x3 and 4x4 (seeded random play, stop at a random depth), assert `value(s) == max(-value(apply_move(s, a)) for a in legal_moves(s))` for non-terminal `s`, and that `best_moves(s)` is exactly the set of `a` achieving that maximum, in board order.
6. **Cache.** After `value(initial_state(3))`, `positions_solved > 0`; calling it again does not change `positions_solved`; a second `Solver()` starts at `0`.
7. **Board-size independence.** `value(initial_state(1)) == 0` (one move, full board, no line). `value(initial_state(2)) == 0` (no line of three fits).
8. **Speed guard.** `value(initial_state(3))` completes in well under a second; `value(initial_state(4))` completes within the pytest run without a timeout marker. If 4x4 turns out to take more than ~30 s, mark it `@pytest.mark.slow` and register the marker in `pyproject.toml`; do not drop the test.

Do **not** assert the actual game value of the empty 3x3 or 4x4 board in the tests yet. It is unknown and is one of the experiment outputs. Once measured, the value goes into the results section and a test pins it (step 5).

## Step 2: solver implementation (`src/ox_zero/game/solver.py`)

Negamax with a transposition table keyed by `State` (frozen, hashable; the module docstring of `rules.py` promises exactly this use):

```
value(s):
    if s in cache: return cache[s]
    if is_terminal(s): v = -1 if s.winner is not None else 0
    else:
        v = -1
        for a in legal_moves(s):
            v = max(v, -value(apply_move(s, a)))
            if v == +1: break          # cannot do better than a win; exact, not a bound
    cache[s] = v
    return v
```

- The `+1` short-circuit is safe because the stored value is exact; do **not** add alpha-beta windows, which would store bounds and break `best_moves` and the consistency test.
- Recursion depth is at most `S²` (16 for 4x4, 25 for 5x5), far below Python's limit.
- `best_moves(s)` calls `value` on every child and filters; it must not use the short-circuit.
- Explain in comments: what negamax is (`value(s) = max_a -value(child)` because the players alternate and the game is zero-sum), why memoisation matters (many move orders reach the same position: transpositions), and why the value is always from the side to move's perspective (same convention as the network's `v` and MCTS backups, see `docs/design/search.md`).

Export `Solver` from `ox_zero.game.__init__` alongside the existing names.

## Step 3: performance check, and canonicalisation only if needed

Measure `Solver().value(initial_state(4))` wall time and `positions_solved`. Target: under 2 minutes in pure Python.

If it is slower, add board-symmetry canonicalisation to the cache key: the 8 dihedral symmetries of the square (4 rotations × optional reflection) map a position to an equivalent one, so the cache key becomes the lexicographically smallest of the 8 transformed boards. Put this in `src/ox_zero/game/symmetry.py` with tests (`tests/game/test_symmetry.py`: each transform is a bijection on cells, applying a transform and its inverse is the identity, the 8 transforms of a board are all distinct for an asymmetric board, `canonical(s) == canonical(transform(s, k))` for all `k`). The mark swap is *not* needed as a key reduction here: a swapped board has the same value but is unreachable with the same `to_move`, so it never shows up.

If 4x4 is fast enough without it, do **not** build `symmetry.py` in this plan. Note the decision and the timing in the results section.

## Step 4: experiment scripts

### `scripts/experiments/random_play.py`

Command line (argparse): `--sizes 3 4 5 6 8 12`, `--games 2000`, `--seed 0`, `--policy random|greedy|both` (default `both`), `--out scripts/experiments/results/random_play.json`.

Policies:
- **random**: uniformly random legal move.
- **greedy**: if a move completes an alternating line, play it (win). Otherwise, among moves that do *not* leave the opponent an immediate win, pick uniformly at random. If every move loses immediately, pick uniformly at random. This is a one-ply lookahead and is meant only as a second, less silly, data point; label it as such in the output.

Per size and policy, collect over `--games` games:
- game length: mean, median, min, max, 10th and 90th percentiles;
- outcome rates: X wins, O wins, draws (fractions);
- **average legal moves per decision**: mean of `len(legal_moves(s))` over every non-terminal position where a move was chosen, across all games. This is the denominator for `α = 10 / avg` in `search.md`. Also report the same average restricted to the first 25% of each game and to the last 25%, to show how much it moves;
- positions per second (total positions generated / wall time), as a first data point for the performance plan.

Output: a Markdown table per policy on stdout (sizes as rows), and a JSON file with every number plus the parameters and the seed. Use the standard library `random` seeded once per (size, policy) pair from `--seed`, so results are reproducible run to run.

12x12 random games are short compared to 144 moves (random marks form alternating lines quickly), but if `--games 2000` on 12x12 takes more than a few minutes, lower the default for large sizes and record the actual count.

### `scripts/experiments/solve_boards.py`

Command line: `--sizes 3 4`, `--out scripts/experiments/results/solve_boards.json`.

For each size: solve the empty board, then report value (as `X wins` / `draw` / `O wins`, since X moves first), `best_moves` for X's first move, `positions_solved`, and wall time. Also report, for the solved sizes, the **length of the game under optimal play** where both sides pick the first best move in board order (play it out with the solver and count moves); this is a rough indicator of how long trained-agent games might be, as opposed to random ones.

Attempt 5x5 **once**, manually, only if 4x4 solved in under a few seconds. Run it with a 10 minute wall-clock limit (`timeout 600 uv run ...` or equivalent) and record whether it finished. Do not build a timeout into the solver.

## Step 5: record results

Append a `## Results (2026-MM-DD)` section to `docs/design/open-questions.md` with:

- the random-play table and the greedy-play table (sizes as rows; columns: mean length, median, P10, P90, X win %, O win %, draw %, avg legal moves overall / first quarter / last quarter);
- the solver table (size, value, best first moves, positions solved, time, optimal-play length), including the 5x5 attempt outcome;
- the 4x4 solver timing and whether canonicalisation was needed;
- positions per second of the pure-Python rules;
- a **"What this suggests"** list of at most six bullets that maps numbers to the open constants (e.g. "12x12 random-play average of N legal moves gives α ≈ 10/N"), phrased as observations, not decisions.

Then pin the measured empty-board values in `tests/game/test_solver.py` as regression tests (`test_empty_3x3_value`, `test_empty_4x4_value`), so a rules change that alters the game is caught.

## Definition of done

- `uv run pytest` is green, including the new solver tests.
- `uv run python scripts/experiments/random_play.py` and `solve_boards.py` run from a clean checkout and regenerate the committed JSON byte-for-byte with the default seed.
- `docs/design/open-questions.md` has the results section; no other design file changed.
- Placeholder `README.md` files in `src/ox_zero/engine/` and `src/ox_zero/training/` are untouched (they are for later plans).
- A pull request from `feat/game-experiments` to `main` summarising the numbers, with the "What this suggests" bullets in the description. On merge, this plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- If the solver's 4x4 run is too slow even with canonicalisation, do not go further (no alpha-beta, no C extension); report the timing.
- If the rules package turns out to have a bug (the solver or the consistency test exposes one), fix nothing in `rules.py` without raising it first: it changes the game.
- If any result looks absurd (e.g. 12x12 random games averaging 100 moves, or 3x3 solving to a first-player loss in two moves), double-check the script before writing it down, then report it as is.
