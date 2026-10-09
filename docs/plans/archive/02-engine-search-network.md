# Plan 02: Engine: search and network

| | |
|---|---|
| Status | Implemented and merged (PR #12, 2026-10-09) |
| Branches | Plan: `plan/engine-search-network`. Implementation: `feat/engine-search-network`. |
| Design references | [search.md](../../design/search.md) (every search rule and constant), [network.md](../../design/network.md) (encoding, tower, heads, loss, symmetries), [engineering.md](../../design/engineering.md) (module layout, evaluator seam, tree representation, testing), [cli-integration.md](../../design/cli-integration.md) (what the analysis generator must yield, read but not implemented here) |
| Depends on | [Plan 01](01-game-experiments.md): `ox_zero.game` including `Solver`. |

## Goal

Build roadmap step 3: the `ox_zero.engine` package. A PUCT tree search that is correct against the exact solver, a residual policy/value network with the AlphaZero loss, and the evaluator seam between them. The search must already have the shape self-play needs (many trees stepped between one batched network call) and the shape the CLI needs (a generator of snapshots), so that plans 03 (CLI adapter) and 04 (training) add code around it without reopening it.

Nothing gets trained and nothing is connected to the CLI. The proof that this plan worked is the test suite plus a benchmark run, not a game.

## Deliverables

1. `src/ox_zero/engine/encoding.py`: `State` to input planes, legal-move masks, the 8 board symmetries on planes, policies and cells.
2. `src/ox_zero/engine/evaluator.py`: the `Evaluator` protocol, `UniformEvaluator`, `TableEvaluator` (scripted, for tests), `NetworkEvaluator`, and `select_device`.
3. `src/ox_zero/engine/mcts.py`: `Node`, PUCT selection, expansion, backup, terminal values, root noise, visit distribution, move sampling, subtree reuse.
4. `src/ox_zero/engine/search.py`: `SearchConfig`, `SearchTree` (the two-phase select / expand-and-backup API), `analyse()` (the snapshot generator), `Snapshot`.
5. `src/ox_zero/engine/network.py`: `NetworkConfig`, `Network`, `alphazero_loss`.
6. `tests/engine/` mirroring the modules, including the solver-fixture tests listed in step 5.
7. `scripts/bench_search.py`: simulations per second for a board size, evaluator and device.
8. Documentation updates listed in step 9, and the placeholder `src/ox_zero/engine/README.md` deleted.

## Out of scope

- The CLI: no adapter, no change to `cli/engine_port.py`, no `--model` loading. `PlaceholderEngine` stays. That is plan 03.
- Checkpoints, self-play driver, replay buffer, trainer, tournaments, `scripts/train.py`. That is plan 04. The per-tree self-play primitives (noise, temperature, subtree reuse, the two-phase API) *are* in scope because they live in the tree.
- Every item in [upgrades.md](../../design/upgrades.md): no MCTS-Solver, no first-play urgency, no evaluation cache, no virtual loss, no global pooling, no symmetry at inference. The search is the paper's baseline on purpose.
- Performance work beyond the benchmark script. Measure, do not optimise. No `torch.compile`, no array tree.
- The `training` and `gui` packages.

## Decisions made in this plan

Code-level choices the design left to the plan, with the reason. Vláďa approved the scope on 2026-09-27; anything here that turns out wrong in review gets changed here, not in code.

| Decision | Choice | Why |
|---|---|---|
| Evaluator return types | NumPy `float32` arrays: policies `[B, S²]`, values `[B]`. | The search stays torch-free, as [engineering.md](../../design/engineering.md#the-evaluator-seam) wants. `NetworkEvaluator` is the only module that converts. |
| Encoding output type | NumPy `float32` `[3, S, S]`; the network evaluator stacks and converts. | Same reason. `network.md` says "tensor" loosely; NumPy is the torch-free reading. |
| Where the loss lives | `engine/network.py`, not `training/trainer.py`. | The "memorise one batch" test is the main proof the network works, and it needs the loss. Approved 2026-09-27; step 9 updates the layout table in `engineering.md`. |
| `N(s)` in PUCT | The parent node's own `visit_count`, incremented on every node of the backed-up path, as in the paper's pseudocode. | Hand-computable and matches the reference. It equals `Σ_a N(s,a)` at the root and `Σ_a N(s,a) + 1` at expanded non-root nodes (the extra one is the visit that expanded the node). `search.md` writes `N(s) = Σ_a N(s,a)`; step 9 adds the clarification there. |
| Root value | `Σ_a W(root,a) / Σ_a N(root,a)` over the root's children, `0.0` with no visits. | The root node's own `value_sum` is accumulated from the perspective of the player who moved *into* the root, i.e. the opponent of the side to move, so `root.q` has the wrong sign. Compute over children and say so in a comment. |
| Root-expansion evaluations and `simulations` | Not counted. The first analysis snapshot has `simulations = 0`. | The design's own recommendation in [cli-integration.md](../../design/cli-integration.md#every-legal-move-gets-a-genuine-score). |
| Expanding the root | Setup, not a simulation: it expands without backing anything up and adds no visits. Root noise is applied right after. | The pseudocode's `evaluate(root)` then `add_exploration_noise(root)` before the loop. |
| Simulations per move after subtree reuse | `n` fresh simulations per move regardless of visits inherited from the reused subtree. | Simplest bookkeeping; the inherited counts are a warm start, not part of the budget. `SearchTree.simulations` resets to 0 on `play`. |
| Temperature cutoff for sizes not in the table | Table `{4: 2, 6: 3, 8: 4, 12: 7}` from `search.md`; other sizes fall back to `max(1, round(S / 2))`, which gives 2, 3, 4, 6 on the tabulated sizes. | Some fallback is needed for 3x3 and 5x5 tests. Nothing was measured for them; the fallback is labelled provisional in code. |
| Symmetry indexing | `k` in `0..7`: `k % 4` quarter-turn rotations after an optional left-right flip when `k >= 4`. `inverse(k) = k` for `k >= 4`, `(-k) % 4` otherwise. | Every reflection of the square is its own inverse, so the inverse is a one-liner and the test can check it. |
| Ties in selection | Board order: the first maximal child wins. | `Node.children` is filled from `legal_moves`, which is in board order, and `max` keeps the first maximum. Makes hand-computed tests deterministic. |
| Randomness | One `numpy.random.Generator` per `SearchTree`, passed in; `None` means `default_rng()`. Torch's RNG is used only inside the network. | [engineering.md](../../design/engineering.md#devices-and-determinism): every source of randomness is seedable and later checkpointable. |

## Conventions that apply

- Test-driven: for each step write the failing tests first, run `uv run pytest tests/engine -q`, then implement until green. Tests run on CPU only and never select `mps` or `cuda`.
- Comment thoroughly, connecting code to theory: PUCT and the `C(s)` formula, why `Q = 0` for unvisited children means "assume a draw", why the sign flips on backup, why terminal leaves skip the network, Dirichlet noise as forced exploration, why π uses `τ = 1`, what a residual block is, why BatchNorm is frozen at inference, why illegal logits are `-∞` and the `0 · -∞ = NaN` trap in the loss, why the cross-entropy has a floor at the entropy of the target. See the project `CLAUDE.md`.
- Coordinates are `(row, col)`, zero-based, row 0 at the top. Policies are flat vectors of length `S²` in board order, index `row * S + col`, the same order as `State.board`.
- Values are in `[-1, 1]` from the perspective of the side to move at the position they describe. The `[0, 1]` mapping is the CLI adapter's job (plan 03), not the engine's.
- Conventional commits, small and atomic, on `feat/engine-search-network`. Suggested sequence: `test(engine): add encoding tests` / `feat(engine): add position encoding and symmetries`, then the same pair for evaluator, mcts, search, network, network evaluator; `feat(scripts): add search benchmark`; `docs: record the engine as built`.
- No new dependencies. NumPy and torch are already in `pyproject.toml`.
- Delete `src/ox_zero/engine/README.md` in the first implementation commit; its module docstrings replace it. Leave `src/ox_zero/training/README.md` alone.

## Interfaces

The contract the tests pin. Names and shapes are fixed; internals are free.

```python
# engine/encoding.py
NUM_PLANES = 3
NUM_SYMMETRIES = 8
def encode(state: State) -> np.ndarray                          # float32 [3, S, S]: side to move, other side, ones
def encode_batch(states: Sequence[State]) -> np.ndarray         # float32 [B, 3, S, S]
def legal_mask(state: State) -> np.ndarray                      # bool [S²]; all False when terminal
def transform_planes(planes: np.ndarray, k: int) -> np.ndarray  # acts on the last two axes; returns a contiguous copy
def transform_policy(policy: np.ndarray, k: int) -> np.ndarray  # acts on the last axis of length S²
def transform_cell(cell: Cell, k: int, size: int) -> Cell
def inverse_symmetry(k: int) -> int

# engine/evaluator.py
type Policies = np.ndarray   # float32 [B, S²]: zero on illegal cells, legal cells sum to 1
type Values = np.ndarray     # float32 [B]: in [-1, 1], side to move's perspective
class Evaluator(Protocol):
    def evaluate(self, states: Sequence[State]) -> tuple[Policies, Values]: ...
class UniformEvaluator: ...                                     # equal priors over legal moves, value 0
class TableEvaluator:                                           # exact scripted answers for tests
    def __init__(self, table: Mapping[State, tuple[np.ndarray, float]], fallback: Evaluator | None = None): ...
class NetworkEvaluator:
    def __init__(self, network: Network, device: torch.device | None = None): ...
def select_device(preference: str | None = None) -> torch.device  # cuda → mps → cpu, or the named one

# engine/mcts.py
@dataclass(slots=True)
class Node:
    state: State
    prior: float                    # P(s,a) of the edge into this node
    visit_count: int = 0            # N(s,a)
    value_sum: float = 0.0          # W(s,a), for the player who played a
    children: dict[Cell, Node] | None = None   # None until expanded; terminal nodes stay None
    @property
    def q(self) -> float: ...       # W / N, 0.0 when N == 0
    @property
    def expanded(self) -> bool: ...
def terminal_value(state: State) -> float                       # -1.0 if someone won, 0.0 if full board
def puct_score(parent_visits: int, child: Node, c_base: float, c_init: float) -> float
def select_child(node: Node, c_base: float, c_init: float) -> tuple[Cell, Node]
def select_leaf(root: Node, c_base: float, c_init: float) -> list[Node]   # path from root to an unexpanded or terminal node
def expand(node: Node, policy: np.ndarray) -> None              # children for legal moves, priors renormalised over them
def backup(path: Sequence[Node], leaf_value: float) -> None     # sign flips per search.md, increments every node on the path
def root_value(root: Node) -> float                             # Σ W / Σ N over children
def add_dirichlet_noise(root: Node, rng: np.random.Generator, epsilon: float, alpha: float) -> None
def visit_distribution(root: Node, size: int) -> np.ndarray     # float32 [S²], the τ = 1 policy target π
def most_visited(root: Node) -> Cell                            # ties in board order
def sample_move(root: Node, rng: np.random.Generator) -> Cell   # proportional to visits (τ = 1)
def reuse_subtree(root: Node, move: Cell) -> Node               # the child becomes the new root

# engine/search.py
@dataclass(frozen=True)
class SearchConfig:
    c_base: float = 19652.0
    c_init: float = 1.25
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float | None = None     # None: 11 / S²
    temperature_cutoff: int | None = None    # None: table, then S/2 fallback
    root_noise: bool = False                 # self-play: True
    expand_root: bool = False                # analysis: True (all root children evaluated up front)
    def alpha(self, size: int) -> float: ...
    def cutoff(self, size: int) -> int: ...
ANALYSIS = SearchConfig(expand_root=True)
SELF_PLAY = SearchConfig(root_noise=True)

class SearchTree:
    def __init__(self, state: State, config: SearchConfig, rng: np.random.Generator | None = None): ...
    root: Node
    simulations: int                          # completed since the last play(); root expansion not counted
    def select(self) -> State | None
        # Descend by PUCT. Returns the leaf state that needs a network evaluation and remembers the path.
        # On a fresh tree the first call returns the root's own state (root expansion, see Decisions).
        # If the leaf is terminal, backs up the exact value itself, counts the simulation, and returns None.
        # Raises if called while a leaf is pending.
    def expand_and_backup(self, policy: np.ndarray, value: float) -> None
        # Expands the pending leaf with the given prior and backs `value` up along the remembered path.
        # For the root: expands, applies noise if configured, backs up nothing, counts nothing.
    def expand_root_children(self, evaluator: Evaluator) -> None
        # Analysis mode: one batched evaluation of every root child, one visit each with W = -v_child.
    def simulate(self, evaluator: Evaluator, n: int = 1) -> None   # convenience: n full simulations, batch size 1
    def policy_target(self) -> np.ndarray                          # visit_distribution(root)
    def choose_move(self, move_index: int) -> Cell                 # sample below the cutoff, most visited from it on
    def play(self, move: Cell) -> None                             # reuse the subtree, re-apply noise, reset simulations

@dataclass(frozen=True)
class Snapshot:
    value: float                 # root value, [-1, 1], side to move
    q: Mapping[Cell, float]      # every legal move in board order: Q(root, a)
    visits: Mapping[Cell, int]   # N(root, a)
    best: Cell                   # most visited
    simulations: int

def analyse(state: State, evaluator: Evaluator, config: SearchConfig = ANALYSIS,
            max_simulations: int | None = None) -> Iterator[Snapshot]
    # Raises ValueError eagerly (before the first next()) if the game is over, like PlaceholderEngine does.
    # Yields after root setup with simulations = 0, then after every simulation. Ends when simulations == max.

# engine/network.py
@dataclass(frozen=True)
class NetworkConfig:
    blocks: int = 4
    filters: int = 64
    value_hidden: int = 256
class Network(nn.Module):
    size: int
    config: NetworkConfig
    def __init__(self, size: int, config: NetworkConfig = NetworkConfig()): ...
    def forward(self, planes: Tensor) -> tuple[Tensor, Tensor]   # [B,3,S,S] → logits [B, S²], value [B] after tanh
def alphazero_loss(logits: Tensor, values: Tensor, pi: Tensor, z: Tensor, legal: Tensor) -> tuple[Tensor, Tensor, Tensor]
    # (total, policy_loss, value_loss), each a scalar mean over the batch. Weight decay is the optimiser's job.
```

The tree keeps one pending leaf at a time (no virtual loss). In lockstep self-play (plan 04) the driver calls `select()` on every tree, evaluates the returned states in one `evaluate()` call, then `expand_and_backup()` on each; trees that returned `None` are skipped that round. `analyse()` is the same loop with one tree and batch size 1.

## Step 1: encoding (`tests/engine/test_encoding.py`, then `encoding.py`)

Tests:

1. **Plane contents.** On a hand-built 4x4 position with X to move, plane 0 is 1 exactly on X's cells, plane 1 on O's, plane 2 all ones. Swap the side to move (add one mark) and check planes 0 and 1 swap roles: the encoding is relative to the side to move, not to X.
2. **Shape and dtype** for sizes 3, 4, 6, 12: `[3, S, S]`, `float32`. `encode_batch` stacks to `[B, 3, S, S]`.
3. **Legal mask** matches `legal_moves` for a mid-game position, and is all `False` on a terminal one.
4. **Symmetries are bijections.** For an asymmetric 4x4 board, the 8 transformed plane stacks are pairwise distinct. `transform_planes(transform_planes(a, k), inverse_symmetry(k)) == a` for every `k`.
5. **Planes, policies and cells agree.** For every `k` and every cell `c`: `transform_policy(onehot(c), k)` is one-hot at `transform_cell(c, k, S)`. And for a random move list, `encode(play(transformed moves)) == transform_planes(encode(play(moves)), k)`.
6. **Contiguity.** The result of `transform_planes` is C-contiguous (`torch.from_numpy` rejects negative strides, which `np.flip` and `np.rot90` produce as views).

Implementation notes: `transform_planes` is `np.flip` on the last axis when `k >= 4`, then `np.rot90` with `k % 4` turns on the last two axes, then `np.ascontiguousarray`. Derive `transform_cell` from the array operation (transform an index grid `np.arange(S²).reshape(S, S)` and look the cell up) rather than writing coordinate formulas by hand, so the two can never disagree. `transform_policy` reshapes the last axis to `[S, S]`, reuses `transform_planes`, and flattens back: a policy is a scalar field over the board, so the same spatial map applies. Explain the D4 group and the mark-swap symmetry that the relative encoding already absorbs ([network.md](../../design/network.md#symmetries)).

## Step 2: uniform and table evaluators (`tests/engine/test_evaluator.py`, then the torch-free half of `evaluator.py`)

Tests:

1. `UniformEvaluator` on a batch of two positions with different legal-move counts: policies have zeros exactly on illegal cells, legal cells are equal and sum to 1 (within `1e-6`), values are all 0, shapes and dtypes as specified.
2. `TableEvaluator` returns the scripted policy and value for a known state and falls back (uniform by default) for an unknown one. It must stack per-state results into arrays of the right shape.

Import torch nowhere in this half; the `NetworkEvaluator` class is added in step 7 and may import torch at module level (the search modules import only the protocol and the two simple evaluators, so keep `NetworkEvaluator` in the same file only if `import torch` stays cheap; otherwise split it into `network_evaluator.py` and say so in the layout table).

## Step 3: tree primitives (`tests/engine/test_mcts.py`, then `mcts.py`)

Tests, each on hand-built `Node` trees or tiny boards with `UniformEvaluator` or `TableEvaluator` doing the evaluation by hand (call `expand` with the evaluator's policy directly; `SearchTree` does not exist yet):

1. **`puct_score` by hand.** With `parent_visits = 100`, `N(s,a) = 10`, `P = 0.5`, `Q = 0.2`, `c_base = 19652`, `c_init = 1.25`: `C = log((1 + 100 + 19652) / 19652) + 1.25`, `U = C · 0.5 · √100 / 11`, score `= 0.2 + U`. Write the arithmetic in the test and assert to `1e-9`. Also: an unvisited child has `Q = 0`, and with `parent_visits = 0` every child scores exactly 0 (so the first selection is by board order).
2. **Selection follows priors when values are equal.** A 2x2 board (no line fits, so every value is 0) with a `TableEvaluator` giving the root priors `(0.7, 0.2, 0.06, 0.04)`: since `Q = 0` everywhere and `C · √N(s)` is common to all children, each selection maximises `P / (1 + N)`. Run 10 selections with `select_child` and hand-derive the visit sequence in comments (`a, a, a, b, a, a, b, a, c, a` or whatever the arithmetic gives; compute it, do not guess). Assert the final counts.
3. **Selection prefers value over prior.** Same tree, but a scripted evaluation gives one low-prior child `Q = +0.8` after its first visit; within a few more selections it is chosen despite the lower prior. Assert it has the most visits after 20 selections.
4. **Backup signs.** A hand-built path of three nodes (root, child, grandchild). `backup(path, +1.0)`: the grandchild (leaf) gets `W = -1`, the child `W = +1`, the root `W = -1`; every `N` is 1. Explain in the test why the leaf's edge receives the negated value ([search.md](../../design/search.md#3-backup)).
5. **Terminal values.** `terminal_value` is `-1.0` after a completed line and `0.0` on a full 2x2 board; raises (or asserts) on a non-terminal state.
6. **Expansion.** `expand` on a position with 5 legal moves and a policy that has mass on an illegal cell: children exist for exactly the legal moves, in board order, priors renormalised over legal moves only and summing to 1. Expanding a terminal node raises.
7. **Root value.** Hand-built root with two visited children: `root_value` equals `(W_a + W_b) / (N_a + N_b)`, and equals `-root.q` when the root itself was backed up through the same path (documents the sign trap). Empty root gives `0.0`.
8. **Dirichlet noise.** With `rng = default_rng(0)`, priors still sum to 1, each prior changed unless `epsilon = 0`, children of children untouched, and the same seed gives the same priors twice.
9. **Visit distribution and move choice.** `visit_distribution` is `float32 [S²]`, zero on illegal cells, `N(root,a) / Σ N` elsewhere. `most_visited` breaks ties in board order. `sample_move` with a fixed seed over 1000 draws lands within a few percent of the visit fractions (or simply assert it never returns a child with `N = 0`; pick one and say why).
10. **Subtree reuse.** After `reuse_subtree(root, a)`, the returned node is the old child with its `visit_count`, `value_sum` and children intact, and the old root is unreachable from it.

## Step 4: the search tree (`tests/engine/test_search.py`, then `SearchConfig` and `SearchTree` in `search.py`)

Tests:

1. **Config resolution.** `SearchConfig().alpha(6) == 11 / 36`; `cutoff(4) == 2`, `cutoff(12) == 7`, `cutoff(5) == 2` (fallback). Explicit values override.
2. **Two-phase protocol.** On a fresh tree, `select()` returns the root state; after `expand_and_backup`, `simulations == 0` and the root is expanded with no visits. The next `select()` returns a child state (an unexpanded node). Calling `select()` twice in a row raises. `expand_and_backup` without a pending leaf raises.
3. **Terminal leaves need no evaluation.** On the 3x3 win-in-1 position `play([(0,0), (0,1)], size=3)`, run `simulate(UniformEvaluator(), 30)`: `simulations == 30`, every simulation that reached the terminal child returned `None` from `select()` (count them via a wrapper or check that `visit_count` sums match), and the winning child `(0,2)` has `Q == 1.0` exactly, because every visit backs up `-terminal_value = +1`.
4. **Root noise only at the root, only when configured.** With `SELF_PLAY` and a seeded rng, root priors differ from the evaluator's; with `ANALYSIS` they are identical. Grandchildren priors are the evaluator's in both.
5. **Move choice and policy target.** After 50 simulations on a 4x4 position with `SELF_PLAY`: `policy_target()` sums to 1 and is zero on illegal cells; `choose_move(move_index=0)` returns a visited move; `choose_move(move_index=cutoff)` returns `most_visited(root)`.
6. **Reuse preserves counts and resets the budget.** Record `N` and `W` of the chosen child's children, `play(move)`, and check the new root's children have the same numbers, `simulations == 0`, and (with `SELF_PLAY`) the new root's priors were re-noised.
7. **Determinism.** Two trees with `default_rng(7)` and `SELF_PLAY` produce identical visit counts after 100 simulations with `UniformEvaluator`.

## Step 5: the analysis generator and the solver fixtures (`tests/engine/test_search.py` continued, then `analyse()` and `Snapshot`)

Generator tests:

1. **Eager error.** `analyse(terminal_state, ...)` raises `ValueError` without a `next()` call.
2. **Snapshot sequence.** With `max_simulations = 5`: six snapshots with `simulations` `0, 1, 2, 3, 4, 5`; every snapshot has a `q` and a `visits` entry for every legal move, in board order; `visits` sums to `simulations + number of legal moves` (root expansion gives one visit each); `best` is the most visited move. Without a cap, `itertools.islice(analyse(...), 20)` yields 20 snapshots and dropping the iterator needs no cleanup.
3. **Root expansion gives a genuine `Q` to every move from the first snapshot.** On the 3x3 win-in-1 position the first snapshot (`simulations = 0`) already has `q[(0,2)] == 1.0` and every other move's `q` from its own evaluation.

Solver-fixture tests. These use `UniformEvaluator` and a session-scoped `Solver()` from `tests/engine/conftest.py`. Budgets below were checked against a scratch PUCT following `search.md` exactly; the real implementation should match. If a fixture fails at the stated budget, doubling it once is fine; beyond that, stop and ask (see the questions at the end).

4. **Win in one.** `play([(0,0), (0,1)], size=3)`, X to move. After 100 simulations: `best == (0,2)` and `value > 0.85`.
5. **Forced loss in two.** The plan-01 position `play([(0,0), (1,1), (1,0), (0,2), (2,0), (1,2), (2,1)], size=3)`, O to move, both moves lose. After 200 simulations: `value < -0.9`. This is the value-sign test: shallow tactics do converge under averaging.
6. **Loss in one avoided.** `from_board_string("____XXO_________", size=4)`, O to move, 13 legal moves, exactly one (`(1,3)`) does not hand X an immediate win. After 200 simulations `best == (1,3)`.
7. **Deep forced wins: the search picks the unique optimal move.** Four reachable 4x4 positions, each O to move, solver value `+1`, exactly one optimal move out of 13, forced win 9 to 11 plies deep:

   | Moves from the empty 4x4 board | Board (rows top to bottom) | Only optimal move |
   |---|---|---|
   | `(1,3), (2,0), (3,0)` | `____ / ___X / O___ / X___` | `(1,0)` |
   | `(0,3), (1,3), (2,0)` | `___X / ___O / X___ / ____` | `(2,3)` |
   | `(1,1), (2,2), (0,0)` | `X___ / _X__ / __O_ / ____` | `(3,3)` |
   | `(0,0), (0,1), (2,3)` | `XO__ / ____ / ___X / ____` | `(0,2)` |

   The test first asserts the solver agrees (`value == 1`, `best_moves == [move]`) so a rules change cannot silently invalidate the fixture. Then after 1000 simulations `best == move`. In the scratch run the optimal move took about 90% of the visits at every budget from 200 up. Do **not** assert the value sign here: the root value stays near 0 on wins this deep because plain averaging never propagates a proof ([upgrades.md](../../design/upgrades.md#search), MCTS-Solver). Write that down in the test; it is the baseline that upgrade will be measured against.
8. **The empty 3x3.** Eight moves draw and the centre loses ([open-questions.md](../../design/open-questions.md#exact-solutions)). After 2000 simulations `best != (1,1)` and `visits[(1,1)]` is the smallest.
9. **Random 3x3 agreement (property).** Sample 30 reachable non-terminal 3x3 positions with `random.Random(0)` whose optimal moves are a strict subset of at least 3 legal moves. After 500 simulations `best` is in `solver.best_moves`. Scratch result: 0 failures in 60. If the real search fails more than one, report rather than tune.

Keep the whole engine suite under about 30 seconds on CPU. The deep fixtures cost roughly 0.1 s each in the scratch run.

## Step 6: the network and the loss (`tests/engine/test_network.py`, then `network.py`)

Use a tiny config in tests (`NetworkConfig(blocks=2, filters=8, value_hidden=16)`) unless the test is about defaults.

Tests:

1. **Shapes.** For sizes 3, 6, 12 and batch 5: logits `[B, S²]`, values `[B]`, values in `[-1, 1]`, all `float32`. The default config on 12x12 has a receptive field covering the board (assert nothing; note the 9-layer arithmetic from [network.md](../../design/network.md#kernel-size) in a comment).
2. **Eval mode is deterministic.** In `eval()` under `torch.inference_mode`, the same batch twice gives identical outputs; in `train()` mode BatchNorm uses batch statistics, so a batch of size 1 raises or differs. Pin whichever behaviour the code has and explain BatchNorm's two modes.
3. **Loss by hand.** Batch of one, 2x2 board: chosen logits, `legal` masking one cell that has the *largest* logit, one-hot `pi` on a legal cell, `z = 1`, `v = 0.5`. Expected `value_loss = 0.25`, `policy_loss = -log_softmax(masked logits)[a]` computed in the test with the masked cell excluded from the softmax; `total` is their sum. Asserting with the large illegal logit proves masking happened before the softmax.
4. **No NaN from masked cells.** `pi` is zero on illegal cells and the masked log-probabilities are `-inf`; a naive `(pi * log_p).sum()` gives `0 · -inf = NaN`. Assert the loss is finite. Implementation: zero the masked entries of `log_p` (`masked_fill`) before multiplying, or use `torch.where`.
5. **Memorise one batch.** 16 random 6x6 positions, `pi` one-hot on a random legal move, `z` random in `{-1, 0, 1}`, AdamW `lr = 1e-2`, no weight decay, up to 300 steps in `train()` mode: `policy_loss < 0.05` and `value_loss < 0.01`. One-hot targets are chosen so the cross-entropy floor is 0; with soft targets the floor is the entropy of `pi`, which the comment should say. Seed torch. If it does not converge, that is a bug, not a tuning problem.
6. **Config is recorded.** `Network(6, cfg).size == 6` and `.config == cfg`, so a checkpoint (plan 04) can rebuild the architecture from the saved config.

Implementation follows [network.md](../../design/network.md) exactly: stem, `blocks` residual blocks, the two heads, `tanh` on the value. Weight decay is not part of the loss (AdamW applies it), so `alphazero_loss` has no regularisation term; say why in the docstring.

## Step 7: the network evaluator and device selection (`tests/engine/test_network_evaluator.py`, then the torch half of `evaluator.py`)

Tests (CPU, tiny network, random weights):

1. **Protocol shape.** `NetworkEvaluator(net, torch.device("cpu")).evaluate(states)` returns arrays of the right shapes and dtypes; policies are exactly 0 on illegal cells and sum to 1 over legal ones (`1e-5`); values in `[-1, 1]`.
2. **Agrees with the raw network.** For one state, the policy equals `softmax` of the masked logits computed directly, and the value equals the network's.
3. **Batch independence.** Evaluating `[a, b]` gives the same numbers for `a` as evaluating `[a]` alone (eval-mode BatchNorm makes this true; it would fail in train mode, which is the point of the test).
4. **Size mismatch.** A 6x6 network given a 4x4 state raises `ValueError`.
5. **`select_device`.** `select_device("cpu")` is CPU; `select_device()` returns a device whose type is one of `cuda`, `mps`, `cpu` and follows availability (patch `torch.cuda.is_available` and `torch.backends.mps.is_available` to test the order without hardware).
6. **Search end to end on a network.** `analyse(4x4 position, NetworkEvaluator(tiny net), max_simulations=50)` runs and the last snapshot is well-formed. No claim about move quality: the weights are random.

Implementation: `encode_batch` → `torch.from_numpy` → `.to(device)` → `net.eval()`, `torch.inference_mode()` → `logits.masked_fill(~legal, -inf)` → `softmax` → `.cpu().numpy()`. Build the legal mask with `legal_mask` from encoding. Never call `.double()`: MPS has no float64.

## Step 8: benchmark script (`scripts/bench_search.py`)

Not test-driven. Arguments: `--size 6`, `--simulations 800`, `--evaluator network|uniform` (default `network`), `--blocks 4`, `--filters 64`, `--device auto|cpu|mps|cuda`, `--repeats 3`, `--seed 0`, `--position` (optional move list; default a fixed seeded random opening of 4 moves so the root is not the empty board). It runs `analyse` to the cap, and prints the parameters, the resolved device, simulations per second (mean over repeats), evaluator calls per second, and the fraction of wall time inside `evaluate`. Print a Markdown row so results can be pasted into the PR.

Run it once on 6x6 and once on 12x12 with the default network on this laptop, with `--device cpu` and `--device mps`, and put the four rows in the PR description. This is the first data point for [engineering.md](../../design/engineering.md#performance-plan); nothing is optimised in response to it in this plan.

Add the script to `scripts/README.md` (a new section, since it is not a game experiment) with one usage line.

## Step 9: documentation

- `docs/architecture.md`: package table row for `engine` becomes "Built (step 3), not yet connected to the CLI: see plan 03"; the "Designed but not built" section shrinks to `training`, and the two `PlaceholderEngine` / port notes stay (they are plan 03's job).
- `docs/design/engineering.md`: module layout gains `alphazero_loss` under `engine/network.py` and removes "the loss" from `training/trainer.py`; adds `scripts/bench_search.py`. Note the approval date.
- `docs/design/search.md`: one clarifying sentence under "Node statistics" on what `N(s)` is (see Decisions). No other change.
- `docs/design/open-questions.md`: the "Root-expansion evaluations and the `simulations` count" row becomes "Decided in plan 02: not counted".
- Module docstrings in `engine/` carry the design summary each module implements, with links to the design files, so the design does not need to be open to read the code.

## Definition of done

- `uv run pytest` is green, all previous tests included, and `uv run pytest tests/engine` finishes in under about 30 seconds on CPU.
- Every solver-fixture test in step 5 passes at the budget stated there, or the deviation is reported in the PR.
- `scripts/bench_search.py` runs from a clean checkout; the four benchmark rows are in the PR description.
- `src/ox_zero/engine/README.md` is gone; `src/ox_zero/training/README.md` is untouched; `cli/` is untouched.
- Every module has a docstring connecting it to the design, and every non-obvious block has a "how this works" comment, per the project `CLAUDE.md`.
- A pull request from `feat/engine-search-network` to `main`. On merge, this plan moves to `docs/plans/archive/`.

## Questions an implementer should stop and ask about

- A solver fixture fails at twice the stated budget. Do not raise budgets further or weaken the assertion; report which fixture, the visit distribution, and the root value.
- The memorise test does not converge in 300 steps at `lr = 1e-2`. Do not change the architecture; report the loss curve.
- The pseudocode of the 2018 paper and `search.md` disagree on something not covered in Decisions above. Follow the pseudocode, and list the disagreement in the PR.
- The benchmark hits an unsupported op on `mps` or a silent CPU fallback. Report it with the op name; do not add device-specific branches.
- Anything in `ox_zero.game` looks wrong (the search or the fixtures expose a rules bug). Change nothing in `game/rules.py` without asking; it changes the game.
- Any design-level choice not listed here seems to be needed. Stop and ask; do not decide it in code.
