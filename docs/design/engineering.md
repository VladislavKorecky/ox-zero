# Engineering

How the design becomes code: layout, seams, data structures, configuration, devices, and tests. The [architecture document](../architecture.md) has the package map and the dependency rule; this file goes one level down.

## Module layout

```
src/ox_zero/
  game/          rules, notation                       (done)
  engine/
    encoding.py  State -> input planes; symmetries (transform planes and policies)
    network.py   the residual tower, heads, NetworkConfig; alphazero_loss
    evaluator.py the Evaluator protocol; UniformEvaluator, TableEvaluator (torch-free)
    network_evaluator.py  NetworkEvaluator, select_device (the only torch on the search path)
    mcts.py      Node, PUCT selection, expansion, backup, tree reuse
    search.py    SearchConfig; SearchTree (two-phase select / expand-and-backup); analyse (snapshot generator)
  training/
    selfplay.py  lockstep self-play of many games; SelfPlayConfig
    replay.py    the replay buffer
    trainer.py   the optimiser, one training step; TrainConfig
    evaluate.py  checkpoint tournaments and Elo
    checkpoint.py save / load of weights + configs + optimiser + RNG state
    run.py       the generation loop, logging, resume
  cli/
    adapter.py   wraps engine.search into the CLI's Engine port (see cli-integration.md)
scripts/
  bench_search.py  simulations per second for a board size, evaluator and device
  train.py       entry point that builds configs and calls training.run
  plot.py        curves from metrics files (until a dashboard exists)
checkpoints/     run outputs, gitignored
```

One module per concept. `engine` never imports `training` or `cli`; `training` never imports `cli`.

Two changes from the original layout, made in [plan 02](../plans/archive/02-engine-search-network.md) (scope approved 2026-09-27): the loss lives in `engine/network.py` rather than `training/trainer.py`, because the network's main test ("memorise one batch") needs it; and `NetworkEvaluator` has its own module, because `import torch` costs about half a second and the search, which imports `evaluator.py`, must stay torch-free.

## The evaluator seam

The search never touches torch. It depends on one protocol:

```
class Evaluator(Protocol):
    def evaluate(self, states: Sequence[State]) -> tuple[Policies, Values]: ...
```

Given a batch of states it returns, for each, a prior over the `S²` cells (illegal moves already zero, legal moves summing to 1) and a value in `[-1, 1]` from the side to move's perspective.

- `NetworkEvaluator` encodes states, runs the model in eval mode without gradients, masks and normalises the policy. Encoding lives on this side of the seam.
- `UniformEvaluator` returns equal priors and value 0. Search tests use it to check visit counts and forced-win detection without loading torch or a device.
- Hand-scripted evaluators (fixed tables of state -> policy/value) give tests exact, deterministic expectations.

This seam is what makes the search testable, and it is also where batching across lockstep games happens: self-play collects leaves from many trees and calls `evaluate` once.

## Tree representation

**Decided:** node objects. A `Node` holds its `State` (for children made by expansion, built lazily on first access; see below), `prior`, `visit_count`, `value_sum`, and a dict `move -> Node` of expanded children; `Q` is a property. This maps one-to-one onto the paper's `N, W, Q, P` and is easy to read and to debug. It is slower than array storage; lockstep batching hides most of the cost, and the array representation is a deferred optimisation ([upgrades.md](upgrades.md#engineering)). A child's `State` is built lazily, on first access, from its parent's state and its move (2026-10-09): most children are never visited, and building every board up front made boards 88% of the tree's memory. That took one 12x12 expansion from 203 KB to 29 KB and made an analysis search about 13× faster, with no API change.

The rules stay immutable and functional; the tree is the one place with mutable statistics, and it is confined to `mcts.py`.

## Configuration and checkpoints

**Decided:** frozen dataclasses with defaults, one per concern: `NetworkConfig` (blocks, filters, value hidden size), `SearchConfig` (`c_base`, `c_init`, noise `ε` and `α`, simulations, temperature cutoff, root expansion), `SelfPlayConfig` (games per generation, parallel games), `TrainConfig` (batch size, steps per generation, learning rate, weight decay, buffer generations `K`). Board size is part of the run configuration and stored alongside.

A checkpoint is a single `torch.save` file containing the model weights, optimiser state, every config, the board size, the generation number, and RNG state. A checkpoint therefore fully describes itself: the loader rebuilds the exact architecture and encoding without any external file. No YAML layer; a run is configured in code (`scripts/train.py`).

`checkpoints/<run-name>/gen_NNN.pt`, gitignored. A model worth sharing is attached to a GitHub release, not committed.

## Devices and determinism

- Device selection at runtime: `cuda` → `mps` → `cpu`. Never hardcoded. Tests run on `cpu`.
- float32 only.
- Every source of randomness (NumPy for Dirichlet noise and sampling, torch for weights and augmentation) is seeded from one run seed and its state is checkpointed, so a resumed run continues rather than restarts.
- Bit-exact reproducibility across devices is not a goal; reproducibility of a run on one device is.

## Testing

Test-driven throughout: failing tests first, then implementation. What is a unit test and what is not:

| Layer | Tests |
|-------|-------|
| Encoding | Plane contents for hand-built positions; each symmetry is a bijection on planes and on policies; applying a symmetry and its inverse is the identity. |
| Search | With `UniformEvaluator` and fixed seeds: exact visit counts on tiny trees computed by hand; a win-in-1 is found and a loss-in-1 avoided; terminal backups have the right sign; tree reuse preserves counts; root noise changes priors only at the root; the generator yields strictly increasing simulation counts and honours the cap. |
| Solver fixture | `ox_zero.game.Solver` gives exact game values on 3x3 and 4x4 (5x5 is out of reach: it ran out of memory before time, see [open-questions.md](open-questions.md#results-2026-09-27)). Search with enough simulations must agree on the value sign and pick a proven-best move. Fixture positions must have a strict, small subset of optimal moves, or the test is vacuous: on the empty 4x4 all 16 moves lose. Good fixtures: the empty 3x3 (eight moves draw, the centre loses), win-in-one and forced-loss positions, and reachable 4x4 positions sampled with the solver and kept only when few moves are optimal. This is the only ground truth the search can have; it is a test tool, not a training metric. |
| Network | Output shapes for several board sizes; masked cells get exactly zero probability; value in `[-1, 1]`; the network can memorise a single batch (loss goes to ~0); the loss matches a hand computation. |
| Training | Replay buffer expiry by generation; augmentation permutes `π` consistently with the planes; checkpoint round-trip restores weights, configs, and RNG. |
| End to end | One full generation on a 4x4 board with a tiny network: a few self-play games, a training step, a checkpoint written, loaded back, and searched. Must run in seconds. |

"Does it learn" is not a pytest. It is answered by the Elo curve of real runs.

## Performance plan

Do not optimise before profiling. Expected hot spots, in the order we expect to hit them: Python overhead in the tree (selection over up to 144 children per node), `State` creation in `apply_move` (tuple copy of `S²` cells), and network inference latency at small batch sizes on `mps`. The deferred fixes for each are listed in [upgrades.md](upgrades.md#engineering). The lockstep design is chosen so that the network is never the bottleneck at batch sizes it can use.

### Measured (2026-09-27)

First data point, from `scripts/bench_search.py` with its defaults: analysis mode (batch size 1, root children evaluated up front), 800 simulations, the default 4×64 network with random weights, mean of 3 runs, on the author's 8 GB Apple Silicon laptop with torch 2.14. The position is a seeded random 4-move opening with no win in one. Nothing was optimised in response ([plan 02](../plans/archive/02-engine-search-network.md)).

| Board | Device | Simulations/s | Evaluate calls/s | Time in `evaluate` |
|---|---|---|---|---|
| 6x6 | cpu | 813 | 770 | 78% |
| 6x6 | mps | 257 | 243 | 93% |
| 12x12 | cpu | 284 | 284 | 38% |
| 12x12 | mps | 166 | 166 | 66% |

With `UniformEvaluator` instead of the network (pure tree cost), 6x6 runs at about 3,500 simulations/s, of which 4% is in `evaluate`.

- **At batch size 1, `mps` is 1.7 to 3 times slower than `cpu`.** Each GPU call pays a fixed latency that a tiny batch cannot amortise. This is the third expected hot spot, and it confirms lockstep batching for self-play. For analysis, CPU is the faster device until virtual loss exists. No unsupported-op errors or fallback warnings appeared on `mps`.
- **On 12x12 CPU, 62% of the time is the Python tree, not the network.** Expanding a node builds a `State` for each of its ~140 children: the `apply_move` hot spot predicted above. At 6x6 the network dominates instead.
