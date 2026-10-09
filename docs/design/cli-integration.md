# CLI integration

The CLI owns its engine interface (`cli/engine_port.py`, see [architecture.md](../architecture.md)). The real engine is connected through one adapter, `cli/adapter.py`, which runs `engine.search` in analysis mode and translates each yielded search snapshot into an `Analysis`.

## Mapping

| `Analysis` field | From the search |
|---|---|
| `value` | The root's mean value `Q(root)`, i.e. `W/N` over all root visits, mapped to `[0, 1]` as `(q + 1) / 2`. Win probability for the side to move. |
| `scores[a]` | For each legal move `a`: `Q(root, a)` mapped to `[0, 1]`. `Q(root, a)` is from the perspective of the player who plays `a`, which is the side to move, so it is exactly "win probability for the side to move after playing `a`", as the spec defines it. |
| `simulations` | The number of completed PUCT simulations. |
| best move | The **most visited** root child (see below). |

## The one port change

**Done** ([plan 03](../plans/archive/03-cli-adapter.md), 2026-10-09): `Analysis.chosen` exists, and `best` returns it.

`Analysis.best` used to derive the best move from the highest score. AlphaZero chooses the **most visited** child, because `Q` on a rarely visited move is noise while the visit count already integrates value and confidence. **Decided:** `Analysis` gains an explicit field for the engine's chosen move, set by the adapter, and `best` returns it. `top(n)` keeps ranking by score, since it exists to show candidate moves with their win estimates. `docs/cli.md` already says `best` prints "the engine's chosen move" and needs only a clarifying sentence. `PlaceholderEngine` sets the field to its highest-scoring move, so nothing else changes.

## Every legal move gets a genuine score

The spec requires a score for every legal move. A root child that PUCT never visited has no `Q`, and showing it the root's value would advertise a possibly terrible move as average. **Decided:** in analysis mode the search evaluates **all root children up front** in one batched network call (at most `S²` positions, one forward pass), before the PUCT loop starts. Every legal move then has one real visit and a `Q` from its own evaluation, from the first snapshot on. No spec change, no missing scores, and off in self-play where the paper's behaviour is kept.

**Open (plan-level detail):** whether the root-expansion evaluations count toward `simulations`. Recommendation: they do not; they are setup, and the first snapshot reports `simulations = 0`, followed by 1, 2, ... This keeps "`--simulations N` ends with exactly `N`" true even when `N` is smaller than the number of legal moves.

## Snapshots and cancellation

The search is a generator. In analysis mode each iteration runs one simulation (batch size 1) and yields a snapshot; building one costs `O(legal moves)`. The CLI already throttles its redraws, and it cancels a search simply by dropping the iterator, which the generator design supports with no extra machinery. Deferred: [virtual loss](upgrades.md#search) to batch leaves within one tree and speed up analysis.

**GIL note (from the architecture document):** the sandbox runs the engine in a background thread. Torch releases the GIL during a forward pass, and the generator yields between simulations, so the UI gets scheduled regularly. The pure-Python tree work between yields is short. Measured once connected (2026-10-09, uniform priors, the worst case since nothing releases the GIL): about 13 ms from a keypress to the screen, 50–70 ms to play a move, and no snapshot of an old position ever displayed.

## Loading a model

`--model PATH` or the newest checkpoint in `checkpoints/` (recursively, by generation), per the spec. The checkpoint carries its configs, so the adapter rebuilds the network from the file alone and selects the device at runtime. A checkpoint trained on a different board size than the position being analysed is an error.

Settled in [plan 03](../plans/archive/03-cli-adapter.md#decisions-made-in-this-plan) on 2026-10-09, with three changes to `docs/cli.md`:

- **No model: uniform-priors search, not the placeholder.** With no checkpoint the CLI runs the real search with `UniformEvaluator` and a notice says so. It finds short tactics with no training at all. `PlaceholderEngine` is test tooling only.
- **`--device auto|cpu|mps|cuda`, default `cpu`.** Analysis runs at batch size 1, where `mps` measured 1.7–3× slower than `cpu` ([engineering.md](engineering.md#measured-2026-09-27)). Revisit the default when virtual loss batches the analysis.
- **`--seed` is documented as having no effect on the real engine:** analysis mode has no noise and no sampling, so it is deterministic.
