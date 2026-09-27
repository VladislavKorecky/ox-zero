"""The AlphaZero core: tree search, the policy/value network, and the encoding between them.

Design: docs/design/ (search.md, network.md, engineering.md).

Modules, in dependency order:

- `encoding`: `State` -> input planes, legal-move masks, the 8 board symmetries.
- `evaluator`: the `Evaluator` protocol (the seam between search and network)
  and its implementations.
- `mcts`: the tree itself: `Node`, PUCT selection, expansion, backup.
- `search`: `SearchTree` (the stepped search self-play drives) and `analyse`
  (the snapshot generator the CLI will consume).
- `network`: the residual tower, its two heads, and the AlphaZero loss.

This package deliberately re-exports nothing: importing `ox_zero.engine.mcts`
or `ox_zero.engine.search` must not pull in torch, and only `network` and the
network half of `evaluator` need it.
"""
