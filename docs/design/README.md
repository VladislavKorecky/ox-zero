# Engine and training design

The decisions behind roadmap steps 3 (search and network) and 4 (self-play training). This folder is the project's **settings**: what we are building and why, with the maths spelled out. It is not a plan. Implementation plans (which features get built in which order, and exactly how) will live in `docs/plans/` and reference these files.

Everything here was decided in a design discussion before any engine code existed. Where a decision waits on data, the file says **Open** and [open-questions.md](open-questions.md) lists what has to be measured first.

## Guiding principles

1. **Version 1 replicates AlphaZero.** The reference is Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*, Science 2018 (preprint: [arXiv 1712.01815](https://arxiv.org/abs/1712.01815)), including its pseudocode. Where the 2017 AlphaGo Zero paper (Nature) differs, we follow the 2018 version and say so. We deviate only where a single machine forces it or where the OXOX rules make a paper feature meaningless.
2. **Deferred is not discarded.** Every improvement we chose *not* to build in version 1 is recorded in [upgrades.md](upgrades.md) with what it is expected to buy. Version 1 is the baseline those upgrades will be measured against.
3. **Board size is a parameter everywhere.** The pipeline is developed and validated on small boards (4x4 to 6x6), where a generation takes minutes and tiny boards can be solved exactly for ground truth. 12x12 is paid for only once the pipeline is proven.
4. **Device-agnostic and resumable.** Development and most training happen on an Apple Silicon laptop (`mps`); the final run may happen on a rented CUDA GPU. Nothing may assume a device, and a run must survive being stopped and restarted.
5. **Fastest working pipeline, understood.** When choices trade off, a simple end-to-end pipeline that learns beats a clever one that does not exist yet. Code comments connect each operation to the theory (see the project `CLAUDE.md`).

## Files

| File | Covers |
|------|--------|
| [search.md](search.md) | Monte Carlo tree search: node statistics, PUCT selection, expansion, backup, terminal handling, root noise, temperature, tree reuse, analysis mode. |
| [network.md](network.md) | Position encoding, the residual tower, policy and value heads, loss, optimiser, symmetry augmentation. |
| [training.md](training.md) | The generation loop, lockstep self-play, training data, replay buffer, evaluation, logging. |
| [engineering.md](engineering.md) | Module layout, the evaluator seam, tree representation, configuration and checkpoints, devices, testing strategy, performance plan. |
| [cli-integration.md](cli-integration.md) | How the search plugs into the CLI's engine port, and the one port change it needs. |
| [upgrades.md](upgrades.md) | Deferred improvements and what each should buy. |
| [open-questions.md](open-questions.md) | Constants and choices that wait on experiments. |

## Notation used throughout

- `S`: board side length; the board has `S²` cells. Cells are `(row, col)`, zero-based, row 0 at the top.
- `s`: a game state (`ox_zero.game.State`); `a`: a move (a cell).
- `N(s,a)`, `W(s,a)`, `Q(s,a)`, `P(s,a)`: visit count, total value, mean value, prior of edge `(s,a)`, as in the paper.
- `v`: the network's value estimate in `[-1, 1]`, always from the perspective of the side to move in the evaluated position.
- `p`: the network's move probabilities over all `S²` cells; `π`: the search's visit distribution at the root (the policy target).
- `z`: the final game result from the perspective of a position's side to move: `+1` win, `0` draw, `-1` loss.
