# Training

Self-play reinforcement learning from zero human data, as in AlphaZero, restructured for one machine.

## The generation loop

AlphaZero ran self-play actors and a trainer continuously and asynchronously across thousands of TPUs. **Decided:** we run **discrete generations**:

```
for generation g = 1, 2, ...:
    games   = self_play(net_g, config)          # many games at once, see below
    buffer.add(games, generation=g)             # positions from the last K generations
    for step in range(train_steps):
        batch = buffer.sample(batch_size)       # uniform over positions
        batch = augment(batch)                  # random board symmetry
        loss  = alphazero_loss(net(batch.x), batch.pi, batch.z)
        optimiser.step()
    checkpoint(net, optimiser, configs, g)
    evaluate(net_g, previous checkpoints)       # monitoring only, see Evaluation
    log(metrics)
```

The maths is identical; only the scheduling differs. Generations are simple to reason about, trivially resumable (restart from the last checkpoint), and easy to test end to end on a tiny board. Deferred: [continuous async pipeline](upgrades.md#training).

**No gating.** AlphaGo Zero promoted a new network only if it beat the current best in 55% of evaluation games. AlphaZero dropped the gate and always self-played with the latest weights. We follow AlphaZero. Evaluation still happens, but only to measure progress, never to decide which network self-plays. Deferred: [gating](upgrades.md#training).

## Self-play in lockstep

The network wants large batches; a Python tree search produces one leaf at a time. **Decided:** play many games at once in a single process, each game with its own tree, and advance them in lockstep:

```
games = [new_game() for _ in range(G)]           # e.g. G = 64
while any game is unfinished:
    leaves = [g.select_leaf() for g in games if g.needs_evaluation()]
    p, v   = evaluator(leaves)                    # ONE batched forward pass
    for g, (p_i, v_i) in zip(...): g.expand_and_backup(p_i, v_i)
    for g in games with n simulations complete:  g.play_move(); g.reuse_subtree()
```

Every tree runs one simulation per step, so no virtual loss is needed, the network sees batches of up to `G`, and the whole thing is one thread with no GIL contention. It behaves the same on `mps` and `cuda`. Deferred: [multiprocessing workers on top](upgrades.md#training).

Rules stay in the pure-Python `ox_zero.game` package. If profiling shows the rules dominate, a tensorised twin is the deferred fix ([upgrades.md](upgrades.md#engineering)).

### Per-game settings

- Root Dirichlet noise on, temperature schedule as in [search.md](search.md#move-selection-self-play), subtree reuse between moves.
- **No resignation.** AlphaZero resigns hopeless games to save time and plays 10% out to calibrate the threshold. OXOX games are expected to be short, and early in training resignation risks poisoning labels. Not planned.
- Games run to a terminal position. A full board is a draw, so there is no need for a move cap.

## Training data

Each move of each self-play game produces one example, recorded when the game ends:

| Field | Content |
|-------|---------|
| `state` | The position (stored as the `State`, or its encoded planes; the plan decides) |
| `π` | The root visit distribution at `τ = 1`, over all `S²` cells (zeros for illegal moves) |
| `z` | The game's result from this position's side to move: `+1`, `0`, `-1` |
| metadata | Generation, game id, move index. For diagnostics and for expiring old generations. |

**Value target: `z`, the game outcome.** Paper-faithful and unbiased. Its variance is high early in training, because one noisy game labels every position in it. Deferred: [blending `z` with the root's search value](upgrades.md#training).

### Replay buffer

**Decided:** the buffer holds every position from the most recent `K` generations and samples uniformly from them. This is the paper's "most recent 500,000 games" at our scale. Old generations expire by generation number, not by count. `K` starts around 10 to 20. **Open:** tune `K`, games per generation, and training steps per generation together, since they set how many times each position is seen. The game experiments give the multiplier: about 13 examples per game on 6x6, 17 on 8x8 and 28 on 12x12 at random strength, and trained games are not expected to be much longer ([open-questions.md](open-questions.md#follow-up-safe-moves-over-a-game)).

### Augmentation

Each sampled batch is transformed by a random one of the 8 board symmetries; `π` is permuted identically ([network.md](network.md#symmetries)).

## Checkpoints

One checkpoint per generation: weights, optimiser state, all configuration dataclasses, board size, generation number, and the RNG state needed to resume. See [engineering.md](engineering.md#configuration-and-checkpoints).

## Evaluation

Loss curves are necessary but not sufficient: a network can lower its loss while playing worse. **Decided:**

1. **Checkpoint tournaments with Elo.** Each new generation plays a fixed number of games against a few recent checkpoints, both as X and as O, with search but without root noise. Results feed an Elo calculation over all checkpoints. This is the primary progress metric. **Open:** games per pairing, simulations per move, how many opponents, and how to avoid identical games between deterministic players (a small temperature or an opening-book of random first moves).
2. **Loss curves.** Policy loss, value loss, and total, per generation.

Not built as metrics (but see [engineering.md](engineering.md#testing) for their role in tests): win rate against random or against the raw network, and agreement with an exact solver on small boards.

## Logging

**Decided:** the training loop writes plain metrics files, one row per generation (losses, Elo, games per second, average game length, draw rate, buffer size, wall time). Any dashboard is a *reader* of those files:

- **TensorBoard** from day one: local, free, live curves during a run, including on a rented machine.
- A **custom Textual training dashboard** is a later roadmap step, built against real runs once it is clear which numbers matter.
- **Weights & Biases** is an optional reader (free personal tier) if cross-machine comparison becomes useful.

The training loop never depends on a dashboard.
