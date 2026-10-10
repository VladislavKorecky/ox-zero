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
| `state` | The position, stored as its encoded planes (`uint8`; plan 04): sampling and augmentation are then array operations, with no re-encoding per step |
| `π` | The root visit distribution at `τ = 1`, over all `S²` cells (zeros for illegal moves) |
| `z` | The game's result from this position's side to move: `+1`, `0`, `-1` |
| metadata | The generation, for expiring old generations. As built it is the replay buffer's key, not a stored column. |

**Value target: `z`, the game outcome.** Paper-faithful and unbiased. Its variance is high early in training, because one noisy game labels every position in it. Deferred: [blending `z` with the root's search value](upgrades.md#training).

### Replay buffer

**Decided:** the buffer holds every position from the most recent `K` generations and samples uniformly from them. This is the paper's "most recent 500,000 games" at our scale. Old generations expire by generation number, not by count. `K` starts around 10 to 20. **Open:** tune `K`, games per generation, and training steps per generation together, since they set how many times each position is seen. The game experiments give a floor for the multiplier: about 13 examples per game on 6x6, 17 on 8x8 and 28 on 12x12 at random strength. Trained games that build walls are expected to be longer, perhaps 50–60 moves on 12x12 ([open-questions.md](open-questions.md#follow-up-safe-moves-over-a-game)); the logged average game length gives the real number.

The constants are set provisionally for 6x6 in [plan 04](../plans/archive/04-training-pipeline.md#decisions-made-in-this-plan) (`K = 10`, 128 games and 50 steps of 256 per generation); the first run's numbers are in [open-questions.md](open-questions.md#results-2026-10-10-the-first-6x6-run).

**Persisted for resume** (plan 04). The buffer is saved with the run, one file per held generation, so a resumed run trains on the same positions it would have trained on without the interruption, rather than on its own new games only. Each save writes only the newest generation and deletes expired ones.

### Augmentation

**Decided (plan 04):** each sampled *example* is transformed by its own random one of the 8 board symmetries; `π` is permuted identically and `z` is unchanged ([network.md](network.md#symmetries)). Per example rather than per batch: eight times the variety per step, at the cost of eight array calls (one per symmetry group) instead of one. Augmentation is free supervision: the game is symmetric, the data is not.

## Checkpoints

One checkpoint per generation: weights, optimiser state, all configuration dataclasses, board size, generation number, and the RNG state needed to resume. See [engineering.md](engineering.md#configuration-and-checkpoints).

## Evaluation

Loss curves are necessary but not sufficient: a network can lower its loss while playing worse. **Decided:**

1. **Checkpoint tournaments with Elo.** Each new generation plays a fixed number of games against a few recent checkpoints, both as X and as O, with search but without root noise. Results feed an Elo calculation over all checkpoints. This is the primary progress metric. The protocol below was decided in [plan 04](../plans/archive/04-training-pipeline.md#decisions-made-in-this-plan) (approved 2026-10-09, ladder added 2026-10-10); its numbers are provisional ([open-questions.md](open-questions.md#constants-waiting-on-the-experiments)).
2. **Loss curves.** Policy loss, value loss, and total, per generation.

Not built as metrics (but see [engineering.md](engineering.md#testing) for their role in tests): win rate against random or against the raw network, and agreement with an exact solver on small boards. `scripts/evaluate.py` can play a checkpoint against the uniform search by hand.

### Tournament protocol

- **Opponents.** Generation `g` plays the previous 3 checkpoints (fewer early on) plus one **ladder** opponent, `g − 8`, when it exists and is not already among them. Generation 0, the random network, is a checkpoint and the anchor.
- **Openings.** Per pairing, 20 random first moves (distinct cells), each played twice with the colours swapped: 40 games. Random first moves give distinct games between deterministic players with no new search code; paired colours cancel the first-move advantage. The alternative, `τ = 1` sampling for the first few moves, duplicates games once the policy is sharp.
- **Play.** After the opening, both sides search without root noise and always play the most-visited move, at 100 simulations per move, the same as self-play (`scripts/train.py --simulations` sets both). Noise exists to explore in self-play; in evaluation it would only add variance to the measurement. Each player has its own tree: a shared tree would hand one network the other's priors and values every ply, and measure a blend of the two. Draws score ½.
- **Elo fit.** A Bradley-Terry maximum-likelihood fit over *every* match of the run so far, generation 0 fixed at 0 (Elo is only defined up to a constant, so it needs an anchor). One virtual draw per pairing regularises the fit, so a 40–0 sweep gives a finite rating instead of infinity.
- **Why the ladder.** With neighbour matches only, a late generation's rating reaches the anchor through a chain of noisy links, about 50 Elo of noise per 40-game link, and the errors add up like a random walk: the absolute rating could be off by well over 100 Elo while the local curve looks fine. One long-range match per generation measures across eight links directly and pins the chain. If ladder matches come out as sweeps (above about 95%), the distance is too long to be informative and should be shortened. In the first 6x6 run they did not ([open-questions.md](open-questions.md#results-2026-10-10-the-first-6x6-run)).

## Logging

**Decided:** the training loop writes plain metrics files, one row per generation (losses, Elo, games per second, average game length, draw rate, buffer size, wall time). Any dashboard is a *reader* of those files. As built (plan 04), in the run directory ([engineering.md](engineering.md#configuration-and-checkpoints)):

- `metrics.jsonl`: one JSON object per generation (self-play counts, game length, draw and X-win rates, simulations per second, the three losses, Elo, the per-opponent scores, the time of each phase). JSON Lines appends without rewriting, a crash mid-row costs only the partial last line, and it nests the per-opponent scores.
- `matches.jsonl`: one object per tournament pairing; the Elo fit is recomputed from all of them.
- `ratings.json`: the current Elo table, rewritten each generation.
- `tensorboard/`: the same scalars as TensorBoard event files.

Readers:

- **TensorBoard** from day one: local, free, live curves during a run, including on a rented machine.
- A **custom Textual training dashboard** is a later roadmap step, built against real runs once it is clear which numbers matter.
- **Weights & Biases** is an optional reader (free personal tier) if cross-machine comparison becomes useful.

The training loop never depends on a dashboard.
