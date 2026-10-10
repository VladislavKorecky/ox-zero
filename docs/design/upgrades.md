# Upgrades

Everything considered during the design and deliberately left out of version 1. Nothing here is rejected. Version 1 is the paper-faithful baseline; each item is a hypothesis to test against it, with a measured Elo difference (see [training.md](training.md#evaluation)) as the verdict.

Columns: what the change is, what it is expected to buy, where the idea comes from, and roughly when it becomes worth trying.

## Search

| Upgrade | Expected benefit | Source | When |
|---|---|---|---|
| **Constant `c_puct`** instead of the log-growing `C(s)` | One knob instead of two; useful if the 2018 constants behave badly at our small simulation counts. | AlphaGo Zero 2017 | If tuning `c_base`/`c_init` proves awkward |
| **First-play urgency (FPU)**: unvisited children get the parent's value minus a penalty instead of `Q = 0` | Commits to promising lines sooner in wide positions (144 moves). Probably one of the larger wins. | Leela Chess Zero | Early, once a baseline Elo curve exists |
| **Parent-`Q` for unvisited children** | Same aim as FPU, one fewer knob. | Common variant | Alongside the FPU test |
| **Evaluation cache** keyed by `State` | Repeated positions cost no forward pass. Search maths untouched. | Common | When profiling shows duplicate evaluations |
| **Transposition DAG**: share nodes across move orders | Fewer evaluations and deeper effective search. Visit-count semantics get subtle. | Various engines | After the cache, if it is not enough |
| **MCTS-Solver**: propagate proven wins and losses | Exact tactics: a forced win is never averaged away. OXOX is almost purely tactical, so this could be large. | Winands et al., 2008 | **First search upgrade to test.** The game experiments show that by mid-game most legal moves lose on the spot and games end by forced loss ([open-questions.md](open-questions.md#follow-up-safe-moves-over-a-game)); averaging relearns those one-ply facts every search. Baseline from plan 02: on the 4x4 forced wins 9 to 11 plies deep, plain PUCT with a uniform evaluator finds the unique winning move (about 90% of 1000 visits) but reports a root value near 0 (−0.03 on the first fixture), because averaging never proves the win. |
| **Virtual loss** / batched leaves in one tree | Much faster analysis in the CLI (paper used 8 leaves per batch). | AlphaGo Zero / AlphaZero | When analysis speed matters, i.e. once a good model exists |
| **Randomised-symmetry inference**: evaluate leaves under a random board symmetry | Acts as a regulariser and averages out orientation bias. | AlphaGo Zero 2017 | Cheap to test once symmetries exist for augmentation |

## Network

| Upgrade | Expected benefit | Source | When |
|---|---|---|---|
| **5×5 stem**, 3×3 blocks after | Layer one sees a full 4-cell line directly. Cheap on a 12×12 board. | Design discussion | A quick ablation once training runs |
| **Global pooling** in the trunk | Whole-board facts (tempo, parity, who is ahead) reach every cell. Convolutions alone cannot count. | KataGo (Wu, 2019) | **First network upgrade to test.** The game experiments showed parity matters: O wins 4x4, and games are decided by who runs out of safe moves first, a whole-board count ([open-questions.md](open-questions.md#follow-up-safe-moves-over-a-game)). Try it when the value loss plateaus. |
| **Empty-count / parity input plane** | Cheaper way to hand the network a global fact. | Design discussion | Low priority: it counts empty cells, but the quantity that decides games is the number of *safe* cells, which the plane does not know. Expect less than pooling. |
| **Win/draw/loss value head** (3-way softmax) | Separates "drawish" from "unclear". Better calibrated when draws are common. | Leela Chess Zero, KataGo | If the measured draw rate is high |
| **SGD with momentum + stepped learning rate** | The paper's optimiser; sometimes generalises better. | AlphaZero 2018 | When we want to compare against a faithful run |
| **Bigger towers** | Strength, at compute cost. | Paper | For the final 12×12 run on a rented GPU |
| **Exclude norms and biases from weight decay** (AdamW parameter groups) | Weight decay on BatchNorm scales and biases shrinks parameters that do not cause overfitting; excluding them is a common refinement. Version 1 decays every parameter, as the paper's `c · ‖θ‖²` does ([plan 04](../plans/04-training-pipeline.md#decisions-made-in-this-plan)). | Common practice | Low priority; if the value loss plateaus |

## Training

| Upgrade | Expected benefit | Source | When |
|---|---|---|---|
| **Blend `z` with the root's search value** as the value target | Much lower variance targets on small compute; faster early learning. | KataGo | Early candidate; a config weight makes it a one-line change |
| **Gating** (promote only a net that beats the previous best) | Protection against regressions between generations. | AlphaGo Zero 2017 | If Elo curves show regressions |
| **Continuous asynchronous pipeline** | Better hardware utilisation: self-play and training overlap. | AlphaZero 2018 | Only for a long final run on a machine with a spare GPU |
| **Multiprocessing self-play workers**, lockstep inside each | Uses all CPU cores for tree work. | Common | When profiling shows the single process is CPU-bound with the GPU idle |
| **Playout cap randomisation** and other KataGo training tricks | Cheaper self-play games per unit of learning. Not discussed in detail yet. | KataGo | Later |
| **Resignation** with calibration | Faster self-play if late-game moves dominate time. | AlphaZero | Not planned; only if profiling forces it |
| **Weights & Biases** logging | Cross-machine run comparison on the free tier. | Tooling | When runs happen on more than one machine |
| **Custom Textual training dashboard** | A live view in the style of the sandbox. | Roadmap | Its own roadmap step, after real runs exist |

## Engineering

| Upgrade | Expected benefit | Source | When |
|---|---|---|---|
| **Array-based tree** (NumPy arrays for `N, W, P`, children as indices) | Large speed-up of selection and backup over node objects. | mctx, many engines | When profiling shows the tree dominates |
| **Tensorised rules**: batched board tensors, win detection by convolution | Removes Python from move application in self-play. Needs cross-check tests against `ox_zero.game`. | pgx-style environments | When profiling shows `apply_move` dominates |
| **`torch.compile`** / fused inference | Lower per-batch latency. | PyTorch | When the network is the bottleneck |
| **YAML/TOML experiment files** | Sweeps without editing code. | Tooling | If we start running many configurations |
| **Sandbox search budget**: stop open-ended analysis at a simulation or memory cap, and show "done" in the status line | Bounded memory. The sandbox searches until the position changes, and the tree only grows: with lazy child states an empty-board search grows by about 175 MB/s with uniform priors (2026-10-09, [PR #16](https://github.com/VladislavKorecky/ox-zero/pull/16)), so leaving the sandbox open can exhaust an 8 GB machine within a minute. Needs a `docs/cli.md` change; a cap near 50k simulations is about 1.5 GB at 29 KB per expansion. | Design discussion | Before the sandbox is used for long sessions; the array-based tree shrinks each simulation but does not bound the total |

## Not yet discussed

Ideas that came up in passing and deserve their own discussion before they go in a plan: Gumbel AlphaZero (sequential-halving move selection at the root, far fewer simulations per move), and KataGo's auxiliary training targets (ownership, score).
