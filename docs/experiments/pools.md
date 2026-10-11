# Benchmark reference pools

The frozen checkpoints every candidate is rated against (`scripts/benchmark.py`), one pool per board size. The pools themselves live in `runs/benchmarks/<S>x<S>/pool/`, gitignored with every other training artefact, together with the match log (`matches.jsonl`) and the ratings derived from it (`ratings.json`). This file is the committed record: enough to rebuild a lost pool and verify the rebuild.

A pool member `<run>-gen_NNN.pt` is the player `<run>/gen_NNN`. The uniform search at 100 and 400 simulations is always also a reference and needs no file. See [`training/benchmark.py`](../../src/ox_zero/training/benchmark.py) for the method.

## Rules

- **Frozen once, only extended.** Changing or replacing a reference changes the scale. To add one, copy a checkpoint into `pool/` and add it below; the next benchmark rates every earlier candidate against it only when that candidate is benchmarked again (pairings already logged are never replayed).
- **The pool never holds a generation the calibration criteria rate** (6x6: generations 20 and 40; 8x8: 15 and 30).
- **Rebuilding**: rerun the origin command below with the same seed on the same machine type, copy the generations listed, and compare `shasum -a 256` with the hashes here. A mismatch means the rebuilt network is a different player; start a new log rather than reusing the old one.
- **Sharing** a pool means sharing its log with it (for example as a GitHub release): without the checkpoints a log is a record of games nobody can replay.

## Manifest

**8x8 is the testbed** ([experiment 01](01-testbed-and-noise-floor.md)); its pool is frozen as of 2026-10-12 and defines the scale every experiment is rated on. The 6x6 pool is kept for 6x6 runs.

| Board | File | Origin (run, seed, generation, command) | sha256 | Frozen |
|---|---|---|---|---|
| 6x6 | `six-a-gen_005.pt` | `six-a`, seed 0, generation 5, `uv run python scripts/train.py --name six-a --size 6 --generations 20 --seed 0 --device cpu` | `05448fcca04a8a43211ee1d263e20e241a17cc178b4b1d1b46b518c2373de7eb` | 2026-10-11 |
| 6x6 | `six-a-gen_010.pt` | `six-a`, seed 0, generation 10, `uv run python scripts/train.py --name six-a --size 6 --generations 20 --seed 0 --device cpu` | `8fa78b3e7959803c26ad08e3ee1145bbcac2002126420295541cf969125d7ec0` | 2026-10-11 |
| 6x6 | `six-a-gen_015.pt` | `six-a`, seed 0, generation 15, `uv run python scripts/train.py --name six-a --size 6 --generations 20 --seed 0 --device cpu` | `a9c06f096ae5fa9b031fb63058742d373d8feb365e400ea5b33a9393e9163dd8` | 2026-10-11 |
| 8x8 | `eight-a-gen_005.pt` | `eight-a`, seed 0, generation 5, `uv run python scripts/train.py --name eight-a --size 8 --generations 30 --seed 0 --device mps` | `bf9db4d1cbc36b15f7311c3878ba601cf868014518b86eaddfe3a29fbd10575f` | 2026-10-11 |
| 8x8 | `eight-a-gen_010.pt` | `eight-a`, seed 0, generation 10, `uv run python scripts/train.py --name eight-a --size 8 --generations 30 --seed 0 --device mps` | `21dc07992512289d00cdaf794069bd25609e2c5e369655024e95aa7cafc7929b` | 2026-10-11 |
| 8x8 | `eight-a-gen_020.pt` | `eight-a`, seed 0, generation 20, `uv run python scripts/train.py --name eight-a --size 8 --generations 30 --seed 0 --device mps` | `d563e24d2f720b309ed66a5e32c41c4b7f8634767e42a50152e409aae1d40ede` | 2026-10-11 |

The 6x6 references come from `six-a`'s first 20 generations, which were interrupted once (Ctrl-C during generation 6) and resumed with the same command ([open-questions.md](../design/open-questions.md#results-2026-10-10-the-first-6x6-run)). That does not affect a rebuild: resume is bit-for-bit on the same device ([engineering.md](../design/engineering.md#devices-and-determinism)).

The 8x8 references were trained on `mps` (faster than `cpu` at 8x8, [experiment 01](01-testbed-and-noise-floor.md)), so a rebuild must also use `mps` on the same machine type: resume and reruns are bit-for-bit per device, not across devices. The benchmark games themselves are always played on `cpu`.
