# Benchmark reference pools

The frozen checkpoints every candidate is rated against (`scripts/benchmark.py`), one pool per board size. The pools themselves live in `runs/benchmarks/<S>x<S>/pool/`, gitignored with every other training artefact, together with the match log (`matches.jsonl`) and the ratings derived from it (`ratings.json`). This file is the committed record: enough to rebuild a lost pool and verify the rebuild.

A pool member `<run>-gen_NNN.pt` is the player `<run>/gen_NNN`. The uniform search at 100 and 400 simulations is always also a reference and needs no file. See [`training/benchmark.py`](../../src/ox_zero/training/benchmark.py) for the method.

## Rules

- **Frozen once, only extended.** Changing or replacing a reference changes the scale. To add one, copy a checkpoint into `pool/` and add it below; the next benchmark rates every earlier candidate against it only when that candidate is benchmarked again (pairings already logged are never replayed).
- **The pool never holds a generation the calibration criteria rate** (6x6: generations 20 and 40; 8x8: 15 and 30).
- **Rebuilding**: rerun the origin command below with the same seed on the same machine type, copy the generations listed, and compare `shasum -a 256` with the hashes here. A mismatch means the rebuilt network is a different player; start a new log rather than reusing the old one.
- **Sharing** a pool means sharing its log with it (for example as a GitHub release): without the checkpoints a log is a record of games nobody can replay.

## Manifest

*Empty until the calibration runs (experiment 01) seed the pools.*

| Board | File | Origin (run, seed, generation, command) | sha256 | Frozen |
|---|---|---|---|---|
