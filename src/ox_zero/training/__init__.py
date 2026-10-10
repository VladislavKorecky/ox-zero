"""The self-play training pipeline (roadmap step 4).

Design: docs/design/training.md; implementation plan: plan 04. Modules:
`selfplay` (lockstep self-play and the `z` labelling), `replay` (the replay
buffer and symmetry augmentation), `trainer` (AdamW training steps),
`evaluate` (checkpoint tournaments and the Elo fit), `metrics` (JSON Lines and
TensorBoard), `checkpoint` (the file format the CLI also reads) and `run` (the
resumable generation loop that ties them together). Start a run with
`scripts/train.py`.
"""
