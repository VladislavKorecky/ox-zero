# ox-zero

AlphaZero engine and analysis tooling for the OXOX board game. Project description, rules, goals, and roadmap:

@README.md

## Code Style

- Comment thoroughly. The author is learning the ML theory alongside the code, so connect implementation to theory: name the concept (e.g. PUCT, policy target, value head, Dirichlet noise), state the formula or paper reference, and explain *why* the operation exists, not just what it does.
- Don't shy away from explaining non-obvious operations (tensor reshapes, masking, normalisation, backprop details). A short "how this works" comment above a tricky block is welcome, not noise.

- Coordinates are `row,col`, zero-based, row 0 at the top, column 0 at the left. Same convention in the CLI, the code, and the tests. Never introduce a second one.
- The CLI's behaviour is specified in `docs/cli.md`. Implement to that spec; change the spec first if it needs changing.

## Workflow

- Test-driven development. For every feature: write the failing tests first, run them to confirm they fail, then implement until they pass.
- Never write implementation code before its tests exist.

## Commands

Managed with [uv](https://docs.astral.sh/uv/). Never `pip install` into the venv directly.

| Command | Description |
|---------|-------------|
| `uv sync` | Create/update `.venv` from `uv.lock` |
| `uv add <pkg>` / `uv add --dev <pkg>` | Add a runtime / dev dependency (updates lock) |
| `uv run pytest` | Run the test suite |
| `uv run python -m ox_zero...` | Run package code inside the venv |

Layout: `src/ox_zero/` is the package, `tests/` mirrors it. Python 3.14 (Torch 2.14 supports it).

## Gotchas

- GPU acceleration on this machine (Apple Silicon) is via PyTorch's `mps` backend, not CUDA. Select the device at runtime (`select_device()`: `mps` if available, else `cpu`) rather than hardcoding; MPS lacks float64 and a few ops silently fall back to CPU. Exception: the CLI's `--device` defaults to `cpu`, because analysis runs at batch size 1, where `mps` measured 1.7–3× slower (docs/design/engineering.md, "Measured"); `--device auto` gives runtime selection.
