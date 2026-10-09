# Architecture

How the code is organised, and the decisions behind it that aren't obvious from any single file. For what the game is and where the project is going, see the [README](../README.md). For what the CLI does, see the [CLI specification](cli.md).

Each module's docstring explains its own design in detail. This document is the map that ties them together.

## Package map

Everything importable lives in `src/ox_zero/`. `tests/` mirrors its layout.

| Package | Owns | Status |
|---------|------|--------|
| `game` | OXOX rules (`rules.py`) and text notation for cells and positions (`notation.py`). Pure Python: no NumPy, no tensors. | Done (roadmap step 1) |
| `engine` | Position analysis: the AlphaZero search and network. | Built (step 3), connected to the CLI ([plan 03](plans/03-cli-adapter.md)) |
| `training` | The self-play training pipeline. | Checkpoint format built (plan 03); the rest designed, not built (step 4): see [docs/design](design/README.md) |
| `cli` | The `ox-zero` command: commands, rendering, JSON reports, the sandbox, and the engine port. | Done (step 2), running the engine |
| `gui` | Graphical analysis interface. | Not started (step 5) |

`scripts/` is reserved for one-off runners outside the package, such as launching training.

## Dependency rule

Packages depend on each other in one direction only:

```
game  ←  engine  ←  training
  ↑        ↑
  cli, gui (on top, importing whatever they need)
```

An arrow points from the importer to what it imports: `engine` may import `game`, never the reverse.

- `game` imports nothing from the other packages.
- `engine` and `training` never import `cli` or `gui`. An interface doesn't get to shape the engine.

## Key decisions

### Game states are immutable values

`State` is a frozen dataclass. `apply_move` returns a new state rather than modifying the old one.

- **Why:** tree search expands many hypothetical positions from the same parent. With immutable states, exploring one branch can never corrupt another, and there is no undo stack. Immutable states are also hashable, so the same position reached by different move orders compares equal, which makes positions usable as dictionary keys.
- **Where:** `game/rules.py` (the module docstring has the full reasoning).

### One coordinate convention

Cells are `row,col`, zero-based, row 0 at the top, column 0 at the left, everywhere: CLI input and output, code, and tests. The board string and every list of moves are in board order (left to right, top to bottom).

### The CLI is specified before it is built

`docs/cli.md` is the source of truth for the CLI's behaviour. A behaviour change starts as a spec change, then tests, then code.

### The CLI owns its engine interface

The CLI needs analysis results, but it must not dictate how an engine works. So the interface belongs to the consumer, not the provider (the "ports and adapters" pattern, a form of dependency inversion):

- `cli/engine_port.py` defines what the CLI needs: `Analysis` (per-move win probabilities, a position value, a simulation count, and the engine's chosen move) and the `Engine` protocol, whose `search()` yields snapshots that improve over time. Incremental results are a real requirement: `analyze --live` and the sandbox show analysis as it deepens.
- `cli/adapter.py` holds `SearchEngine`, the adapter that maps `ox_zero.engine.search.analyse` snapshots onto `Analysis`, and `load_engine()`, the single point where the CLI obtains an engine: the given or newest checkpoint, else the same search with uniform priors. The engine package itself stays free of any CLI-driven design.
- `cli/placeholder.py` holds `PlaceholderEngine`, a fast, deterministic stand-in with meaningless numbers. It is test tooling for the CLI and never returned by `load_engine()`.

### Output: results on stdout, everything else on stderr

- **Why:** `ox-zero best ... | other-tool` and `ox-zero analyze ... --json | jq` must always receive clean data, so progress bars, notices, and errors go to stderr.
- **Colour:** decided by Rich, never by our code. Rich drops styling when the stream isn't a terminal or `NO_COLOR` is set.
- **Separation:** rendering (`cli/render.py`) builds Rich objects without printing anything, and JSON (`cli/report.py`) never goes through Rich. The machine-readable output can't be affected by terminal capabilities.

### Sandbox: logic separate from the screen

- `cli/sandbox/session.py` holds the position, undo/redo history, and command language, with no UI. That makes every command unit-testable with plain function calls.
- `cli/sandbox/app.py` is the Textual screen: a thin layer that feeds input into `Session` and redraws from it.
- The engine runs in a background thread. Every position change cancels the search and starts a new one, and snapshots belonging to an old position are dropped.
- **Constraint for any engine plugged in:** the thread shares Python's GIL with the UI. A search that computes without ever releasing it will make the screen lag. Measured with the real search (2026-10-09, uniform priors): about 13 ms from a keypress to the screen and 50–70 ms to play a move, against a few milliseconds with the placeholder. Noticeable on paper, fine in use.

### Tooling

- [uv](https://docs.astral.sh/uv/) manages the environment and dependencies.
- The CLI is built on [Typer](https://typer.tiangolo.com/) (commands and flags) and [Rich](https://rich.readthedocs.io/) (styled output). The sandbox is built on [Textual](https://textual.textualize.io/) (full-screen terminal apps).
- Tests use pytest. pytest-asyncio is needed for Textual's async test harness.
- Development is test-driven: tests are written first and seen to fail before the implementation.

## Designed but not built

The `training` package has a complete design in [docs/design](design/README.md): the training loop ([training.md](design/training.md)) and its place in the code structure ([engineering.md](design/engineering.md)). The `engine` it drives is built (search, network, and the evaluator seam between them, designed in [search.md](design/search.md), [network.md](design/network.md) and [engineering.md](design/engineering.md)). Improvements deferred from version 1 are in [upgrades.md](design/upgrades.md), and constants that wait on experiments in [open-questions.md](design/open-questions.md).

The engine is connected to the CLI ([plan 03](plans/03-cli-adapter.md), [cli-integration.md](design/cli-integration.md)): `Analysis` carries the engine's chosen move, and the placeholder is test tooling only.

## Deliberately undecided

- **`gui`:** everything.
