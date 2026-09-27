# Architecture

How the code is organised, and the decisions behind it that aren't obvious from any single file. For what the game is and where the project is going, see the [README](../README.md). For what the CLI does, see the [CLI specification](cli.md).

Each module's docstring explains its own design in detail. This document is the map that ties them together.

## Package map

Everything importable lives in `src/ox_zero/`. `tests/` mirrors its layout.

| Package | Owns | Status |
|---------|------|--------|
| `game` | OXOX rules (`rules.py`) and text notation for cells and positions (`notation.py`). Pure Python: no NumPy, no tensors. | Done (roadmap step 1) |
| `engine` | Position analysis: the AlphaZero search and network. | Designed, not built (step 3): see [docs/design](design/README.md) |
| `training` | The self-play training pipeline. | Designed, not built (step 4): see [docs/design](design/README.md) |
| `cli` | The `ox-zero` command: commands, rendering, JSON reports, the sandbox, and the engine port. | Done (step 2), on a placeholder engine |
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

- `cli/engine_port.py` defines what the CLI needs: `Analysis` (per-move win probabilities, a position value, and a simulation count) and the `Engine` protocol, whose `search()` yields snapshots that improve over time. Incremental results are a real requirement: `analyze --live` and the sandbox show analysis as it deepens.
- `cli/placeholder.py` holds `PlaceholderEngine`, a stand-in with meaningless numbers, and `load_engine()`, the single point where the CLI obtains an engine.
- Connecting a real engine means writing one adapter in `cli/` that wraps whatever `ox_zero.engine` provides into the `Engine` protocol, and returning it from `load_engine()`. The engine package itself stays free of any CLI-driven design.

### Output: results on stdout, everything else on stderr

- **Why:** `ox-zero best ... | other-tool` and `ox-zero analyze ... --json | jq` must always receive clean data, so progress bars, notices, and errors go to stderr.
- **Colour:** decided by Rich, never by our code. Rich drops styling when the stream isn't a terminal or `NO_COLOR` is set.
- **Separation:** rendering (`cli/render.py`) builds Rich objects without printing anything, and JSON (`cli/report.py`) never goes through Rich. The machine-readable output can't be affected by terminal capabilities.

### Sandbox: logic separate from the screen

- `cli/sandbox/session.py` holds the position, undo/redo history, and command language, with no UI. That makes every command unit-testable with plain function calls.
- `cli/sandbox/app.py` is the Textual screen: a thin layer that feeds input into `Session` and redraws from it.
- The engine runs in a background thread. Every position change cancels the search and starts a new one, and snapshots belonging to an old position are dropped.
- **Constraint for any engine plugged in:** the thread shares Python's GIL with the UI. A search that computes without ever releasing it will make the screen lag.

### Tooling

- [uv](https://docs.astral.sh/uv/) manages the environment and dependencies.
- The CLI is built on [Typer](https://typer.tiangolo.com/) (commands and flags) and [Rich](https://rich.readthedocs.io/) (styled output). The sandbox is built on [Textual](https://textual.textualize.io/) (full-screen terminal apps).
- Tests use pytest. pytest-asyncio is needed for Textual's async test harness.
- Development is test-driven: tests are written first and seen to fail before the implementation.

## Designed but not built

The `engine` and `training` packages have a complete design in [docs/design](design/README.md): the search ([search.md](design/search.md)), the network ([network.md](design/network.md)), the training loop ([training.md](design/training.md)), the code structure ([engineering.md](design/engineering.md)), and how the engine connects to the CLI ([cli-integration.md](design/cli-integration.md)). Improvements deferred from version 1 are in [upgrades.md](design/upgrades.md), and constants that wait on experiments in [open-questions.md](design/open-questions.md).

Two things in the current code are affected once the engine is built:

- `PlaceholderEngine` imitates only how the CLI expects results to *behave* (they arrive over time and settle). It is not a model for the real engine.
- The `Engine` protocol in `cli/engine_port.py` stays the CLI's contract, reached through an adapter. It will gain one field: the engine's chosen move, because AlphaZero picks the most visited move rather than the highest-scoring one (see [cli-integration.md](design/cli-integration.md)).

## Deliberately undecided

- **`gui`:** everything.
