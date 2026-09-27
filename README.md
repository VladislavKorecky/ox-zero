# ox-zero

An [AlphaZero](https://arxiv.org/abs/1712.01815)-style engine for **OXOX**, a two-player board game derived from Tic-tac-toe, together with tooling for analyzing positions and studying the game.

> **Status:** early development. The game rules (roadmap step 1) and the CLI (step 2) are done. The CLI runs on a placeholder engine with meaningless scores until the real one (step 3) exists.

## The game

OXOX is a two-player, zero-sum, perfect-information game played on a 12×12 board (experimental, subject to change). Players take turns placing X and O marks on empty cells, exactly as in Tic-tac-toe. If the board fills up with no winner, the game is a draw.

The difference is the winning condition. In Tic-tac-toe you win by lining up three of your own marks (`XXX` or `OOO`). In OXOX you win by completing an **alternating** line of three: `OXO` or `XOX`, in any row, column, or diagonal.

Ownership of the marks does not matter. Whoever places the mark that completes an alternating pattern wins, regardless of who placed the other two. Since every move can be used by either player toward the same pattern, the game turns into a constant back-and-forth of setting up threats and denying them to the opponent.

## Project goals

- **Engine.** Implement the AlphaZero algorithm (Monte Carlo tree search guided by a self-play-trained neural network) and train it on OXOX from scratch, with no human game data.
- **Analysis tools.** Provide a CLI and a GUI for analyzing board positions, finding optimal moves, and exploring lines of play.
- **Training aid.** Use the engine to help human players understand the game and improve at it.

## Roadmap

1. Game rules, board representation, and move generation.
2. CLI for playing the game, later extended with engine analysis and play against the engine.
3. Monte Carlo tree search and neural network.
4. Self-play training pipeline.
5. GUI for interactive analysis.

## Quickstart

```bash
uv sync                                    # install
uv run ox-zero analyze 5,5 6,6 5,6         # score every move in a position
uv run ox-zero best 5,5 6,6 5,6            # print just the engine's move
uv run ox-zero sandbox                     # interactive analysis screen
```

Positions are move lists (`row,col`, zero-based, X first) or 144-character board strings. See the [CLI specification](docs/cli.md) for every command, flag, and output format.

## Documentation

- [CLI specification](docs/cli.md): commands, flags, input formats, text and JSON output.
- [Architecture](docs/architecture.md): how the code is organised and the key design decisions.

## Development

Python 3.14, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # create the virtual environment and install dependencies
uv run pytest    # run the test suite
```

## License

[MIT](LICENSE)
