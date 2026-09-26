# oxox-zero

An [AlphaZero](https://arxiv.org/abs/1712.01815)-style engine for **OXOX**, a two-player board game derived from Tic-tac-toe, together with tooling for analyzing positions and studying the game.

> **Status:** early development. There is no working engine yet; this repository currently holds the project skeleton.

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
2. Monte Carlo tree search and neural network.
3. Self-play training pipeline.
4. CLI for position analysis and play against the engine.
5. GUI for interactive analysis.

## Tech stack

Python. Further dependencies will be documented as the implementation takes shape.

## License

[MIT](LICENSE)
