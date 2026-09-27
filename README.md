# ox-zero

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
2. CLI for playing the game, later extended with engine analysis and play against the engine.
3. Monte Carlo tree search and neural network.
4. Self-play training pipeline.
5. GUI for interactive analysis.

## Usage

> **Not implemented yet.** This section is the design specification for the CLI (roadmap step 2). It is written as if the tool existed so that the implementation has a fixed target.

The command is `ox-zero`. `analyze` and `best` take a position, print a human-readable report by default, and print JSON with `--json`. `sandbox` is an interactive screen.

### Positions and moves

- Cells are written `row,col`, zero-based. Row 0 is the top row, column 0 is the left column. This convention is used everywhere: CLI, code, and tests.
- A position is given either as a **move list** or as a **board string**. The format is auto-detected.
  - Move list: cells in the order they were played, X first, e.g. `5,5 6,6 5,6`.
  - Board string: 144 characters, row 0 first, `X`, `O`, or `_` for empty, e.g. `____...X..._` (12 rows × 12 columns).
- The side to move is derived from the mark counts. X always moves first, so equal counts mean X to move and one extra X means O to move. Any other count is rejected as illegal.

### Commands

| Command | Description |
|---------|-------------|
| `ox-zero analyze <position>` | Evaluate a position and score every legal move. |
| `ox-zero best <position>` | Print only the engine's chosen move. |
| `ox-zero sandbox [<position>]` | Interactive analysis screen. See below. |

Flags. Not every flag makes sense for every command; the last three columns say where each one applies.

| Flag | Description | `analyze` | `best` | `sandbox` |
|------|-------------|:---:|:---:|:---:|
| `--simulations N` | Tree-search playouts per analysis. Strength versus speed. | ✓ | ✓ | |
| `--model PATH` | Checkpoint to load. Defaults to the newest in `checkpoints/`. | ✓ | ✓ | ✓ |
| `--top N` | Number of candidate moves in the top section. Default 3. | ✓ | | ✓ |
| `--seed N` | Fix randomness for reproducible output. | ✓ | ✓ | ✓ |
| `--json` | Machine-readable output instead of text. | ✓ | ✓ | |

Why the gaps:

- `--simulations` does not apply to the sandbox because analysis there is open-ended. The engine keeps searching until the position changes or analysis is switched off, so there is no fixed playout budget to set.
- `--top` does not apply to `best` because `best` prints exactly one move. For a ranked list of candidates, use `analyze --top N`.
- `--json` does not apply to the sandbox because it is an interactive screen, not a report.

### `analyze`

Scores are the engine's estimated win probability for the side to move after playing that cell, as a percentage. Every legal move gets a score, shown in place on the board. Occupied cells show their mark. Cells are fixed-width so the grid keeps its shape regardless of the values.

```
$ ox-zero analyze 5,5 6,6 5,6

O to move (3 marks on board)

       0   1   2   3   4   5   6   7   8   9  10  11
  0   12  10   9   8   8   8   8   8   9  10  11  12
  1   10   9   8   7   7   7   7   7   8   9  10  11
  2    9   8   7   6   6   6   6   6   7   8   9  10
  3    8   7   6   5   5   9   9   5   6   7   8   9
  4    8   7   6   5  14  31  38  14   6   7   8   9
  5    8   7   6   5  15   X   X  47   6   7   8   9
  6    8   7   6   5  16  36   O  29   6   7   8   9
  7    8   7   6   5   9  12  16   9   6   7   8   9
  8    9   8   7   6   6   6   6   6   7   8   9  10
  9   10   9   8   7   7   7   7   7   8   9  10  11
 10   11  10   9   8   8   8   8   8   9  10  11  12
 11   12  11  10   9   9   9   9   9  10  11  12  13

Eval: 47% for O

Top 3
  1. 5,7   47%
  2. 4,6   38%
  3. 6,5   36%
```

> The numbers in this example are made up to show the layout. They are not the output of any engine, and a trained model will produce different values.

If the position is already decided, the report says so instead of running the engine:

```
$ ox-zero analyze 5,5 6,6 5,6 5,7
Game over: O wins (X O X at 5,5 5,6 5,7)
```

With `--json`, the same information is emitted as one object:

```json
{
  "board": "____...X..._",
  "to_move": "O",
  "value": 0.47,
  "moves": [{"move": [0, 0], "score": 0.12}, {"move": [0, 1], "score": 0.10}],
  "top": [{"move": [5, 7], "score": 0.47}, {"move": [4, 6], "score": 0.38}, {"move": [6, 5], "score": 0.36}],
  "result": null
}
```

`moves` lists every legal move in board order (left to right, top to bottom). `result` is `null` for a live position, otherwise `"X"`, `"O"`, or `"draw"`.

### `best`

Prints the chosen move and nothing else, so the output can be fed straight into another command.

```
$ ox-zero best 5,5 6,6 5,6
5,7
```

With `--json`: `{"move": [5, 7], "score": 0.47}`.

### `sandbox`

An interactive screen for studying the game, in the spirit of a chess GUI's analysis mode. The sandbox takes over the terminal with a single view that is redrawn in place; there is no scrolling transcript. You play moves for both sides, and the engine analyses the current position continuously in the background, updating the scores on screen as its search deepens. Start from an empty board or from a given position.

The screen shows, top to bottom:

- **Status line.** Side to move, mark count, and the engine state: `analysing` with the number of simulations so far, or `paused`.
- **Board.** The same board as `analyze`, with a score in every empty cell. While analysis is paused, empty cells show `.` instead of a score.
- **Eval and top N candidates.** As in `analyze`.
- **Move history.** The moves played so far, in order.
- **Message line.** Feedback for the last command: errors, the output of `board`, or nothing.
- **Prompt.** Where you type commands.

```
$ ox-zero sandbox 5,5 6,6 5,6

O to move (3 marks on board)              analysing: 12400 simulations

[board with scores, as in analyze]

Eval: 47% for O

Top 3
  1. 5,7   47%
  2. 4,6   38%
  3. 6,5   36%

Moves: 5,5 6,6 5,6


>
```

Whenever the position changes (a move is played or undone, a position is loaded, the board is reset), the engine drops its current search and starts on the new position. The screen redraws at once with the new board, and the scores fill in and settle as the search progresses. If the position is decided, the engine stops and the status line shows the result instead, e.g. `Game over: O wins (X O X at 5,5 5,6 5,7)`.

Commands available inside the sandbox:

| Input | Effect |
|-------|--------|
| `row,col` | Play a move for the side to move. |
| `undo` / `redo` | Step back through the move history, or forward again. Playing a new move discards the redo history. |
| `auto on` / `auto off` | Turn continuous analysis on or off. With it off, the board and history still update as you play, but no scores are shown. |
| `load <position>` | Jump to a position, given as a move list or board string. |
| `reset` | Clear to an empty board. |
| `board` | Show the current position as a board string in the message line, ready to paste into `analyze`. |
| `quit` | Leave the sandbox. |

## Development

Python 3.14, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # create the virtual environment and install dependencies
uv run pytest    # run the test suite
```

## License

[MIT](LICENSE)
