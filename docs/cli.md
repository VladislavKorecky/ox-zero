# CLI specification

This is the specification of the `ox-zero` command line: every command, flag, input format, and output. The implementation in `src/ox_zero/cli/` follows it, and any change to the CLI's behaviour starts here.

> The CLI runs the AlphaZero search. Until the training pipeline (roadmap step 4) produces a checkpoint, it searches with uniform priors: it finds short tactics, but has no positional judgement. The CLI says which engine it loaded on stderr each time it starts.

The command is `ox-zero`. `analyze` and `best` take a position, print a human-readable report by default, and print JSON with `--json`. `sandbox` is an interactive screen.

## Positions and moves

- Cells are written `row,col`, zero-based. Row 0 is the top row, column 0 is the left column. This convention is used everywhere: CLI, code, and tests.
- A position is given either as a **move list** or as a **board string**. The format is auto-detected.
  - Move list: cells in the order they were played, X first, e.g. `5,5 6,6 5,6`.
  - Board string: 144 characters, row 0 first, `X`, `O`, or `_` for empty, e.g. `____...X..._` (12 rows × 12 columns).
- The side to move is derived from the mark counts. X always moves first, so equal counts mean X to move and one extra X means O to move. Any other count is rejected as illegal.
- A board string does not record move order, so a decided board is read as follows: the side that moved last (the side *not* to move) placed the winning mark. The board is rejected as unreachable if no single mark of the last mover lies on every alternating line, because one move must have completed all of them at once.

## Commands

| Command | Description |
|---------|-------------|
| `ox-zero analyze <position>` | Evaluate a position and score every legal move. |
| `ox-zero best <position>` | Print only the engine's chosen move. |
| `ox-zero sandbox [<position>]` | Interactive analysis screen. See below. |

Flags. Not every flag makes sense for every command; the last three columns say where each one applies.

| Flag | Description | `analyze` | `best` | `sandbox` |
|------|-------------|:---:|:---:|:---:|
| `--simulations N` | Tree-search playouts per analysis. Strength versus speed. Default 800. With `--live`, an optional cap (no cap by default). | ✓ | ✓ | |
| `--live` | Keep analysing and stream reports until interrupted. See [Live analysis](#live-analysis). | ✓ | | |
| `--model PATH` | Checkpoint to load. Defaults to the newest in `checkpoints/`. With no checkpoint available, the search runs with uniform priors and a notice says so. A path that does not exist, a file that is not a readable checkpoint, and a checkpoint for another board size are errors (status 2). | ✓ | ✓ | ✓ |
| `--top N` | Number of candidate moves in the top section. Default 3. | ✓ | | ✓ |
| `--device NAME` | Where the network runs: `cpu` (default), `mps`, `cuda`, or `auto` for the best available. Ignored when no model is loaded. `cpu` is the default because analysis evaluates one position at a time, where a GPU is slower (see [engineering.md](design/engineering.md#measured-2026-09-27)). A device this machine lacks is an error (status 2). | ✓ | ✓ | ✓ |
| `--seed N` | Fix randomness for reproducible output. The real engine's analysis is deterministic; the seed only affects engines that sample. | ✓ | ✓ | ✓ |
| `--json` | Machine-readable output instead of text. | ✓ | ✓ | |

Why the gaps:

- `--simulations` does not apply to the sandbox because analysis there is open-ended. The engine keeps searching until the position changes or analysis is paused, so there is no fixed playout budget to set.
- `--live` does not apply to `best`, which exists to print one move and exit, nor to the sandbox, which is always live.
- `--top` does not apply to `best` because `best` prints exactly one move. For a ranked list of candidates, use `analyze --top N`.
- `--json` does not apply to the sandbox because it is an interactive screen, not a report.

## Output conventions

- **stdout is for results, stderr is for everything else.** Progress bars, notices, and error messages go to stderr, so `ox-zero best ... | other-tool` and `ox-zero analyze ... --json | jq` always see clean output.
- **Colour only on a terminal.** Colour and styling are used when the output is a terminal and dropped when it is piped or redirected. The [`NO_COLOR`](https://no-color.org/) environment variable turns colour off everywhere.
- `best` output and all `--json` output are always plain.
- Invalid input exits with status 2 and an error that echoes the input with a caret under the offending token:

  ```
  $ ox-zero analyze 5,5 6,6 5,5
  Error: cannot play 5,5: cell is occupied
    5,5 6,6 5,5
            ^^^
  ```

## `analyze`

Scores are the engine's estimated win probability for the side to move after playing that cell, as a percentage. Every legal move gets a score, shown in place on the board. Occupied cells show their mark. Cells are fixed-width so the grid keeps its shape regardless of the values.

On a colour terminal the board is a heatmap: each empty cell's background runs from red (low score) through yellow to green (high score). X and O are drawn in two distinct bold colours, the top N candidates are emphasised, the last move played is underlined, and a winning line is highlighted. The eval line includes a bar showing the win probability for the side to move.

While the search runs, a progress bar with the simulation count and rate is shown on stderr. It disappears once the report is printed, and it is not shown when stderr is not a terminal.

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

Eval ███████████▊░░░░░░░░░░░░░ 47% for O

Top 3
  1. 5,7   47%
  2. 4,6   38%
  3. 6,5   36%
```

> The numbers in this example are made up to show the layout. They are not the output of any engine, and a trained model will produce different values.

If the position is already decided, the report says so instead of running the engine, and shows the board with the winning line highlighted:

```
$ ox-zero analyze 5,5 6,6 5,7 5,6
Game over: O wins (X O X at 5,5 5,6 5,7)

       0   1   2   3   4   5   6   7   8   9  10  11
  0    .   .   .   .   .   .   .   .   .   .   .   .
  ...
  5    .   .   .   .   .   X   O   X   .   .   .   .
  6    .   .   .   .   .   .   O   .   .   .   .   .
  ...
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

### Live analysis

Without `--live`, `analyze` runs `--simulations` playouts, prints one report, and exits. With `--live`, the engine keeps searching and the report is refreshed as the search deepens, roughly twice a second. It runs until you press Ctrl-C, or until the `--simulations` cap is reached if one is given.

In text mode the report is redrawn in place, so the terminal shows one board whose numbers settle over time, with the simulation count (and the cap, if any) on the status line.

With `--json`, the output is [JSON Lines](https://jsonlines.org/): every refresh writes one complete report object on its own line, in the same shape as above plus a `simulations` count. A consumer reads the stream line by line and keeps the most recent object; there is nothing to reassemble. When the cap is reached, the last line is the final report and the process exits with status 0. On Ctrl-C the stream simply ends, with the conventional status 130.

```
$ ox-zero analyze 5,5 6,6 5,6 --live --json
{"simulations": 400, "board": "____...X..._", "to_move": "O", "value": 0.51, "moves": [...], "top": [...], "result": null}
{"simulations": 1200, "board": "____...X..._", "to_move": "O", "value": 0.48, "moves": [...], "top": [...], "result": null}
{"simulations": 2000, "board": "____...X..._", "to_move": "O", "value": 0.47, "moves": [...], "top": [...], "result": null}
```

## `best`

Prints the chosen move and nothing else, so the output can be fed straight into another command.

The chosen move is the one the search visited most, which is what AlphaZero plays. It can differ from the highest score shown by `analyze`: a move visited only a few times can have a lucky high average, while the visit count reflects how strongly the search kept preferring a move.

```
$ ox-zero best 5,5 6,6 5,6
5,7
```

With `--json`: `{"move": [5, 7], "score": 0.47}`.

If the position is already decided there is no move to print. `best` then writes the game-over line to stderr, or `{"move": null, "score": null, "result": "O"}` with `--json`, and exits with status 1 so scripts notice.

## `sandbox`

An interactive screen for studying the game, in the spirit of a chess GUI's analysis mode. The sandbox takes over the terminal with a single view that is redrawn in place; there is no scrolling transcript. You play moves for both sides, and the engine analyses the current position continuously in the background, updating the scores on screen as its search deepens. Start from an empty board or from a given position.

The screen shows:

- **Status line** (top). Side to move, mark count, and the engine state: `analysing` with the number of simulations so far, or `paused`.
- **Board** (left). The same heatmap board as `analyze`, with a score in every empty cell. While analysis is paused, empty cells show `.` instead of a score.
- **Side panel** (right of the board). Eval bar and top N candidates as in `analyze`, then the move history: the moves played so far, numbered in X-and-O pairs as in chess notation.
- **Message line** (below the board). Feedback for the last command: errors, the output of `export`, or nothing.
- **Prompt** (bottom). Where you type commands, above a footer listing the keyboard shortcuts.

```
$ ox-zero sandbox 5,5 6,6 5,6

O to move (3 marks on board)      analysing: 12,400 simulations

       0   1   2   3   4 …  11         Eval ███████████▊░░░░░░░░░░░░░ 47% for O
  0   12  10   9   8   8 …  12
  …                                    Top 3
  5    8   7   6   5  15   X   X  47     1. 5,7   47%
  6    8   7   6   5  16  36   O  29     2. 4,6   38%
  …                                      3. 6,5   36%

                                       Moves: 1. 5,5 6,6  2. 5,6

unknown command 'hello' (type help for the list)

┌──────────────────────────────────────────────────────────────┐
│ row,col to play  ·  help for commands                        │
└──────────────────────────────────────────────────────────────┘
 ^q Quit  ^z Undo  ^y Redo  ^p Pause/resume
```

Whenever the position changes (a move is played or undone, a position is loaded, the board is reset), the engine drops its current search and starts on the new position. The screen redraws at once with the new board, and the scores fill in and settle as the search progresses. If the position is decided, the engine stops and the status line shows the result instead, e.g. `Game over: O wins (X O X at 5,5 5,6 5,7)`.

Commands available inside the sandbox:

| Input | Effect |
|-------|--------|
| `row,col` | Play a move for the side to move. |
| `undo` / `redo` | Step back through the move history, or forward again. Playing a new move discards the redo history. |
| `pause` / `resume` | Pause or resume continuous analysis. While paused, the board and history still update as you play, but no scores are shown. |
| `load <position>` | Jump to a position, given as a move list or board string. |
| `reset` | Clear to an empty board. |
| `export` | Show the current position as a board string in the message line, ready to paste into `analyze`. |
| `help` | List the commands in the message line. |
| `quit` | Leave the sandbox. `exit` works too. |

A position loaded as a board string has no move history. It becomes the starting point: `undo` stops there, and the move history lists only the moves played after it.

Shortcuts, for when typing a command is slower than pressing a key:

| Key / action | Effect |
|-------|--------|
| Click an empty cell | Play that cell. |
| `ctrl+z` / `ctrl+y` | `undo` / `redo`. |
| `ctrl+p` | Toggle `pause` / `resume`. |
| `ctrl+q` | `quit`. |
