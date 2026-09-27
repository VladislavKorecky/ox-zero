"""Rich renderables for positions and analyses.

Shared by `analyze`, live analysis, and the sandbox, so every view draws the
board the same way. Every function here is pure: it takes game/engine values
and returns a Rich `Text` (or a group of them). Nothing is printed here, and
nothing decides whether colour is used. That is the `Console`'s job: Rich
drops all styling automatically when output is not a terminal or `NO_COLOR`
is set, so the plain-text layout below is exactly what pipes receive.

Layout (from the README): a 4-character row label, then one right-aligned
4-character column per cell, so `100` still fits with a space before it.
"""

from __future__ import annotations

from rich.color import Color
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text

from ox_zero.engine import Analysis
from ox_zero.game import Cell, Mark, State, format_cell, is_draw

CELL_WIDTH = 4
EVAL_BAR_WIDTH = 25

# Explicit RGB rather than named ANSI colours: named colours are remapped by
# every terminal theme (bright cyan can come out a muddy teal), while RGB is
# stable and Rich downgrades it to the nearest 256/16 colour when needed.
MARK_STYLES = {
    Mark.X: Style(color="#4fc3f7", bold=True),  # sky blue
    Mark.O: Style(color="#ff6b9a", bold=True),  # pink
}
_LABEL = Style(dim=True)
_EMPTY_DOT = Style(dim=True)
_SCORE_TEXT = Style(color="grey93")
_TOP_TEXT = Style(color="bright_white", bold=True)
_WINNING_CELL = Style(color="black", bgcolor="#ffd54f", bold=True)

# Heatmap colour stops: (score, RGB). Deliberately dark tints, so light text
# stays readable on top and the board doesn't glare. Scores in between are
# linearly interpolated between the two nearest stops.
_HEAT_STOPS = (
    (0.0, (120, 28, 38)),   # red: bad for the side to move
    (0.5, (135, 105, 20)),  # amber: even
    (1.0, (28, 120, 62)),   # green: good for the side to move
)

# Brighter versions of the same stops, for text and bars drawn on the
# terminal's own (usually dark) background, where the dark tints would vanish.
_HEAT_STOPS_BRIGHT = (
    (0.0, (239, 96, 96)),
    (0.5, (236, 190, 64)),
    (1.0, (96, 214, 128)),
)

# Partial blocks in eighths, for a smooth eval bar: index k is k/8 of a cell.
_EIGHTHS = " ▏▎▍▌▋▊▉"


def heat_color(score: float, *, bright: bool = False) -> Color:
    """The heatmap colour for a win probability in [0, 1].

    `bright=True` gives the foreground variant, for text and bars.
    """
    score = min(1.0, max(0.0, score))
    stops = _HEAT_STOPS_BRIGHT if bright else _HEAT_STOPS
    for (lo, lo_rgb), (hi, hi_rgb) in zip(stops, stops[1:]):
        if score <= hi:
            t = (score - lo) / (hi - lo)
            r, g, b = (round(a + (b - a) * t) for a, b in zip(lo_rgb, hi_rgb))
            return Color.from_rgb(r, g, b)
    raise AssertionError("unreachable: score is clamped to [0, 1]")


def percent(score: float) -> int:
    return round(score * 100)


def board(
    state: State,
    analysis: Analysis | None,
    *,
    top_n: int = 3,
    last_move: Cell | None = None,
) -> Text:
    """The board grid with a score in every empty cell.

    Args:
        analysis: Scores to show. `None` shows `.` in empty cells instead
            (the sandbox while paused, or a finished game).
        top_n: How many of the best-scoring cells to emphasise.
        last_move: Cell to underline, if known. A board string carries no
            move order, so callers pass `None` in that case.
    """
    top = {cell for cell, _ in analysis.top(top_n)} if analysis else set()
    winning = {cell for line in state.winning_lines for cell in line}

    text = Text(no_wrap=True)
    text.append(" " * CELL_WIDTH)
    for col in range(state.size):
        text.append(f"{col:>{CELL_WIDTH}}", _LABEL)

    for row in range(state.size):
        text.append("\n")
        text.append(f"{row:>3} ", _LABEL)
        for col in range(state.size):
            cell = (row, col)
            # Each cell is drawn as padding + content. `background` covers the
            # whole cell (so heatmap blocks touch), while `style` applies to
            # the content only, so an underline marks just the glyph.
            mark = state[cell]
            background = Style()
            if mark is not None:
                content, style = mark.value, MARK_STYLES[mark]
                if cell in winning:
                    background = style = _WINNING_CELL
            elif analysis is not None and cell in analysis.scores:
                score = analysis.scores[cell]
                content = str(percent(score))
                background = Style(bgcolor=heat_color(score))
                style = background + (_TOP_TEXT if cell in top else _SCORE_TEXT)
            else:
                content, style = ".", _EMPTY_DOT
            if cell == last_move:
                style += Style(underline=True)
            text.append(" " * (CELL_WIDTH - len(content)), background)
            text.append(content, style)
    return text


def eval_bar(value: float, side: Mark) -> Text:
    """`Eval ███████████▊░░░░ 47% for O`: the position's value as a bar.

    How the bar is drawn: the filled length is `value * width` cells. The
    whole part is drawn with full blocks, the fractional part with one of
    the eighth-width block characters, and the rest with light shade.
    """
    eighths = round(min(1.0, max(0.0, value)) * EVAL_BAR_WIDTH * 8)
    full, part = divmod(eighths, 8)
    filled = "█" * full + (_EIGHTHS[part] if part else "")
    empty = "░" * (EVAL_BAR_WIDTH - len(filled))

    text = Text()
    text.append("Eval ", "bold")
    text.append(filled, Style(color=heat_color(value, bright=True)))
    text.append(empty, _LABEL)
    text.append(f" {percent(value)}%", Style(bold=True))
    text.append(" for ")
    text.append(side.value, MARK_STYLES[side])
    return text


def top_list(analysis: Analysis, n: int) -> Text:
    """The ranked candidate moves: `  1. 5,7   47%`."""
    ranked = analysis.top(n)
    rank_width = len(str(len(ranked)))
    text = Text()
    text.append(f"Top {n}", "bold")
    for rank, (cell, score) in enumerate(ranked, start=1):
        text.append(f"\n  {rank:>{rank_width}}. ")
        # 5 characters fit the widest cell on a 12x12 board, `10,11`.
        text.append(f"{format_cell(cell):<5}", Style(bold=rank == 1))
        text.append(f" {percent(score):>2}%", Style(color=heat_color(score, bright=True), bold=True))
    return text


def status_line(state: State, right: Text | str | None = None) -> Text:
    """`O to move (3 marks on board)`, optionally followed by engine status."""
    marks = state.move_count
    text = Text()
    text.append(state.to_move.value, MARK_STYLES[state.to_move])
    text.append(f" to move ({marks} mark{'' if marks == 1 else 's'} on board)")
    if right is not None:
        text.append("      ")
        text.append(right)
    return text


def game_over(state: State) -> Text:
    """`Game over: O wins (X O X at 5,5 5,6 5,7)` or `Game over: draw (board full)`."""
    text = Text()
    text.append("Game over: ", "bold")
    if is_draw(state):
        text.append("draw (board full)")
        return text
    assert state.winner is not None
    text.append(state.winner.value, MARK_STYLES[state.winner])
    text.append(" wins (")
    for index, line in enumerate(state.winning_lines):
        if index:
            text.append("; ")
        for position, cell in enumerate(line):
            mark = state[cell]
            assert mark is not None
            text.append(("" if position == 0 else " ") + mark.value, MARK_STYLES[mark])
        text.append(" at " + " ".join(format_cell(cell) for cell in line))
    text.append(")")
    return text


def report(
    state: State,
    analysis: Analysis | None,
    *,
    top_n: int = 3,
    last_move: Cell | None = None,
    status: Text | str | None = None,
) -> RenderableType:
    """The full `analyze` report: status, board, eval, and top candidates.

    For a finished game: the game-over line and the board with the winning
    line highlighted. `status` is extra text for the right of the status line
    (e.g. the simulation count in live mode).
    """
    if analysis is None:
        return Group(game_over(state), "", board(state, None, last_move=last_move))
    return Group(
        status_line(state, status),
        "",
        board(state, analysis, top_n=top_n, last_move=last_move),
        "",
        eval_bar(analysis.value, state.to_move),
        "",
        top_list(analysis, top_n),
    )
