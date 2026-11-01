"""
Shared terminal look for every CLI surface (one-shot runs, the REPL, the
setup wizard, the pipeline's progress output).

One theme and one Console, so a green tick, a warning, or a panel looks the
same no matter which module printed it. This module is a leaf: it must not
import config, storage, or the pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme({
    "brand": "bold cyan",     # titles, the wordmark
    "cmd": "bold cyan",       # commands and flags the user can type
    "arg": "yellow",          # placeholders: ID, QUESTION, {a,b,c}
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "muted": "dim",
})

# highlight=False: rich's auto-highlighter colors every number and quoted
# string, which fights the explicit palette above.
console = Console(theme=THEME, highlight=False)
err_console = Console(theme=THEME, highlight=False, stderr=True)

_UNICODE = {
    "ok": "✓", "fail": "✗", "warn": "!", "bullet": "•", "arrow": "›",
    "dot": "·", "branch": "↳", "on": "●", "off": "○", "mark": "◆", "dash": "—",
}
_ASCII = {
    "ok": "+", "fail": "x", "warn": "!", "bullet": "*", "arrow": ">",
    "dot": "-", "branch": "^", "on": "*", "off": "o", "mark": "*", "dash": "-",
}


def glyph(name: str) -> str:
    """Unicode symbol, or an ASCII stand-in when the terminal can't show it."""
    enc = (console.encoding or "").lower().replace("-", "")
    return (_UNICODE if enc.startswith("utf") else _ASCII)[name]


# --- one-line messages --------------------------------------------------
# Callers escape() any user-controlled text they interpolate; these helpers
# deliberately accept rich markup so a message can embed [cmd]...[/cmd].

_problems = 0


def problem_count() -> int:
    """How many warn()/fail() lines have been printed so far. The REPL compares
    before/after a command to tell a clean success (nothing to show) from one
    that has something the user needs to read."""
    return _problems


def ok(msg: str) -> None:
    console.print(f"[ok]{glyph('ok')}[/ok] {msg}")


def warn(msg: str) -> None:
    global _problems
    _problems += 1
    console.print(f"[warn]{glyph('warn')}[/warn] {msg}")


def fail(msg: str, hint: Optional[str] = None) -> None:
    global _problems
    _problems += 1
    console.print(f"[err]{glyph('fail')}[/err] {msg}")
    if hint:
        console.print(f"  [muted]{hint}[/muted]")


def note(msg: str) -> None:
    console.print(f"[muted]{msg}[/muted]")


def section(title: str) -> None:
    """A left-aligned rule with a title: `Findings ─────────────`."""
    console.print()
    console.rule(f"[brand]{title}[/brand]", align="left", style="muted")


# --- building blocks ----------------------------------------------------

def panel(body, title: Optional[str] = None, border: str = "cyan") -> Panel:
    return Panel(
        body,
        title=f"[brand]{title}[/brand]" if title else None,
        title_align="left",
        border_style=border,
        padding=(0, 1),
    )


def kv_grid(rows: Sequence[tuple[str, object]], key_style: str = "muted") -> Table:
    """Two-column key/value grid (keys dim, values as given)."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=key_style, no_wrap=True)
    grid.add_column()
    for key, value in rows:
        grid.add_row(key, value)
    return grid


def numbered(items: Sequence[str]) -> Table:
    """A right-aligned, dim-numbered list. Items are plain text (not markup)."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="muted", no_wrap=True)
    grid.add_column()
    for i, item in enumerate(items, 1):
        grid.add_row(f"{i}.", Text(item))
    return grid


def bullets(items: Sequence[str]) -> Table:
    """A dim-bulleted list. Items are plain text (not markup)."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="muted", no_wrap=True)
    grid.add_column()
    for item in items:
        grid.add_row(glyph("bullet"), Text(item))
    return grid


# --- formatting helpers -------------------------------------------------

def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def short_path(path: Path) -> str:
    """Display form of a path: relative to cwd when inside it, else ~-collapsed."""
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        pass
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def truncate(text: str, width: int) -> str:
    text = " ".join(text.split())  # collapse newlines/tabs from stored queries
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def tty() -> bool:
    import sys
    return sys.stdin is not None and sys.stdin.isatty()


def no_color() -> bool:
    return bool(os.environ.get("NO_COLOR"))


__all__ = [
    "console", "err_console", "glyph", "problem_count", "ok", "warn", "fail", "note", "section", "bullets",
    "panel", "kv_grid", "numbered", "fmt_duration", "short_path", "truncate", "plural",
    "tty", "escape",
]
