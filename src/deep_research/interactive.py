"""
Transient terminal UI for the REPL: the option picker and the closable viewer.

Both draw below the cursor and erase themselves when they finish, so choosing
something or reading a screen leaves nothing behind in the scrollback.
"""
from __future__ import annotations

import re
import shutil
from typing import Optional

from .ui import glyph

# Rich emits OSC-8 hyperlink escapes for linked text; prompt_toolkit's ANSI
# parser doesn't understand them, so drop them before parsing.
_OSC8 = re.compile(r"\x1b\]8;[^\x1b\x07]*(?:\x1b\\|\x07)")


def pick_option(
    title: str, choices: list, current: Optional[str] = None, *, input=None, output=None,
) -> Optional[str]:
    """Arrow-key selector, like Claude Code's menus.

    ↑/↓ (or j/k) move, Enter selects, 1-9 jump straight to an item, Esc or
    Ctrl-C cancels. Returns the chosen value, or None when cancelled. The
    list erases itself when done, so only the command's confirmation remains.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import FormattedTextControl, Layout, Window
    from prompt_toolkit.styles import Style

    values = [value for value, _ in choices]
    state = {"index": values.index(current) if current in values else 0}
    on, off = glyph("on"), glyph("off")
    width = max((len(v) for v in values), default=0)

    def render():
        frags = [("class:title", f" {title}\n")]
        for i, (value, blurb) in enumerate(choices):
            here = i == state["index"]
            frags.append(("class:pointer", " ❯ " if here else "   "))
            frags.append(("class:current" if value == current else "class:dot", f"{on if value == current else off} "))
            frags.append(("class:selected" if here else "class:item", value.ljust(width)))
            if blurb:
                frags.append(("class:blurb", f"  {blurb}"))
            frags.append(("", "\n"))
        frags.append(("class:hint", " ↑/↓ move · Enter select · 1-9 jump · Esc cancel"))
        return frags

    kb = KeyBindings()

    def move(step):
        def _handler(event):
            state["index"] = (state["index"] + step) % len(values)
        return _handler

    for key in ("up", "k", "c-p"):
        kb.add(key)(move(-1))
    for key in ("down", "j", "c-n"):
        kb.add(key)(move(1))

    @kb.add("enter")
    def _select(event):
        event.app.exit(result=values[state["index"]])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("c-d")
    @kb.add("q")
    def _cancel(event):
        event.app.exit(result=None)

    def _jump(n):
        def _handler(event):
            event.app.exit(result=values[n])
        return _handler

    for n in range(min(len(values), 9)):
        kb.add(str(n + 1))(_jump(n))

    app = Application(
        layout=Layout(Window(FormattedTextControl(render, show_cursor=False), dont_extend_height=True)),
        key_bindings=kb,
        style=Style.from_dict({
            "title": "bold ansicyan",
            "pointer": "bold ansicyan",
            "selected": "bold ansicyan",
            "item": "",
            "current": "ansigreen",
            "dot": "ansibrightblack",
            "blurb": "ansibrightblack",
            "hint": "ansibrightblack",
        }),
        full_screen=False,
        erase_when_done=True,
        input=input,
        output=output,
    )
    app.ttimeoutlen = 0.05  # make a lone Esc register promptly
    if not values:
        return None
    return app.run()


def view_lines(text: str) -> list:
    """Captured console output (with ANSI colors) -> a list of lines, each a
    list of (style, text) fragments, without leading/trailing blank lines."""
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text

    lines: list = [[]]
    for frag in to_formatted_text(ANSI(_OSC8.sub("", text))):
        style, chunk = frag[0], frag[1]
        for i, part in enumerate(chunk.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append((style, part))

    def blank(line):
        return not "".join(t for _, t in line).strip()

    while lines and blank(lines[0]):
        lines.pop(0)
    while lines and blank(lines[-1]):
        lines.pop()
    return lines


def show_view(text: str, *, rows: Optional[int] = None, input=None, output=None) -> None:
    """Show captured output in a closable, scrollable viewer.

    Enter, Esc or q closes it (Ctrl-C too); ↑/↓, j/k, PgUp/PgDn, g/G scroll
    when the text is taller than the screen. The viewer erases itself on exit,
    so nothing from the command stays on screen.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import FormattedTextControl, Layout, Window
    from prompt_toolkit.styles import Style

    lines = view_lines(text)
    if not lines:
        return
    rows = rows or shutil.get_terminal_size().lines
    height = max(3, min(len(lines), rows - 3))
    top_max = max(0, len(lines) - height)
    scroll = {"top": 0}

    def render():
        top = scroll["top"]
        frags: list = []
        for line in lines[top:top + height]:
            frags.extend(line)
            frags.append(("", "\n"))
        if top_max:
            frags.append(("class:hint", f" ↑/↓ scroll · Enter/Esc close   lines {top + 1}-{top + height} of {len(lines)}"))
        else:
            frags.append(("class:hint", " Enter / Esc to close"))
        return frags

    def move(delta=None, to=None):
        def _handler(event):
            if to is not None:
                scroll["top"] = to if to >= 0 else top_max
            else:
                scroll["top"] = min(top_max, max(0, scroll["top"] + delta))
        return _handler

    kb = KeyBindings()
    for key in ("up", "k", "c-p"):
        kb.add(key)(move(-1))
    for key in ("down", "j", "c-n"):
        kb.add(key)(move(1))
    for key in ("pageup", "b"):
        kb.add(key)(move(-height))
    for key in ("pagedown", "space", "f"):
        kb.add(key)(move(height))
    kb.add("home")(move(to=0))
    kb.add("g")(move(to=0))
    kb.add("end")(move(to=-1))
    kb.add("G")(move(to=-1))

    @kb.add("enter")
    @kb.add("escape", eager=True)
    @kb.add("q")
    @kb.add("c-c")
    @kb.add("c-d")
    def _close(event):
        event.app.exit(result=None)

    app = Application(
        layout=Layout(Window(FormattedTextControl(render, show_cursor=False), dont_extend_height=True)),
        key_bindings=kb,
        style=Style.from_dict({"hint": "ansibrightblack"}),
        full_screen=False,
        erase_when_done=True,
        input=input,
        output=output,
    )
    app.ttimeoutlen = 0.05  # make a lone Esc register promptly
    app.run()


def ask_text(
    title: str, *, hint: str = "", secret: bool = False, initial: str = "",
    esc: str = "cancel", input=None, output=None,
) -> Optional[str]:
    """One-line text prompt that erases itself when done.

    Enter submits (an empty line returns ""); Esc or Ctrl-C cancels (None).
    `esc` is only the word the footer uses for what Esc does ("cancel", "skip").
    `secret=True` masks what is typed, for API keys. Pasting works.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.layout.processors import PasswordProcessor
    from prompt_toolkit.styles import Style

    buffer = Buffer(multiline=False)
    if initial:
        buffer.insert_text(initial)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.app.exit(result=buffer.text)

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    processors = [PasswordProcessor()] if secret else []
    layout = Layout(HSplit([
        Window(FormattedTextControl([("class:title", f" {title}")]), height=1),
        VSplit([
            Window(FormattedTextControl([("class:pointer", " ❯ ")]), width=3, height=1),
            Window(BufferControl(buffer=buffer, input_processors=processors), height=1),
        ]),
        Window(FormattedTextControl(
            [("class:hint", f" {hint + ' · ' if hint else ''}Enter to confirm · Esc to {esc}")]
        ), height=1),
    ]), focused_element=None)
    app = Application(
        layout=layout,
        key_bindings=kb,
        style=Style.from_dict({"title": "bold ansicyan", "pointer": "bold ansicyan", "hint": "ansibrightblack"}),
        full_screen=False,
        erase_when_done=True,
        input=input,
        output=output,
    )
    app.ttimeoutlen = 0.05
    app.layout.focus(layout.container.children[1].children[1])
    return app.run()
