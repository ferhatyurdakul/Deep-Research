"""
Command registry and the rich-rendered `--help`.

The registry (`COMMANDS`) is the single list of subcommands: it drives the
top-level help, per-command help, `deep-research help`, and dispatch in
main.py. Option and argument rows are read straight from the argparse
parsers, so help can't drift from what the parser accepts; option *values*
are explained from options.CHOICE_DOCS.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from rich.console import Group
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from . import __version__
from .options import CHOICE_DOCS
from .ui import console, err_console, glyph

PROG = "deep-research"


@dataclass(frozen=True)
class CliCommand:
    name: str
    group: str
    summary: str
    args: str = ""            # usage tail, e.g. "ID [options]"
    details: str = ""         # longer explanation shown by `<command> --help`
    examples: tuple = ()      # ((command line, comment), ...)


GROUPS = ("Session", "Credential", "System")

COMMANDS: tuple = (
    CliCommand(
        "history", "Session", "List recent research sessions", "[-n N]",
        examples=((f"{PROG} history", "the 25 most recent"), (f"{PROG} history -n 100", "")),
    ),
    CliCommand(
        "inspect", "Session", "Show a stored session",
        "ID [--full] [--source N]",
        examples=(
            (f"{PROG} inspect 3", "metadata, sources, iterations, summary"),
            (f"{PROG} inspect 3 --full", "also print the full detailed findings"),
            (f"{PROG} inspect 3 --source 5", "one source: findings and verbatim evidence"),
        ),
    ),
    CliCommand(
        "continue", "Session", "Deepen a stored session",
        "ID [options]",
        details="Runs another gap-analysis pass over an existing session's sources and "
                "saves the result as a new session linked to the original.",
        examples=(
            (f"{PROG} continue 3", ""),
            (f"{PROG} continue 3 --depth deep --no-papers", "go deeper, web sources only"),
        ),
    ),
    CliCommand(
        "resynthesize", "Session", "Re-write a report from stored sources",
        "ID [options]",
        details="No searching: re-reads the stored sources and writes a fresh report, "
                "e.g. in a different style. Saved as a new session linked to the original.",
        examples=(
            (f"{PROG} resynthesize 3", ""),
            (f"{PROG} resynthesize 3 --style academic", "rewrite as an academic report"),
        ),
    ),
    CliCommand(
        "fork", "Session", "Copy a session into a new branch", "ID [QUERY...]",
        details="Copies the session (optionally with a new query) so you can continue it "
                "without touching the original.",
        examples=(
            (f"{PROG} fork 3", ""),
            (f'{PROG} fork 3 "narrower angle on X"', "fork with a different query"),
        ),
    ),
    CliCommand(
        "delete", "Session", "Delete a stored session", "ID",
        details="Asks for confirmation on a terminal; scripts proceed without prompting.",
    ),
    CliCommand(
        "setup", "Credential", "Choose provider, API key and models",
        details="Opens the setup screen: provider, API key (kept in the OS keyring), an optional "
                "Tavily key, and a model for each pipeline stage. Needs a terminal; Esc skips it.",
    ),
    CliCommand(
        "login", "Credential", "Store a single API key", "[KEY]",
        details="Without KEY, sets the active provider's API key.",
        examples=((f"{PROG} login", "the active provider's key"), (f"{PROG} login TAVILY_API_KEY", "")),
    ),
    CliCommand(
        "logout", "Credential", "Remove stored API keys", "[KEY|all]",
        details="Without KEY, clears the active provider's key only.",
        examples=((f"{PROG} logout", ""), (f"{PROG} logout all", "")),
    ),
    CliCommand(
        "keys", "Credential", "Show which keys are stored (never prints values)",
    ),
    CliCommand(
        "status", "System", "Show provider, stage models, credentials, paths",
    ),
    CliCommand(
        "migrate", "System", "Import a legacy research.db", "[PATH]",
    ),
    CliCommand(
        "help", "System", "Show help for a command", "[COMMAND]",
        examples=((f"{PROG} help continue", ""),),
    ),
)
INDEX = {c.name: c for c in COMMANDS}

TOP_EXAMPLES: tuple = (
    (PROG, "open the interactive prompt"),
    (f'{PROG} "How do solid-state batteries work?"', "one query, then exit"),
    (f'{PROG} --papers "CRISPR off-target effects"', "always include research papers"),
    (f'{PROG} --template market --style brief "EV charging market"', ""),
    (f"{PROG} history", "past sessions"),
    (f"{PROG} continue 3 --depth deep", "dig further into session 3"),
)


# --- parser -------------------------------------------------------------

class _HelpAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        parser.exit()


class _VersionAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        console.print(f"{PROG} {__version__}")
        parser.exit()


class CliParser(argparse.ArgumentParser):
    """argparse with rich help and styled usage errors."""

    def __init__(self, command: Optional[CliCommand] = None, **kwargs):
        prog = PROG if command is None else f"{PROG} {command.name}"
        super().__init__(prog=prog, add_help=False, **kwargs)
        self.command = command
        self.add_argument("-h", "--help", action=_HelpAction, help="Show this help and exit")
        if command is None:
            self.add_argument("-V", "--version", action=_VersionAction, help="Show the version and exit")

    def usage_line(self) -> str:
        if self.command is None:
            return f'{PROG} [options] ["QUESTION"]'
        return f"{self.prog} {self.command.args}".rstrip()

    def print_help(self, file=None) -> None:
        if self.command is None:
            print_main_help(self)
        else:
            print_command_help(self)

    def error(self, message: str) -> None:
        err_console.print(f"[err]{glyph('fail')}[/err] {escape(message)}")
        err_console.print(f"  [muted]usage:[/muted] {escape(self.usage_line())}")
        err_console.print(f"  [muted]more:[/muted]  [cmd]{self.prog} --help[/cmd]")
        self.exit(2)


# --- rendering ----------------------------------------------------------

def _heading(title: str) -> None:
    console.print()
    console.print(f"[brand]{title}[/brand]")


def _indented(renderable) -> Padding:
    return Padding(renderable, (0, 0, 0, 2))


def _grid() -> Table:
    grid = Table.grid(padding=(0, 3))
    grid.add_column(no_wrap=True)
    grid.add_column()
    return grid


def _metavar(action: argparse.Action) -> str:
    if action.nargs == 0:
        return ""
    if action.choices and action.dest in CHOICE_DOCS:
        return action.dest.upper()  # the values are listed (with meanings) beneath
    if action.choices:
        return "{" + ",".join(str(c) for c in action.choices) + "}"
    if isinstance(action.metavar, str):
        return action.metavar
    return action.dest.upper()


def _flag_label(action: argparse.Action) -> Text:
    label = Text()
    label.append(", ".join(action.option_strings), style="cmd")
    metavar = _metavar(action)
    if metavar:
        label.append(" " + metavar, style="arg")
    return label


def _has_default(action: argparse.Action) -> bool:
    return action.default not in (None, False, argparse.SUPPRESS, "", [])


def _option_description(action: argparse.Action) -> Text:
    head = Text(action.help or "")
    if _has_default(action):
        head.append(f"  (default: {action.default})", style="muted")
    return head


def _is_meta(action: argparse.Action) -> bool:
    return isinstance(action, (_HelpAction, _VersionAction))


def _option_table(parser: argparse.ArgumentParser, *, with_choices: bool, meta: str = "exclude") -> Table:
    """Rows for the parser's options. `meta` controls -h/--version: "exclude"
    (research options only), "only" (just those), or "include" (all, meta last)."""
    actions = [a for a in parser._actions if a.option_strings and a.help is not argparse.SUPPRESS]
    regular = [a for a in actions if not _is_meta(a)]
    special = [a for a in actions if _is_meta(a)]
    chosen = {"exclude": regular, "only": special, "include": regular + special}[meta]
    grid = _grid()
    for action in chosen:
        grid.add_row(_flag_label(action), _option_description(action))
        docs = CHOICE_DOCS.get(action.dest)
        if with_choices and action.choices and docs:
            for value, blurb in docs():
                grid.add_row(Text("    " + value, style="cmd"), Text(blurb, style="muted"))
    return grid


def _command_table(commands) -> Table:
    grid = _grid()
    for c in commands:
        label = Text(c.name, style="cmd")
        if c.args:
            label.append(" " + c.args, style="arg")
        grid.add_row(label, Text(c.summary))
    return grid


def _examples(examples) -> Group:
    """Comment line above each command (when there is one), so long commands
    never fight their comment for width."""
    lines = []
    for command, comment in examples:
        if comment:
            lines.append(Text(f"# {comment}", style="muted"))
        lines.append(Text(command, style="bold"))
    return Group(*lines)


def print_main_help(parser: argparse.ArgumentParser) -> None:
    console.print()
    console.print(Text.assemble(
        (f"{glyph('mark')} Deep Research", "brand"), (f"  v{__version__}", "muted"),
    ))
    console.print(
        "Breaks a question into sub-questions, searches the web and research papers,\n"
        "reads and analyzes what it finds, and writes a cited report."
    )

    _heading("Usage")
    usage = _grid()
    usage.add_row(Text(PROG, style="bold"), Text("Open the interactive prompt"))
    usage.add_row(Text.assemble((PROG, "bold"), (' [options] "QUESTION"', "arg")),
                  Text("Research one question, then exit"))
    usage.add_row(Text.assemble((PROG, "bold"), (" COMMAND [args]", "arg")),
                  Text("Sessions, credentials, configuration"))
    console.print(_indented(usage))

    _heading("Research options")
    console.print(_indented(_option_table(parser, with_choices=True)))

    for group in GROUPS:
        _heading(f"{group} commands")
        console.print(_indented(_command_table([c for c in COMMANDS if c.group == group])))

    _heading("General")
    console.print(_indented(_option_table(parser, with_choices=False, meta="only")))

    _heading("Examples")
    console.print(_indented(_examples(TOP_EXAMPLES)))

    console.print()
    console.print(f"[muted]Details on a command:[/muted] [cmd]{PROG} COMMAND --help[/cmd]")
    console.print("[muted]Inside the prompt, type[/muted] [cmd]/[/cmd] [muted]to browse slash commands.[/muted]")
    console.print()


def print_command_help(parser: argparse.ArgumentParser) -> None:
    command: CliCommand = parser.command  # type: ignore[attr-defined]
    console.print()
    console.print(Text.assemble((f"{PROG} {command.name}", "brand"), (f"  {glyph('dash')}  {command.summary}", "")))
    if command.details:
        console.print(Text(command.details, style="muted"))

    _heading("Usage")
    console.print(_indented(Text.assemble(
        (f"{PROG} {command.name}", "bold"),
        (f" {command.args}" if command.args else "", "arg"),
    )))

    positionals = [a for a in parser._actions if not a.option_strings]
    if positionals:
        _heading("Arguments")
        grid = _grid()
        for a in positionals:
            name = a.metavar if isinstance(a.metavar, str) else a.dest.upper()
            description = a.help or ""
            if a.nargs in ("?", "*"):
                description += " (optional)"
            grid.add_row(Text(name, style="arg"), Text(description))
        console.print(_indented(grid))

    _heading("Options")
    console.print(_indented(_option_table(parser, with_choices=True, meta="include")))

    if command.examples:
        _heading("Examples")
        console.print(_indented(_examples(command.examples)))
    console.print()


def print_command_list() -> None:
    """Compact list used by `help` when given an unknown command."""
    for group in GROUPS:
        _heading(f"{group} commands")
        console.print(_indented(_command_table([c for c in COMMANDS if c.group == group])))
    console.print()
