"""
Interactive REPL for Deep Research.

Behaves like the `python` or `node` REPL: a persistent prompt where you type
either a research query (anything not starting with /) or a slash command
that mutates session state, browses history, or shows status.

The REPL is purely a thin wrapper over the same async pipeline that the
one-shot CLI uses. State that lives in `settings` (provider, route, fallback)
is mutated in place; per-research toggles (depth, style, agent, ...) live in
a `ReplState` instance and are read each time you start a query.
"""
from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .capabilities import CapabilityViolation, known_profiles
from .config import settings
from .models import ReportStyle, ResearchConfig, ResearchDepth, ResearchReport
from .output.report import display_report, save_report
from .setup_wizard import (
    run_login,
    run_logout,
    run_setup_wizard,
    show_keys_status,
)
from .storage.db import (
    delete_session,
    list_sessions,
    load_report as db_load,
    save_report as db_save,
)
from .pipeline.orchestrator import acontinue_research, arun_research
from .pipeline.agent import arun_agent_research

console = Console()

VERSION = "0.2.0"


@dataclass
class ReplState:
    """Per-session toggles. Mirrors the one-shot CLI flags."""
    depth: ResearchDepth = ResearchDepth.STANDARD
    style: ReportStyle = ReportStyle.STANDARD
    template: Optional[str] = None
    thinking: bool = False
    academic: bool = False
    agent: bool = False
    fresh: bool = False

    def to_config(self) -> ResearchConfig:
        config = ResearchConfig.from_depth(self.depth)
        config.report_style = self.style
        if self.thinking:
            config.use_thinking = True
        if self.academic:
            config.use_academic_search = True
        if self.template:
            config.template = self.template
        if self.fresh:
            config.fresh = True
        config.load_model_routes()
        return config


@dataclass
class Command:
    name: str
    handler: Callable[["ReplState", list[str]], Optional[bool]]
    summary: str
    usage: str = ""
    aliases: tuple[str, ...] = ()

    def all_names(self) -> tuple[str, ...]:
        return (self.name,) + self.aliases


# --- handlers -----------------------------------------------------------

def _toggle(state: ReplState, attr: str, label: str) -> None:
    setattr(state, attr, not getattr(state, attr))
    console.print(f"  {label}: [bold]{'on' if getattr(state, attr) else 'off'}[/bold]")


def _cmd_depth(state: ReplState, args: list[str]) -> None:
    if not args:
        console.print(f"  Depth: [bold]{state.depth.value}[/bold]")
        return
    try:
        state.depth = ResearchDepth(args[0])
    except ValueError:
        console.print(f"  [red]Unknown depth {args[0]!r}.[/red] Valid: quick, standard, deep")
        return
    console.print(f"  Depth: [bold]{state.depth.value}[/bold]")


def _cmd_style(state: ReplState, args: list[str]) -> None:
    if not args:
        console.print(f"  Style: [bold]{state.style.value}[/bold]")
        return
    try:
        state.style = ReportStyle(args[0])
    except ValueError:
        console.print(f"  [red]Unknown style {args[0]!r}.[/red] Valid: brief, standard, academic")
        return
    console.print(f"  Style: [bold]{state.style.value}[/bold]")


def _cmd_template(state: ReplState, args: list[str]) -> None:
    if not args:
        console.print(f"  Template: [bold]{state.template or 'none'}[/bold]")
        return
    val = args[0].lower()
    if val in ("none", "off", "clear"):
        state.template = None
    elif val in ("technology", "science", "market", "literature"):
        state.template = val
    else:
        console.print(f"  [red]Unknown template {val!r}.[/red] Valid: technology, science, market, literature, none")
        return
    console.print(f"  Template: [bold]{state.template or 'none'}[/bold]")


def _cmd_thinking(state: ReplState, args: list[str]) -> None:
    _toggle(state, "thinking", "Thinking")


def _cmd_academic(state: ReplState, args: list[str]) -> None:
    _toggle(state, "academic", "Academic sources")


def _cmd_agent(state: ReplState, args: list[str]) -> None:
    _toggle(state, "agent", "Agent mode")


def _cmd_fresh(state: ReplState, args: list[str]) -> None:
    _toggle(state, "fresh", "Fresh mode")


def _cmd_provider(state: ReplState, args: list[str]) -> None:
    if not args:
        console.print(f"  Provider: [bold]{settings.active_provider}[/bold]")
        return
    val = args[0]
    original = settings.llm_provider
    settings.llm_provider = val
    try:
        info = settings.validate_routes()
    except (CapabilityViolation, ValueError) as e:
        settings.llm_provider = original
        console.print(f"  [red]Cannot switch to {val!r}: {e}[/red]")
        return
    console.print(f"  Provider: [bold]{settings.active_provider}[/bold] ({info['endpoint_family']})")


def _cmd_route(state: ReplState, args: list[str]) -> None:
    if not args:
        profiles = known_profiles(settings.active_provider) or ["(none)"]
        console.print(f"  Route: [bold]{settings.llm_route}[/bold]  (available: {', '.join(profiles)})")
        return
    original = settings.llm_route
    settings.llm_route = args[0]
    try:
        settings.validate_routes()
    except (CapabilityViolation, ValueError) as e:
        settings.llm_route = original
        console.print(f"  [red]Cannot switch route: {e}[/red]")
        return
    console.print(f"  Route: [bold]{settings.llm_route}[/bold]")


def _cmd_status(state: ReplState, args: list[str]) -> None:
    try:
        info = settings.validate_routes()
    except CapabilityViolation as e:
        console.print(f"  [red]Route is invalid: {e}[/red]")
        info = {}
    body = (
        f"Provider:   [bold]{settings.active_provider}[/bold] "
        f"({info.get('endpoint_family', '?')})\n"
        f"Profile:    [bold]{settings.llm_route}[/bold]\n"
        f"Active model: {settings.active_model}\n"
        f"Fallback:   {settings.llm_fallback_provider or '(none)'}\n"
        f"\n"
        f"Depth:      [bold]{state.depth.value}[/bold]\n"
        f"Style:      [bold]{state.style.value}[/bold]\n"
        f"Template:   {state.template or '(none)'}\n"
        f"Thinking:   {'on' if state.thinking else 'off'}\n"
        f"Academic:   {'on' if state.academic else 'off'}\n"
        f"Agent mode: {'on' if state.agent else 'off'}\n"
        f"Fresh mode: {'on' if state.fresh else 'off'}"
    )
    if "stages" in info:
        stage_lines = "\n".join(f"  {k:13} {v}" for k, v in info["stages"].items())
        body += f"\n\nStage models:\n{stage_lines}"
    console.print(Panel(body, title="Status", border_style="cyan"))


def _cmd_history(state: ReplState, args: list[str]) -> None:
    sessions = list_sessions(limit=25)
    if not sessions:
        console.print("  [dim]No research sessions found.[/dim]")
        return
    table = Table(title="Research History")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Parent", style="magenta", justify="right")
    table.add_column("Query", style="white", max_width=60)
    table.add_column("Depth", style="yellow")
    table.add_column("Sources", justify="right")
    table.add_column("Date", style="dim")
    for s in sessions[:25]:  # cap recent 25 in REPL output
        date_str = s["created_at"][:16].replace("T", " ")
        parent = s.get("parent_session_id")
        table.add_row(
            str(s["id"]),
            str(parent) if parent else "",
            s["query"][:60],
            s["depth"],
            str(s["source_count"]),
            date_str,
        )
    console.print(table)
    console.print("  [dim]Use /continue <id> to deepen, /inspect <id> to view.[/dim]")


def _cmd_inspect(state: ReplState, args: list[str]) -> None:
    if not args or not args[0].isdigit():
        console.print("  [red]Usage: /inspect <session_id>[/red]")
        return
    sid = int(args[0])
    report = db_load(sid)
    if report is None:
        console.print(f"  [red]Session {sid} not found.[/red]")
        return
    console.print(Panel(
        f"[bold]Query:[/bold] {report.query}\n"
        f"[bold]Sources:[/bold] {len(report.sources)}\n"
        f"[bold]Iterations:[/bold] {len(report.iterations)}\n"
        f"[bold]Created:[/bold] {report.created_at}",
        title=f"Session {sid}",
        border_style="cyan",
    ))
    if report.executive_summary:
        console.print(Markdown(report.executive_summary))


def _cmd_continue(state: ReplState, args: list[str]) -> None:
    if not args or not args[0].isdigit():
        console.print("  [red]Usage: /continue <session_id>[/red]")
        return
    sid = int(args[0])
    prev = db_load(sid)
    if prev is None:
        console.print(f"  [red]Session {sid} not found.[/red]")
        return
    config = state.to_config()
    if state.agent:
        console.print(
            "  [yellow]Note: agent mode has no effect on /continue; "
            "continuation always uses the gap-analysis loop.[/yellow]"
        )
    console.print(f"\n[bold]Continuing session {sid}:[/bold] {prev.query}")
    try:
        report = asyncio.run(acontinue_research(prev, config))
    except Exception as e:
        console.print(f"  [red]Continuation failed: {e}[/red]")
        return
    display_report(report)
    path = save_report(report)
    new_id = db_save(report, depth=config.depth.value, parent_session_id=sid)
    report.id = new_id
    console.print(f"  [green]Saved as session {new_id}[/green] [dim](parent: {sid})[/dim] -> {path}")


def _cmd_delete(state: ReplState, args: list[str]) -> None:
    if not args or not args[0].isdigit():
        console.print("  [red]Usage: /delete <session_id>[/red]")
        return
    sid = int(args[0])
    confirm = Prompt.ask(f"  Delete session {sid}?", choices=["y", "n"], default="n")
    if confirm.lower() != "y":
        return
    if delete_session(sid):
        console.print(f"  [green]Session {sid} deleted.[/green]")
    else:
        console.print(f"  [red]Session {sid} not found.[/red]")


def _cmd_clear(state: ReplState, args: list[str]) -> None:
    console.clear()


def _cmd_setup(state: ReplState, args: list[str]) -> None:
    run_setup_wizard()


def _cmd_login(state: ReplState, args: list[str]) -> None:
    run_login(args[0] if args else None)


def _cmd_logout(state: ReplState, args: list[str]) -> None:
    run_logout(args[0] if args else None)


def _cmd_keys(state: ReplState, args: list[str]) -> None:
    show_keys_status()


def _cmd_help(state: ReplState, args: list[str]) -> None:
    rows = [
        (cmd.name + (" " + cmd.usage if cmd.usage else ""), cmd.summary)
        for cmd in COMMANDS
    ]
    table = Table(title="Slash Commands", show_header=False, border_style="dim")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    for name, summary in rows:
        table.add_row(name, summary)
    console.print(table)
    console.print(
        "  [dim]Type a research question (anything not starting with /) to run it.[/dim]"
    )


def _cmd_quit(state: ReplState, args: list[str]) -> bool:
    return True  # signal to exit


# --- registry -----------------------------------------------------------

COMMANDS: tuple[Command, ...] = (
    Command("/depth",     _cmd_depth,     "Show or set research depth", "[quick|standard|deep]"),
    Command("/style",     _cmd_style,     "Show or set report style", "[brief|standard|academic]"),
    Command("/template",  _cmd_template,  "Set domain template (or none)", "[technology|science|market|literature|none]"),
    Command("/thinking",  _cmd_thinking,  "Toggle thinking mode"),
    Command("/academic",  _cmd_academic,  "Toggle academic sources (arXiv + Semantic Scholar)"),
    Command("/agent",     _cmd_agent,     "Toggle autonomous agent loop"),
    Command("/fresh",     _cmd_fresh,     "Toggle fresh mode (skip URLs from past sessions)"),
    Command("/provider",  _cmd_provider,  "Show or switch LLM provider", "[zai|opencode-go]"),
    Command("/route",     _cmd_route,     "Show or set route profile", "[balanced|budget|quality]"),
    Command("/status",    _cmd_status,    "Show current settings and resolved route"),
    Command("/history",   _cmd_history,   "List recent research sessions"),
    Command("/inspect",   _cmd_inspect,   "Show a stored session", "<id>"),
    Command("/continue",  _cmd_continue,  "Deepen a stored session", "<id>"),
    Command("/delete",    _cmd_delete,    "Delete a stored session", "<id>"),
    Command("/clear",     _cmd_clear,     "Clear the screen"),
    Command("/setup",     _cmd_setup,     "Run the credential setup wizard"),
    Command("/login",     _cmd_login,     "Set a single API key in the keyring", "[KEY_NAME]"),
    Command("/logout",    _cmd_logout,    "Clear stored credentials", "[KEY_NAME|all]"),
    Command("/keys",      _cmd_keys,      "Show which credentials are stored (never shows values)"),
    Command("/help",      _cmd_help,      "Show this command list", aliases=("/?",)),
    Command("/quit",      _cmd_quit,      "Exit the REPL", aliases=("/exit", "/q")),
)

_COMMAND_INDEX: dict[str, Command] = {
    name: cmd for cmd in COMMANDS for name in cmd.all_names()
}


def dispatch(line: str, state: ReplState) -> Optional[bool]:
    """
    Run a slash command. Returns True to signal "exit the REPL", None otherwise.
    Raises KeyError when the command is unknown — the caller decides how to surface it.
    """
    parts = shlex.split(line)
    if not parts:
        return None
    cmd_name, *args = parts
    cmd = _COMMAND_INDEX.get(cmd_name)
    if cmd is None:
        raise KeyError(cmd_name)
    return cmd.handler(state, args)


# --- runtime ------------------------------------------------------------

def _enable_history() -> None:
    """Wire stdlib readline to a persistent history file (best effort)."""
    try:
        import readline  # noqa: F401  (auto-attaches to input())
    except ImportError:
        return
    history_dir = Path(os.path.expanduser("~/.config/deep-research"))
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    history_file = history_dir / "repl_history"
    try:
        if history_file.exists():
            readline.read_history_file(str(history_file))
        readline.set_history_length(1000)
        import atexit
        atexit.register(lambda: _save_history(readline, history_file))
    except (OSError, AttributeError):
        pass


def _save_history(readline_mod, history_file: Path) -> None:
    try:
        readline_mod.write_history_file(str(history_file))
    except OSError:
        pass


def _print_banner(state: ReplState) -> None:
    try:
        info = settings.validate_routes()
        family = info["endpoint_family"]
    except Exception:
        family = "?"
    console.print(Panel(
        f"[bold]Deep Research[/bold] v{VERSION}\n"
        f"[dim]Provider: {settings.active_provider} ({family}) | "
        f"Profile: {settings.llm_route} | "
        f"Depth: {state.depth.value} | Style: {state.style.value}[/dim]\n"
        f"[dim]Type a research question, or /help for commands.[/dim]",
        border_style="blue",
    ))


async def _run_query(state: ReplState, query: str) -> None:
    config = state.to_config()
    try:
        if state.agent:
            report = await arun_agent_research(query, config)
        else:
            report = await arun_research(query, config)
    except Exception as e:
        console.print(f"  [red]Research failed: {e}[/red]")
        return
    display_report(report)
    path = save_report(report)
    new_id = db_save(report, depth=config.depth.value)
    report.id = new_id
    console.print(f"  [green]Saved as session {new_id}[/green] -> {path}")


def run_repl(initial_state: Optional[ReplState] = None) -> None:
    """Entry point: print the banner, loop on input, dispatch."""
    state = initial_state or ReplState()
    _enable_history()
    _print_banner(state)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue

        if line.startswith("/"):
            try:
                should_exit = dispatch(line, state)
            except KeyError as e:
                console.print(f"  [red]Unknown command {e.args[0]!r}.[/red] /help for the list.")
                continue
            if should_exit:
                break
            continue

        # Plain text → research query
        try:
            asyncio.run(_run_query(state, line))
        except KeyboardInterrupt:
            console.print("\n  [yellow]Interrupted.[/yellow]")

    console.print("[dim]Goodbye.[/dim]")
