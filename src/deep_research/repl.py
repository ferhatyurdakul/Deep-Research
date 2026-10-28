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
import difflib
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from rich.console import Group
from rich.markup import escape
from rich.padding import Padding
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from . import __version__, views
from . import options as _options
from . import userconfig
from .capabilities import (
    PIPELINE_STAGES, CapabilityViolation, lookup, registered_providers_for,
)
from .config import _PROVIDER_PROTOCOLS, _xdg_config_home, settings
from .interactive import ask_text, pick_option, show_view
from .models import ReportStyle, ResearchConfig, ResearchDepth
from .onboarding import needs_setup, run_setup
from .options import (
    ACADEMIC_MODES, STAGE_BLURBS, academic_choices, academic_mode_name, academic_mode_value,
    academic_reason, depth_choices, model_choices, style_choices, template_choices,
)
from .setup_wizard import (
    run_login,
    run_logout,
    show_keys_status,
)
from .storage.db import (
    delete_session,
    fork_session,
    load_report as db_load,
)
from .ui import console, fail, glyph, note, ok, problem_count, warn

VERSION = __version__


@dataclass
class ReplState:
    """Per-session toggles. Mirrors the one-shot CLI flags."""
    depth: ResearchDepth = ResearchDepth.STANDARD
    style: ReportStyle = ReportStyle.STANDARD
    template: Optional[str] = None
    academic: Optional[bool] = None  # None = auto: the template and depth decide
    agent: bool = False
    fresh: bool = False

    def to_config(self) -> ResearchConfig:
        config = ResearchConfig.from_depth(self.depth)
        config.report_style = self.style
        config.academic = self.academic
        config.apply_academic_source()
        if self.template:
            config.template = self.template
        if self.fresh:
            config.fresh = True
        config.load_model_routes()
        return config

    def academic_reason(self) -> Optional[str]:
        """Why research papers will be searched (None if they won't be)."""
        return academic_reason(self.depth.value, self.template, self.academic)

    def academic_label(self) -> str:
        """`/status` wording: `on`, `off`, or `auto — on (science template)`."""
        if self.academic is not None:
            return academic_mode_name(self.academic)
        reason = self.academic_reason()
        return f"auto — on ({reason})" if reason else "auto — off"

    def shape(self) -> str:
        return views.shape_summary(
            self.depth.value, self.style.value, self.template,
            self.academic, self.agent, self.fresh,
        )


Choices = Callable[[], list]  # -> [(value, description), ...]

GROUPS = ("Research", "Model", "Sessions", "Credentials", "General")


@dataclass
class Command:
    name: str
    handler: Callable[["ReplState", list], Optional[bool]]
    summary: str
    usage: str = ""
    aliases: tuple = ()
    group: str = "General"
    choices: Optional[Choices] = None   # drives `/cmd` listings and argument completion
    current: Optional[Callable[["ReplState"], Optional[str]]] = None  # value to mark in the picker
    details: str = ""                   # shown by `/help <command>`
    # Completion for commands with more than one argument: receives the
    # arguments already typed, returns the choices for the next one.
    arg_choices: Optional[Callable[[list], list]] = None
    # A custom interactive flow for a command typed in a terminal. Receives
    # (state, args, line); returns the line to run, or None if cancelled.
    interactive: Optional[Callable[["ReplState", list, str], Optional[str]]] = None
    # What the user sees after the command runs in a terminal:
    #   "silent" - nothing on success (the status line already shows the new
    #              state); anything that went wrong is shown in the viewer
    #   "view"   - the output is shown in a closable viewer, then vanishes
    #   "live"   - printed normally (runs, wizards, prompts)
    display: str = "live"

    def all_names(self) -> tuple:
        return (self.name,) + self.aliases

    def short_usage(self) -> str:
        """Usage for dense listings: the value list is shown separately (and
        by `/<cmd>` itself), so `[value]` stands in for it here."""
        return "[value]" if self.choices and not (self.interactive or self.arg_choices) else self.usage


# --- handlers -----------------------------------------------------------

def _indent(renderable) -> Padding:
    return Padding(renderable, (0, 0, 0, 2))


def _print_choices(
    label: str, current: Optional[str], choices: list, command: str, clearable: bool = False,
) -> None:
    """The option picker: every valid value with what it does, current marked."""
    console.print(f"[brand]{label}[/brand]")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column()
    for value, blurb in choices:
        chosen = value == current
        grid.add_row(
            Text(glyph("on") if chosen else glyph("off"), style="brand" if chosen else "muted"),
            Text(value, style="cmd" if chosen else ""),
            Text(blurb, style="" if chosen else "muted"),
        )
    if clearable:
        chosen = current is None
        grid.add_row(
            Text(glyph("on") if chosen else glyph("off"), style="brand" if chosen else "muted"),
            Text("none", style="cmd" if chosen else ""),
            Text("No template", style="" if chosen else "muted"),
        )
    console.print(_indent(grid))
    note(f"Change with {command} <value>.")


def _choice_command(
    state: ReplState, args: list, *, command: str, attr: str, label: str,
    choices: list, coerce: Callable, clearable: bool = False,
) -> None:
    raw = getattr(state, attr)
    current = raw.value if isinstance(raw, Enum) else raw
    valid = [v for v, _ in choices]
    if not args:
        _print_choices(label, current, choices, command, clearable)
        return
    value = args[0].lower()
    if clearable and value in ("none", "off", "clear"):
        setattr(state, attr, None)
    elif value in valid:
        setattr(state, attr, coerce(value))
    else:
        options = valid + (["none"] if clearable else [])
        fail(f"Unknown {label.lower()} {escape(repr(args[0]))}.", hint=f"Valid: {', '.join(options)}")
        return
    new = getattr(state, attr)
    shown = (new.value if isinstance(new, Enum) else new) or "none"
    blurb = dict(choices).get(shown, "")
    ok(f"{label}: [bold]{shown}[/bold]" + (f"  [muted]{escape(blurb)}[/muted]" if blurb else ""))


def _cmd_depth(state: ReplState, args: list) -> None:
    _choice_command(state, args, command="/depth", attr="depth", label="Depth",
                    choices=depth_choices(), coerce=ResearchDepth)


def _cmd_style(state: ReplState, args: list) -> None:
    _choice_command(state, args, command="/style", attr="style", label="Style",
                    choices=style_choices(), coerce=ReportStyle)


def _cmd_template(state: ReplState, args: list) -> None:
    _choice_command(state, args, command="/template", attr="template", label="Template",
                    choices=template_choices(), coerce=str, clearable=True)


def _toggle(state: ReplState, attr: str, label: str, meaning: str) -> None:
    setattr(state, attr, not getattr(state, attr))
    on = getattr(state, attr)
    ok(f"{label}: [bold]{'on' if on else 'off'}[/bold]" + (f"  [muted]{meaning}[/muted]" if on else ""))


def _cmd_papers(state: ReplState, args: list) -> None:
    choices = academic_choices(state.depth.value, state.template)
    if not args:
        _print_choices("Papers", academic_mode_name(state.academic), choices, "/papers")
        return
    value = args[0].lower()
    if value not in ACADEMIC_MODES:
        fail(f"Unknown papers mode {escape(repr(args[0]))}.", hint=f"Valid: {', '.join(ACADEMIC_MODES)}")
        return
    state.academic = academic_mode_value(value)
    blurb = dict(academic_choices(state.depth.value, state.template))[value]
    ok(f"Papers: [bold]{value}[/bold]  [muted]{escape(blurb)}[/muted]")


def _papers_interactive(state: ReplState, args: list, line: str) -> Optional[str]:
    """Bare `/papers` -> a picker whose `auto` row says what it means right
    now. A typed value passes through."""
    if args:
        return line
    choices = academic_choices(state.depth.value, state.template)
    value = pick_option("Papers", choices, academic_mode_name(state.academic))
    return None if value is None else f"/papers {value}"


def _cmd_agent(state: ReplState, args: list) -> None:
    _toggle(state, "agent", "Agent mode", "the agent decides when to search more or stop")


def _cmd_fresh(state: ReplState, args: list) -> None:
    _toggle(state, "fresh", "Fresh mode", "pages from past sessions are revisited")


def _template_choices_with_none() -> list:
    return template_choices() + [("none", "No template")]


def _provider_choices() -> list:
    return [(name, f"endpoints: {', '.join(families)}") for name, families in _PROVIDER_PROTOCOLS.items()]


def _cmd_provider(state: ReplState, args: list) -> None:
    if not args:
        try:
            current = settings.active_provider
        except Exception:
            current = None
        _print_choices("Provider", current, _provider_choices(), "/provider")
        return
    val = args[0]
    original = settings.llm_provider
    settings.llm_provider = val
    try:
        info = settings.validate_routes()
    except (CapabilityViolation, ValueError) as e:
        settings.llm_provider = original
        fail(f"Cannot switch to {escape(repr(val))}: {escape(str(e))}")
        return
    userconfig.set_provider(settings.active_provider)  # chosen in the app, so it sticks
    ok(f"Provider: [bold]{settings.active_provider}[/bold] ({info['endpoint_family']})")


def _cmd_status(state: ReplState, args: list) -> None:
    views.print_status(shape=[
        ("depth", state.depth.value),
        ("style", state.style.value),
        ("template", state.template or "(none)"),
        ("papers", state.academic_label()),
        ("agent mode", "on" if state.agent else "off"),
        ("fresh mode", "on" if state.fresh else "off"),
    ])


def _cmd_history(state: ReplState, args: list) -> None:
    limit = int(args[0]) if args and args[0].isdigit() else 25
    views.print_history(max(1, limit), repl=True)


def _parse_id(args: list, usage: str) -> Optional[int]:
    if not args or not args[0].isdigit():
        fail(f"Usage: {usage}")
        return None
    return int(args[0])


def _cmd_inspect(state: ReplState, args: list) -> None:
    usage = "/inspect <session_id> [--full] [--source N]"
    sid = _parse_id(args, usage)
    if sid is None:
        return
    full = "--full" in args
    source: Optional[int] = None
    if "--source" in args:
        i = args.index("--source")
        if i + 1 >= len(args) or not args[i + 1].isdigit():
            fail(f"Usage: {usage}")
            return
        source = int(args[i + 1])
    report = db_load(sid)
    if report is None:
        fail(f"Session {sid} not found.")
        return
    if source is not None:
        views.print_source(report, source)
    else:
        views.print_session(report, sid, full=full, repl=True)


def _cmd_continue(state: ReplState, args: list) -> None:
    sid = _parse_id(args, "/continue <session_id>")
    if sid is None:
        return
    if db_load(sid) is None:
        fail(f"Session {sid} not found.")
        return
    from .main import async_continue

    if state.agent:
        warn("Agent mode has no effect on /continue; continuation always uses the gap-analysis loop.")
    try:
        asyncio.run(async_continue(sid, state.to_config(), followup=False, repl=True))
    except SystemExit:
        # async_continue reports its own errors and exits; inside the REPL
        # that must not kill the session.
        pass


def _cmd_delete(state: ReplState, args: list) -> None:
    confirmed = "--yes" in args
    sid = _parse_id([a for a in args if a != "--yes"], "/delete <session_id>")
    if sid is None:
        return
    if not confirmed and not Confirm.ask(f"Delete session {sid}?", default=False):
        note("Cancelled.")
        return
    if delete_session(sid):
        ok(f"Session {sid} deleted.")
    else:
        fail(f"Session {sid} not found.")


def _cmd_fork(state: ReplState, args: list) -> None:
    sid = _parse_id(args, "/fork <session_id> [new query]")
    if sid is None:
        return
    new_query = " ".join(args[1:]).strip() or None
    new_id = fork_session(sid, new_query=new_query)
    if new_id is None:
        fail(f"Session {sid} not found.")
        return
    ok(f"Forked session {sid} {glyph('arrow')} [brand]{new_id}[/brand].")
    if new_query:
        note(f"Fork query: {escape(new_query)}")
    note(f"Refine it with /continue {new_id}.")


def _cmd_resynthesize(state: ReplState, args: list) -> None:
    sid = _parse_id(args, "/resynthesize <session_id>")
    if sid is None:
        return
    if db_load(sid) is None:
        fail(f"Session {sid} not found.")
        return
    from .main import async_resynthesize

    try:
        asyncio.run(async_resynthesize(sid, state.to_config(), repl=True))
    except SystemExit:
        # async_resynthesize reports its own errors via console + sys.exit;
        # inside the REPL that must not kill the session.
        pass


_STAGE_NAMES = tuple(PIPELINE_STAGES)
_DEFAULT_WORDS = ("default", "profile", "reset")
_SOURCE_LABELS = {"environment": "from your environment", "saved": "saved", "recommended": "recommended"}


def _stage_models() -> dict:
    try:
        return settings.effective_stage_models()
    except Exception:
        return {}


def _stage_sources() -> dict:
    try:
        return settings.stage_sources()
    except Exception:
        return {}


def _model_stage_choices() -> list:
    """`all` plus each stage: what it does and the model it uses now."""
    stages, sources = _stage_models(), _stage_sources()
    rows = [("all", "Every stage")]
    for stage in PIPELINE_STAGES:
        model = stages.get(stage, "?")
        rows.append((stage, f"{model} ({_SOURCE_LABELS.get(sources.get(stage, ''), '')}) — {STAGE_BLURBS[stage]}"))
    return rows


def _model_pick_list(stage: Optional[str] = None, account: Optional[list] = None) -> list:
    """Models for the active provider, an escape hatch for any other ID, and
    the way back to the recommendation. `account` is the provider's live model
    list when we have it (see options.fetch_account_models)."""
    try:
        provider = settings.active_provider
    except Exception:
        return []
    rows = model_choices(provider, stage if stage in _STAGE_NAMES else None, account)
    return rows + [
        ("other", "Type another model ID"),
        ("default", "Back to the recommended model"),
    ]


def _model_arg_choices(previous: list) -> list:
    if not previous:
        return _model_stage_choices()
    if len(previous) == 1:
        return [row for row in _model_pick_list(previous[0]) if row[0] != "other"]
    return []


def _cmd_model(state: ReplState, args: list) -> None:
    if not args:
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="cmd", no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(style="muted")
        sources = _stage_sources()
        for stage, model in _stage_models().items():
            grid.add_row(stage, model, _SOURCE_LABELS.get(sources.get(stage, ""), ""))
        console.print(_indent(grid))
        note("Change one with /model <stage|all> <model|default>.")
        return

    stage = args[0].lower().replace("-", "_")
    stages = list(PIPELINE_STAGES) if stage == "all" else [stage]
    if stage != "all" and stage not in _STAGE_NAMES:
        fail(f"Unknown stage {escape(repr(args[0]))}.", hint="Stages: all, " + ", ".join(PIPELINE_STAGES))
        return
    if len(args) < 2:
        fail(f"Usage: /model {stage} <model|default>",
             hint="Models: " + ", ".join(v for v, _ in _model_pick_list() if v != "other"))
        return

    model = args[1]
    clear = model.lower() in _DEFAULT_WORDS
    provider = settings.active_provider
    if not clear:
        if lookup(provider, model) is None:
            elsewhere = registered_providers_for(model)
            if elsewhere:
                fail(f"{escape(model)} belongs to {', '.join(elsewhere)}, not {provider}.",
                     hint="Models for " + provider + ": " + ", ".join(v for v, _ in model_choices(provider)))
                return
            warn(f"{escape(model)} is not in the capability registry; thinking and endpoint support are unverified.")

    previous = userconfig.stage_models(provider)
    for s in stages:
        userconfig.set_stage_model(provider, s, None if clear else model)
    try:
        settings.validate_routes()
    except (CapabilityViolation, ValueError) as e:
        userconfig.save_choices(provider, previous)  # put back exactly what was saved
        fail(f"Cannot use {escape(model)}: {escape(str(e))}")
        return

    target = "every stage" if stage == "all" else stage
    ok(f"Model for {target}: [bold]{'the recommended model' if clear else escape(model)}[/bold]")
    shadowed = [s for s in stages if os.getenv(f"MODEL_{s.upper()}")]
    if shadowed:
        warn("Saved, but " + ", ".join(f"MODEL_{s.upper()}" for s in shadowed)
             + " is set in your environment and overrides it.")


def _model_interactive(state: ReplState, args: list, line: str) -> Optional[str]:
    """Bare `/model` -> pick a stage, then a model. Fully typed lines pass through."""
    if len(args) >= 2:
        return line
    stage = args[0].lower().replace("-", "_") if args else None
    if stage is None:
        stage = pick_option("Model · which stage?", _model_stage_choices(), None)
        if stage is None:
            return None
    elif stage != "all" and stage not in _STAGE_NAMES:
        return line  # let the handler explain the typo
    stages = _stage_models()
    if stage == "all":
        used = set(stages.values())
        current = used.pop() if len(used) == 1 else None
    else:
        current = stages.get(stage)
    try:
        account = _options.fetch_account_models(settings.active_provider)
    except Exception:
        account = None
    while True:
        models = _model_pick_list(stage, account)
        if not models:
            return line
        title = f"Model for {'every stage' if stage == 'all' else stage}"
        model = pick_option(title, models, current)
        if model != "other":
            break
        model = ask_text(title, hint="Any ID your provider serves")
        if model is None:
            return None
        if model.strip():
            model = model.strip()
            break
    return None if model is None else f"/model {stage} {model}"


def _delete_interactive(state: ReplState, args: list, line: str) -> Optional[str]:
    """`/delete 3` -> a Yes/No picker instead of a typed prompt."""
    if not args or not args[0].isdigit() or "--yes" in args or db_load(int(args[0])) is None:
        return line  # nothing to confirm (the handler reports a missing session)
    answer = pick_option(
        f"Delete session {args[0]}?",
        [("no", "Keep it"), ("yes", "Delete it permanently")], "no",
    )
    return f"/delete {args[0]} --yes" if answer == "yes" else None


def _cmd_clear(state: ReplState, args: list) -> None:
    console.clear()


def _cmd_setup(state: ReplState, args: list) -> None:
    run_setup()


def _cmd_login(state: ReplState, args: list) -> None:
    run_login(args[0] if args else None)


def _cmd_logout(state: ReplState, args: list) -> None:
    run_logout(args[0] if args else None)


def _cmd_keys(state: ReplState, args: list) -> None:
    show_keys_status()


def _label(cmd: Command, *, compact: bool = False) -> Text:
    label = Text(cmd.name, style="cmd")
    usage = cmd.short_usage() if compact else cmd.usage
    if usage:
        label.append(" " + usage, style="arg")
    return label


def _cmd_help(state: ReplState, args: list) -> None:
    if args:
        _help_for(args[0])
        return
    for group in GROUPS:
        cmds = [c for c in COMMANDS if c.group == group]
        grid = Table.grid(padding=(0, 3))
        grid.add_column(no_wrap=True)
        grid.add_column()
        for cmd in cmds:
            description = Text(cmd.summary)
            if cmd.choices:
                values = " · ".join(v for v, _ in cmd.choices())
                description = Group(description, Text(values, style="muted"))
            grid.add_row(_label(cmd, compact=True), description)
        console.print(f"\n[brand]{group}[/brand]")
        console.print(_indent(grid))
    console.print()
    note("Type a question (anything not starting with /) to research it. "
         "/help <command> explains one command; Tab completes commands and values.")


def _help_for(name: str) -> None:
    key = name if name.startswith("/") else "/" + name
    cmd = _COMMAND_INDEX.get(key)
    if cmd is None:
        _unknown_command(key)
        return
    console.print()
    console.print(Text.assemble(_label(cmd), (f"  {glyph('dash')}  {cmd.summary}", "")))
    if cmd.details:
        note(cmd.details)
    if cmd.aliases:
        note("Aliases: " + ", ".join(cmd.aliases))
    if cmd.choices:
        console.print()
        choices = cmd.choices()
        grid = Table.grid(padding=(0, 3))
        grid.add_column(no_wrap=True)
        grid.add_column()
        for value, blurb in choices:
            grid.add_row(Text(value, style="cmd"), Text(blurb, style="muted"))
        console.print(_indent(grid))
    console.print()


def _cmd_quit(state: ReplState, args: list) -> bool:
    return True  # signal to exit


# --- registry -----------------------------------------------------------

COMMANDS: tuple = (
    Command("/depth", _cmd_depth, "Show or set research depth", "[quick|standard|deep]",
            group="Research", choices=depth_choices, current=lambda s: s.depth.value,
            details="How many sub-questions, search results and research passes a run uses.", display="silent"),
    Command("/style", _cmd_style, "Show or set report style", "[brief|standard|academic]",
            group="Research", choices=style_choices, current=lambda s: s.style.value,
            details="The shape and length of the written report.", display="silent"),
    Command("/template", _cmd_template, "Set a domain template (or none)",
            "[technology|science|market|literature|none]",
            group="Research", choices=_template_choices_with_none,
            current=lambda s: s.template or "none",
            details="Domain-specific prompts and search strategy.", display="silent"),
    Command("/papers", _cmd_papers, "Search research papers: auto, on or off (arXiv + Semantic Scholar)",
            "[auto|on|off]", aliases=("/academic",), group="Research", choices=academic_choices,
            current=lambda s: academic_mode_name(s.academic), interactive=_papers_interactive,
            details="auto lets the template and depth decide (deep and the science / literature "
                    "templates search papers); on always does; off never does. Not the report "
                    "style: /style academic is about how the report reads.", display="silent"),
    Command("/agent", _cmd_agent, "Toggle the autonomous agent loop",
            group="Research",
            details="After each pass the agent decides whether to search more, go deeper, broaden, "
                    "or stop. Does not apply to /continue.", display="silent"),
    Command("/fresh", _cmd_fresh, "Toggle fresh mode (revisit URLs from past sessions)",
            group="Research",
            details="By default, pages analyzed in earlier sessions are skipped so each run finds new material.", display="silent"),
    Command("/provider", _cmd_provider, "Show or switch the LLM provider", "[zai|opencode-go]",
            group="Model", choices=_provider_choices, current=lambda s: _safe(lambda: settings.active_provider), display="silent"),
    Command("/model", _cmd_model, "Choose the model for each pipeline stage", "[stage] [model]",
            group="Model", choices=_model_stage_choices, arg_choices=_model_arg_choices,
            interactive=_model_interactive, display="silent",
            details="Stages: decompose, analyze, gap_analysis, synthesize, or all. Your choice is saved; "
                    "'default' returns a stage to the recommended model. A MODEL_<STAGE> environment "
                    "variable overrides it."),
    Command("/status", _cmd_status, "Show settings, resolved models and credentials", group="Model", display="view"),
    Command("/history", _cmd_history, "List recent research sessions", "[count]", group="Sessions", display="view"),
    Command("/inspect", _cmd_inspect, "Show a stored session", "<id> [options]",
            group="Sessions",
            details="Options: --full prints the detailed findings; --source N shows one source's "
                    "findings and evidence.", display="view"),
    Command("/continue", _cmd_continue, "Deepen a stored session", "<id>", group="Sessions",
            details="Another gap-analysis pass using the current settings; saved as a new linked session."),
    Command("/fork", _cmd_fork, "Fork a session into a new branch", "<id> [query]", group="Sessions",
            details="Copies a session so you can continue it without touching the original.", display="view"),
    Command("/resynthesize", _cmd_resynthesize, "Re-write the report from stored sources", "<id>",
            group="Sessions",
            details="No searching: re-runs only synthesis, using the current style and template."),
    Command("/delete", _cmd_delete, "Delete a stored session", "<id>", group="Sessions", display="silent", interactive=_delete_interactive),
    Command("/setup", _cmd_setup, "Set up provider, API key and models", group="Credentials"),
    Command("/login", _cmd_login, "Store a single API key in the keyring", "[KEY_NAME]", group="Credentials"),
    Command("/logout", _cmd_logout, "Clear stored credentials", "[KEY_NAME|all]", group="Credentials"),
    Command("/keys", _cmd_keys, "Show which credentials are stored (never shows values)", group="Credentials", display="view"),
    Command("/clear", _cmd_clear, "Clear the screen", group="General"),
    Command("/help", _cmd_help, "Show commands, or explain one", "[command]", aliases=("/?",), group="General", display="view"),
    Command("/quit", _cmd_quit, "Exit", aliases=("/exit", "/q"), group="General"),
)

_COMMAND_INDEX: dict = {
    name: cmd for cmd in COMMANDS for name in cmd.all_names()
}


def suggest_command(name: str) -> Optional[str]:
    """Closest known command to a mistyped one, or None."""
    matches = difflib.get_close_matches(name, _COMMAND_INDEX, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _unknown_command(name: str) -> None:
    guess = suggest_command(name)
    hint = f"Did you mean {guess}?  /help lists every command." if guess else "/help lists every command."
    fail(f"Unknown command {escape(repr(name))}.", hint=hint)


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


# --- interactive picker --------------------------------------------------

def _safe(fn: Callable[[], Optional[str]]) -> Optional[str]:
    try:
        return fn()
    except Exception:
        return None


def _pick_for_command(line: str, state: "ReplState") -> Optional[str]:
    """Turn a bare setting command (`/depth`) into a chosen one (`/depth deep`)
    through the picker. Anything else comes back unchanged. None means the
    user cancelled the picker."""
    try:
        parts = shlex.split(line)
    except ValueError:
        return line
    cmd = _COMMAND_INDEX.get(parts[0]) if parts else None
    if cmd is not None and cmd.interactive is not None:
        return cmd.interactive(state, parts[1:], line)
    if len(parts) != 1 or cmd is None or cmd.choices is None:
        return line
    try:
        choices = cmd.choices()
    except Exception:
        return line
    if not choices:
        return line
    current = cmd.current(state) if cmd.current else None
    value = pick_option(cmd.name.lstrip("/").capitalize(), choices, current)
    return None if value is None else f"{cmd.name} {value}"


# --- runtime ------------------------------------------------------------

def _dispatch_reporting(line: str, state: "ReplState") -> Optional[bool]:
    """dispatch(), with unknown commands and unparseable arguments reported
    as messages instead of exceptions. True means exit the REPL."""
    try:
        return dispatch(line, state)
    except KeyError as e:
        _unknown_command(e.args[0])
    except ValueError as e:  # shlex: unbalanced quote in the arguments
        fail(f"Could not parse that command: {escape(str(e))}")
    return None


def _lookup_command(line: str) -> Optional[Command]:
    try:
        parts = shlex.split(line)
    except ValueError:
        return None
    return _COMMAND_INDEX.get(parts[0]) if parts else None


def _run_transient(line: str, state: "ReplState") -> bool:
    """Run a slash command typed in a terminal so it leaves nothing behind.

    Settings succeed silently (the status line shows the new state) and
    anything that went wrong, plus read-only screens like /help, is shown in
    a viewer that Enter or Esc closes. Commands whose output is the point
    (runs, wizards, prompts) print normally. True means exit the REPL.
    """
    cmd = _lookup_command(line)
    # Unknown commands and unparseable lines are reported in the viewer too.
    display = cmd.display if cmd is not None else "view"
    if display == "live":
        return bool(_dispatch_reporting(line, state))

    before = problem_count()
    with console.capture() as captured:
        should_exit = _dispatch_reporting(line, state)
    text = captured.get()
    if text.strip() and (display == "view" or problem_count() > before):
        show_view(text)
    return bool(should_exit)


def _history_file() -> Optional[Path]:
    """Path to the persistent REPL history (best effort)."""
    history_dir = _xdg_config_home() / "deep-research"
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return history_dir / "repl_history"


def _command_completer():
    """Slash-command completer. Typing "/" lists the primary commands; more
    characters filter fuzzily (aliases join in once you type past the slash);
    after "/depth " the command's own values complete. Plain (non-slash)
    queries never trigger completions."""
    from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter, WordCompleter

    def _meta(cmd: Command) -> str:
        return cmd.summary

    primary = FuzzyCompleter(WordCompleter(
        [c.name for c in COMMANDS],
        meta_dict={c.name: _meta(c) for c in COMMANDS}, sentence=True, ignore_case=True,
    ))
    everything = FuzzyCompleter(WordCompleter(
        list(_COMMAND_INDEX),
        meta_dict={n: _meta(c) for n, c in _COMMAND_INDEX.items()}, sentence=True, ignore_case=True,
    ))

    class _SlashOnly(Completer):
        # Must subclass Completer so get_completions_async comes from the
        # base — the async path is what complete_while_typing uses.
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            if " " not in text:
                inner = primary if text == "/" else everything
                yield from inner.get_completions(document, complete_event)
                return
            name, _, rest = text.partition(" ")
            cmd = _COMMAND_INDEX.get(name)
            if cmd is None:
                return
            *done, typed = rest.split(" ")
            done = [t for t in done if t]
            try:
                if cmd.arg_choices is not None:
                    choices = cmd.arg_choices(done)
                elif cmd.choices is not None and not done:
                    choices = cmd.choices()
                else:
                    return
            except Exception:
                return
            for value, blurb in choices:
                if value.lower().startswith(typed.lower()):
                    yield Completion(value, start_position=-len(typed), display_meta=blurb)

    return _SlashOnly()


def _status_fragments(state: "ReplState") -> list:
    """Colored, minimal status fragments: depth always; template,
    non-standard style, and active toggles only when set — no noise for
    defaults, no always-on flags like thinking."""
    try:
        provider = settings.active_provider
    except Exception:
        provider = "invalid config"
    frags = [("class:depth", f" {state.depth.value}")]
    if state.template:
        frags += [("class:sep", " · "), ("class:tpl", state.template)]
    if state.style != ReportStyle.STANDARD:
        frags += [("class:sep", " · "), ("class:style", state.style.value)]
    for label, on in (
        ("papers", state.academic_reason() is not None),
        ("agent", state.agent), ("fresh", state.fresh),
    ):
        if on:
            frags += [("class:sep", " · "), ("class:on", label)]
    frags += [("class:sep", " · "), ("class:provider", provider)]
    return frags


_MENU_ROWS = 8


def _commits_box(text: str) -> bool:
    """Whether a submitted line leaves its input box in the scrollback.

    Only a research question does — the box then doubles as the record of
    what was asked and with which settings. Slash commands and empty Enters
    erase their box, so changing a setting never stacks up boxes.
    """
    text = text.strip()
    return bool(text) and not text.startswith("/")


class _BoxedPrompt:
    """Claude-Code-style input box: rounded border with the run state in the
    top edge (tmux-style titled border), rebuilt on every prompt. No separate
    status line."""

    def __init__(self, state: "ReplState", *, input=None, output=None) -> None:
        from prompt_toolkit.application import Application

        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
        from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings
        from prompt_toolkit.key_binding.bindings.emacs import load_emacs_bindings
        from prompt_toolkit.layout import (
            ConditionalContainer, FormattedTextControl, HSplit, Layout, VSplit, Window,
        )
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
        from prompt_toolkit.styles import Style

        # Rounded-box drawing built from Windows (Frame hard-codes square
        # corners in prompt_toolkit 3.0.x).
        def edge(left: str, mid: str, right: str) -> "HSplit":
            # The middle cell must be allowed to stretch (no
            # dont_extend_width) or the border collapses to just corners.
            return HSplit([VSplit([
                Window(width=1, height=1, char=left, style="class:border"),
                Window(height=1, char=mid, style="class:border"),
                Window(width=1, height=1, char=right, style="class:border"),
            ])])

        history_path = _history_file()
        history = FileHistory(str(history_path)) if history_path is not None else None

        kb = KeyBindings()

        @kb.add("enter")
        def _accept(event):
            text = event.app.current_buffer.text
            # Claude-Code-style command expansion, deterministic: an exact
            # command submits as typed; a UNIQUE prefix ("/dep⏎") expands to
            # "/depth"; anything ambiguous or with arguments is untouched.
            # (complete_state is async and may be gone by Enter, so don't
            # rely on it.)
            if text.strip() == "/":
                text = ""  # a bare "/" is a closed menu, not a command
            elif text.startswith("/") and " " not in text.strip() and text not in _COMMAND_INDEX:
                matches = [k for k in _COMMAND_INDEX if k.startswith(text)]
                if len(matches) == 1:
                    text = matches[0]
            # Only a research question keeps its box in the scrollback.
            event.app.erase_when_done = not _commits_box(text)
            event.app.exit(result=text)

        @kb.add("escape", filter=Condition(lambda: self._buffer.text.startswith("/")), eager=True)
        def _dismiss_menu(event):
            # Esc closes the command menu and the "/" with it, like backing
            # out of the pickers. Eager only while a slash line is open, so
            # Alt-key editing shortcuts are untouched the rest of the time.
            event.app.current_buffer.reset()

        @kb.add("c-c")
        def _cancel(event):
            buf = event.app.current_buffer
            if buf.text:
                buf.reset()  # first Ctrl-C clears the line, like Claude Code
            else:
                event.app.erase_when_done = True
                event.app.exit(exception=KeyboardInterrupt)

        @kb.add("c-d")
        def _eof(event):
            if not event.app.current_buffer.text:
                event.app.erase_when_done = True
                event.app.exit(exception=EOFError)

        @kb.add("tab")
        def _complete_next(event):
            event.app.current_buffer.complete_next()

        @kb.add("s-tab")
        def _complete_prev(event):
            event.app.current_buffer.complete_previous()

        @kb.add("c-u")
        def _clear_line(event):
            # readline parity: delete from cursor to start of line.
            buf = event.app.current_buffer
            buf.delete_before_cursor(len(buf.text))

        self._buffer = Buffer(
            completer=_command_completer(), history=history,
            complete_while_typing=True, multiline=False,
        )
        def top_border():
            # Titled border: the status text lives inside the box's top
            # edge (tmux-style) — no separate status line above the input.
            cols = shutil.get_terminal_size().columns
            inner = _status_fragments(state)
            inner_len = sum(len(t) for _, t in inner) + 2  # padding spaces
            frags = [("class:border", "╭─ ")] + inner + [("class:border", " ")]
            frags += [("class:border", "─" * max(1, cols - inner_len - 4) + "╮")]
            return frags

        # Empty-box hint, gone as soon as anything is typed.
        placeholder = ConditionalProcessor(
            AfterInput("Ask a research question, or type / for commands", style="class:placeholder"),
            filter=Condition(lambda: not self._buffer.text),
        )
        input_row = VSplit([
            Window(width=1, char="│", style="class:border"),
            Window(width=2, height=1,
                   content=FormattedTextControl([("class:prompt", "❯ ")]),
                   dont_extend_width=True),
            Window(BufferControl(buffer=self._buffer, input_processors=[placeholder]),
                   wrap_lines=False, dont_extend_height=True),
            Window(width=1, char="│", style="class:border"),
        ])
        box = HSplit([
            Window(FormattedTextControl(top_border), height=1),
            input_row,
            edge("╰", "─", "╯"),
        ])

        # Inline completion area BELOW the box — part of the layout itself,
        # not a float, so it renders identically in terminals that don't
        # answer cursor-position requests. Hidden entirely when no
        # completions are active (i.e. for any non-slash input). Command
        # rows are aligned name | usage | summary; value rows (after
        # "/depth ") are value | meaning. The window scrolls with the
        # selection so every command is reachable.
        def render_completions():
            frags = []
            cs = self._buffer.complete_state
            if cs is None:
                return frags
            comps = cs.completions
            selected = cs.complete_index
            start = 0 if selected is None else max(0, selected - _MENU_ROWS + 1)
            window = comps[start:start + _MENU_ROWS]
            width = max((len(c.text) for c in window), default=0)
            usage_width = max(
                (len(cmd.short_usage()) for c in window if (cmd := _COMMAND_INDEX.get(c.text))),
                default=0,
            )
            for offset, comp in enumerate(window):
                is_sel = selected is not None and start + offset == selected
                cmd = _COMMAND_INDEX.get(comp.text)
                name_style = "class:selected" if is_sel else "class:menuname"
                frags.append((name_style, f" {comp.text.ljust(width)}"))
                if cmd is not None:
                    frags.append(("class:menuusage", f"  {cmd.short_usage().ljust(usage_width)}"))
                    frags.append(("class:menumeta", f"  {cmd.summary}"))
                elif comp.display_meta_text:
                    frags.append(("class:menumeta", f"  {comp.display_meta_text}"))
                frags.append(("", "\n"))
            hidden_above, hidden_below = start, len(comps) - start - len(window)
            if hidden_above or hidden_below:
                frags.append(("class:dim", f" {len(comps)} matches — Tab/Shift-Tab to move"))
            else:
                frags.append(("class:dim", " Tab to cycle · Enter to run"))
            return frags

        has_completions = Condition(lambda: self._buffer.complete_state is not None)
        menu = ConditionalContainer(
            Window(FormattedTextControl(render_completions), dont_extend_height=True),
            filter=has_completions,
        )
        self._app = Application(
            layout=Layout(HSplit([box, menu])),
            key_bindings=merge_key_bindings([
                load_basic_bindings(), load_emacs_bindings(), kb,
            ]),
            style=Style.from_dict({
                "depth": "bold ansicyan",
                "sep": "ansibrightblack",
                "tpl": "ansimagenta",
                "style": "ansiyellow",
                "on": "ansigreen",
                "provider": "ansiblue",
                "border": "ansicyan",
                "prompt": "bold ansicyan",
                "selected": "bold ansicyan reverse",
                "menuname": "bold ansicyan",
                "menuusage": "ansiyellow",
                "menumeta": "ansibrightblack",
                "placeholder": "ansibrightblack",
                "dim": "ansibrightblack",
            }),
            full_screen=False,
            input=input,
            output=output,
        )
        self._app.ttimeoutlen = 0.05  # make a lone Esc register promptly

    def prompt(self) -> str:
        self._app.erase_when_done = False  # the key bindings decide per submit
        text = self._app.run()
        self._buffer.reset(append_to_history=True)
        return text or ""


def _print_banner(state: ReplState) -> None:
    views.print_banner(
        shape=state.shape(),
        tips="[muted]Ask a question to start  ·  [/muted][cmd]/[/cmd][muted] commands  ·  "
             "[/muted][cmd]/help[/cmd][muted]  ·  [/muted][cmd]/q[/cmd][muted] quit[/muted]",
    )


async def _run_query(state: ReplState, query: str) -> None:
    from .main import check_config, research_query

    # Same gate as a one-shot run, minus the setup-wizard offer (that is /setup
    # here): fail with a pointer instead of an SDK "missing credentials" error.
    if not check_config():
        note("Run /setup to store a key, or /status to inspect the resolved route.")
        return

    config = state.to_config()
    started = time.monotonic()
    try:
        report = await research_query(query, config, state.agent)
    except Exception as e:
        console.print()
        fail(f"Research failed: {escape(str(e))}")
        return
    views.present_report(report, depth=config.depth.value, started=started, repl=True)


def run_repl(initial_state: Optional[ReplState] = None) -> None:
    """Entry point: print the banner, loop on input, dispatch."""
    state = initial_state or ReplState()

    # Nothing to run on yet (no API key): open setup first. Terminal only, so
    # pipes and scripts are never blocked, and Esc skips it.
    if sys.stdin is not None and sys.stdin.isatty() and needs_setup():
        run_setup(first_run=True)

    _print_banner(state)

    # prompt_toolkit drives the interactive UI (boxed input, slash-command
    # popup, persistent history). Pipes and other non-tty stdin — scripts,
    # tests, `echo "/q" | deep-research` — fall back to plain input().
    boxed = None
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            boxed = _BoxedPrompt(state)
        except Exception:
            boxed = None

    while True:
        try:
            if boxed is not None:
                line = boxed.prompt().strip()
            else:
                line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue

        if line.startswith("/"):
            if boxed is None:
                should_exit = _dispatch_reporting(line, state)
            else:
                # Interactive terminal: bare setting commands open a picker,
                # and the result leaves nothing behind (see _run_transient).
                line = _pick_for_command(line, state)
                if line is None:
                    continue  # cancelled
                should_exit = _run_transient(line, state)
            if should_exit:
                break
            continue

        # Plain text → research query
        console.print()
        try:
            asyncio.run(_run_query(state, line))
        except KeyboardInterrupt:
            console.print()
            warn("Interrupted.")

    note("Goodbye.")
