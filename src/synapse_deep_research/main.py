from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.text import Text

from . import views
from .cli_help import INDEX, PROG, CliParser, print_command_list
from .config import settings
from .logging_config import setup_logging
from .models import ReportStyle, ResearchConfig, ResearchDepth, ResearchReport
from .options import DEPTH_VALUES, STYLE_VALUES, TEMPLATE_VALUES
from .storage.db import delete_session, fork_session, load_report as db_load
from .ui import (
    console, emit_json_error, enable_json_output, fail, glyph, json_output, note, ok,
    section, tty, warn,
)


def check_config() -> bool:
    """
    Verify the API key and route capability before a run. Prints only
    problems; the banner already shows the resolved model and search setup.
    """
    from .capabilities import CapabilityViolation

    ready = True
    try:
        if not settings.active_api_key:
            fail(
                f"{settings.active_api_key_env_name} is not set "
                f"(LLM_PROVIDER={escape(settings.llm_provider)})"
            )
            ready = False
        settings.validate_routes()
    except CapabilityViolation as e:
        # Includes a typo'd LLM_PROVIDER / LLM_ENDPOINT_FAMILY: those surface
        # as CapabilityViolation from the Settings properties.
        fail(f"Capability check failed: {escape(str(e))}")
        ready = False
    return ready


# --- parsers --------------------------------------------------------------

def _validated_default_depth() -> str:
    """RESEARCH_DEPTH env guard: argparse doesn't validate `default=` against
    `choices`, so an invalid value would crash ResearchDepth(...) later."""
    depth = settings.default_depth
    if depth not in DEPTH_VALUES:
        warn(
            f"Ignoring invalid RESEARCH_DEPTH={escape(repr(depth))} "
            f"(expected {', '.join(DEPTH_VALUES)}); using standard."
        )
        return "standard"
    return depth


def add_shape_args(parser: argparse.ArgumentParser, *, agent: bool = False) -> None:
    """Research-shape flags, shared by one-shot runs and continue / resynthesize."""
    parser.add_argument(
        "--depth", choices=DEPTH_VALUES, default=_validated_default_depth(),
        help="How thorough the research is",
    )
    parser.add_argument(
        "--style", choices=STYLE_VALUES, default="standard",
        help="Shape and length of the final report",
    )
    parser.add_argument(
        "--template", choices=TEMPLATE_VALUES,
        help="Domain-specific prompts and search strategy",
    )
    # Research papers (arXiv + Semantic Scholar) are auto — the template and
    # depth decide — unless one of these says otherwise: args.academic is None,
    # True or False. (Called "papers", not "academic", so it can't be mistaken
    # for --style academic.)
    papers = parser.add_mutually_exclusive_group()
    papers.add_argument(
        "--papers", dest="academic", action="store_const", const=True, default=None,
        help="Always also search research papers (arXiv, Semantic Scholar); default: the template and depth decide",
    )
    papers.add_argument(
        "--no-papers", dest="academic", action="store_const", const=False,
        help="Web search only, even for --depth deep or the science / literature templates",
    )
    if agent:
        parser.add_argument(
            "--agent", action="store_true",
            help="Autonomous mode: the agent decides when to search more or stop",
        )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Revisit pages already analyzed in past sessions",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """How a run reports back, shared by one-shot runs, continue and resynthesize."""
    parser.add_argument(
        "--json", action="store_true",
        help="Print the result as one JSON object on stdout, progress and messages on stderr (for scripts and agents)",
    )


def build_run_parser() -> CliParser:
    parser = CliParser()
    parser.add_argument(
        "query", nargs="*", metavar="QUESTION",
        help="What to research (omit to open the interactive prompt)",
    )
    add_shape_args(parser, agent=True)
    add_output_args(parser)
    return parser


def build_command_parser(name: str) -> CliParser:
    parser = CliParser(command=INDEX[name])
    if name in ("inspect", "fork", "delete", "continue", "resynthesize"):
        parser.add_argument("id", type=int, metavar="ID", help="Session ID (see `history`)")
    if name == "history":
        parser.add_argument("-n", "--limit", type=int, default=25, metavar="N",
                            help="How many sessions to show")
    elif name == "inspect":
        parser.add_argument("--full", action="store_true",
                            help="Also print the full detailed findings")
        parser.add_argument("--source", type=int, metavar="N",
                            help="Deep dive on one source by its 1-based number")
    elif name == "fork":
        parser.add_argument("query", nargs="*", metavar="QUERY",
                            help="New query for the fork (default: keep the original)")
    elif name in ("continue", "resynthesize"):
        add_shape_args(parser)
        add_output_args(parser)
    elif name in ("login", "logout"):
        parser.add_argument("key", nargs="?", metavar="KEY",
                            help="Credential name, e.g. TAVILY_API_KEY" + (" or `all`" if name == "logout" else ""))
    elif name == "migrate":
        parser.add_argument("path", nargs="?", metavar="PATH",
                            help="Legacy research.db to import (default: search common locations)")
    elif name == "help":
        parser.add_argument("topic", nargs="?", metavar="COMMAND", help="Command to describe")
    return parser


def build_config(args: argparse.Namespace) -> ResearchConfig:
    config = ResearchConfig.from_depth(ResearchDepth(args.depth))
    config.report_style = ReportStyle(args.style)
    config.academic = args.academic
    config.apply_academic_source()
    if args.template:
        config.template = args.template
    if args.fresh:
        config.fresh = True
    config.load_model_routes()
    return config


# --- research flows ---------------------------------------------------------

async def research_query(query: str, config: ResearchConfig, agent_mode: bool) -> ResearchReport:
    # Imported here: the LLM/search client stack is slow to import and
    # `--help` / `history` / `status` shouldn't pay for it.
    if agent_mode:
        from .pipeline.agent import arun_agent_research
        return await arun_agent_research(query, config)
    from .pipeline.orchestrator import arun_research
    return await arun_research(query, config)


async def async_main(query: str, config: ResearchConfig, agent_mode: bool = False) -> None:
    started = time.monotonic()
    try:
        report = await research_query(query, config, agent_mode)
    except Exception as e:
        console.print()
        fail(f"Research failed: {escape(str(e))}")
        sys.exit(1)

    views.present_report(report, depth=config.depth.value, started=started)
    await followup_loop(report, config, agent_mode=agent_mode)


async def async_resynthesize(
    session_id: int, config: ResearchConfig, banner_shape: Optional[str] = None,
    *, repl: bool = False,
) -> None:
    from .pipeline.synthesizer import asynthesize_report
    from .pipeline.templates import get_template
    from .usage import new_run_tracker, stage

    prev = db_load(session_id)
    if not prev:
        fail(f"Session {session_id} not found.")
        sys.exit(1)
    if not prev.sources:
        fail(f"Session {session_id} has no sources to synthesize.")
        sys.exit(1)

    tmpl = get_template(config.template) if config.template else None
    template_guidance = tmpl.synthesis_guidance if tmpl else ""
    system_prompt = tmpl.system_prompt if tmpl else ""

    if banner_shape:
        views.print_banner(shape=banner_shape, task=f"Re-synthesize session {session_id}", query=prev.query)
    else:
        section(f"Re-synthesizing session {session_id}")
        console.print(Text(prev.query, style="bold"))
    suffix = f", template={tmpl.name}" if tmpl else ""
    note(
        f"{len(prev.sources)} sources, {len(prev.iterations)} iterations "
        f"{glyph('arrow')} style={config.report_style.value}{suffix}"
    )
    console.print()

    started = time.monotonic()
    tracker = new_run_tracker()
    try:
        with console.status("[brand]Synthesizing report from existing sources..."):
            async with stage("synthesize"):
                executive, detailed, follow_ups = await asynthesize_report(
                    prev.query,
                    prev.sub_questions,
                    prev.sources,
                    thinking=config.use_thinking,
                    model=config.models.get("synthesize"),
                    template_guidance=template_guidance,
                    system_prompt=system_prompt,
                    report_style=config.report_style,
                    iteration_count=max(1, len(prev.iterations)),
                )
    except Exception as e:
        console.print()
        fail(f"Synthesis failed: {escape(str(e))}")
        sys.exit(1)

    new_report = ResearchReport(
        query=prev.query,
        sub_questions=prev.sub_questions,
        sources=prev.sources,
        iterations=prev.iterations,
        searched_urls=prev.searched_urls,
        executive_summary=executive,
        detailed_findings=detailed,
        follow_up_questions=follow_ups,
        parent_session_id=session_id,
        usage=tracker.snapshot(),
    )
    note(tracker.format_summary())
    views.present_report(
        new_report, depth=config.depth.value, parent_session_id=session_id,
        started=started, relation=f"resynthesized from session {session_id}", repl=repl,
    )


async def async_continue(
    session_id: int, config: ResearchConfig, banner_shape: Optional[str] = None,
    *, followup: bool = True, repl: bool = False,
) -> None:
    from .pipeline.orchestrator import acontinue_research

    prev = db_load(session_id)
    if not prev:
        fail(f"Session {session_id} not found.")
        sys.exit(1)

    if banner_shape:
        views.print_banner(shape=banner_shape, task=f"Continue session {session_id}", query=prev.query)
    else:
        section(f"Continuing session {session_id}")
        console.print(Text(prev.query, style="bold"))
    note(f"Previous research: {len(prev.sources)} sources across {len(prev.iterations)} iterations")
    console.print()

    started = time.monotonic()
    try:
        report = await acontinue_research(prev, config)
    except Exception as e:
        console.print()
        fail(f"Research failed: {escape(str(e))}")
        sys.exit(1)

    views.present_report(
        report, depth=config.depth.value, parent_session_id=session_id,
        started=started, relation=f"continues session {session_id}", repl=repl,
    )
    if followup:
        await followup_loop(report, config)


async def followup_loop(report: ResearchReport, config: ResearchConfig, agent_mode: bool = False) -> None:
    if not tty() or json_output():
        return

    while True:
        console.print()
        answer = Prompt.ask(
            f"[brand]{glyph('arrow')} follow-up[/brand] "
            "[muted](a number from the list, a new question, or Enter to finish)[/muted]",
            default="", show_default=False,
        ).strip()
        if not answer or answer.lower() in ("q", "quit", "exit"):
            break

        if answer.isdigit() and report.follow_up_questions:
            idx = int(answer) - 1
            if 0 <= idx < len(report.follow_up_questions):
                answer = report.follow_up_questions[idx]

        section("Follow-up")
        console.print(Text(answer, style="bold"))
        console.print()
        started = time.monotonic()
        try:
            report = await research_query(answer, config, agent_mode)
            views.present_report(report, depth=config.depth.value, started=started)
        except Exception as e:
            console.print()
            fail(f"Research failed: {escape(str(e))}")


# --- session / maintenance commands ---------------------------------------

def _delete_session_flow(session_id: int) -> None:
    # Interactive callers get the same confirmation the REPL asks for;
    # non-tty (scripts) proceed without prompting.
    if tty():
        if not Confirm.ask(f"[bold]Delete session {session_id}?[/bold]", default=False):
            note("Cancelled.")
            return
    if delete_session(session_id):
        ok(f"Session {session_id} deleted.")
    else:
        fail(f"Session {session_id} not found.")


def _fork_session_flow(session_id: int, new_query: Optional[str]) -> None:
    new_id = fork_session(session_id, new_query=new_query)
    if new_id is None:
        fail(f"Session {session_id} not found.")
        sys.exit(1)
    ok(f"Forked session {session_id} {glyph('arrow')} new session [brand]{new_id}[/brand].")
    if new_query:
        note(f"Fork query: {escape(new_query)}")
    note(f"Refine it without touching session {session_id}: {PROG} continue {new_id}")


def _run_migrate(path: Optional[str]) -> None:
    """Move a legacy <repo>/data/research.db into the XDG data dir."""
    import shutil
    from .config import DATA_DIR, DB_PATH

    if path:
        candidate = Path(path).expanduser().resolve()
    else:
        # Try a couple of obvious places. PROJECT_ROOT under pipx-install
        # points at the venv, so we look from cwd up.
        candidates = [
            Path.cwd() / "data" / "research.db",
            Path.cwd().parent / "data" / "research.db",
            Path.home() / "Documents" / "Projects" / "Deep-Research" / "data" / "research.db",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            fail("Could not find a legacy research.db.",
                 hint=f"Pass an explicit path: {PROG} migrate /path/to/research.db")
            return
        candidate = found

    if not candidate.exists():
        fail(f"No file at {candidate}.")
        return
    if candidate == DB_PATH:
        note(f"{candidate} is already the active DB. Nothing to do.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        backup = DB_PATH.with_suffix(".db.bak")
        warn(f"Backing up existing {DB_PATH} {glyph('arrow')} {backup}")
        shutil.copy2(DB_PATH, backup)

    shutil.copy2(candidate, DB_PATH)
    ok(f"Migrated {candidate} {glyph('arrow')} {DB_PATH}")
    note("The original is untouched; delete it manually once you've confirmed.")


def _ensure_ready() -> None:
    """check_config gate: offer the wizard when no key is configured, hard-exit
    on a broken route so a run never starts with mismatched models."""
    if check_config():
        return
    try:
        key_missing = not settings.active_api_key
    except Exception:
        # Invalid provider/family name — not a wizard case; exit below.
        key_missing = False
    if not key_missing:
        fail(
            "Provider or model configuration is invalid.",
            hint="Fix LLM_PROVIDER / MODEL_<STAGE> above and re-run, or choose models with "
                 f"`{PROG} setup` (`{PROG} status` shows what is resolved).",
        )
        sys.exit(1)

    console.print()
    missing = f"No {settings.active_api_key_env_name} found in env, .env, or keyring."
    if not tty():
        fail(missing, hint=f"Run `{PROG} setup` to store credentials, or set {settings.active_api_key_env_name} in your environment.")
        sys.exit(1)
    warn(missing)

    # The setup screen is the offer: Esc skips it.
    from .onboarding import run_setup
    run_setup(first_run=True)
    settings.reload_credentials()
    if not settings.active_api_key:
        fail(f"Still no {settings.active_api_key_env_name} configured. Exiting.")
        sys.exit(1)
    if not check_config():
        fail("Provider or model configuration is invalid.", hint="Fix the configuration above and re-run.")
        sys.exit(1)


def _run_llm_command(name: str, ns: argparse.Namespace) -> None:
    # continue / resynthesize make LLM calls: validate config first.
    if ns.json:
        enable_json_output()
    config = build_config(ns)
    _ensure_ready()
    shape = views.shape_from_namespace(ns)
    coro = async_continue if name == "continue" else async_resynthesize
    asyncio.run(coro(ns.id, config, banner_shape=shape))


def _do_history(ns: argparse.Namespace) -> None:
    views.print_history(max(1, ns.limit))


def _do_inspect(ns: argparse.Namespace) -> None:
    report = db_load(ns.id)
    if not report:
        fail(f"Session {ns.id} not found.")
        sys.exit(1)
    if ns.source is not None:
        if not views.print_source(report, ns.source):
            sys.exit(1)
    else:
        views.print_session(report, ns.id, full=ns.full)


def _do_fork(ns: argparse.Namespace) -> None:
    _fork_session_flow(ns.id, " ".join(ns.query).strip() or None)


def _do_delete(ns: argparse.Namespace) -> None:
    _delete_session_flow(ns.id)


def _do_continue(ns: argparse.Namespace) -> None:
    _run_llm_command("continue", ns)


def _do_resynthesize(ns: argparse.Namespace) -> None:
    _run_llm_command("resynthesize", ns)


def _do_setup(ns: argparse.Namespace) -> None:
    from .onboarding import run_setup
    if not run_setup() and not tty():
        sys.exit(1)  # nothing can be set up without a terminal; scripts should see that


def _do_login(ns: argparse.Namespace) -> None:
    from .setup_wizard import run_login
    run_login(ns.key)


def _do_logout(ns: argparse.Namespace) -> None:
    from .setup_wizard import run_logout
    run_logout(ns.key)


def _do_keys(ns: argparse.Namespace) -> None:
    from .setup_wizard import show_keys_status
    show_keys_status()


def _do_status(ns: argparse.Namespace) -> None:
    views.print_status()


def _do_migrate(ns: argparse.Namespace) -> None:
    _run_migrate(ns.path)


def _do_help(ns: argparse.Namespace) -> None:
    if ns.topic is None:
        build_run_parser().print_help()
    elif ns.topic in INDEX:
        build_command_parser(ns.topic).print_help()
    else:
        fail(f"Unknown command {escape(repr(ns.topic))}.")
        print_command_list()
        sys.exit(2)


# One handler per registered command (cli_help.COMMANDS); a test keeps the
# two in step.
HANDLERS = {
    "history": _do_history,
    "inspect": _do_inspect,
    "continue": _do_continue,
    "resynthesize": _do_resynthesize,
    "fork": _do_fork,
    "delete": _do_delete,
    "setup": _do_setup,
    "login": _do_login,
    "logout": _do_logout,
    "keys": _do_keys,
    "status": _do_status,
    "migrate": _do_migrate,
    "help": _do_help,
}


def run_command(name: str, argv: list) -> None:
    HANDLERS[name](build_command_parser(name).parse_args(argv))


def _warn_retired_settings() -> None:
    """Routes were replaced by per-stage models; say so once if the old
    setting is still around instead of silently ignoring it."""
    if os.getenv("LLM_ROUTE"):
        warn(
            "LLM_ROUTE is no longer used — routes were replaced by a model per stage. "
            f"Remove it from your environment or .env, and choose models with `{PROG} setup` or /model."
        )


def main() -> None:
    try:
        _main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            emit_json_error()  # a --json run always ends with a JSON object, success or not
        raise
    except Exception as exc:  # noqa: BLE001 — still a JSON object for the caller, then the traceback
        emit_json_error(f"{type(exc).__name__}: {exc}")
        raise


def _main() -> None:
    # Warnings and errors only by default: the INFO stream is per-call route
    # dumps and result counts that clutter the UI. SYNAPSE_DEEP_RESEARCH_LOG_LEVEL=INFO
    # (or DEBUG) brings them back.
    setup_logging(os.getenv("SYNAPSE_DEEP_RESEARCH_LOG_LEVEL", "WARNING"), console=console)
    argv = sys.argv[1:]

    # Subcommands are dispatched on the first argument so the free-text
    # QUESTION positional of a one-shot run never fights with them.
    if argv and argv[0] in INDEX:
        run_command(argv[0], argv[1:])
        return

    parser = build_run_parser()
    args = parser.parse_args(argv)
    if args.json:
        if not args.query:
            parser.error("--json needs a QUESTION to research")
        enable_json_output()

    # No positional query → launch the interactive REPL. Direct one-shot
    # usage like `synapse-deep-research "topic"` skips the REPL.
    if not args.query:
        from .repl import ReplState, run_repl
        _warn_retired_settings()
        run_repl(ReplState(
            depth=ResearchDepth(args.depth),
            style=ReportStyle(args.style),
            template=args.template,
            academic=args.academic,
            agent=args.agent,
            fresh=args.fresh,
        ))
        return

    query = " ".join(args.query).strip()
    if not query:
        fail("No query provided.")
        sys.exit(1)

    config = build_config(args)
    _warn_retired_settings()
    views.print_banner(shape=views.shape_from_namespace(args), query=query)

    # Heuristic warning: deep depth + agent mode on opencode-go can burn
    # through the $12/5h cap quickly. We can't query remaining budget — the
    # OpenCode Go usage API isn't published — so this is purely advisory.
    # Reading the provider raises on a typo'd LLM_PROVIDER; the banner already
    # reports that, so the advisory is simply skipped.
    try:
        if (
            settings.active_provider == "opencode-go"
            and config.depth == ResearchDepth.DEEP
            and args.agent
            and not settings.llm_fallback_provider
        ):
            warn(
                "--depth deep + --agent on OpenCode Go can consume a significant share of "
                "the $12/5h cap. Consider setting LLM_FALLBACK_PROVIDER (e.g. zai) so the "
                "run can survive a 429."
            )
    except ValueError:  # includes CapabilityViolation
        pass

    _ensure_ready()
    console.print()
    asyncio.run(async_main(query, config, agent_mode=args.agent))


if __name__ == "__main__":
    main()
