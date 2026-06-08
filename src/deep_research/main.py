from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import settings
from .logging_config import setup_logging
from .models import ReportStyle, ResearchConfig, ResearchDepth, ResearchReport
from .output.report import display_report, save_report
from .pipeline.orchestrator import arun_research, acontinue_research
from .pipeline.agent import arun_agent_research
from .storage.db import (
    save_report as db_save,
    load_report as db_load,
    list_sessions,
    delete_session,
    fork_session,
)

console = Console()


def check_config(quiet: bool = False) -> bool:
    """
    Verify API keys and route capability before a one-shot run.

    `quiet=True` suppresses the route dump (used when the REPL is about to
    take over and will surface settings via /status).
    """
    from .capabilities import CapabilityViolation

    ok = True
    if not settings.active_api_key:
        console.print(
            f"[red]Error: {settings.active_api_key_env_name} not set in .env file "
            f"(LLM_PROVIDER={settings.active_provider})[/red]"
        )
        ok = False
    if not settings.tavily_api_key and not quiet:
        console.print("[yellow]Warning: TAVILY_API_KEY not set — falling back to DuckDuckGo for web search[/yellow]")
    try:
        route = settings.validate_routes()
        if not quiet:
            console.print(f"[dim]Route: {route}[/dim]")
    except CapabilityViolation as e:
        console.print(f"[red]Capability check failed: {e}[/red]")
        ok = False
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research - AI-powered research framework")
    parser.add_argument("query", nargs="*", help="Research query")
    parser.add_argument(
        "--depth", choices=["quick", "standard", "deep"],
        default=settings.default_depth,
        help="Research depth (default: %(default)s)",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable thinking mode for deeper analysis",
    )
    parser.add_argument(
        "--academic", action="store_true",
        help="Include arXiv and Semantic Scholar in search",
    )
    parser.add_argument(
        "--template", choices=["technology", "science", "market", "literature"],
        help="Use a domain-specific research template",
    )
    parser.add_argument(
        "--style", choices=["brief", "standard", "academic"],
        default="standard",
        help="Report style (default: %(default)s)",
    )
    parser.add_argument(
        "--agent", action="store_true",
        help="Use autonomous agent mode (decides when to search more or stop)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore previously analyzed URLs (start fresh without cross-session dedup)",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="List past research sessions",
    )
    parser.add_argument(
        "--continue-session", type=int, metavar="ID",
        help="Continue a previous research session by ID",
    )
    parser.add_argument(
        "--fork-session", type=int, metavar="ID",
        help="Fork a previous session into a new session (copy, leaves original untouched)",
    )
    parser.add_argument(
        "--fork-query", type=str, default="",
        help="When used with --fork-session, set a different query on the fork",
    )
    parser.add_argument(
        "--delete", type=int, metavar="ID",
        help="Delete a research session by ID",
    )
    parser.add_argument(
        "--resynthesize", type=int, metavar="ID",
        help="Re-run only the synthesis step on an existing session's sources, saved as a new fork",
    )
    parser.add_argument(
        "--inspect", type=int, metavar="ID",
        help="Show metadata, sub-questions, sources, iterations, and summary for a session",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="With --inspect: also print the full detailed findings",
    )
    parser.add_argument(
        "--source", type=int, metavar="N",
        help="With --inspect: deep dive on a specific source by 1-based index",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ResearchConfig:
    config = ResearchConfig.from_depth(ResearchDepth(args.depth))
    config.report_style = ReportStyle(args.style)
    if args.thinking:
        config.use_thinking = True
    if args.academic:
        config.use_academic_search = True
    if args.template:
        config.template = args.template
    if args.fresh:
        config.fresh = True
    config.load_model_routes()
    return config


def show_history() -> None:
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No research sessions found.[/dim]")
        return

    table = Table(title="Research History")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Parent", style="magenta", justify="right")
    table.add_column("Query", style="white", max_width=60)
    table.add_column("Depth", style="yellow")
    table.add_column("Sources", justify="right")
    table.add_column("Date", style="dim")

    for s in sessions:
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
    console.print(
        "\n[dim]--continue-session ID to deepen a session; "
        "--fork-session ID to copy it into a new branch[/dim]"
    )


def _print_source_detail(report, index_1based: int) -> None:
    if index_1based < 1 or index_1based > len(report.sources):
        console.print(
            f"[red]Source {index_1based} out of range. Session has {len(report.sources)} sources.[/red]"
        )
        sys.exit(1)
    source = report.sources[index_1based - 1]
    meta_lines = [
        f"[bold]Title:[/bold]    {source.title}",
        f"[bold]URL:[/bold]      {source.url}",
        f"[bold]Type:[/bold]     {source.source_type.value}",
        f"[bold]Depth:[/bold]    {source.content_depth.value}",
    ]
    if source.authors:
        meta_lines.append(f"[bold]Authors:[/bold]  {', '.join(source.authors)}")
    if source.published_date:
        meta_lines.append(f"[bold]Date:[/bold]     {source.published_date}")
    if source.relevance:
        meta_lines.append(f"[bold]Relevance:[/bold] {source.relevance}")
    console.print(Panel("\n".join(meta_lines), title=f"Source #{index_1based}", border_style="cyan"))

    if source.summary:
        console.print(Panel(source.summary, title="Summary", border_style="dim"))

    if source.key_findings:
        console.print("\n[bold]Key findings:[/bold]")
        for f in source.key_findings:
            console.print(f"  - {f}")

    if source.key_evidence:
        console.print("\n[bold]Key evidence (verbatim):[/bold]")
        for e in source.key_evidence:
            console.print(f'  "{e}"')

    if source.evidenced_findings:
        console.print("\n[bold]Evidenced findings:[/bold]")
        for ef in source.evidenced_findings:
            console.print(f"  - [{ef.confidence}] {ef.finding}")
            if ef.evidence:
                console.print(f'    evidence: "{ef.evidence}"')


def show_session_detail(session_id: int, full: bool = False, source_index: int | None = None) -> None:
    report = db_load(session_id)
    if not report:
        console.print(f"[red]Session {session_id} not found.[/red]")
        sys.exit(1)

    if source_index is not None:
        _print_source_detail(report, source_index)
        return

    # Header
    header_lines = [
        f"[bold]Query:[/bold]   {report.query}",
        f"[bold]Sources:[/bold] {len(report.sources)}    "
        f"[bold]Iterations:[/bold] {len(report.iterations)}    "
        f"[bold]Sub-questions:[/bold] {len(report.sub_questions)}",
        f"[bold]Created:[/bold] {report.created_at.strftime('%Y-%m-%d %H:%M')}",
    ]
    if report.parent_session_id:
        header_lines.append(f"[bold]Parent:[/bold]  session {report.parent_session_id}")
    console.print(Panel("\n".join(header_lines), title=f"Session {session_id}", border_style="blue"))

    # Sub-questions
    if report.sub_questions:
        console.print(
            Panel(
                "\n".join(f"  {i+1}. {sq.question}" for i, sq in enumerate(report.sub_questions)),
                title="Sub-Questions",
                border_style="cyan",
            )
        )

    # Sources table
    if report.sources:
        table = Table(title=f"Sources ({len(report.sources)})")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Type", style="yellow")
        table.add_column("Depth", style="dim")
        table.add_column("Title", max_width=70)
        table.add_column("Findings", justify="right")
        table.add_column("Evid.", justify="right")
        for i, src in enumerate(report.sources, 1):
            table.add_row(
                str(i),
                src.source_type.value,
                src.content_depth.value,
                src.title[:70],
                str(len(src.key_findings)),
                str(len(src.key_evidence) + len(src.evidenced_findings)),
            )
        console.print(table)
        console.print(
            "[dim]Use --inspect ID --source N to see findings and evidence for source #N[/dim]"
        )

    # Iterations
    if report.iterations:
        lines = []
        for it in report.iterations:
            lines.append(
                f"  [bold]Iter {it.iteration_number}[/bold]: "
                f"{len(it.sources)} sources, {len(it.knowledge_gaps)} gaps"
            )
            for g in it.knowledge_gaps:
                lines.append(f"      - [{g.priority}] {g.gap_description}")
                if g.suggested_query:
                    lines.append(f"        query: {g.suggested_query}")
        console.print(Panel("\n".join(lines), title=f"Iterations ({len(report.iterations)})", border_style="magenta"))

    # Follow-ups
    if report.follow_up_questions:
        console.print(
            Panel(
                "\n".join(f"  {i+1}. {q}" for i, q in enumerate(report.follow_up_questions)),
                title="Suggested Follow-ups",
                border_style="green",
            )
        )

    # Executive summary (always)
    if report.executive_summary:
        console.print(Panel(report.executive_summary, title="Executive Summary", border_style="white"))

    # Detailed findings only with --full
    if full and report.detailed_findings:
        console.print(Panel(report.detailed_findings, title="Detailed Findings", border_style="white"))
    elif report.detailed_findings:
        console.print(
            f"\n[dim]Detailed findings: {len(report.detailed_findings)} chars. "
            f"Pass --full to print, or open the saved markdown in outputs/.[/dim]"
        )


async def async_main(query: str, config: ResearchConfig, agent_mode: bool = False) -> None:
    console.print(f"\n[bold]Researching:[/bold] {query}")
    if agent_mode:
        console.print("[dim]Agent mode: autonomous research loop[/dim]")
    console.print()

    try:
        if agent_mode:
            report = await arun_agent_research(query, config)
        else:
            report = await arun_research(query, config)
    except Exception as e:
        console.print(f"\n[red]Research failed: {e}[/red]")
        sys.exit(1)

    display_report(report)

    path = save_report(report)
    session_id = db_save(report, depth=config.depth.value)
    report.id = session_id
    console.print(f"[green]Report saved to:[/green] {path}")
    console.print(f"[green]Session ID:[/green] {session_id}\n")

    await followup_loop(report, config)


async def async_resynthesize(session_id: int, config: ResearchConfig) -> None:
    from .pipeline.synthesizer import asynthesize_report
    from .pipeline.templates import get_template
    from .usage import new_run_tracker, stage

    prev = db_load(session_id)
    if not prev:
        console.print(f"[red]Session {session_id} not found.[/red]")
        sys.exit(1)
    if not prev.sources:
        console.print(f"[red]Session {session_id} has no sources to synthesize.[/red]")
        sys.exit(1)

    tmpl = get_template(config.template) if config.template else None
    template_guidance = tmpl.synthesis_guidance if tmpl else ""
    system_prompt = tmpl.system_prompt if tmpl else ""

    suffix = f", template={tmpl.name}" if tmpl else ""
    console.print(f"\n[bold]Re-synthesizing session {session_id}:[/bold] {prev.query}")
    console.print(
        f"[dim]{len(prev.sources)} sources, {len(prev.iterations)} iterations "
        f"-> style={config.report_style.value}{suffix}[/dim]\n"
    )

    tracker = new_run_tracker()
    try:
        with console.status("[cyan]Synthesizing report from existing sources..."):
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
        console.print(f"\n[red]Synthesis failed: {e}[/red]")
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
    console.print(f"\n[dim]{tracker.format_summary()}[/dim]")

    display_report(new_report)
    path = save_report(new_report)
    new_id = db_save(new_report, depth=config.depth.value, parent_session_id=session_id)
    new_report.id = new_id
    console.print(f"[green]Report saved to:[/green] {path}")
    console.print(f"[green]New session ID:[/green] {new_id} [dim](parent: {session_id}, resynthesized)[/dim]\n")


async def async_continue(session_id: int, config: ResearchConfig) -> None:
    prev = db_load(session_id)
    if not prev:
        console.print(f"[red]Session {session_id} not found.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Continuing session {session_id}:[/bold] {prev.query}")
    console.print(f"[dim]Previous research had {len(prev.sources)} sources across {len(prev.iterations)} iterations[/dim]\n")

    try:
        report = await acontinue_research(prev, config)
    except Exception as e:
        console.print(f"\n[red]Research failed: {e}[/red]")
        sys.exit(1)

    display_report(report)

    path = save_report(report)
    new_id = db_save(report, depth=config.depth.value, parent_session_id=session_id)
    report.id = new_id
    console.print(f"[green]Report saved to:[/green] {path}")
    console.print(f"[green]New session ID:[/green] {new_id} [dim](parent: {session_id})[/dim]\n")

    await followup_loop(report, config)


async def followup_loop(report, config: ResearchConfig) -> None:
    while True:
        followup = Prompt.ask(
            "[bold]Research a follow-up question? (enter number, new query, or 'q' to quit)[/bold]",
            default="q",
        )
        if followup.lower() in ("q", "quit", "exit"):
            break

        if followup.isdigit() and report.follow_up_questions:
            idx = int(followup) - 1
            if 0 <= idx < len(report.follow_up_questions):
                followup = report.follow_up_questions[idx]

        console.print(f"\n[bold]Researching:[/bold] {followup}\n")
        try:
            report = await arun_research(followup, config)
            display_report(report)
            path = save_report(report)
            session_id = db_save(report, depth=config.depth.value)
            console.print(f"[green]Report saved to:[/green] {path}")
            console.print(f"[green]Session ID:[/green] {session_id}\n")
        except Exception as e:
            console.print(f"\n[red]Research failed: {e}[/red]")

    console.print("[dim]Goodbye![/dim]")


_SUBCOMMANDS: tuple[str, ...] = ("setup", "migrate", "login", "logout", "keys")


def _handle_subcommand(argv: list[str]) -> bool:
    """
    Intercept first-positional-arg subcommands (setup / migrate / login /
    logout / keys) before the main argparse runs. Returns True when a
    subcommand was handled, telling main() to exit.
    """
    if len(argv) < 2 or argv[1] not in _SUBCOMMANDS:
        return False
    name, *rest = argv[1:]
    if name == "setup":
        from .setup_wizard import run_setup_wizard
        run_setup_wizard()
    elif name == "login":
        from .setup_wizard import run_login
        run_login(rest[0] if rest else None)
    elif name == "logout":
        from .setup_wizard import run_logout
        run_logout(rest[0] if rest else None)
    elif name == "keys":
        from .setup_wizard import show_keys_status
        show_keys_status()
    elif name == "migrate":
        _run_migrate(rest)
    return True


def _run_migrate(args: list[str]) -> None:
    """Move a legacy <repo>/data/research.db into the XDG data dir."""
    import shutil
    from .config import DATA_DIR, DB_PATH

    if args:
        candidate = Path(args[0]).expanduser().resolve()
    else:
        # Try a couple of obvious places. PROJECT_ROOT under pipx-install
        # points at the venv, so we look from cwd up.
        candidates = [
            Path.cwd() / "data" / "research.db",
            Path.cwd().parent / "data" / "research.db",
            Path.home() / "Documents" / "Projects" / "Deep-Research" / "data" / "research.db",
        ]
        candidate = next((p for p in candidates if p.exists()), None)
        if candidate is None:
            console.print(
                "[red]Could not find a legacy research.db.[/red] "
                "Pass an explicit path: deep-research migrate /path/to/research.db"
            )
            return

    if not candidate.exists():
        console.print(f"[red]No file at {candidate}.[/red]")
        return
    if candidate == DB_PATH:
        console.print(f"[dim]{candidate} is already the active DB. Nothing to do.[/dim]")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        backup = DB_PATH.with_suffix(".db.bak")
        console.print(f"[yellow]Backing up existing {DB_PATH} -> {backup}[/yellow]")
        shutil.copy2(DB_PATH, backup)

    shutil.copy2(candidate, DB_PATH)
    console.print(
        f"[green]Migrated[/green] {candidate} -> {DB_PATH}\n"
        f"[dim]The original is untouched; delete it manually once you've confirmed.[/dim]"
    )


def main() -> None:
    setup_logging()

    # Intercept subcommands (setup / login / logout / keys / migrate) so
    # users get a Claude-Code-style flow without the main argparse fighting
    # for the first positional argument.
    if _handle_subcommand(sys.argv):
        return

    args = parse_args()

    if args.history:
        show_history()
        return

    if args.delete:
        if delete_session(args.delete):
            console.print(f"[green]Session {args.delete} deleted.[/green]")
        else:
            console.print(f"[red]Session {args.delete} not found.[/red]")
        return

    if args.inspect:
        show_session_detail(args.inspect, full=args.full, source_index=args.source)
        return

    if args.fork_session:
        new_query = args.fork_query.strip() or None
        new_id = fork_session(args.fork_session, new_query=new_query)
        if new_id is None:
            console.print(f"[red]Session {args.fork_session} not found.[/red]")
            sys.exit(1)
        console.print(
            f"[green]Forked session {args.fork_session} -> new session {new_id}.[/green]"
        )
        if new_query:
            console.print(f"[dim]Fork query: {new_query}[/dim]")
        console.print(
            f"[dim]Run --continue-session {new_id} to refine the fork "
            f"without touching session {args.fork_session}.[/dim]"
        )
        return

    config = build_config(args)

    # If the user launched bare `deep-research`, the REPL prints its own
    # banner — skip the one-shot Panel so we don't double up on welcome text.
    is_repl_launch = (
        not args.query
        and not args.resynthesize
        and not args.continue_session
    )

    if not is_repl_launch:
        search_engine = "Tavily" if settings.tavily_api_key else "DuckDuckGo"
        console.print(
            Panel(
                "[bold]Deep Research[/bold]\n"
                f"[dim]Provider: {settings.active_provider} ({settings.effective_endpoint_family}) | "
                f"Profile: {settings.llm_route} | Model: {settings.active_model} | "
                f"Search: {search_engine} | Depth: {config.depth.value}[/dim]",
                border_style="blue",
            )
        )

        # Heuristic warning: deep depth + agent mode on opencode-go can burn
        # through the $12/5h cap quickly. We can't query remaining budget — the
        # OpenCode Go usage API isn't published — so this is purely advisory.
        if (
            settings.active_provider == "opencode-go"
            and config.depth == ResearchDepth.DEEP
            and getattr(args, "agent", False)
            and not settings.llm_fallback_provider
        ):
            console.print(
                "[yellow]Warning: --depth deep + --agent on OpenCode Go can consume a "
                "significant share of the $12/5h cap. Consider setting "
                "LLM_FALLBACK_PROVIDER (e.g. zai) so the run can survive a 429.[/yellow]"
            )

    if not check_config(quiet=is_repl_launch):
        if not settings.active_api_key:
            console.print(
                f"\n[yellow]No {settings.active_api_key_env_name} found in env, .env, or keyring.[/yellow]"
            )
            from rich.prompt import Confirm
            from .setup_wizard import run_setup_wizard
            if Confirm.ask("[bold]Run the credential setup wizard now?[/bold]", default=True):
                run_setup_wizard()
                settings.reload_credentials()
                if not settings.active_api_key:
                    console.print(f"[red]Still no {settings.active_api_key_env_name} configured. Exiting.[/red]")
                    sys.exit(1)
            else:
                console.print(
                    "[dim]Run `deep-research setup` to store credentials, or set "
                    f"{settings.active_api_key_env_name} in your environment.[/dim]"
                )
                sys.exit(1)

    if args.resynthesize:
        asyncio.run(async_resynthesize(args.resynthesize, config))
        return

    if args.continue_session:
        if args.agent:
            console.print(
                "[yellow]Note: --agent has no effect with --continue-session; "
                "continuation always uses the gap-analysis loop.[/yellow]"
            )
        asyncio.run(async_continue(args.continue_session, config))
        return

    # No positional query and no one-shot flag → launch the interactive REPL.
    # Direct one-shot usage like `deep-research "topic"` still skips the REPL.
    if not args.query:
        from .repl import ReplState, run_repl
        run_repl(ReplState(
            depth=ResearchDepth(args.depth),
            style=ReportStyle(args.style),
            template=args.template,
            thinking=args.thinking,
            academic=args.academic,
            agent=args.agent,
            fresh=args.fresh,
        ))
        return

    query = " ".join(args.query)

    if not query.strip():
        console.print("[red]No query provided.[/red]")
        sys.exit(1)

    asyncio.run(async_main(query, config, agent_mode=args.agent))


if __name__ == "__main__":
    main()
