from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import settings
from .logging_config import setup_logging
from .models import ReportStyle, ResearchConfig, ResearchDepth
from .output.report import display_report, save_report
from .pipeline.orchestrator import arun_research, acontinue_research
from .pipeline.agent import arun_agent_research
from .storage.db import save_report as db_save, load_report as db_load, list_sessions, delete_session

console = Console()


def check_config() -> bool:
    ok = True
    if not settings.zai_api_key:
        console.print("[red]Error: ZAI_API_KEY not set in .env file[/red]")
        ok = False
    if not settings.tavily_api_key:
        console.print("[yellow]Warning: TAVILY_API_KEY not set — falling back to DuckDuckGo for web search[/yellow]")
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
        "--history", action="store_true",
        help="List past research sessions",
    )
    parser.add_argument(
        "--continue-session", type=int, metavar="ID",
        help="Continue a previous research session by ID",
    )
    parser.add_argument(
        "--delete", type=int, metavar="ID",
        help="Delete a research session by ID",
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
    config.load_model_routes()
    return config


def show_history() -> None:
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No research sessions found.[/dim]")
        return

    table = Table(title="Research History")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Query", style="white", max_width=60)
    table.add_column("Depth", style="yellow")
    table.add_column("Sources", justify="right")
    table.add_column("Date", style="dim")

    for s in sessions:
        date_str = s["created_at"][:16].replace("T", " ")
        table.add_row(
            str(s["id"]),
            s["query"][:60],
            s["depth"],
            str(s["source_count"]),
            date_str,
        )

    console.print(table)
    console.print(
        "\n[dim]Use --continue-session ID to deepen a previous session[/dim]"
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
    new_id = db_save(report, depth=config.depth.value)
    report.id = new_id
    console.print(f"[green]Report saved to:[/green] {path}")
    console.print(f"[green]New session ID:[/green] {new_id}\n")

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


def main() -> None:
    setup_logging()
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

    config = build_config(args)

    search_engine = "Tavily" if settings.tavily_api_key else "DuckDuckGo"
    console.print(
        Panel(
            "[bold]Deep Research[/bold]\n"
            f"[dim]Model: {settings.glm_model} | Search: {search_engine} | Depth: {config.depth.value}[/dim]",
            border_style="blue",
        )
    )

    if not check_config():
        if not settings.zai_api_key:
            console.print("\n[yellow]Copy .env.example to .env and add your API keys.[/yellow]")
            sys.exit(1)

    if args.continue_session:
        asyncio.run(async_continue(args.continue_session, config))
        return

    if args.query:
        query = " ".join(args.query)
    else:
        query = Prompt.ask("\n[bold]What would you like to research?[/bold]")

    if not query.strip():
        console.print("[red]No query provided.[/red]")
        sys.exit(1)

    asyncio.run(async_main(query, config, agent_mode=args.agent))


if __name__ == "__main__":
    main()
