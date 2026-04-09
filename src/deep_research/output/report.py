from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from deep_research.config import OUTPUTS_DIR
from deep_research.models import ResearchReport

from .citations import format_citation

console = Console()


def format_report_markdown(report: ResearchReport) -> str:
    lines = [
        f"# Deep Research Report",
        f"",
        f"**Query:** {report.query}",
        f"**Date:** {report.created_at.strftime('%Y-%m-%d %H:%M')}",
        f"**Sources analyzed:** {len(report.sources)}",
        f"**Research iterations:** {len(report.iterations)}",
        f"",
        f"---",
        f"",
        report.executive_summary,
        f"",
        report.detailed_findings,
        f"",
        f"---",
        f"",
    ]

    # Only add References if not already present in detailed_findings
    # (synthesizer's replace_references_in_report may have already added them)
    if "## references" not in report.detailed_findings.lower():
        lines.append(f"## References")
        lines.append(f"")
        for i, source in enumerate(report.sources, 1):
            lines.append(format_citation(i, source))

    # Extracted structured data section — with source attribution
    has_extraction = any(s.extracted_data for s in report.sources if s.extracted_data)
    if has_extraction:
        attributed_stats: list[tuple[str, int]] = []
        all_entities: set[str] = set()
        attributed_claims: list[tuple[str, int]] = []
        for i, s in enumerate(report.sources, 1):
            if s.extracted_data:
                for stat in s.extracted_data.statistics:
                    attributed_stats.append((stat, i))
                all_entities.update(s.extracted_data.entities)
                for claim in s.extracted_data.claims:
                    attributed_claims.append((claim, i))

        if attributed_stats or all_entities or attributed_claims:
            lines.append("")
            lines.append("## Key Data Points")
            lines.append("")
            if attributed_stats:
                lines.append("### Statistics")
                for stat, src_idx in attributed_stats[:15]:
                    lines.append(f"- {stat} [{src_idx}]")
            if all_entities:
                lines.append("")
                lines.append("### Key Entities")
                lines.append(", ".join(sorted(all_entities)[:20]))
            if attributed_claims:
                lines.append("")
                lines.append("### Notable Claims")
                for claim, src_idx in attributed_claims[:10]:
                    lines.append(f"- {claim} [{src_idx}]")

    if report.iterations and len(report.iterations) > 1:
        lines.append("")
        lines.append("## Research Depth")
        lines.append(f"Completed {len(report.iterations)} research iterations.")
        for it in report.iterations[1:]:
            if it.knowledge_gaps:
                lines.append(f"\n**Iteration {it.iteration_number} — Gaps filled:**")
                for g in it.knowledge_gaps:
                    lines.append(f"- {g.gap_description}")

    if report.follow_up_questions:
        lines.append("")
        lines.append("## Follow-up Questions")
        lines.append("")
        for q in report.follow_up_questions:
            lines.append(f"- {q}")

    return "\n".join(lines)


def save_report(report: ResearchReport) -> Path:
    markdown = format_report_markdown(report)

    safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in report.query)
    safe_name = safe_name.strip().replace(" ", "_")[:60]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_name}.md"

    path = OUTPUTS_DIR / filename
    path.write_text(markdown, encoding="utf-8")
    return path


def display_report(report: ResearchReport) -> None:
    console.print()
    console.print(Panel("[bold green]Research Complete!", border_style="green"))
    console.print()

    console.print(Panel(Markdown(report.executive_summary), title="[bold]Executive Summary", border_style="blue"))
    console.print()

    console.print(Markdown(report.detailed_findings))
    console.print()

    console.print("[bold]References:[/bold]")
    for i, source in enumerate(report.sources, 1):
        console.print(f"  [{i}] [link={source.url}]{source.title}[/link] [dim]({source.relevance})[/dim]")
    console.print()

    if report.follow_up_questions:
        console.print("[bold]Follow-up Questions:[/bold]")
        for q in report.follow_up_questions:
            console.print(f"  - {q}")
        console.print()
