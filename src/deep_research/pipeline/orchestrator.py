from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from deep_research.config import settings
from deep_research.extraction.parser import ContentParser
from deep_research.models import (
    ResearchConfig,
    ResearchDepth,
    ResearchIteration,
    ResearchReport,
    SourceAnalysis,
    SourceType,
)
from deep_research.search.aggregator import SearchAggregator
from deep_research.storage.db import get_known_urls

from .analyzer import aanalyze_source, aidentify_gaps, format_analyses
from .decomposer import adecompose_query
from .synthesizer import asynthesize_report
from .templates import get_template

logger = logging.getLogger(__name__)
console = Console()


def _build_aggregator() -> SearchAggregator:
    return SearchAggregator(
        tavily_api_key=settings.tavily_api_key,
        semantic_scholar_api_key=getattr(settings, "semantic_scholar_api_key", ""),
    )


async def _search_and_analyze(
    query: str,
    queries_to_search: list[str],
    seen_urls: set[str],
    config: ResearchConfig,
    progress: Progress,
    aggregator: SearchAggregator,
) -> tuple[list[SourceAnalysis], list[SourceAnalysis]]:
    """Search all providers -> extract content -> analyze sources."""
    # Determine which search sources to use
    sources_to_search = None  # None = all available
    if not config.use_academic_search:
        sources_to_search = ["web"]
        if "ddg" in aggregator.available_sources:
            sources_to_search.append("ddg")

    task = progress.add_task(
        f"[cyan]Searching {len(queries_to_search)} queries...",
        total=len(queries_to_search),
    )

    search_results = await aggregator.search(
        queries=queries_to_search,
        sources=sources_to_search,
        max_results_per_query=config.max_search_results,
        max_total=config.max_sources,
    )
    progress.update(task, completed=True, description=f"[green]Found {len(search_results)} results")

    # Filter already-seen URLs
    new_results = [r for r in search_results if r.url not in seen_urls]
    for r in new_results:
        seen_urls.add(r.url)

    console.print(f"  [dim]{len(new_results)} new unique sources[/dim]")

    if not new_results:
        return [], []

    # Extract content using ContentParser
    task = progress.add_task("[cyan]Extracting content...", total=None)
    async with ContentParser() as parser:
        scraped = await parser.parse_many(new_results, max_concurrent=5)
    progress.update(task, completed=True, description=f"[green]Extracted {len(scraped)} pages")

    # Build URL → SearchResult lookup for metadata
    url_to_result = {r.url: r for r in new_results}

    # Analyze sources
    analyzed = []
    if scraped:
        task = progress.add_task(
            f"[cyan]Analyzing {len(scraped)} sources...", total=len(scraped)
        )
        analyze_tasks = []
        for page in scraped:
            sr = url_to_result.get(page.url)
            analyze_tasks.append(aanalyze_source(
                query, page.url, page.title, page.content,
                thinking=config.use_thinking, model=config.models.get("analyze"),
                extract_data=config.use_extraction,
                source_type=sr.source_type if sr else SourceType.WEB,
                authors=sr.authors if sr else [],
                published_date=sr.published_date if sr else "",
                extra=sr.extra if sr else {},
            ))
        analyzed = list(await asyncio.gather(*analyze_tasks))
        progress.update(task, completed=True, description="[green]Source analysis complete")

    # Snippet fallbacks for unscraped results
    scraped_urls = {s.url for s in scraped}
    snippet_sources = [
        SourceAnalysis(
            url=r.url, title=r.title,
            key_findings=[r.snippet], relevance="snippet only",
            source_type=r.source_type,
            authors=r.authors,
            published_date=r.published_date,
            extra=r.extra,
        )
        for r in new_results
        if r.url not in scraped_urls and r.snippet
    ][:5]

    return analyzed, snippet_sources


async def arun_research(query: str, config: ResearchConfig | None = None) -> ResearchReport:
    config = config or ResearchConfig.from_depth(ResearchDepth.STANDARD)
    aggregator = _build_aggregator()

    tmpl = get_template(config.template) if config.template else None
    if tmpl:
        if tmpl.use_academic:
            config.use_academic_search = True
        if tmpl.use_extraction:
            config.use_extraction = True
        console.print(f"  [dim]Using template: {tmpl.name}[/dim]")

    report = ResearchReport(query=query)
    previously_known = get_known_urls()
    seen_urls: set[str] = set(previously_known)
    all_searched_queries: list[str] = []
    if previously_known:
        console.print(f"  [dim]Skipping {len(previously_known)} URLs known from previous sessions[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Decompose
        task = progress.add_task("[cyan]Decomposing research question...", total=None)
        report.sub_questions = await adecompose_query(
            query, max_q=config.max_sub_questions, model=config.models.get("decompose"),
            template_guidance=tmpl.decompose_guidance if tmpl else "",
        )
        progress.update(task, completed=True, description="[green]Decomposed into sub-questions")

        console.print(
            Panel(
                "\n".join(f"  {i+1}. {sq.question}" for i, sq in enumerate(report.sub_questions)),
                title="[bold]Sub-Questions",
                border_style="cyan",
            )
        )

        # Iteration loop
        for iteration_num in range(1, config.max_iterations + 1):
            iteration = ResearchIteration(iteration_number=iteration_num)

            if iteration_num == 1:
                queries_to_search = [sq.question for sq in report.sub_questions]
            else:
                task = progress.add_task(
                    f"[cyan]Iteration {iteration_num}: Analyzing knowledge gaps...", total=None
                )
                gaps, is_sufficient = await aidentify_gaps(
                    query, report.sub_questions, report.sources, all_searched_queries,
                    thinking=config.use_thinking,
                    model=config.models.get("gap_analysis"),
                )
                iteration.knowledge_gaps = gaps
                progress.update(task, completed=True, description="[green]Gap analysis complete")

                if is_sufficient or not gaps:
                    console.print("[green]  Research coverage is sufficient.[/green]")
                    report.iterations.append(iteration)
                    break

                console.print(
                    Panel(
                        "\n".join(f"  - {g.gap_description}" for g in gaps),
                        title=f"[bold yellow]Iteration {iteration_num}: Filling Knowledge Gaps",
                        border_style="yellow",
                    )
                )
                queries_to_search = [g.suggested_query for g in gaps]

            all_searched_queries.extend(queries_to_search)

            analyzed, snippets = await _search_and_analyze(
                query, queries_to_search, seen_urls, config, progress, aggregator
            )
            iteration.sources = analyzed
            report.sources.extend(analyzed)
            report.sources.extend(snippets)
            report.iterations.append(iteration)

        report.searched_urls = list(seen_urls)

        # Synthesize
        task = progress.add_task("[cyan]Synthesizing final report...", total=None)
        executive, detailed, follow_ups = await asynthesize_report(
            query, report.sub_questions, report.sources,
            thinking=config.use_thinking, model=config.models.get("synthesize"),
            template_guidance=tmpl.synthesis_guidance if tmpl else "",
            system_prompt=tmpl.system_prompt if tmpl else "",
        )
        report.executive_summary = executive
        report.detailed_findings = detailed
        report.follow_up_questions = follow_ups
        progress.update(task, completed=True, description="[green]Report complete!")

    return report


async def acontinue_research(
    previous: ResearchReport, config: ResearchConfig | None = None,
) -> ResearchReport:
    """Continue a previous research session."""
    config = config or ResearchConfig.from_depth(ResearchDepth.STANDARD)
    aggregator = _build_aggregator()

    report = ResearchReport(
        query=previous.query,
        sub_questions=previous.sub_questions,
        sources=list(previous.sources),
        iterations=list(previous.iterations),
        searched_urls=list(previous.searched_urls),
    )
    seen_urls: set[str] = set(previous.searched_urls)
    all_searched_queries: list[str] = [sq.question for sq in previous.sub_questions]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for iteration_num in range(1, config.max_iterations + 1):
            actual_num = len(report.iterations) + 1
            iteration = ResearchIteration(iteration_number=actual_num)

            task = progress.add_task(
                f"[cyan]Iteration {actual_num}: Analyzing knowledge gaps...", total=None
            )
            gaps, is_sufficient = await aidentify_gaps(
                report.query, report.sub_questions, report.sources, all_searched_queries,
                thinking=config.use_thinking,
                model=config.models.get("gap_analysis"),
            )
            iteration.knowledge_gaps = gaps
            progress.update(task, completed=True, description="[green]Gap analysis complete")

            if is_sufficient or not gaps:
                console.print("[green]  Research coverage is sufficient.[/green]")
                report.iterations.append(iteration)
                break

            console.print(
                Panel(
                    "\n".join(f"  - {g.gap_description}" for g in gaps),
                    title=f"[bold yellow]Iteration {actual_num}: Filling Knowledge Gaps",
                    border_style="yellow",
                )
            )
            queries_to_search = [g.suggested_query for g in gaps]
            all_searched_queries.extend(queries_to_search)

            analyzed, snippets = await _search_and_analyze(
                report.query, queries_to_search, seen_urls, config, progress, aggregator
            )
            iteration.sources = analyzed
            report.sources.extend(analyzed)
            report.sources.extend(snippets)
            report.iterations.append(iteration)

        report.searched_urls = list(seen_urls)

        task = progress.add_task("[cyan]Synthesizing updated report...", total=None)
        executive, detailed, follow_ups = await asynthesize_report(
            report.query, report.sub_questions, report.sources,
            thinking=config.use_thinking, model=config.models.get("synthesize"),
        )
        report.executive_summary = executive
        report.detailed_findings = detailed
        report.follow_up_questions = follow_ups
        progress.update(task, completed=True, description="[green]Report complete!")

    return report


def run_research(query: str, config: ResearchConfig | None = None) -> ResearchReport:
    return asyncio.run(arun_research(query, config))
