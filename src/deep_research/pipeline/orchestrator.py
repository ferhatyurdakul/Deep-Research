from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from deep_research.config import settings
from deep_research.extraction.parser import ContentParser
from deep_research.models import (
    ContentDepth,
    ResearchConfig,
    ResearchDepth,
    ResearchIteration,
    ResearchReport,
    SourceAnalysis,
    SourceType,
)
from deep_research.search.aggregator import SearchAggregator
from deep_research.storage.db import get_known_urls
from deep_research.urls import normalize_url
from deep_research.usage import new_run_tracker, stage

from .analyzer import aanalyze_source, aidentify_gaps, format_analyses, set_llm_concurrency
from .decomposer import adecompose_query
from .synthesizer import asynthesize_report_safe
from .templates import get_template

logger = logging.getLogger(__name__)
console = Console()


def _build_aggregator(config: ResearchConfig) -> SearchAggregator:
    return SearchAggregator(
        tavily_api_key=settings.tavily_api_key,
        semantic_scholar_api_key=getattr(settings, "semantic_scholar_api_key", ""),
        max_concurrency=config.max_search_concurrency,
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
        academic_weight=0.65 if config.use_academic_search else 0.5,
    )
    progress.update(task, completed=True, description=f"[green]Found {len(search_results)} results")

    # Filter already-seen URLs. Comparisons use the normalized form so
    # scheme/www/trailing-slash variants of a known page can't slip past
    # the dedup (the aggregator already dedups this way within a search).
    new_results = [r for r in search_results if normalize_url(r.url) not in seen_urls]
    for r in new_results:
        seen_urls.add(normalize_url(r.url))

    console.print(f"  [dim]{len(new_results)} new unique sources[/dim]")

    if not new_results:
        return [], []

    # Extract content using ContentParser
    task = progress.add_task("[cyan]Extracting content...", total=None)
    async with ContentParser() as parser:
        # Honor the depth preset's scrape budget; the rest fall through to
        # the snippet-fallback list below rather than being fetched.
        scraped = await parser.parse_many(
            new_results[:config.max_scrape_pages],
            max_concurrent=config.max_search_concurrency,
        )
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
            # Determine content depth: arXiv abstracts are short summaries
            if sr and sr.source_type == SourceType.ARXIV and page.word_count < 1000:
                depth = ContentDepth.ABSTRACT
            else:
                depth = ContentDepth.FULL_TEXT
            analyze_tasks.append(aanalyze_source(
                query, page.url, page.title, page.content,
                thinking=config.use_thinking, model=config.models.get("analyze"),
                extract_data=config.use_extraction,
                source_type=sr.source_type if sr else SourceType.WEB,
                authors=sr.authors if sr else [],
                published_date=sr.published_date if sr else "",
                extra=sr.extra if sr else {},
                content_depth=depth,
            ))
        results = await asyncio.gather(*analyze_tasks, return_exceptions=True)
        analyzed, failed = [], []
        for page, result in zip(scraped, results):
            if isinstance(result, BaseException):
                failed.append(result)
                logger.warning("Analysis failed for %s: %r", page.url, result)
            else:
                analyzed.append(result)
        if failed and not analyzed:
            # Every source failed — systemic outage (e.g. the provider's rate
            # limit outlived the whole retry budget), not one bad page. Raise
            # rather than synthesize a hollow, source-less report.
            raise failed[0]
        if failed:
            console.print(
                f"  [yellow]{len(failed)}/{len(results)} source analyses failed; "
                "continuing with the rest[/yellow]"
            )
        progress.update(
            task, completed=True,
            description=f"[green]Analyzed {len(analyzed)}/{len(results)} sources",
        )

    # Snippet fallbacks for unscraped results
    scraped_urls = {s.url for s in scraped}
    snippet_sources = [
        SourceAnalysis(
            url=r.url, title=r.title,
            key_findings=[r.snippet], relevance="snippet only",
            content_depth=ContentDepth.SNIPPET,
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
    tracker = new_run_tracker()
    set_llm_concurrency(config.max_llm_concurrency)
    aggregator = _build_aggregator(config)

    tmpl = get_template(config.template) if config.template else None
    if tmpl:
        if tmpl.use_academic:
            config.use_academic_search = True
        if tmpl.use_extraction:
            config.use_extraction = True
        console.print(f"  [dim]Using template: {tmpl.name}[/dim]")

    report = ResearchReport(query=query)
    known_urls: set[str] = set()
    if not config.fresh:
        known_urls = {normalize_url(u) for u in get_known_urls()}
    seen_urls: set[str] = set(known_urls)
    all_searched_queries: list[str] = []
    if known_urls:
        console.print(f"  [dim]Skipping {len(known_urls)} URLs known from previous sessions[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Decompose
        task = progress.add_task("[cyan]Decomposing research question...", total=None)
        async with stage("decompose"):
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

        # Iteration loop. If an iteration fails after sources were already
        # collected (e.g. a 429 that outlives the retry budget), stop the loop
        # and synthesize from what exists — the session is saved and
        # --continue-session can resume it. With nothing collected there is no
        # partial result worth saving, so the error propagates.
        for iteration_num in range(1, config.max_iterations + 1):
            try:
                iteration = ResearchIteration(iteration_number=iteration_num)

                if iteration_num == 1:
                    queries_to_search = [sq.question for sq in report.sub_questions]
                else:
                    task = progress.add_task(
                        f"[cyan]Iteration {iteration_num}: Analyzing knowledge gaps...", total=None
                    )
                    async with stage("gap_analysis"):
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

                async with stage("analyze"):
                    analyzed, snippets = await _search_and_analyze(
                        query, queries_to_search, seen_urls, config, progress, aggregator
                    )
                iteration.sources = analyzed
                report.sources.extend(analyzed)
                report.sources.extend(snippets)
                report.iterations.append(iteration)
            except Exception as e:
                if not report.sources:
                    raise
                logger.warning("Iteration %d aborted: %r", iteration_num, e)
                console.print(
                    f"\n[yellow]Iteration {iteration_num} failed after {len(report.sources)} "
                    f"sources were collected: {e}\nContinuing to synthesis — the session "
                    "will be saved; re-run with --continue-session <id> to resume research.[/yellow]"
                )
                break

        # Run-scoped: only URLs first seen in THIS run. The global cross-session
        # history lives in analyzed_urls; storing it per session made every
        # session row (and every fork of it) grow without bound.
        report.searched_urls = sorted(seen_urls - known_urls)

        if not report.sources:
            console.print(
                "\n[yellow]No analyzable sources were found — skipping synthesis. "
                "URLs already analyzed in earlier sessions are skipped; try "
                "--fresh to revisit them, or a broader query.[/yellow]"
            )

        # Synthesize
        task = progress.add_task("[cyan]Synthesizing final report...", total=None)
        async with stage("synthesize"):
            executive, detailed, follow_ups, ok = await asynthesize_report_safe(
                query, report.sub_questions, report.sources,
                thinking=config.use_thinking, model=config.models.get("synthesize"),
                template_guidance=tmpl.synthesis_guidance if tmpl else "",
                system_prompt=tmpl.system_prompt if tmpl else "",
                report_style=config.report_style,
                iteration_count=len(report.iterations),
            )
        report.executive_summary = executive
        report.detailed_findings = detailed
        report.follow_up_questions = follow_ups
        if ok:
            progress.update(task, completed=True, description="[green]Report complete!")
        else:
            progress.update(task, completed=True, description="[yellow]Synthesis failed — sources preserved")

    report.usage = tracker.snapshot()
    if not ok and report.sources:
        console.print(
            f"\n[yellow]Synthesis did not complete ({len(report.sources)} sources collected). "
            "The session will be saved; re-run with --resynthesize <id> to retry just synthesis.[/yellow]"
        )
    console.print(f"\n[dim]{tracker.format_summary()}[/dim]")
    return report


async def acontinue_research(
    previous: ResearchReport, config: ResearchConfig | None = None,
) -> ResearchReport:
    """Continue a previous research session."""
    config = config or ResearchConfig.from_depth(ResearchDepth.STANDARD)
    tracker = new_run_tracker()
    set_llm_concurrency(config.max_llm_concurrency)
    aggregator = _build_aggregator(config)

    # Same template handling as arun_research: templates toggle academic
    # search/extraction and steer synthesis prompts.
    tmpl = get_template(config.template) if config.template else None
    if tmpl:
        if tmpl.use_academic:
            config.use_academic_search = True
        if tmpl.use_extraction:
            config.use_extraction = True
        console.print(f"  [dim]Using template: {tmpl.name}[/dim]")

    report = ResearchReport(
        query=previous.query,
        sub_questions=previous.sub_questions,
        sources=list(previous.sources),
        iterations=list(previous.iterations),
        searched_urls=list(previous.searched_urls),
    )
    # Seed with everything the previous session touched plus the global
    # cross-session registry (normalized), so continuation doesn't re-fetch
    # pages the original run already analyzed.
    known_urls: set[str] = set() if config.fresh else {normalize_url(u) for u in get_known_urls()}
    known_urls |= {normalize_url(u) for u in previous.searched_urls}
    known_urls |= {normalize_url(s.url) for s in previous.sources}
    seen_urls: set[str] = set(known_urls)
    all_searched_queries: list[str] = [sq.question for sq in previous.sub_questions]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Same partial-result policy as arun_research: a failed iteration
        # after sources exist stops the loop and synthesizes from what was
        # already gathered; with nothing new collected the error propagates.
        for iteration_num in range(1, config.max_iterations + 1):
            try:
                actual_num = len(report.iterations) + 1
                iteration = ResearchIteration(iteration_number=actual_num)

                task = progress.add_task(
                    f"[cyan]Iteration {actual_num}: Analyzing knowledge gaps...", total=None
                )
                async with stage("gap_analysis"):
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

                async with stage("analyze"):
                    analyzed, snippets = await _search_and_analyze(
                        report.query, queries_to_search, seen_urls, config, progress, aggregator
                    )
                iteration.sources = analyzed
                report.sources.extend(analyzed)
                report.sources.extend(snippets)
                report.iterations.append(iteration)
            except Exception as e:
                if not report.sources:
                    raise
                logger.warning("Continuation iteration %d aborted: %r", iteration_num, e)
                console.print(
                    f"\n[yellow]Iteration {len(report.iterations) + 1} failed after "
                    f"{len(report.sources)} sources were collected: {e}\nContinuing to "
                    "synthesis — the session will be saved; re-run with "
                    "--continue-session <id> to resume research.[/yellow]"
                )
                break

        report.searched_urls = sorted(seen_urls - known_urls)

        task = progress.add_task("[cyan]Synthesizing updated report...", total=None)
        async with stage("synthesize"):
            executive, detailed, follow_ups, ok = await asynthesize_report_safe(
                report.query, report.sub_questions, report.sources,
                thinking=config.use_thinking, model=config.models.get("synthesize"),
                template_guidance=tmpl.synthesis_guidance if tmpl else "",
                system_prompt=tmpl.system_prompt if tmpl else "",
                report_style=config.report_style,
                iteration_count=len(report.iterations),
            )
        report.executive_summary = executive
        report.detailed_findings = detailed
        report.follow_up_questions = follow_ups
        if ok:
            progress.update(task, completed=True, description="[green]Report complete!")
        else:
            progress.update(task, completed=True, description="[yellow]Synthesis failed — sources preserved")

    report.usage = tracker.snapshot()
    if not ok and report.sources:
        console.print(
            f"\n[yellow]Synthesis did not complete ({len(report.sources)} sources collected). "
            "The session will be saved; re-run with --resynthesize <id> to retry just synthesis.[/yellow]"
        )
    console.print(f"\n[dim]{tracker.format_summary()}[/dim]")
    return report


def run_research(query: str, config: ResearchConfig | None = None) -> ResearchReport:
    return asyncio.run(arun_research(query, config))
