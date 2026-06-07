from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from deep_research.config import settings
from deep_research.extraction.parser import ContentParser
from deep_research.llm import achat_json
from deep_research.models import (
    ContentDepth,
    KnowledgeGap,
    ResearchConfig,
    ResearchDepth,
    ResearchIteration,
    ResearchReport,
    SourceAnalysis,
    SourceType,
)
from deep_research.search.aggregator import SearchAggregator
from deep_research.storage.db import get_known_urls

from .analyzer import aanalyze_source, format_analyses
from .decomposer import adecompose_query
from .synthesizer import asynthesize_report
from .templates import get_template

logger = logging.getLogger(__name__)
console = Console()

MAX_AGENT_ITERATIONS = 8

# Agent mode runs its own decision loop whose budget is intentionally decoupled
# from the fixed-pipeline `max_iterations` (which is tuned for the non-agent loop
# and tops out at 3). Depth still scales the budget so `quick` stays lighter than
# `deep`, but every depth gets at least one decision cycle and the hard ceiling is
# MAX_AGENT_ITERATIONS. The agent stops early as soon as it judges coverage
# sufficient, so the budget is a ceiling rather than a target.
_AGENT_ITERATION_BUDGET: dict[ResearchDepth, int] = {
    ResearchDepth.QUICK: 3,
    ResearchDepth.STANDARD: 5,
    ResearchDepth.DEEP: 8,
}


def agent_iteration_budget(depth: ResearchDepth) -> int:
    """Iteration ceiling for the autonomous agent loop at a given depth.

    Guaranteed to be at least 2 so the decision loop (`range(2, budget + 1)`)
    always runs at least one cycle, and never exceeds MAX_AGENT_ITERATIONS.
    """
    budget = _AGENT_ITERATION_BUDGET.get(depth, MAX_AGENT_ITERATIONS)
    return max(2, min(budget, MAX_AGENT_ITERATIONS))

AGENT_DECISION_PROMPT = """You are an autonomous research agent. Evaluate the current research state and decide what to do next.

Original research question: {query}

Sub-questions investigated:
{sub_questions}

Findings so far ({source_count} sources analyzed):
{findings_summary}

Previously searched queries:
{searched_queries}

Current iteration: {iteration} of max {max_iterations}

Evaluate the research and choose ONE action:

1. "search_more" — Current findings are insufficient. Provide 2-3 targeted search queries to fill specific gaps.
2. "go_deeper" — Findings are broad but shallow on a key subtopic. Provide 1-2 focused queries to go deeper.
3. "broaden" — Findings cover the main topic but miss important related perspectives. Provide 1-2 queries exploring adjacent areas.
4. "sufficient" — Research comprehensively answers the original question. No more searching needed.

Respond in JSON:
{{
  "action": "search_more|go_deeper|broaden|sufficient",
  "reasoning": "brief explanation of why this action was chosen",
  "confidence": 0.0-1.0,
  "queries": ["query1", "query2"],
  "focus_area": "what aspect these queries target"
}}

Be critical. Only choose "sufficient" when findings truly cover the question comprehensively with multiple corroborating sources."""


def _build_aggregator() -> SearchAggregator:
    return SearchAggregator(
        tavily_api_key=settings.tavily_api_key,
        semantic_scholar_api_key=getattr(settings, "semantic_scholar_api_key", ""),
    )


async def _agent_search_and_analyze(
    query: str,
    queries_to_search: list[str],
    seen_urls: set[str],
    config: ResearchConfig,
    aggregator: SearchAggregator,
) -> tuple[list[SourceAnalysis], list[SourceAnalysis]]:
    """Search, extract, and analyze — returns (analyzed, snippet_fallbacks)."""
    sources_to_search = None
    if not config.use_academic_search:
        sources_to_search = ["web"]
        if "ddg" in aggregator.available_sources:
            sources_to_search.append("ddg")

    search_results = await aggregator.search(
        queries=queries_to_search,
        sources=sources_to_search,
        max_results_per_query=config.max_search_results,
        max_total=config.max_sources,
        academic_weight=0.65 if config.use_academic_search else 0.5,
    )

    new_results = [r for r in search_results if r.url not in seen_urls]
    for r in new_results:
        seen_urls.add(r.url)

    if not new_results:
        return [], []

    async with ContentParser() as parser:
        scraped = await parser.parse_many(new_results, max_concurrent=5)

    url_to_result = {r.url: r for r in new_results}

    analyzed = []
    if scraped:
        analyze_tasks = []
        for p in scraped:
            sr = url_to_result.get(p.url)
            if sr and sr.source_type == SourceType.ARXIV and p.word_count < 1000:
                depth = ContentDepth.ABSTRACT
            else:
                depth = ContentDepth.FULL_TEXT
            analyze_tasks.append(aanalyze_source(
                query, p.url, p.title, p.content,
                thinking=config.use_thinking,
                model=config.models.get("analyze"),
                extract_data=config.use_extraction,
                source_type=sr.source_type if sr else SourceType.WEB,
                authors=sr.authors if sr else [],
                published_date=sr.published_date if sr else "",
                extra=sr.extra if sr else {},
                content_depth=depth,
            ))
        analyzed = list(await asyncio.gather(*analyze_tasks))

    scraped_urls = {s.url for s in scraped}
    snippets = [
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

    return analyzed, snippets


async def arun_agent_research(query: str, config: ResearchConfig | None = None) -> ResearchReport:
    """Autonomous agent research loop."""
    config = config or ResearchConfig.from_depth(ResearchDepth.DEEP)
    aggregator = _build_aggregator()

    tmpl = get_template(config.template) if config.template else None
    if tmpl:
        if tmpl.use_academic:
            config.use_academic_search = True
        if tmpl.use_extraction:
            config.use_extraction = True
        console.print(f"  [dim]Using template: {tmpl.name}[/dim]")

    report = ResearchReport(query=query)
    if config.fresh:
        previously_known = set()
    else:
        previously_known = get_known_urls()
    seen_urls: set[str] = set(previously_known)
    all_searched_queries: list[str] = []
    max_iters = agent_iteration_budget(config.depth)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Decompose
        task = progress.add_task("[cyan]Agent: Decomposing research question...", total=None)
        report.sub_questions = await adecompose_query(
            query, max_q=config.max_sub_questions,
            model=config.models.get("decompose"),
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

        # Initial search
        queries_to_search = [sq.question for sq in report.sub_questions]
        all_searched_queries.extend(queries_to_search)

        task = progress.add_task("[cyan]Agent: Initial search...", total=None)
        analyzed, snippets = await _agent_search_and_analyze(
            query, queries_to_search, seen_urls, config, aggregator
        )
        iteration = ResearchIteration(iteration_number=1, sources=analyzed)
        report.sources.extend(analyzed)
        report.sources.extend(snippets)
        report.iterations.append(iteration)
        progress.update(task, completed=True, description=f"[green]Found {len(analyzed)} sources")

        # Agent decision loop
        for iter_num in range(2, max_iters + 1):
            task = progress.add_task(
                f"[cyan]Agent: Evaluating research quality (iteration {iter_num})...", total=None
            )

            sq_text = "\n".join(f"- {sq.question}" for sq in report.sub_questions)
            searched_text = "\n".join(f"- {q}" for q in all_searched_queries)
            decision_prompt = AGENT_DECISION_PROMPT.format(
                query=query,
                sub_questions=sq_text,
                findings_summary=format_analyses(report.sources),
                searched_queries=searched_text,
                source_count=len(report.sources),
                iteration=iter_num,
                max_iterations=max_iters,
            )
            decision = await achat_json(
                decision_prompt,
                thinking=config.use_thinking,
                model=config.models.get("gap_analysis"),
            )

            action = decision.get("action", "sufficient")
            reasoning = decision.get("reasoning", "")
            confidence = decision.get("confidence", 1.0)
            new_queries = decision.get("queries", [])
            focus = decision.get("focus_area", "")

            progress.update(task, completed=True, description=f"[green]Agent decided: {action}")

            if action == "sufficient":
                console.print(
                    Panel(
                        f"[green]{reasoning}[/green]\n[dim]Confidence: {confidence:.0%}[/dim]",
                        title="[bold green]Agent: Research Sufficient",
                        border_style="green",
                    )
                )
                break

            action_labels = {
                "search_more": "Searching for more information",
                "go_deeper": "Going deeper",
                "broaden": "Broadening perspective",
            }
            label = action_labels.get(action, action)
            console.print(
                Panel(
                    f"[yellow]{reasoning}[/yellow]\n"
                    f"[dim]Focus: {focus} | Confidence: {confidence:.0%}[/dim]\n"
                    f"Queries: {', '.join(new_queries)}",
                    title=f"[bold yellow]Agent: {label} (iteration {iter_num})",
                    border_style="yellow",
                )
            )

            if not new_queries:
                break

            all_searched_queries.extend(new_queries)

            task = progress.add_task(f"[cyan]Agent: {label}...", total=None)
            analyzed, snippets = await _agent_search_and_analyze(
                query, new_queries, seen_urls, config, aggregator
            )

            iteration = ResearchIteration(
                iteration_number=iter_num,
                sources=analyzed,
                knowledge_gaps=[
                    KnowledgeGap(
                        gap_description=focus,
                        suggested_query=q,
                        priority="high" if action == "search_more" else "medium",
                    )
                    for q in new_queries
                ],
            )
            report.sources.extend(analyzed)
            report.sources.extend(snippets)
            report.iterations.append(iteration)
            progress.update(task, completed=True, description=f"[green]Added {len(analyzed)} sources")

        report.searched_urls = list(seen_urls)

        task = progress.add_task("[cyan]Agent: Synthesizing final report...", total=None)
        executive, detailed, follow_ups = await asynthesize_report(
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
        progress.update(task, completed=True, description="[green]Report complete!")

    return report
