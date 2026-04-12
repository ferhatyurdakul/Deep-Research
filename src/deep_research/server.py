from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import markdown
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import settings
from .models import ContentDepth, ReportStyle, ResearchConfig, ResearchDepth, ResearchReport, SourceType
from .output.report import format_report_markdown, save_report
from .pipeline.templates import list_templates, get_template
from .storage.db import (
    save_report as db_save,
    load_report as db_load,
    list_sessions,
    delete_session,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Deep Research", version="0.2.0")

STATIC_DIR = Path(__file__).parent / "static"


# --- WebSocket progress bridge ---

class WebSocketProgress:
    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def send(self, event: str, data: dict | str):
        msg = json.dumps({"event": event, "data": data})
        try:
            await self.ws.send_text(msg)
        except Exception:
            pass

    async def update(self, step: str, detail: str = ""):
        await self.send("progress", {"step": step, "detail": detail})


# --- Research runner with WS progress ---

async def run_research_with_ws(
    query: str,
    config: ResearchConfig,
    progress: WebSocketProgress,
) -> ResearchReport:
    from .llm import achat_json
    from .extraction.parser import ContentParser
    from .models import ResearchIteration, SourceAnalysis
    from .pipeline.analyzer import aanalyze_source, aidentify_gaps
    from .pipeline.decomposer import adecompose_query
    from .pipeline.synthesizer import asynthesize_report
    from .search.aggregator import SearchAggregator
    from .storage.db import get_known_urls

    aggregator = SearchAggregator(
        tavily_api_key=settings.tavily_api_key,
        semantic_scholar_api_key=getattr(settings, "semantic_scholar_api_key", ""),
    )

    report = ResearchReport(query=query)
    previously_known = get_known_urls()
    seen_urls: set[str] = set(previously_known)
    all_searched_queries: list[str] = []

    # Step 1: Decompose
    await progress.update("decompose", "Breaking down research question...")
    report.sub_questions = await adecompose_query(
        query, max_q=config.max_sub_questions, model=config.models.get("decompose")
    )
    await progress.send("sub_questions", [
        {"question": sq.question, "reasoning": sq.reasoning}
        for sq in report.sub_questions
    ])

    # Iteration loop
    for iteration_num in range(1, config.max_iterations + 1):
        iteration = ResearchIteration(iteration_number=iteration_num)

        if iteration_num == 1:
            queries_to_search = [sq.question for sq in report.sub_questions]
        else:
            await progress.update("gap_analysis", f"Iteration {iteration_num}: Finding knowledge gaps...")
            gaps, is_sufficient = await aidentify_gaps(
                query, report.sub_questions, report.sources, all_searched_queries,
                thinking=config.use_thinking,
                model=config.models.get("gap_analysis"),
            )
            iteration.knowledge_gaps = gaps

            if is_sufficient or not gaps:
                await progress.update("gap_analysis", "Research coverage is sufficient")
                report.iterations.append(iteration)
                break

            await progress.send("gaps", [
                {"description": g.gap_description, "query": g.suggested_query}
                for g in gaps
            ])
            queries_to_search = [g.suggested_query for g in gaps]

        all_searched_queries.extend(queries_to_search)

        # Search via aggregator
        sources_to_search = None
        if not config.use_academic_search:
            sources_to_search = ["web"]
            if "ddg" in aggregator.available_sources:
                sources_to_search.append("ddg")

        await progress.update("search", f"Searching {len(queries_to_search)} queries...")
        search_results = await aggregator.search(
            queries=queries_to_search,
            sources=sources_to_search,
            max_results_per_query=config.max_search_results,
            max_total=config.max_sources,
        )
        await progress.update("search", f"Found {len(search_results)} results")

        new_results = [r for r in search_results if r.url not in seen_urls]
        for r in new_results:
            seen_urls.add(r.url)

        if not new_results:
            report.iterations.append(iteration)
            continue

        # Extract content
        await progress.update("scrape", f"Extracting content from {len(new_results)} pages...")
        async with ContentParser() as parser:
            scraped = await parser.parse_many(new_results, max_concurrent=5)
        await progress.update("scrape", f"Extracted {len(scraped)} pages")

        url_to_result = {r.url: r for r in new_results}

        # Analyze
        if scraped:
            await progress.update("analyze", f"Analyzing {len(scraped)} sources...")
            analyze_tasks = []
            for p in scraped:
                sr = url_to_result.get(p.url)
                if sr and sr.source_type == SourceType.ARXIV and p.word_count < 1000:
                    depth = ContentDepth.ABSTRACT
                else:
                    depth = ContentDepth.FULL_TEXT
                analyze_tasks.append(aanalyze_source(
                    query, p.url, p.title, p.content,
                    thinking=config.use_thinking, model=config.models.get("analyze"),
                    extract_data=config.use_extraction,
                    source_type=sr.source_type if sr else SourceType.WEB,
                    authors=sr.authors if sr else [],
                    published_date=sr.published_date if sr else "",
                    extra=sr.extra if sr else {},
                    content_depth=depth,
                ))
            analyzed = list(await asyncio.gather(*analyze_tasks))
            iteration.sources = analyzed
            report.sources.extend(analyzed)
            await progress.update("analyze", f"Analyzed {len(analyzed)} sources")

        # Snippet fallbacks
        scraped_urls = {s.url for s in scraped}
        for r in new_results:
            if r.url not in scraped_urls and r.snippet:
                report.sources.append(SourceAnalysis(
                    url=r.url, title=r.title,
                    key_findings=[r.snippet], relevance="snippet only",
                    content_depth=ContentDepth.SNIPPET,
                    source_type=r.source_type, authors=r.authors,
                    published_date=r.published_date, extra=r.extra,
                ))

        report.iterations.append(iteration)

    report.searched_urls = list(seen_urls)

    # Synthesize
    await progress.update("synthesize", "Generating final report...")
    executive, detailed, follow_ups = await asynthesize_report(
        query, report.sub_questions, report.sources,
        thinking=config.use_thinking, model=config.models.get("synthesize"),
        report_style=config.report_style,
        iteration_count=len(report.iterations),
    )
    report.executive_summary = executive
    report.detailed_findings = detailed
    report.follow_up_questions = follow_ups

    return report


async def run_agent_with_ws(
    query: str,
    config: ResearchConfig,
    progress: WebSocketProgress,
) -> ResearchReport:
    from .llm import achat_json
    from .extraction.parser import ContentParser
    from .models import ResearchIteration, SourceAnalysis, KnowledgeGap
    from .pipeline.analyzer import aanalyze_source, format_analyses
    from .pipeline.decomposer import adecompose_query
    from .pipeline.synthesizer import asynthesize_report
    from .pipeline.agent import AGENT_DECISION_PROMPT
    from .search.aggregator import SearchAggregator
    from .storage.db import get_known_urls

    aggregator = SearchAggregator(
        tavily_api_key=settings.tavily_api_key,
        semantic_scholar_api_key=getattr(settings, "semantic_scholar_api_key", ""),
    )

    report = ResearchReport(query=query)
    tmpl = get_template(config.template) if config.template else None
    if tmpl:
        if tmpl.use_academic:
            config.use_academic_search = True
        if tmpl.use_extraction:
            config.use_extraction = True

    previously_known = get_known_urls()
    seen_urls: set[str] = set(previously_known)
    all_searched_queries: list[str] = []
    from .pipeline.agent import MAX_AGENT_ITERATIONS
    max_iters = min(config.max_iterations, MAX_AGENT_ITERATIONS)

    # Decompose
    await progress.update("decompose", "Agent: Breaking down research question...")
    report.sub_questions = await adecompose_query(
        query, max_q=config.max_sub_questions,
        model=config.models.get("decompose"),
        template_guidance=tmpl.decompose_guidance if tmpl else "",
    )
    await progress.send("sub_questions", [
        {"question": sq.question, "reasoning": sq.reasoning}
        for sq in report.sub_questions
    ])

    # Initial search
    queries_to_search = [sq.question for sq in report.sub_questions]
    all_searched_queries.extend(queries_to_search)

    sources_to_search = None
    if not config.use_academic_search:
        sources_to_search = ["web"]
        if "ddg" in aggregator.available_sources:
            sources_to_search.append("ddg")

    await progress.update("search", f"Agent: Initial search ({len(queries_to_search)} queries)...")
    search_results = await aggregator.search(
        queries=queries_to_search,
        sources=sources_to_search,
        max_results_per_query=config.max_search_results,
        max_total=config.max_sources,
    )

    new_results = [r for r in search_results if r.url not in seen_urls]
    for r in new_results:
        seen_urls.add(r.url)

    await progress.update("scrape", f"Agent: Extracting {len(new_results)} pages...")
    async with ContentParser() as parser:
        scraped = await parser.parse_many(new_results, max_concurrent=5)

    url_to_result = {r.url: r for r in new_results}

    analyzed = []
    if scraped:
        await progress.update("analyze", f"Agent: Analyzing {len(scraped)} sources...")
        analyze_tasks = []
        for p in scraped:
            sr = url_to_result.get(p.url)
            if sr and sr.source_type == SourceType.ARXIV and p.word_count < 1000:
                depth = ContentDepth.ABSTRACT
            else:
                depth = ContentDepth.FULL_TEXT
            analyze_tasks.append(aanalyze_source(
                query, p.url, p.title, p.content,
                thinking=config.use_thinking, model=config.models.get("analyze"),
                extract_data=config.use_extraction,
                source_type=sr.source_type if sr else SourceType.WEB,
                authors=sr.authors if sr else [],
                published_date=sr.published_date if sr else "",
                extra=sr.extra if sr else {},
                content_depth=depth,
            ))
        analyzed = list(await asyncio.gather(*analyze_tasks))

    scraped_urls = {s.url for s in scraped}
    for r in new_results:
        if r.url not in scraped_urls and r.snippet:
            report.sources.append(SourceAnalysis(
                url=r.url, title=r.title, key_findings=[r.snippet], relevance="snippet only",
                content_depth=ContentDepth.SNIPPET,
                source_type=r.source_type, authors=r.authors,
                published_date=r.published_date, extra=r.extra,
            ))

    iteration = ResearchIteration(iteration_number=1, sources=analyzed)
    report.sources.extend(analyzed)
    report.iterations.append(iteration)

    # Agent decision loop
    for iter_num in range(2, max_iters + 1):
        await progress.update("agent_decision", f"Agent: Evaluating quality (iteration {iter_num})...")

        sq_text = "\n".join(f"- {sq.question}" for sq in report.sub_questions)
        searched_text = "\n".join(f"- {q}" for q in all_searched_queries)
        decision_prompt = AGENT_DECISION_PROMPT.format(
            query=query, sub_questions=sq_text,
            findings_summary=format_analyses(report.sources),
            searched_queries=searched_text,
            source_count=len(report.sources),
            iteration=iter_num, max_iterations=max_iters,
        )
        decision = await achat_json(decision_prompt, thinking=config.use_thinking,
                                     model=config.models.get("gap_analysis"))

        action = decision.get("action", "sufficient")
        reasoning = decision.get("reasoning", "")
        new_queries = decision.get("queries", [])
        focus = decision.get("focus_area", "")

        await progress.send("agent_decision", {
            "action": action, "reasoning": reasoning,
            "queries": new_queries, "focus": focus,
            "iteration": iter_num,
        })

        if action == "sufficient" or not new_queries:
            await progress.update("agent_decision", "Agent: Research sufficient")
            break

        all_searched_queries.extend(new_queries)

        await progress.update("search", f"Agent: {action} ({len(new_queries)} queries)...")
        search_results = await aggregator.search(
            queries=new_queries,
            sources=sources_to_search,
            max_results_per_query=config.max_search_results,
            max_total=config.max_sources,
        )

        new_results = [r for r in search_results if r.url not in seen_urls]
        for r in new_results:
            seen_urls.add(r.url)

        if new_results:
            url_to_result2 = {r.url: r for r in new_results}
            async with ContentParser() as parser:
                scraped = await parser.parse_many(new_results, max_concurrent=5)
            if scraped:
                analyze_tasks = []
                for p in scraped:
                    sr = url_to_result2.get(p.url)
                    if sr and sr.source_type == SourceType.ARXIV and p.word_count < 1000:
                        depth = ContentDepth.ABSTRACT
                    else:
                        depth = ContentDepth.FULL_TEXT
                    analyze_tasks.append(aanalyze_source(
                        query, p.url, p.title, p.content,
                        thinking=config.use_thinking, model=config.models.get("analyze"),
                        extract_data=config.use_extraction,
                        source_type=sr.source_type if sr else SourceType.WEB,
                        authors=sr.authors if sr else [],
                        published_date=sr.published_date if sr else "",
                        extra=sr.extra if sr else {},
                        content_depth=depth,
                    ))
                analyzed = list(await asyncio.gather(*analyze_tasks))
            else:
                analyzed = []

            scraped_urls = {s.url for s in scraped}
            for r in new_results:
                if r.url not in scraped_urls and r.snippet:
                    report.sources.append(SourceAnalysis(
                        url=r.url, title=r.title, key_findings=[r.snippet], relevance="snippet only",
                        content_depth=ContentDepth.SNIPPET,
                        source_type=r.source_type, authors=r.authors,
                        published_date=r.published_date, extra=r.extra,
                    ))

            iter_obj = ResearchIteration(
                iteration_number=iter_num, sources=analyzed,
                knowledge_gaps=[KnowledgeGap(gap_description=focus, suggested_query=q, priority="high")
                                for q in new_queries],
            )
            report.sources.extend(analyzed)
            report.iterations.append(iter_obj)

    report.searched_urls = list(seen_urls)

    # Synthesize
    await progress.update("synthesize", "Agent: Generating final report...")
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

    return report


# --- API Routes ---

@app.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Deep Research</h1><p>Frontend not found.</p>")


@app.get("/api/config")
async def get_config():
    return {
        "model": settings.glm_model,
        "default_depth": settings.default_depth,
        "depths": ["quick", "standard", "deep"],
        "styles": ["brief", "standard", "academic"],
        "templates": list_templates(),
    }


@app.get("/api/history")
async def get_history():
    sessions = list_sessions(limit=50)
    return {"sessions": sessions}


@app.get("/api/session/{session_id}")
async def get_session(session_id: int):
    report = db_load(session_id)
    if not report:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {
        "id": report.id,
        "query": report.query,
        "executive_summary": report.executive_summary,
        "detailed_findings": report.detailed_findings,
        "follow_up_questions": report.follow_up_questions,
        "sources": [s.model_dump() for s in report.sources],
        "sub_questions": [sq.model_dump() for sq in report.sub_questions],
        "iterations": [it.model_dump() for it in report.iterations],
        "created_at": report.created_at.isoformat(),
        "markdown": format_report_markdown(report),
        "html": markdown.markdown(format_report_markdown(report), extensions=["tables", "fenced_code"]),
    }


@app.delete("/api/session/{session_id}")
async def remove_session(session_id: int):
    if delete_session(session_id):
        return {"status": "deleted"}
    return JSONResponse({"error": "Session not found"}, status_code=404)


@app.get("/api/compare")
async def compare_sessions(ids: str):
    try:
        session_ids = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError:
        return JSONResponse({"error": "Invalid session IDs"}, status_code=400)

    if len(session_ids) < 2 or len(session_ids) > 4:
        return JSONResponse({"error": "Provide 2-4 session IDs"}, status_code=400)

    results = []
    for sid in session_ids:
        report = db_load(sid)
        if not report:
            return JSONResponse({"error": f"Session {sid} not found"}, status_code=404)
        results.append({
            "id": report.id,
            "query": report.query,
            "executive_summary": report.executive_summary,
            "detailed_findings": report.detailed_findings,
            "follow_up_questions": report.follow_up_questions,
            "sources": [s.model_dump() for s in report.sources],
            "source_count": len(report.sources),
            "iterations_count": len(report.iterations),
            "created_at": report.created_at.isoformat(),
            "html": markdown.markdown(format_report_markdown(report), extensions=["tables", "fenced_code"]),
        })

    return {"sessions": results}


@app.get("/api/session/{session_id}/markdown")
async def get_session_markdown(session_id: int):
    report = db_load(session_id)
    if not report:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    md = format_report_markdown(report)
    return JSONResponse(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=research_{session_id}.md"},
    )


PDF_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.6; color: #1a1a2e; }
h1 { font-size: 1.6em; color: #2d3436; border-bottom: 2px solid #6c5ce7; padding-bottom: 4px; }
h2 { font-size: 1.25em; color: #2d3436; margin-top: 1.2em; border-bottom: 1px solid #ddd; padding-bottom: 3px; }
h3 { font-size: 1.1em; color: #2d3436; }
a { color: #6c5ce7; text-decoration: none; }
code { background: #f1f2f6; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
blockquote { border-left: 3px solid #6c5ce7; padding-left: 10px; color: #636e72; }
hr { border: none; border-top: 1px solid #ddd; margin: 1em 0; }
"""


@app.get("/api/session/{session_id}/pdf")
async def get_session_pdf(session_id: int):
    report = db_load(session_id)
    if not report:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    md = format_report_markdown(report)
    html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    full_html = f"<html><head><style>{PDF_CSS}</style></head><body>{html_body}</body></html>"

    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=full_html).write_pdf()
    except Exception as e:
        return JSONResponse({"error": f"PDF generation failed: {e}"}, status_code=500)

    from fastapi.responses import Response
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=research_{session_id}.pdf"},
    )


@app.websocket("/ws/research")
async def ws_research(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_json()
        query = data.get("query", "").strip()
        depth = data.get("depth", "standard")
        style = data.get("style", "standard")
        thinking = data.get("thinking", False)
        academic = data.get("academic", False)
        template = data.get("template", "")
        agent_mode = data.get("agent", False)

        if not query:
            await ws.send_json({"event": "error", "data": "No query provided"})
            await ws.close()
            return

        config = ResearchConfig.from_depth(ResearchDepth(depth))
        config.report_style = ReportStyle(style)
        if thinking:
            config.use_thinking = True
        if academic:
            config.use_academic_search = True
        if template:
            config.template = template
        config.load_model_routes()

        progress = WebSocketProgress(ws)
        await progress.send("started", {"query": query, "depth": depth})

        try:
            if agent_mode:
                report = await run_agent_with_ws(query, config, progress)
            else:
                report = await run_research_with_ws(query, config, progress)

            path = save_report(report)
            session_id = db_save(report, depth=depth)
            report.id = session_id

            md = format_report_markdown(report)
            html = markdown.markdown(md, extensions=["tables", "fenced_code"])

            await progress.send("complete", {
                "session_id": session_id,
                "query": report.query,
                "executive_summary": report.executive_summary,
                "detailed_findings": report.detailed_findings,
                "follow_up_questions": report.follow_up_questions,
                "sources": [s.model_dump() for s in report.sources],
                "iterations_count": len(report.iterations),
                "markdown": md,
                "html": html,
                "file_path": str(path),
            })
        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            await progress.send("error", str(e))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


def run():
    from .logging_config import setup_logging
    setup_logging()
    search_engine = "Tavily" if settings.tavily_api_key else "DuckDuckGo"
    print(f"\n  Deep Research Server")
    print(f"  Model: {settings.glm_model} | Search: {search_engine} | http://localhost:8000\n")
    uvicorn.run(
        "deep_research.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
