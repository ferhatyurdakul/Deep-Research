"""
Shared screens: banner, history, session inspector, status, and the closing
"saved" card.

The one-shot CLI and the REPL both render through these functions, so
`deep-research history` and `/history` (and `inspect`, `status`, ...) are the
same screen by construction.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import settings
from .models import ResearchReport
from .options import academic_reason, stage_summary
from .ui import (
    bullets, console, fmt_duration, glyph, kv_grid, note, numbered, panel, section,
    plural, short_path,
)

PROG = "deep-research"


# --- banner -------------------------------------------------------------

def shape_summary(
    depth: str, style: str, template: Optional[str] = None,
    academic: bool = False, agent: bool = False, fresh: bool = False,
) -> str:
    """`standard depth · standard style · science template · academic`: always
    depth and style; the optional parts only when on. `academic` shows when a
    run will search academic sources — whether switched on directly or implied
    by the template or depth (science, literature, deep)."""
    parts = [f"{depth} depth", f"{style} style"]
    if template:
        parts.append(f"{template} template")
    academic = academic_reason(depth, template, academic) is not None
    parts += [label for label, on in (("academic", academic), ("agent", agent), ("fresh", fresh)) if on]
    return f" {glyph('dot')} ".join(parts)


def shape_from_namespace(ns: argparse.Namespace) -> str:
    return shape_summary(
        ns.depth, getattr(ns, "style", "standard"), getattr(ns, "template", None),
        getattr(ns, "academic", False), getattr(ns, "agent", False), getattr(ns, "fresh", False),
    )


def _model_line() -> Text:
    try:
        info = settings.validate_routes()
        dot = glyph("dot")
        return Text(
            f"{stage_summary(info.get('stages') or {}) if info.get('stages') else settings.active_model} "
            f"via {settings.active_provider} ({info['endpoint_family']}) {dot} {settings.llm_route}"
        )
    except Exception:
        return Text(f"configuration invalid {glyph('dash')} run `{PROG} status`", style="err")


def _search_line() -> Text:
    if settings.tavily_api_key:
        return Text("Tavily + DuckDuckGo")
    line = Text("DuckDuckGo")
    line.append("  (set TAVILY_API_KEY for better results)", style="muted")
    return line


def print_banner(
    *, shape: str, query: Optional[str] = None, task: Optional[str] = None,
    tips: Optional[str] = None,
) -> None:
    rows: list[tuple[str, object]] = [
        ("model", _model_line()),
        ("search", _search_line()),
        ("shape", Text(shape)),
    ]
    if task:
        rows.append(("task", Text(task)))
    if query:
        rows.append(("query", Text(query, style="bold")))
    body = kv_grid(rows)
    if tips:
        body = Group(body, Text(""), Text.from_markup(tips))
    console.print(panel(body, title=f"{glyph('mark')} Deep Research v{__version__}"))


# --- history ------------------------------------------------------------

def print_history(limit: int = 25, *, repl: bool = False) -> None:
    from .storage.db import list_sessions

    sessions = list_sessions(limit=limit)
    if not sessions:
        note("No research sessions found.")
        note("Type a question to create one." if repl
             else f'Run one with: {PROG} "your question"')
        return

    has_parent = any(s.get("parent_session_id") for s in sessions)
    table = Table(box=box.SIMPLE_HEAD, header_style="muted", pad_edge=False, expand=True)
    table.add_column("ID", style="cyan", justify="right", no_wrap=True)
    if has_parent:
        table.add_column("From", style="muted", justify="right", no_wrap=True)
    table.add_column("Query", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("Depth", no_wrap=True)
    table.add_column("Sources", justify="right", no_wrap=True)
    table.add_column("Date", style="muted", no_wrap=True)

    for s in sessions:
        row = [str(s["id"])]
        if has_parent:
            parent = s.get("parent_session_id")
            row.append(f"{glyph('branch')} {parent}" if parent else "")
        row += [
            Text(" ".join(s["query"].split())),
            s["depth"],
            str(s["source_count"]),
            s["created_at"][:16].replace("T", " "),
        ]
        table.add_row(*row)

    section(f"Research history ({len(sessions)} most recent)")
    console.print(table)
    console.print(Text("Next: ", style="muted") + _next_steps(("inspect", "continue", "fork"), "ID", repl=repl))


def _next_steps(commands: tuple, arg: str, *, repl: bool) -> Text:
    dot = f" {glyph('dot')} "
    prefix = "/" if repl else f"{PROG} "
    out = Text()
    for i, name in enumerate(commands):
        if i:
            out.append(dot, style="muted")
        out.append(f"{prefix if (repl or i == 0) else ''}{name}", style="cmd")
        out.append(f" {arg}", style="arg")
    return out


# --- session inspector --------------------------------------------------

def print_source(report: ResearchReport, index_1based: int) -> bool:
    """Print one source in detail. False (after printing why) when out of range."""
    if index_1based < 1 or index_1based > len(report.sources):
        console.print(
            f"[err]{glyph('fail')}[/err] Source {index_1based} out of range. "
            f"Session has {len(report.sources)} sources."
        )
        return False
    s = report.sources[index_1based - 1]
    rows: list[tuple[str, object]] = [
        ("title", Text(s.title, style="bold")),
        ("url", Text(s.url, style="blue underline")),
        ("type", Text(f"{s.source_type.value} {glyph('dot')} {s.content_depth.value}")),
    ]
    if s.authors:
        rows.append(("authors", Text(", ".join(s.authors))))
    if s.published_date:
        rows.append(("date", Text(s.published_date)))
    if s.relevance:
        rows.append(("relevance", Text(s.relevance)))
    console.print(panel(kv_grid(rows), title=f"Source #{index_1based}"))

    if s.summary:
        section("Summary")
        console.print(Text(s.summary))
    if s.key_findings:
        section("Key findings")
        console.print(bullets(s.key_findings))
    if s.key_evidence:
        section("Key evidence (verbatim)")
        console.print(bullets([f'"{e}"' for e in s.key_evidence]))
    if s.evidenced_findings:
        section("Evidenced findings")
        for ef in s.evidenced_findings:
            line = Text()
            line.append(f"{glyph('bullet')} ", style="muted")
            line.append(f"[{ef.confidence}] ", style="warn")
            line.append(ef.finding)
            console.print(line)
            if ef.evidence:
                console.print(Text(f'    evidence: "{ef.evidence}"', style="muted"))
    return True


def print_session(report: ResearchReport, session_id: int, *, full: bool = False, repl: bool = False) -> None:
    rows: list[tuple[str, object]] = [
        ("query", Text(report.query, style="bold")),
        ("created", Text(report.created_at.strftime("%Y-%m-%d %H:%M"))),
        ("shape", Text(
            f"{plural(len(report.sources), 'source')} {glyph('dot')} "
            f"{plural(len(report.iterations), 'iteration')} {glyph('dot')} "
            f"{plural(len(report.sub_questions), 'sub-question')}"
        )),
    ]
    if report.parent_session_id:
        rows.append(("from", Text(f"session {report.parent_session_id}")))
    console.print(panel(kv_grid(rows), title=f"Session {session_id}"))

    if report.sub_questions:
        section("Sub-questions")
        console.print(numbered([sq.question for sq in report.sub_questions]))

    if report.sources:
        table = Table(box=box.SIMPLE_HEAD, header_style="muted", pad_edge=False, expand=True)
        table.add_column("#", style="cyan", justify="right", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Depth", style="muted", no_wrap=True)
        table.add_column("Title", ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column("Findings", justify="right", no_wrap=True)
        table.add_column("Evidence", justify="right", no_wrap=True)
        for i, src in enumerate(report.sources, 1):
            table.add_row(
                str(i), src.source_type.value, src.content_depth.value,
                Text(" ".join(src.title.split())),
                str(len(src.key_findings)),
                str(len(src.key_evidence) + len(src.evidenced_findings)),
            )
        section(f"Sources ({len(report.sources)})")
        console.print(table)
        cmd = f"{'/' if repl else PROG + ' '}inspect {session_id} --source <n>"
        note(f"{cmd} shows findings and evidence for one source.")

    if report.iterations:
        section(f"Iterations ({len(report.iterations)})")
        for it in report.iterations:
            head = Text()
            head.append(f"Iteration {it.iteration_number}", style="bold")
            head.append(f"  {plural(len(it.sources), 'source')}, {plural(len(it.knowledge_gaps), 'gap')}", style="muted")
            console.print(head)
            for g in it.knowledge_gaps:
                console.print(Text(f"  {glyph('bullet')} [{g.priority}] {g.gap_description}"))
                if g.suggested_query:
                    console.print(Text(f"      query: {g.suggested_query}", style="muted"))

    if report.follow_up_questions:
        section("Suggested follow-ups")
        console.print(numbered(report.follow_up_questions))

    if report.executive_summary:
        console.print()
        console.print(panel(Markdown(report.executive_summary), title="Executive summary"))

    if report.detailed_findings:
        if full:
            section("Detailed findings")
            console.print(Markdown(report.detailed_findings))
        else:
            flag = f"/inspect {session_id} --full" if repl else f"{PROG} inspect {session_id} --full"
            note(f"Detailed findings: {len(report.detailed_findings):,} chars. "
                 f"Add --full to print them ({flag}), or open the saved markdown report.")


# --- status -------------------------------------------------------------

def print_status(shape: Optional[list] = None) -> None:
    """Resolved configuration. `shape` is an optional list of (label, value)
    rows for the REPL's per-session toggles."""
    from .capabilities import CapabilityViolation
    from .config import DB_PATH, OUTPUTS_DIR

    info: dict = {}
    model_rows: list[tuple[str, object]] = []
    try:
        info = settings.validate_routes()
        model_rows = [
            ("provider", Text(f"{settings.active_provider} ({info.get('endpoint_family', '?')})")),
            ("profile", Text(settings.llm_route)),
            ("model", Text(settings.active_model)),
            ("fallback", Text(settings.llm_fallback_provider or "(none)")),
        ]
    except (CapabilityViolation, ValueError) as e:
        model_rows = [("route", Text(f"invalid: {e}", style="err"))]
    console.print(panel(kv_grid(model_rows), title="Model"))

    if "stages" in info:
        console.print(panel(
            kv_grid([(k, Text(str(v))) for k, v in info["stages"].items()]),
            title="Stage models",
        ))

    def _key_state(is_set: bool) -> Text:
        return Text(f"{glyph('on')} set", style="ok") if is_set else Text(f"{glyph('off')} not set", style="muted")

    try:
        active_key = bool(settings.active_api_key)
        key_name = settings.active_api_key_env_name
    except Exception:
        active_key, key_name = False, "API key"
    console.print(panel(kv_grid([
        (key_name, _key_state(active_key)),
        ("TAVILY_API_KEY", Text.assemble(_key_state(bool(settings.tavily_api_key)),
                                         ("" if settings.tavily_api_key else "  (falls back to DuckDuckGo)", "muted"))),
    ]), title="Credentials"))

    if shape:
        console.print(panel(kv_grid([(k, Text(str(v))) for k, v in shape]), title="This session"))

    console.print(panel(kv_grid([
        ("sessions db", Text(short_path(DB_PATH), overflow="fold")),
        ("reports", Text(short_path(OUTPUTS_DIR), overflow="fold")),
    ]), title="Files"))
    note(f"Change credentials with `{PROG} setup`. Environment variables and .env files override stored defaults.")


# --- run completion -----------------------------------------------------

def print_done(
    *, session_id: int, path: Path, report: ResearchReport,
    started: Optional[float] = None, relation: Optional[str] = None, repl: bool = False,
) -> None:
    dot = glyph("dot")
    session = Text(str(session_id), style="brand")
    if relation:
        session.append(f"  {relation}", style="muted")
    stats = f"{plural(len(report.sources), 'source')} {dot} {plural(len(report.iterations), 'iteration')}"
    if started is not None:
        stats += f" {dot} {fmt_duration(time.monotonic() - started)}"
    console.print()
    console.print(panel(
        kv_grid([
            ("session", session),
            ("report", Text(short_path(path), overflow="fold")),
            ("stats", Text(stats)),
            ("next", _next_steps(("continue", "inspect", "fork"), str(session_id), repl=repl)),
        ]),
        title=f"{glyph('ok')} Saved",
        border="green",
    ))


def present_report(
    report: ResearchReport, *, depth: str, parent_session_id: Optional[int] = None,
    started: Optional[float] = None, relation: Optional[str] = None, repl: bool = False,
) -> int:
    """Show a finished report, persist it (markdown + DB), print the saved
    card, and return the new session id. The one place that does this, so
    one-shot runs, follow-ups, continue, resynthesize and the REPL agree."""
    from .output.report import display_report, save_report
    from .storage.db import save_report as db_save

    display_report(report)
    path = save_report(report)
    session_id = db_save(report, depth=depth, parent_session_id=parent_session_id)
    report.id = session_id
    print_done(
        session_id=session_id, path=path, report=report,
        started=started, relation=relation, repl=repl,
    )
    return session_id


__all__ = [
    "print_banner", "print_history", "print_session", "print_source",
    "print_status", "print_done", "present_report", "shape_summary",
    "shape_from_namespace",
]
