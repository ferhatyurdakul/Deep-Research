from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from deep_research.config import DB_PATH
from deep_research.models import (
    ContentDepth,
    EvidencedFinding,
    KnowledgeGap,
    ResearchIteration,
    ResearchReport,
    SourceAnalysis,
    SourceType,
    SubQuestion,
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    depth TEXT NOT NULL DEFAULT 'standard',
    executive_summary TEXT NOT NULL DEFAULT '',
    detailed_findings TEXT NOT NULL DEFAULT '',
    follow_up_questions TEXT NOT NULL DEFAULT '[]',
    sub_questions TEXT NOT NULL DEFAULT '[]',
    iterations TEXT NOT NULL DEFAULT '[]',
    searched_urls TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    key_findings TEXT NOT NULL DEFAULT '[]',
    relevance TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);

CREATE TABLE IF NOT EXISTS analyzed_urls (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    times_seen INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_sources_session ON sources(session_id);
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
CREATE INDEX IF NOT EXISTS idx_sessions_query ON research_sessions(query);
"""

_MIGRATIONS = [
    "ALTER TABLE sources ADD COLUMN summary TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE sources ADD COLUMN source_type TEXT NOT NULL DEFAULT 'web'",
    "ALTER TABLE sources ADD COLUMN authors TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE sources ADD COLUMN published_date TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE sources ADD COLUMN extra TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE sources ADD COLUMN key_evidence TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE sources ADD COLUMN evidenced_findings TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE sources ADD COLUMN content_depth TEXT NOT NULL DEFAULT 'full_text'",
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.executescript(_CREATE_SQL)
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists


def save_report(report: ResearchReport, depth: str = "standard") -> int:
    init_db()
    with _get_conn() as conn:
        cursor = conn.execute(
            """INSERT INTO research_sessions
               (query, depth, executive_summary, detailed_findings,
                follow_up_questions, sub_questions, iterations, searched_urls, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.query,
                depth,
                report.executive_summary,
                report.detailed_findings,
                json.dumps(report.follow_up_questions),
                json.dumps([sq.model_dump() for sq in report.sub_questions]),
                json.dumps([it.model_dump() for it in report.iterations]),
                json.dumps(report.searched_urls),
                report.created_at.isoformat(),
            ),
        )
        session_id = cursor.lastrowid

        for source in report.sources:
            conn.execute(
                """INSERT INTO sources (session_id, url, title, key_findings, relevance,
                   summary, source_type, authors, published_date, extra, key_evidence,
                   evidenced_findings, content_depth)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source.url,
                    source.title,
                    json.dumps(source.key_findings),
                    source.relevance,
                    source.summary,
                    source.source_type.value,
                    json.dumps(source.authors),
                    source.published_date,
                    json.dumps(source.extra),
                    json.dumps(source.key_evidence),
                    json.dumps([ef.model_dump() for ef in source.evidenced_findings]),
                    source.content_depth.value,
                ),
            )
            now = datetime.now().isoformat()
            conn.execute(
                """INSERT INTO analyzed_urls (url, title, first_seen, last_seen, times_seen)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(url) DO UPDATE SET
                       last_seen = excluded.last_seen,
                       times_seen = times_seen + 1""",
                (source.url, source.title, now, now),
            )

        conn.commit()
        return session_id


def load_report(session_id: int) -> ResearchReport | None:
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM research_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None

        source_rows = conn.execute(
            "SELECT * FROM sources WHERE session_id = ?", (session_id,)
        ).fetchall()

        sources = []
        for s in source_rows:
            keys = s.keys()
            ef_raw = json.loads(s["evidenced_findings"]) if "evidenced_findings" in keys else []
            evidenced = [EvidencedFinding(**ef) for ef in ef_raw]
            depth_val = s["content_depth"] if "content_depth" in keys else "full_text"
            try:
                content_depth = ContentDepth(depth_val)
            except ValueError:
                content_depth = ContentDepth.FULL_TEXT
            sa = SourceAnalysis(
                url=s["url"],
                title=s["title"],
                key_findings=json.loads(s["key_findings"]),
                key_evidence=json.loads(s["key_evidence"]) if "key_evidence" in keys else [],
                evidenced_findings=evidenced,
                content_depth=content_depth,
                relevance=s["relevance"],
                summary=s["summary"] if "summary" in keys else "",
                source_type=SourceType(s["source_type"]) if "source_type" in keys else SourceType.WEB,
                authors=json.loads(s["authors"]) if "authors" in keys else [],
                published_date=s["published_date"] if "published_date" in keys else "",
                extra=json.loads(s["extra"]) if "extra" in keys else {},
            )
            sources.append(sa)

        sub_questions = [SubQuestion(**sq) for sq in json.loads(row["sub_questions"])]

        iterations_data = json.loads(row["iterations"])
        iterations = []
        for it_data in iterations_data:
            iterations.append(ResearchIteration(
                iteration_number=it_data["iteration_number"],
                sub_questions=[SubQuestion(**sq) for sq in it_data.get("sub_questions", [])],
                sources=[SourceAnalysis(**s) for s in it_data.get("sources", [])],
                knowledge_gaps=[KnowledgeGap(**g) for g in it_data.get("knowledge_gaps", [])],
            ))

        return ResearchReport(
            id=row["id"],
            query=row["query"],
            sub_questions=sub_questions,
            sources=sources,
            iterations=iterations,
            searched_urls=json.loads(row["searched_urls"]),
            executive_summary=row["executive_summary"],
            detailed_findings=row["detailed_findings"],
            follow_up_questions=json.loads(row["follow_up_questions"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )


def list_sessions(limit: int = 20) -> list[dict]:
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT id, query, depth, created_at,
                      (SELECT COUNT(*) FROM sources WHERE session_id = research_sessions.id) as source_count
               FROM research_sessions
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_known_urls() -> set[str]:
    init_db()
    with _get_conn() as conn:
        rows = conn.execute("SELECT url FROM analyzed_urls").fetchall()
        return {r["url"] for r in rows}


def delete_session(session_id: int) -> bool:
    init_db()
    with _get_conn() as conn:
        conn.execute("DELETE FROM sources WHERE session_id = ?", (session_id,))
        cursor = conn.execute("DELETE FROM research_sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0
