"""Mechanical checks for Deep Research reports.

These are deterministic, cheap, and catch regressions in:
  - citation quality (density, coverage, invalid refs)
  - structural integrity (required sections per style)
  - tone (banned superlative/marketing phrases)
  - source hygiene (orphan sources, snippet ratio)

All checks take a ResearchReport and return a dict of metrics.
"""
from __future__ import annotations

import re
from typing import Any

from deep_research.models import ContentDepth, ReportStyle, ResearchReport

# --- Tone: phrases banned by the synthesis prompt ---
BANNED_PHRASES = [
    "revolutionary",
    "groundbreaking",
    "game-changing",
    "game changer",
    "game-changer",
    "next-generation",
    "next generation",
    "cutting-edge",
    "cutting edge",
    "bleeding-edge",
    "best-in-class",
    "world-class",
    "industry-leading",
    "paradigm shift",
    "paradigm-shifting",
    "unparalleled",
    "seamless integration",
    "robust solution",
]

# --- Required sections per report style ---
REQUIRED_SECTIONS = {
    ReportStyle.BRIEF: ["## Summary", "## Key Findings", "## References"],
    ReportStyle.STANDARD: [
        "## Executive Summary",
        "## Key Findings",
        "## Limitations",
        "## References",
    ],
    ReportStyle.ACADEMIC: [
        "## Abstract",
        "## 1. Introduction",
        "## 2. Methodology",
        "## 3. Findings",
        "## 4. Discussion",
        "## 5. Limitations",
        "## 6. Conclusion",
        "## References",
    ],
}

_CITATION_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"\b\w+\b")


def _report_text(report: ResearchReport) -> str:
    return f"{report.executive_summary}\n\n{report.detailed_findings}"


def citation_density(text: str) -> dict[str, float]:
    """Citations per 100 words."""
    citations = _CITATION_RE.findall(text)
    word_count = len(_WORD_RE.findall(text))
    density = (len(citations) / word_count * 100) if word_count else 0.0
    return {
        "citations": len(citations),
        "words": word_count,
        "per_100_words": round(density, 2),
    }


def citation_coverage(text: str) -> dict[str, Any]:
    """Percentage of substantive paragraphs containing at least one [N] citation.

    Excludes headings, code blocks, bullet lists, and very short paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    substantive = []
    in_code = False
    for p in paragraphs:
        if p.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not p:
            continue
        if p.startswith("#"):
            continue
        if len(p) < 80:
            continue
        substantive.append(p)

    with_cite = sum(1 for p in substantive if _CITATION_RE.search(p))
    coverage = (with_cite / len(substantive)) if substantive else 0.0
    return {
        "substantive_paragraphs": len(substantive),
        "with_citation": with_cite,
        "coverage_ratio": round(coverage, 3),
    }


def invalid_citations(text: str, source_count: int) -> dict[str, Any]:
    """Citations that reference [N] where N > source_count or N < 1."""
    nums = [int(n) for n in _CITATION_RE.findall(text)]
    bad = sorted({n for n in nums if n < 1 or n > source_count})
    return {
        "source_count": source_count,
        "invalid_refs": bad,
        "count": len(bad),
    }


def orphan_sources(text: str, source_count: int) -> dict[str, Any]:
    """Sources (by index 1..N) that are never cited in the report body."""
    cited = {int(n) for n in _CITATION_RE.findall(text)}
    valid_cited = {n for n in cited if 1 <= n <= source_count}
    orphans = sorted(set(range(1, source_count + 1)) - valid_cited)
    return {
        "uncited_indices": orphans,
        "count": len(orphans),
        "ratio": round(len(orphans) / source_count, 3) if source_count else 0.0,
    }


def snippet_ratio(report: ResearchReport) -> dict[str, Any]:
    """Fraction of sources that were snippet-only (no full-text analysis)."""
    total = len(report.sources)
    snippet_only = sum(
        1 for s in report.sources if s.content_depth == ContentDepth.SNIPPET
    )
    return {
        "total": total,
        "snippet_only": snippet_only,
        "ratio": round(snippet_only / total, 3) if total else 0.0,
    }


def banned_phrases(text: str) -> dict[str, Any]:
    """Count of banned superlative/marketing phrases."""
    lowered = text.lower()
    hits: dict[str, int] = {}
    for phrase in BANNED_PHRASES:
        count = lowered.count(phrase)
        if count:
            hits[phrase] = count
    return {
        "total": sum(hits.values()),
        "hits": hits,
    }


def structure_check(text: str, style: ReportStyle) -> dict[str, Any]:
    """Required sections present for the given style."""
    required = REQUIRED_SECTIONS.get(style, [])
    missing = [s for s in required if s not in text]
    return {
        "required_count": len(required),
        "missing": missing,
        "pass": not missing,
    }


def run_all(report: ResearchReport, style: ReportStyle) -> dict[str, Any]:
    """Run every mechanical check and return a single structured result."""
    text = _report_text(report)
    return {
        "citation_density": citation_density(text),
        "citation_coverage": citation_coverage(text),
        "invalid_citations": invalid_citations(text, len(report.sources)),
        "orphan_sources": orphan_sources(text, len(report.sources)),
        "snippet_ratio": snippet_ratio(report),
        "banned_phrases": banned_phrases(text),
        "structure": structure_check(text, style),
    }


def summary_line(checks: dict[str, Any]) -> str:
    """One-line human-readable summary of a checks dict."""
    cd = checks["citation_density"]["per_100_words"]
    cov = checks["citation_coverage"]["coverage_ratio"]
    inv = checks["invalid_citations"]["count"]
    orph = checks["orphan_sources"]["count"]
    snip = checks["snippet_ratio"]["ratio"]
    banned = checks["banned_phrases"]["total"]
    struct = "ok" if checks["structure"]["pass"] else f"missing={checks['structure']['missing']}"
    return (
        f"cite_density={cd}/100w cov={cov} invalid={inv} orphans={orph} "
        f"snippet_ratio={snip} banned={banned} struct={struct}"
    )
