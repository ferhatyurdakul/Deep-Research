from __future__ import annotations

import re
from datetime import datetime

from deep_research.models import SourceAnalysis, SourceType


def _format_authors(authors: list[str]) -> str:
    """Format author list for academic citation."""
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def format_citation(index: int, source: SourceAnalysis) -> str:
    """Format a single source as a bibliographic reference."""
    today = datetime.now().strftime("%Y-%m-%d")
    authors = _format_authors(source.authors)
    author_prefix = f"{authors}. " if authors else ""

    if source.source_type == SourceType.ARXIV:
        arxiv_id = source.extra.get("arxiv_id", "")
        year = source.published_date[:4] if source.published_date else "n.d."
        arxiv_ref = f"arXiv:{arxiv_id} " if arxiv_id else ""
        return (
            f"[{index}] {author_prefix}"
            f"{source.title}. "
            f"{arxiv_ref}({year}). "
            f"{source.url}"
        )

    if source.source_type == SourceType.SEMANTIC_SCHOLAR:
        venue = source.extra.get("venue", "")
        doi = source.extra.get("doi", "")
        year = source.published_date or "n.d."

        parts = [f"[{index}] {author_prefix}{source.title}."]
        if venue:
            parts.append(f" {venue},")
        parts.append(f" {year}.")
        if doi:
            parts.append(f" DOI: {doi}.")
        parts.append(f" {source.url} (retrieved {today})")
        return "".join(parts)

    # Web source
    return (
        f"[{index}] {author_prefix}"
        f"{source.title}. "
        f"{source.url} (retrieved {today})"
    )


def build_references_section(sources: list[SourceAnalysis]) -> str:
    """Build a full References section with proper academic citations."""
    lines = ["## References", ""]
    for i, source in enumerate(sources, 1):
        lines.append(format_citation(i, source))
    return "\n".join(lines)


def replace_references_in_report(markdown: str, sources: list[SourceAnalysis]) -> str:
    """Replace LLM-generated references with verified ones from actual sources."""
    references = build_references_section(sources)

    lower = markdown.lower()
    for marker in ["## references", "## references:", "## sources"]:
        idx = lower.find(marker)
        if idx != -1:
            return markdown[:idx] + references

    return markdown.rstrip() + "\n\n" + references


# --- Citation validation ---

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def find_all_citations(markdown: str) -> list[int]:
    """Extract all [N] citation numbers from markdown text."""
    nums: list[int] = []
    for match in _CITE_RE.finditer(markdown):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                nums.append(int(part))
    return nums


def validate_citations(markdown: str, num_sources: int) -> dict:
    """Check inline citations for validity.

    Returns a dict with:
      - valid: list of in-range citation numbers used
      - invalid: list of out-of-range citation numbers
      - unused_sources: list of source numbers never cited
      - uncited_paragraphs: count of substantive paragraphs with no citation
    """
    # Split off references section for analysis
    lower = markdown.lower()
    body = markdown
    for marker in ["## references", "## sources"]:
        idx = lower.find(marker)
        if idx != -1:
            body = markdown[:idx]
            break

    cited = find_all_citations(body)
    valid = sorted({n for n in cited if 1 <= n <= num_sources})
    invalid = sorted({n for n in cited if n < 1 or n > num_sources})
    all_source_nums = set(range(1, num_sources + 1))
    unused = sorted(all_source_nums - set(valid))

    # Count substantive paragraphs without citations
    uncited_count = 0
    for para in body.split("\n\n"):
        text = para.strip()
        # Skip headings, short lines, metadata, bullets with just a few words
        if not text or text.startswith("#") or len(text) < 80:
            continue
        if not _CITE_RE.search(text):
            uncited_count += 1

    return {
        "valid": valid,
        "invalid": invalid,
        "unused_sources": unused,
        "uncited_paragraphs": uncited_count,
    }


def clean_invalid_citations(markdown: str, num_sources: int) -> str:
    """Remove out-of-range [N] citations from the report body."""

    def _replace(match: re.Match) -> str:
        nums = [p.strip() for p in match.group(1).split(",")]
        kept = [n for n in nums if n.isdigit() and 1 <= int(n) <= num_sources]
        if not kept:
            return ""
        return "[" + ", ".join(kept) + "]"

    # Only clean the body, not the references section
    lower = markdown.lower()
    ref_idx = len(markdown)
    for marker in ["## references", "## sources"]:
        idx = lower.find(marker)
        if idx != -1:
            ref_idx = min(ref_idx, idx)

    body = markdown[:ref_idx]
    tail = markdown[ref_idx:]
    cleaned_body = _CITE_RE.sub(_replace, body)
    return cleaned_body + tail
