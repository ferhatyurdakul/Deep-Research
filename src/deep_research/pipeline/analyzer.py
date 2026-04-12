from __future__ import annotations

import asyncio

from deep_research.llm import achat, achat_json
from deep_research.models import (
    ContentDepth,
    EvidencedFinding,
    ExtractedData,
    KnowledgeGap,
    SourceAnalysis,
    SourceType,
    SubQuestion,
)

# Max concurrent LLM calls to avoid rate limits — lazily created per event loop
_semaphore_cache: dict[int, asyncio.Semaphore] = {}


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if loop_id not in _semaphore_cache:
        _semaphore_cache[loop_id] = asyncio.Semaphore(5)
    return _semaphore_cache[loop_id]

ANALYZE_PROMPT = """You are a research analyst. Analyze the following source material and extract key findings relevant to the research question. Each finding MUST be paired with verbatim evidence from the source.

Research question: {query}

Source URL: {url}
Source title: {title}
Content:
{content}

Respond in JSON format:
{{
  "evidenced_findings": [
    {{
      "finding": "your interpreted finding (1-2 sentences)",
      "evidence": "verbatim quote or specific data point copied exactly from the source that supports this finding",
      "confidence": "supported|partially_supported|inferred"
    }}
  ],
  "relevance": "high/medium/low and brief explanation"
}}

RULES:
- Return 3-6 evidenced findings, ordered by relevance to the research question
- "evidence" must be a VERBATIM excerpt (1-2 sentences) copied directly from the source text — do NOT paraphrase
- If the source has specific numbers, percentages, or metrics, include them exactly in the evidence
- "confidence" levels: "supported" = evidence directly states the finding; "partially_supported" = evidence implies it but doesn't state it directly; "inferred" = finding is your interpretation beyond what the text explicitly says
- If you cannot find verbatim evidence for a finding, set evidence to "" and confidence to "inferred"
- Prefer findings with concrete evidence over vague interpretations"""

EXTRACT_DATA_PROMPT = """You are a data extraction specialist. From the following source content, extract structured data.

Research question: {query}
Source: {title}
Content:
{content}

Extract the following categories. Only include items that are explicitly stated in the content. Be precise and quote numbers exactly.

Respond in JSON:
{{
  "statistics": ["specific numbers, percentages, metrics found"],
  "entities": ["key people, organizations, products, technologies mentioned"],
  "dates": ["significant dates, deadlines, timelines"],
  "claims": ["notable claims, conclusions, or predictions"]
}}

If a category has no relevant data, return an empty list for it."""

SUMMARIZE_PROMPT = """You are a research assistant. Given a source text and a research topic, write a concise summary (2-4 sentences) of the source's relevance to the topic. Focus on key claims, findings, and data points. Be factual, not promotional."""

GAP_ANALYSIS_PROMPT = """You are a research quality reviewer. Given the original research question and the findings gathered so far, identify specific knowledge gaps — areas where the research is incomplete, contradictory, or lacking depth.

Original research question: {query}

Sub-questions investigated:
{sub_questions}

Findings so far:
{findings_summary}

Previously searched queries (do NOT repeat these):
{searched_queries}

Identify 2-4 specific knowledge gaps and suggest a targeted search query for each.
Respond in JSON:
{{
  "gaps": [
    {{
      "gap_description": "what information is missing",
      "suggested_query": "specific search query to fill this gap",
      "priority": "high/medium/low"
    }}
  ],
  "is_sufficient": false
}}

Set "is_sufficient" to true ONLY if the current findings comprehensively answer the original question. Be critical — look for missing perspectives, outdated information, and unsupported claims."""


def format_analyses(analyses: list[SourceAnalysis]) -> str:
    parts = []
    for i, a in enumerate(analyses, 1):
        authors = ", ".join(a.authors[:3]) if a.authors else "N/A"
        if len(a.authors) > 3:
            authors += " et al."
        date = a.published_date or "N/A"
        depth_label = a.content_depth.value.replace("_", " ")

        lines = [
            f"[{i}] {a.title}",
            f"URL: {a.url}",
            f"Authors: {authors} | Date: {date} | Type: {a.source_type.value} | Content: {depth_label}",
        ]
        if a.summary:
            lines.append(f"Summary: {a.summary}")
        if a.relevance:
            lines.append(f"Relevance: {a.relevance}")

        # Prefer evidenced findings (paired finding+evidence) over flat lists
        if a.evidenced_findings:
            lines.append("Findings with evidence:")
            for ef in a.evidenced_findings:
                conf = f" [{ef.confidence}]" if ef.confidence != "supported" else ""
                lines.append(f"  - {ef.finding}{conf}")
                if ef.evidence:
                    lines.append(f'    Evidence: "{ef.evidence}"')
        else:
            # Fallback to flat lists for backward compat
            if a.key_findings:
                lines.append("Findings:")
                for f in a.key_findings:
                    lines.append(f"  - {f}")
            if a.key_evidence:
                lines.append("Direct evidence (verbatim from source):")
                for ev in a.key_evidence:
                    lines.append(f'  > "{ev}"')

        if a.extracted_data:
            ed = a.extracted_data
            if ed.statistics:
                lines.append("Statistics: " + "; ".join(ed.statistics[:5]))
            if ed.claims:
                lines.append("Key claims: " + "; ".join(ed.claims[:3]))

        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


async def asummarize_source(
    query: str, title: str, content: str, model: str | None = None,
) -> str:
    """Generate a concise 2-4 sentence summary of a source."""
    async with _get_semaphore():
        prompt = f"Research topic: {query}\n\nSource: {title}\n\n{content[:3000]}"
        return await achat(prompt, system=SUMMARIZE_PROMPT, temperature=0.2, model=model)


async def aextract_data(
    query: str, title: str, content: str, model: str | None = None,
) -> ExtractedData:
    """Extract structured data from source content."""
    async with _get_semaphore():
        prompt = EXTRACT_DATA_PROMPT.format(query=query, title=title, content=content)
        data = await achat_json(prompt, model=model)
    return ExtractedData(
        statistics=data.get("statistics", []),
        entities=data.get("entities", []),
        dates=data.get("dates", []),
        claims=data.get("claims", []),
    )


async def aanalyze_source(
    query: str, url: str, title: str, content: str,
    thinking: bool = False, model: str | None = None,
    extract_data: bool = False,
    source_type: SourceType = SourceType.WEB,
    authors: list[str] | None = None,
    published_date: str = "",
    extra: dict | None = None,
    content_depth: ContentDepth = ContentDepth.FULL_TEXT,
) -> SourceAnalysis:
    async with _get_semaphore():
        prompt = ANALYZE_PROMPT.format(query=query, url=url, title=title, content=content)
        data = await achat_json(prompt, thinking=thinking, model=model)

    summary = await asummarize_source(query, title, content, model=model)

    # Parse evidenced findings from new format
    raw_ef = data.get("evidenced_findings", [])
    evidenced = []
    key_findings = []
    key_evidence = []
    for ef in raw_ef:
        if isinstance(ef, dict):
            finding = ef.get("finding", "")
            evidence = ef.get("evidence", "")
            confidence = ef.get("confidence", "supported")
            if finding:
                evidenced.append(EvidencedFinding(
                    finding=finding, evidence=evidence, confidence=confidence,
                ))
                key_findings.append(finding)
                if evidence:
                    key_evidence.append(evidence)

    # Fallback: if LLM returned old flat format instead
    if not evidenced:
        key_findings = data.get("key_findings", [])
        key_evidence = data.get("key_evidence", [])

    sa = SourceAnalysis(
        url=url, title=title,
        key_findings=key_findings,
        key_evidence=key_evidence,
        evidenced_findings=evidenced,
        content_depth=content_depth,
        relevance=data.get("relevance", ""),
        summary=summary.strip(),
        source_type=source_type,
        authors=authors or [],
        published_date=published_date,
        extra=extra or {},
    )
    if extract_data:
        sa.extracted_data = await aextract_data(query, title, content, model=model)
    return sa


async def aidentify_gaps(
    query: str,
    sub_questions: list[SubQuestion],
    analyses: list[SourceAnalysis],
    searched_queries: list[str],
    thinking: bool = False,
    model: str | None = None,
) -> tuple[list[KnowledgeGap], bool]:
    sq_text = "\n".join(f"- {sq.question}" for sq in sub_questions)
    searched_text = "\n".join(f"- {q}" for q in searched_queries)
    prompt = GAP_ANALYSIS_PROMPT.format(
        query=query, sub_questions=sq_text,
        findings_summary=format_analyses(analyses),
        searched_queries=searched_text,
    )
    data = await achat_json(prompt, thinking=thinking, model=model)
    gaps = [KnowledgeGap(**g) for g in data.get("gaps", [])]
    is_sufficient = data.get("is_sufficient", False)
    return gaps, is_sufficient
