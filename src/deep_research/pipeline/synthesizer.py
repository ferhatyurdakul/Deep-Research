from __future__ import annotations

from datetime import datetime

from deep_research.llm import achat, achat_json
from deep_research.models import SourceAnalysis, SubQuestion
from deep_research.output.citations import (
    clean_invalid_citations,
    replace_references_in_report,
    validate_citations,
)

from .analyzer import format_analyses

SYNTHESIZE_PROMPT = """You are a research report writer. Given a research topic and gathered sources, write a comprehensive, well-structured research report in markdown format.

CITATION RULES (strictly enforced):
1. EVERY factual claim, statistic, date, or specific finding MUST have an inline [N] citation
2. Only use source numbers from the provided list below (1 to {source_count})
3. NEVER invent, fabricate, or guess URLs, authors, or source titles
4. When sources conflict, present both viewpoints with their respective citations and explicitly flag the disagreement
5. Do NOT include any claim you cannot attribute to a specific source — if you cannot cite it, either omit it or clearly mark it as "general knowledge" or "inferred from multiple sources [N, M]"
6. Use the "Direct evidence" quotes from sources when available — prefer quoting over paraphrasing for key data points
7. Do NOT use source numbers beyond {source_count}

Report structure:
- Title: "# Research Report: {{query}}"
- Metadata line: "**Date:** {{date}} | **Sources analyzed:** {{source_count}}"
- ## Executive Summary (2-3 paragraphs synthesizing key insights, with citations)
- ## Key Findings (bulleted list, 5-8 items, EACH item must have at least one [N] citation)
- ## [Thematic Section 1] (with subsections as needed, inline citations throughout)
- ## [Thematic Section 2] ...
- ## Analysis & Synthesis (cross-cutting themes, comparisons across sources with citations)
- ## Limitations (what could not be covered, source quality caveats — note which sources were snippet-only)
- ## References (will be auto-generated — write ONLY "## References" as a placeholder, nothing else)

Write in an objective, academic tone. Be thorough but concise. Target 2000-4000 words.
Organize findings by THEME, not by source. Every substantive paragraph needs at least one [N] citation.
When a source provides a direct evidence quote, you may use it verbatim in quotation marks with its [N] citation.

Original research question: {query}

Sub-questions investigated:
{sub_questions}

Sources gathered ({source_count} total):
---
{analyses}
---
"""


async def asynthesize_report(
    query: str,
    sub_questions: list[SubQuestion],
    analyses: list[SourceAnalysis],
    thinking: bool = False,
    model: str | None = None,
    template_guidance: str = "",
    system_prompt: str = "",
) -> tuple[str, str, list[str]]:
    sq_text = "\n".join(f"- {sq.question}" for sq in sub_questions)
    prompt = SYNTHESIZE_PROMPT.format(
        query=query,
        sub_questions=sq_text,
        analyses=format_analyses(analyses),
        date=datetime.now().strftime("%Y-%m-%d"),
        source_count=len(analyses),
    )
    if template_guidance:
        prompt += f"\n\nDomain-specific guidance: {template_guidance}"
    system = system_prompt or "You are a helpful research assistant."
    report_text = await achat(prompt, system=system, temperature=0.4, thinking=thinking, model=model)

    # Clean invalid citations and replace LLM-generated references with verified ones
    report_text = clean_invalid_citations(report_text, len(analyses))
    report_text = replace_references_in_report(report_text, analyses)

    followup_data = await achat_json(
        f"Based on this research report, suggest 3-5 follow-up questions for further investigation.\n\n{report_text}\n\nRespond as JSON: {{\"follow_up_questions\": [\"q1\", \"q2\", ...]}}",
        model=model,
    )
    follow_ups = followup_data.get("follow_up_questions", [])

    parts = report_text.split("\n## ", 1)
    executive = parts[0].strip()
    detailed = ("## " + parts[1]) if len(parts) > 1 else report_text

    return executive, detailed, follow_ups
