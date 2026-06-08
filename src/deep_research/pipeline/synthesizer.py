from __future__ import annotations

from datetime import datetime

from deep_research.llm import achat, achat_json
from deep_research.models import ReportStyle, SourceAnalysis, SourceType, SubQuestion
from deep_research.output.citations import (
    clean_invalid_citations,
    replace_references_in_report,
)

from .analyzer import format_analyses

# --- Citation rules shared across all styles ---

_CITATION_RULES = """CITATION RULES (strictly enforced):
1. EVERY factual claim, statistic, date, or specific finding MUST have an inline [N] citation
2. Only use source numbers from the provided list below (1 to {source_count})
3. NEVER invent, fabricate, or guess URLs, authors, or source titles
4. When sources conflict, present both viewpoints with their respective citations and explicitly flag the disagreement
5. Do NOT include any claim you cannot attribute to a specific source — if you cannot cite it, either omit it or clearly mark it as "general knowledge" or "inferred from multiple sources [N, M]"
6. Use the "Direct evidence" quotes from sources when available — prefer quoting over paraphrasing for key data points
7. Do NOT use source numbers beyond {source_count}"""

# --- Uncertainty and limitation language shared across standard/academic ---

_UNCERTAINTY_RULES = """
UNCERTAINTY AND EVIDENCE QUALITY:
- When a claim rests on a single source, note this: "According to [N]..." rather than stating it as established fact
- When evidence is limited to snippets or abstracts (marked as "snippet only" in relevance), explicitly caveat: "Based on limited available text from [N]..."
- Distinguish between: established consensus (multiple corroborating sources), emerging findings (1-2 recent sources), and speculation (no direct source support)
- Use hedging language proportional to evidence strength: "evidence suggests", "preliminary findings indicate", "one study found" vs. "research consistently shows", "multiple studies confirm"
- In the limitations section, explicitly list: sources that were snippet-only, topics where only a single source was found, areas where sources contradict each other, and any notable gaps in coverage"""

# --- Style-specific report structure prompts ---

_BRIEF_STRUCTURE = """Report structure:
- Title: "# Research Brief: {{query}}"
- Metadata line: "**Date:** {{date}} | **Sources:** {{source_count}}"
- ## Summary (1-2 paragraphs, key takeaways with citations)
- ## Key Findings (bulleted list, 4-6 items, each with [N] citations)
- ## References (write ONLY "## References" as a placeholder — auto-generated)

Write concisely. Target 500-1000 words. Every finding needs a citation."""

_STANDARD_STRUCTURE = """Report structure:
- Title: "# Research Report: {{query}}"
- Metadata line: "**Date:** {{date}} | **Sources analyzed:** {{source_count}}"
- ## Executive Summary (2-3 paragraphs synthesizing key insights, with citations)
- ## Key Findings (bulleted list, 5-8 items, EACH item must have at least one [N] citation)
- ## [Thematic Section 1] (with subsections as needed, inline citations throughout)
- ## [Thematic Section 2] ...
- ## Analysis & Synthesis (cross-cutting themes, comparisons across sources with citations)
- ## Limitations (what could not be covered, source quality caveats — note which sources were snippet-only)
- ## References (write ONLY "## References" as a placeholder — auto-generated)

Write in an objective, analytical tone. Be thorough but concise. Target 2000-4000 words.
Organize findings by THEME, not by source. Every substantive paragraph needs at least one [N] citation.
When a source provides a direct evidence quote, you may use it verbatim in quotation marks with its [N] citation."""

_ACADEMIC_STRUCTURE = """Report structure (follow this EXACTLY):
- Title: "# {{query}}"
- Metadata line: "**Date:** {{date}} | **Sources analyzed:** {{source_count}}"
- ## Abstract (single paragraph, 150-250 words: background, objective, methods summary, key findings, conclusion — with select citations)
- ## 1. Introduction (research context, why this question matters, scope of this review)
- ## 2. Methodology (describe the retrieval process: search providers used, number of queries, source types retrieved, deduplication, total sources analyzed — present this as a systematic search methodology)
  - Search strategy: {search_methodology}
- ## 3. Findings (organized by theme with subsections numbered 3.1, 3.2, etc. — every claim cited)
  - ## 3.1 [Theme 1]
  - ## 3.2 [Theme 2] ...
- ## 4. Discussion (synthesis across themes, areas of consensus and disagreement, evidence strength assessment, implications)
- ## 5. Limitations and Uncertainty (source coverage gaps, snippet-only sources, single-source claims, conflicting evidence, temporal limitations of sources, areas requiring further research)
- ## 6. Conclusion (2-3 paragraphs: what the evidence shows, confidence level, recommended next steps)
- ## References (write ONLY "## References" as a placeholder — auto-generated)

Write in formal academic prose. Use passive voice where appropriate. Target 3000-5000 words.
Organize findings strictly by theme under numbered sections. Every substantive sentence needs a citation.
When a source provides a direct evidence quote, present it as a block quote with citation.
Distinguish clearly between what the evidence shows vs. what it suggests vs. what remains unknown."""


def _build_search_methodology(
    analyses: list[SourceAnalysis],
    iterations: int,
    sub_question_count: int,
) -> str:
    """Build a methodology description from source metadata."""
    web_count = sum(1 for a in analyses if a.source_type == SourceType.WEB)
    arxiv_count = sum(1 for a in analyses if a.source_type == SourceType.ARXIV)
    scholar_count = sum(1 for a in analyses if a.source_type == SourceType.SEMANTIC_SCHOLAR)
    snippet_count = sum(1 for a in analyses if a.relevance and "snippet" in a.relevance.lower())
    full_text_count = len(analyses) - snippet_count

    parts = []
    parts.append(f"The research question was decomposed into {sub_question_count} sub-questions.")
    parts.append(f"A total of {len(analyses)} sources were retrieved across {iterations} research iteration(s).")

    source_breakdown = []
    if web_count:
        source_breakdown.append(f"{web_count} web sources")
    if arxiv_count:
        source_breakdown.append(f"{arxiv_count} arXiv papers")
    if scholar_count:
        source_breakdown.append(f"{scholar_count} Semantic Scholar entries")
    if source_breakdown:
        parts.append(f"Source composition: {', '.join(source_breakdown)}.")

    if full_text_count and snippet_count:
        parts.append(f"Of these, {full_text_count} were analyzed from full text and {snippet_count} from snippets/abstracts only.")

    parts.append("Results were deduplicated by URL normalization and title similarity across all providers.")

    return " ".join(parts)


def _fill_structure(structure: str, query: str, date: str, source_count: int) -> str:
    """Substitute the real query/date/source-count into a structure template.

    The structure templates carry doubled-brace tokens ({{query}}, {{date}},
    {{source_count}}) so they survive untouched when inserted as a *value* into
    SYNTHESIZE_PROMPT.format(). We resolve them here with str.replace (not
    str.format, which would choke on the templates' literal braces) so the model
    receives the actual title/metadata values instead of literal placeholders.
    """
    return (
        structure
        .replace("{{query}}", query)
        .replace("{{date}}", date)
        .replace("{{source_count}}", str(source_count))
    )


def _get_style_prompt(
    style: ReportStyle,
    analyses: list[SourceAnalysis],
    iterations: int,
    sub_question_count: int,
    query: str,
    date: str,
) -> tuple[str, str]:
    """Return (structure_prompt, system_prompt) for the given report style."""
    methodology = _build_search_methodology(analyses, iterations, sub_question_count)
    source_count = len(analyses)

    def _resolved(structure: str) -> str:
        return _fill_structure(structure, query, date, source_count)

    if style == ReportStyle.BRIEF:
        return _resolved(_BRIEF_STRUCTURE), "You are a concise research analyst. Produce a focused research brief."

    if style == ReportStyle.ACADEMIC:
        structure = _resolved(_ACADEMIC_STRUCTURE.replace("{search_methodology}", methodology))
        system = (
            "You are an academic researcher writing a systematic review. "
            "Use formal academic language, numbered sections, and rigorous evidence attribution. "
            "Clearly distinguish between what the evidence establishes, what it suggests, and what remains unknown."
        )
        return structure, system

    # STANDARD (default)
    system = (
        "You are an analytical research writer. Write in a neutral, evidence-grounded tone. "
        "Present findings comparatively — show where sources agree and diverge rather than flattening nuance into a single narrative. "
        "Avoid superlatives, promotional language, and unqualified generalizations. "
        "Let the strength of the evidence drive the confidence of your claims."
    )
    return _resolved(_STANDARD_STRUCTURE), system


SYNTHESIZE_PROMPT = """You are a research report writer. Given a research topic and gathered sources, write a comprehensive, well-structured research report in markdown format.

TONE AND FRAMING:
- Write as a careful analyst, not an advocate. Do not promote, endorse, or editorialize.
- Avoid superlatives ("revolutionary", "groundbreaking", "the best") and marketing language ("game-changing", "next-generation", "cutting-edge") unless directly quoting a source — and if so, attribute the language.
- Frame comparisons with trade-offs: instead of "X is better than Y", write "X offers [advantage] at the cost of [disadvantage], according to [N]".
- When evidence is mixed or thin, say so directly rather than choosing a side.
- Prefer specific, quantified claims over vague qualitative ones.

{citation_rules}

{uncertainty_rules}

{structure}

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
    report_style: ReportStyle = ReportStyle.STANDARD,
    iteration_count: int = 1,
) -> tuple[str, str, list[str]]:
    sq_text = "\n".join(f"- {sq.question}" for sq in sub_questions)
    today = datetime.now().strftime("%Y-%m-%d")

    style_structure, style_system = _get_style_prompt(
        report_style, analyses, iteration_count, len(sub_questions),
        query=query, date=today,
    )

    # Include uncertainty rules for standard and academic
    uncertainty = _UNCERTAINTY_RULES if report_style != ReportStyle.BRIEF else ""

    prompt = SYNTHESIZE_PROMPT.format(
        citation_rules=_CITATION_RULES.format(source_count=len(analyses)),
        uncertainty_rules=uncertainty,
        structure=style_structure,
        query=query,
        sub_questions=sq_text,
        analyses=format_analyses(analyses),
        source_count=len(analyses),
    )
    if template_guidance:
        prompt += f"\n\nDomain-specific guidance: {template_guidance}"

    # Template system prompt overrides style system prompt
    system = system_prompt or style_system
    report_text = await achat(prompt, system=system, temperature=0.4, thinking=thinking, model=model)

    # Clean invalid citations and replace LLM-generated references with verified ones
    report_text = clean_invalid_citations(report_text, len(analyses))
    report_text = replace_references_in_report(report_text, analyses)

    followup_data = await achat_json(
        f"Based on this research report, suggest 3-5 follow-up questions for further investigation.\n\n{report_text}\n\nRespond as JSON: {{\"follow_up_questions\": [\"q1\", \"q2\", ...]}}",
        model=model,
    )
    follow_ups = followup_data.get("follow_up_questions", [])

    executive, detailed = _split_report(report_text)
    return executive, detailed, follow_ups


def _split_report(report_text: str) -> tuple[str, str]:
    """Split synthesized markdown into (executive_summary_section, detailed_findings).

    The model is asked to emit a leading H1 title and a "**Date:** …" metadata
    line, but output/report.py regenerates those itself — so we drop them here to
    avoid duplicated titles/metadata in the saved report. The first "## " section
    (Executive Summary / Summary / Abstract) becomes the executive summary; every
    section after it becomes the detailed findings.

    Falls back to ("", text) when the output has no recognizable "## " section
    structure, so the full report still renders without being duplicated.
    """
    lines = report_text.strip().split("\n")
    # Drop a leading H1 title ("# ...", not "## ...") and the metadata line.
    while lines and (
        (lines[0].startswith("# ") and not lines[0].startswith("## "))
        or lines[0].strip().startswith("**Date:")
        or not lines[0].strip()
    ):
        lines.pop(0)
    text = "\n".join(lines).strip()

    if not text.startswith("## "):
        return "", text

    parts = text[len("## "):].split("\n## ", 1)
    executive = ("## " + parts[0]).strip()
    detailed = ("## " + parts[1]).strip() if len(parts) > 1 else ""
    return executive, detailed
