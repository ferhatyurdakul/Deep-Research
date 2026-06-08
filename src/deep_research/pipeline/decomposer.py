from __future__ import annotations

from deep_research.llm import achat_json
from deep_research.models import SubQuestion

DECOMPOSE_PROMPT = """You are a research strategist. Given a research question, break it down into {max_q} focused sub-questions that together would provide a comprehensive understanding of the topic.

Research question: {query}

Respond in JSON format:
{{
  "sub_questions": [
    {{"question": "...", "reasoning": "why this sub-question matters"}}
  ]
}}"""


async def adecompose_query(
    query: str, max_q: int = 5, model: str | None = None,
    template_guidance: str = "",
) -> list[SubQuestion]:
    prompt = DECOMPOSE_PROMPT.format(query=query, max_q=max_q)
    if template_guidance:
        prompt += f"\n\nDomain guidance: {template_guidance}"
    data = await achat_json(prompt, model=model)
    raw = data.get("sub_questions", []) if isinstance(data, dict) else []
    sub_questions = [
        SubQuestion(**sq) for sq in raw if isinstance(sq, dict) and sq.get("question")
    ]
    # Degenerate fallback: if the model returned no usable sub-questions (bad
    # JSON, wrong shape, empty list), research the original query directly rather
    # than proceeding with an empty plan that yields a hollow, source-less report.
    if not sub_questions:
        sub_questions = [SubQuestion(question=query, reasoning="Original research question (decomposition returned nothing usable).")]
    return sub_questions
