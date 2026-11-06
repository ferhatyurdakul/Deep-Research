from __future__ import annotations

from pydantic import BaseModel


class ResearchTemplate(BaseModel):
    """Domain-specific research strategy."""
    name: str
    description: str
    system_prompt: str
    decompose_guidance: str
    synthesis_guidance: str
    suggested_depth: str = "standard"
    use_academic: bool = False
    use_extraction: bool = False


TEMPLATES: dict[str, ResearchTemplate] = {
    "technology": ResearchTemplate(
        name="Technology",
        description="Technical topics: software, hardware, AI, infrastructure",
        system_prompt="You are a senior technology analyst specializing in evaluating technical approaches, architectures, and implementations. Focus on practical trade-offs, benchmarks, and real-world adoption.",
        decompose_guidance="Break down into: current state of the art, key technical approaches, performance comparisons, adoption trends, limitations, and future directions.",
        synthesis_guidance="Structure findings by technical approach. Include performance metrics, trade-offs, and practical recommendations. Highlight consensus vs. debate in the field.",
        suggested_depth="standard",
        use_academic=False,
        use_extraction=True,
    ),
    "science": ResearchTemplate(
        name="Science",
        description="Scientific research: biology, physics, chemistry, medicine",
        system_prompt="You are a scientific research analyst. Prioritize peer-reviewed findings, statistical significance, reproducibility, and methodological rigor. Distinguish between established consensus, emerging evidence, and speculation.",
        decompose_guidance="Break down into: current scientific understanding, recent discoveries, methodology advances, open questions, competing hypotheses, and clinical/practical implications.",
        synthesis_guidance="Structure findings by evidence strength. Lead with peer-reviewed consensus, then emerging findings. Flag any contradictions between studies. Include sample sizes and statistical significance where available.",
        suggested_depth="deep",
        use_academic=True,
        use_extraction=True,
    ),
    "market": ResearchTemplate(
        name="Market Analysis",
        description="Market research: industries, competitors, trends, sizing",
        system_prompt="You are a market research analyst. Focus on market size, growth rates, competitive landscape, key players, pricing, and strategic trends. Distinguish between verified data and projections.",
        decompose_guidance="Break down into: market size and growth, key players and market share, competitive dynamics, customer segments, pricing trends, regulatory factors, and future outlook.",
        synthesis_guidance="Structure findings around market dynamics. Lead with hard numbers (TAM, growth rates, market share). Include competitive positioning matrix. Distinguish between reported data and analyst projections.",
        suggested_depth="standard",
        use_academic=False,
        use_extraction=True,
    ),
    "literature": ResearchTemplate(
        name="Literature Review",
        description="Academic literature review: systematic survey of a research field",
        system_prompt="You are an academic research assistant conducting a systematic literature review. Focus on identifying key papers, research themes, methodological approaches, findings, and gaps in the literature.",
        decompose_guidance="Break down into: foundational works, major research themes, methodological approaches used, key findings and consensus, contradictions and debates, and identified gaps for future research.",
        synthesis_guidance="Structure as a formal literature review. Group by research theme, not chronologically. Identify methodological trends, areas of agreement/disagreement, and clear gaps for future work. Cite extensively.",
        suggested_depth="deep",
        use_academic=True,
        use_extraction=False,
    ),
}


def get_template(name: str) -> ResearchTemplate | None:
    return TEMPLATES.get(name.lower())


def list_templates() -> list[dict]:
    return [
        {"name": t.name, "key": k, "description": t.description}
        for k, t in TEMPLATES.items()
    ]
