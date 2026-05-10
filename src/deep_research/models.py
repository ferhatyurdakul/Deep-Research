from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ResearchDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ReportStyle(str, Enum):
    BRIEF = "brief"          # executive summary + key findings + references
    STANDARD = "standard"    # current default structure
    ACADEMIC = "academic"    # abstract, research question, methodology, findings, discussion, limitations, references


class ModelRouter(BaseModel):
    """Routes different pipeline stages to different models."""
    decompose: str = ""     # fast model for query decomposition
    analyze: str = ""       # thinking model for source analysis
    synthesize: str = ""    # best model for final synthesis
    gap_analysis: str = ""  # thinking model for gap detection

    def get(self, stage: str) -> Optional[str]:
        """Return model for a stage, or None to use the default."""
        val = getattr(self, stage, "")
        return val if val else None


class ResearchConfig(BaseModel):
    depth: ResearchDepth = ResearchDepth.STANDARD
    report_style: ReportStyle = ReportStyle.STANDARD
    max_sub_questions: int = 5
    max_search_results: int = 5
    max_scrape_pages: int = 3
    max_iterations: int = 2
    max_sources: int = 30
    use_thinking: bool = False
    use_academic_search: bool = False
    use_extraction: bool = False
    template: Optional[str] = None  # template key from templates.py
    fresh: bool = False
    models: ModelRouter = Field(default_factory=ModelRouter)

    @classmethod
    def from_depth(cls, depth: ResearchDepth) -> ResearchConfig:
        presets = {
            ResearchDepth.QUICK: dict(
                max_sub_questions=3, max_search_results=3,
                max_scrape_pages=2, max_iterations=1, use_thinking=False,
            ),
            ResearchDepth.STANDARD: dict(
                max_sub_questions=5, max_search_results=5,
                max_scrape_pages=3, max_iterations=2, use_thinking=False,
            ),
            ResearchDepth.DEEP: dict(
                max_sub_questions=7, max_search_results=7,
                max_scrape_pages=5, max_iterations=3, use_thinking=True,
                use_academic_search=True, use_extraction=True,
            ),
        }
        return cls(depth=depth, **presets[depth])

    def load_model_routes(self) -> None:
        """
        Load model routing from settings.

        Per-stage env vars (MODEL_DECOMPOSE etc.) win when set; the active
        LLM_ROUTE profile fills the rest. Falls back to single-model routing
        if the profile is unknown for the active provider — no hard fail
        here, since validate_routes() runs separately at startup.
        """
        from .config import settings
        from .capabilities import CapabilityViolation
        try:
            stages = settings.effective_stage_models()
            self.models = ModelRouter(
                decompose=stages["decompose"],
                analyze=stages["analyze"],
                synthesize=stages["synthesize"],
                gap_analysis=stages["gap_analysis"],
            )
        except CapabilityViolation:
            self.models = ModelRouter(
                decompose=settings.model_decompose,
                analyze=settings.model_analyze,
                synthesize=settings.model_synthesize,
                gap_analysis=settings.model_gap_analysis,
            )


class SubQuestion(BaseModel):
    question: str
    reasoning: str = ""


class SourceType(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    WEB = "web"


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source_type: SourceType = SourceType.WEB
    authors: list[str] = Field(default_factory=list)
    published_date: str = ""
    extra: dict = Field(default_factory=dict)  # doi, venue, arxiv_id, etc.


class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    word_count: int = 0


class ContentDepth(str, Enum):
    """How much of a source's content was available for analysis."""
    FULL_TEXT = "full_text"   # full page/PDF extracted
    ABSTRACT = "abstract"     # arXiv abstract or similar
    SNIPPET = "snippet"       # search result snippet only


class EvidencedFinding(BaseModel):
    """A finding paired with its supporting evidence from the source."""
    finding: str                    # interpreted finding (may be paraphrased)
    evidence: str = ""              # verbatim quote or data point from source
    confidence: str = "supported"   # supported / partially_supported / inferred


class ExtractedData(BaseModel):
    """Structured data extracted from a source."""
    statistics: list[str] = Field(default_factory=list)   # numbers, percentages, metrics
    entities: list[str] = Field(default_factory=list)      # people, organizations, products
    dates: list[str] = Field(default_factory=list)         # key dates and timelines
    claims: list[str] = Field(default_factory=list)        # notable claims or conclusions


class SourceAnalysis(BaseModel):
    url: str
    title: str
    key_findings: list[str]
    key_evidence: list[str] = Field(default_factory=list)  # verbatim quotes / specific data points
    evidenced_findings: list[EvidencedFinding] = Field(default_factory=list)  # paired finding+evidence
    content_depth: ContentDepth = ContentDepth.FULL_TEXT
    relevance: str = ""
    summary: str = ""  # concise 2-4 sentence summary for synthesis
    source_type: SourceType = SourceType.WEB
    authors: list[str] = Field(default_factory=list)
    published_date: str = ""
    extra: dict = Field(default_factory=dict)  # doi, venue, arxiv_id, etc.
    extracted_data: Optional[ExtractedData] = None

    @property
    def citation_key(self) -> str:
        return f"[{self.url}]"


class KnowledgeGap(BaseModel):
    gap_description: str
    suggested_query: str
    priority: str = "medium"


class ResearchIteration(BaseModel):
    iteration_number: int
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    sources: list[SourceAnalysis] = Field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = Field(default_factory=list)


class ResearchReport(BaseModel):
    id: Optional[int] = None
    parent_session_id: Optional[int] = None
    query: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    sources: list[SourceAnalysis] = Field(default_factory=list)
    iterations: list[ResearchIteration] = Field(default_factory=list)
    searched_urls: list[str] = Field(default_factory=list)
    executive_summary: str = ""
    detailed_findings: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
