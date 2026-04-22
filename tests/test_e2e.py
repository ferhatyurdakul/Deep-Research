"""
End-to-end tests for the Deep Research pipeline.

These tests hit live APIs (Z.AI GLM, Tavily, arXiv, Semantic Scholar)
and verify the full research pipeline works correctly.

Run with:
    python -m pytest tests/test_e2e.py -v -s

Requires .env with valid ZAI_API_KEY and TAVILY_API_KEY.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research.config import settings
from deep_research.models import ReportStyle, ResearchConfig, ResearchDepth


# --- Guards ---

def _has_api_keys() -> bool:
    return bool(settings.zai_api_key and settings.tavily_api_key)


skip_no_keys = pytest.mark.skipif(
    not _has_api_keys(),
    reason="ZAI_API_KEY and TAVILY_API_KEY required",
)


# --- Unit-level: individual components ---

@skip_no_keys
class TestLLM:
    def test_sync_chat(self):
        from deep_research.llm import chat
        result = chat("Say 'hello' and nothing else.", temperature=0.0)
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_async_chat(self):
        from deep_research.llm import achat
        result = await achat("Say 'hello' and nothing else.", temperature=0.0)
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_async_chat_json(self):
        from deep_research.llm import achat_json
        result = await achat_json('Return this exact JSON: {"status": "ok"}')
        assert result["status"] == "ok"


@skip_no_keys
class TestSearch:
    @pytest.mark.asyncio
    async def test_tavily_search(self):
        from deep_research.search.web_search import TavilySearchProvider
        provider = TavilySearchProvider(settings.tavily_api_key)
        results = await provider.search("Python programming language", max_results=3)
        assert len(results) > 0
        assert results[0].url.startswith("http")
        assert results[0].title

    @pytest.mark.asyncio
    async def test_duckduckgo_search(self):
        from deep_research.search.web_search import DuckDuckGoSearchProvider
        provider = DuckDuckGoSearchProvider()
        results = await provider.search("Python programming language", max_results=3)
        # DDG may fail on older Python/SSL (LibreSSL < 3.x doesn't support TLS 1.3)
        if len(results) > 0:
            assert results[0].url.startswith("http")
        else:
            pytest.skip("DuckDuckGo unavailable (likely SSL/TLS incompatibility)")


class TestAcademicSearch:
    @pytest.mark.asyncio
    async def test_arxiv(self):
        from deep_research.search.arxiv_search import ArxivSearchProvider
        provider = ArxivSearchProvider()
        results = await provider.search("transformer neural network", max_results=3)
        assert len(results) > 0
        assert "arxiv.org" in results[0].url

    @pytest.mark.asyncio
    async def test_semantic_scholar(self):
        from deep_research.search.semantic_scholar import SemanticScholarProvider
        provider = SemanticScholarProvider()
        results = await provider.search("transformer neural network", max_results=3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_aggregator(self):
        from deep_research.search.aggregator import SearchAggregator
        agg = SearchAggregator(tavily_api_key="", semantic_scholar_api_key="")
        # Should have ddg as web fallback + arxiv + scholar
        assert "web" in agg.available_sources
        assert "arxiv" in agg.available_sources
        assert "scholar" in agg.available_sources

    @pytest.mark.asyncio
    async def test_aggregator_dedup(self):
        from deep_research.search.aggregator import deduplicate_results
        from deep_research.models import SearchResult, SourceType
        results = [
            SearchResult(title="Paper A", url="https://arxiv.org/abs/1234", snippet="", source_type=SourceType.ARXIV),
            SearchResult(title="Paper A", url="https://arxiv.org/abs/1234/", snippet="", source_type=SourceType.WEB),
            SearchResult(title="Paper B", url="https://example.com/b", snippet="", source_type=SourceType.WEB),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 2  # "Paper A" deduped by URL


class TestDeduplicationAndScoring:
    """Tests for enhanced dedup (DOI/arXiv ID) and composite scoring."""

    def test_dedup_by_arxiv_id(self):
        """arXiv and Semantic Scholar returning same paper should dedup."""
        from deep_research.search.aggregator import deduplicate_results
        from deep_research.models import SearchResult, SourceType
        results = [
            SearchResult(
                title="Attention Is All You Need",
                url="https://arxiv.org/abs/1706.03762",
                snippet="We propose the Transformer.",
                source_type=SourceType.ARXIV,
                extra={"arxiv_id": "1706.03762"},
            ),
            SearchResult(
                title="Attention Is All You Need",
                url="https://api.semanticscholar.org/paper/abc",
                snippet="We propose the Transformer.",
                source_type=SourceType.SEMANTIC_SCHOLAR,
                extra={"arxiv_id": "1706.03762", "doi": "10.5555/abc"},
            ),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 1
        # arXiv should win (higher priority)
        assert deduped[0].source_type == SourceType.ARXIV

    def test_dedup_by_doi(self):
        from deep_research.search.aggregator import deduplicate_results
        from deep_research.models import SearchResult, SourceType
        results = [
            SearchResult(
                title="Some Journal Paper",
                url="https://nature.com/article/123",
                snippet="...", source_type=SourceType.WEB,
                extra={"doi": "10.1038/s41586-024-00001"},
            ),
            SearchResult(
                title="Some Journal Paper (preprint)",
                url="https://api.semanticscholar.org/paper/xyz",
                snippet="...", source_type=SourceType.SEMANTIC_SCHOLAR,
                extra={"doi": "10.1038/s41586-024-00001"},
            ),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 1

    def test_dedup_preserves_unique(self):
        from deep_research.search.aggregator import deduplicate_results
        from deep_research.models import SearchResult, SourceType
        results = [
            SearchResult(title="Paper A", url="https://a.com", snippet="", source_type=SourceType.WEB),
            SearchResult(title="Paper B", url="https://b.com", snippet="", source_type=SourceType.WEB),
            SearchResult(title="Paper C", url="https://c.com", snippet="", source_type=SourceType.ARXIV),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 3

    def test_composite_score_academic_bonus(self):
        from deep_research.search.aggregator import compute_source_score
        from deep_research.models import SearchResult, SourceType
        web = SearchResult(title="Blog", url="https://blog.com/x", snippet="...", source_type=SourceType.WEB)
        arxiv = SearchResult(title="Paper", url="https://arxiv.org/abs/1234", snippet="...", source_type=SourceType.ARXIV)
        assert compute_source_score(arxiv) > compute_source_score(web)

    def test_composite_score_citation_count(self):
        from deep_research.search.aggregator import compute_source_score
        from deep_research.models import SearchResult, SourceType
        low = SearchResult(title="Paper", url="https://arxiv.org/abs/1", snippet="...",
                           source_type=SourceType.SEMANTIC_SCHOLAR, extra={"citation_count": 2})
        high = SearchResult(title="Paper", url="https://arxiv.org/abs/2", snippet="...",
                            source_type=SourceType.SEMANTIC_SCHOLAR, extra={"citation_count": 500})
        assert compute_source_score(high) > compute_source_score(low)

    def test_composite_score_recency(self):
        from deep_research.search.aggregator import compute_source_score
        from deep_research.models import SearchResult, SourceType
        old = SearchResult(title="Old", url="https://a.com", snippet="...",
                           source_type=SourceType.WEB, published_date="2015-01-01")
        new = SearchResult(title="New", url="https://b.com", snippet="...",
                           source_type=SourceType.WEB, published_date="2024-06-01")
        assert compute_source_score(new) > compute_source_score(old)

    def test_domain_reliability_high(self):
        from deep_research.search.aggregator import domain_reliability_score
        assert domain_reliability_score("https://nature.com/articles/123") == 1.0
        assert domain_reliability_score("https://pubmed.ncbi.nlm.nih.gov/12345") == 1.0
        assert domain_reliability_score("https://arxiv.org/abs/1234") == 1.0
        assert domain_reliability_score("https://www.stanford.edu/paper") == 1.0

    def test_domain_reliability_low(self):
        from deep_research.search.aggregator import domain_reliability_score
        assert domain_reliability_score("https://www.pinterest.com/pin/123") == 0.2
        assert domain_reliability_score("https://quora.com/question/what") == 0.2

    def test_domain_reliability_neutral(self):
        from deep_research.search.aggregator import domain_reliability_score
        score = domain_reliability_score("https://randomsite.com/page")
        assert score == 0.5


class TestExtraction:
    @pytest.mark.asyncio
    async def test_html_scrape(self):
        from deep_research.extraction.parser import ContentParser
        async with ContentParser() as parser:
            result = await parser.parse_url("https://example.com")
        assert result is not None
        assert result.title == "Example Domain"
        assert result.word_count > 0

    @pytest.mark.asyncio
    async def test_pdf_scrape_arxiv(self):
        from deep_research.extraction.parser import ContentParser
        async with ContentParser() as parser:
            result = await parser.parse_url("https://arxiv.org/abs/1706.03762")
        assert result is not None
        assert result.word_count > 100
        assert "attention" in result.content.lower()

    @pytest.mark.asyncio
    async def test_arxiv_abstract_shortcut(self):
        """arXiv papers should use abstract directly without fetching."""
        from deep_research.extraction.parser import ContentParser
        from deep_research.models import SearchResult, SourceType
        sr = SearchResult(
            title="Test Paper", url="https://arxiv.org/abs/2301.00001",
            snippet="This is the abstract of the paper about transformers.",
            source_type=SourceType.ARXIV,
        )
        async with ContentParser() as parser:
            result = await parser.parse_search_result(sr)
        assert result is not None
        assert "abstract" in result.content.lower()
        assert result.word_count > 0


class TestReportStyles:
    def test_methodology_builder(self):
        from deep_research.pipeline.synthesizer import _build_search_methodology
        from deep_research.models import SourceAnalysis, SourceType

        sources = [
            SourceAnalysis(url="https://a.com", title="A", key_findings=["f"], source_type=SourceType.WEB),
            SourceAnalysis(url="https://b.com", title="B", key_findings=["f"], source_type=SourceType.ARXIV),
            SourceAnalysis(url="https://c.com", title="C", key_findings=["f"], source_type=SourceType.WEB, relevance="snippet only"),
        ]
        result = _build_search_methodology(sources, iterations=2, sub_question_count=5)
        assert "5 sub-questions" in result
        assert "3 sources" in result
        assert "2 web sources" in result
        assert "1 arXiv" in result
        assert "2 were analyzed from full text" in result
        assert "1 from snippets" in result

    def test_style_prompt_brief(self):
        from deep_research.pipeline.synthesizer import _get_style_prompt
        structure, system = _get_style_prompt(ReportStyle.BRIEF, [], 1, 3)
        assert "Research Brief" in structure
        assert "500-1000 words" in structure

    def test_style_prompt_academic(self):
        from deep_research.pipeline.synthesizer import _get_style_prompt
        from deep_research.models import SourceAnalysis, SourceType
        sources = [SourceAnalysis(url="https://a.com", title="A", key_findings=["f"], source_type=SourceType.WEB)]
        structure, system = _get_style_prompt(ReportStyle.ACADEMIC, sources, 1, 3)
        assert "Abstract" in structure
        assert "Methodology" in structure
        assert "Discussion" in structure
        assert "Limitations and Uncertainty" in structure
        assert "academic" in system.lower()

    def test_style_prompt_standard(self):
        from deep_research.pipeline.synthesizer import _get_style_prompt
        structure, system = _get_style_prompt(ReportStyle.STANDARD, [], 1, 3)
        assert "Executive Summary" in structure
        assert "Key Findings" in structure


class TestEvidenceExtraction:
    def test_format_analyses_shows_evidenced_findings(self):
        from deep_research.pipeline.analyzer import format_analyses
        from deep_research.models import (
            ContentDepth,
            EvidencedFinding,
            SourceAnalysis,
            SourceType,
        )

        sa = SourceAnalysis(
            url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            key_findings=["Transformer replaces recurrence with attention."],
            evidenced_findings=[
                EvidencedFinding(
                    finding="Transformer replaces recurrence with attention.",
                    evidence="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                    confidence="supported",
                ),
                EvidencedFinding(
                    finding="Training is faster than RNNs.",
                    evidence="",
                    confidence="inferred",
                ),
            ],
            content_depth=ContentDepth.ABSTRACT,
            source_type=SourceType.ARXIV,
        )
        out = format_analyses([sa])
        assert "Findings with evidence:" in out
        assert "Transformer replaces recurrence" in out
        assert "based solely on attention mechanisms" in out
        # Non-"supported" confidence should be annotated
        assert "[inferred]" in out
        # Content depth label should appear
        assert "abstract" in out

    def test_format_analyses_snippet_depth_label(self):
        from deep_research.pipeline.analyzer import format_analyses
        from deep_research.models import ContentDepth, SourceAnalysis, SourceType

        sa = SourceAnalysis(
            url="https://example.com",
            title="Ex",
            key_findings=["partial data from snippet"],
            content_depth=ContentDepth.SNIPPET,
            source_type=SourceType.WEB,
        )
        out = format_analyses([sa])
        assert "snippet" in out

    def test_format_analyses_falls_back_to_flat_lists(self):
        """Old-format SourceAnalysis (no evidenced_findings) should still render."""
        from deep_research.pipeline.analyzer import format_analyses
        from deep_research.models import SourceAnalysis, SourceType

        sa = SourceAnalysis(
            url="https://example.com",
            title="Legacy",
            key_findings=["legacy finding"],
            key_evidence=["legacy evidence quote"],
            source_type=SourceType.WEB,
        )
        out = format_analyses([sa])
        assert "legacy finding" in out
        assert "legacy evidence quote" in out

    def test_analyze_prompt_requests_evidenced_findings(self):
        """The ANALYZE_PROMPT must ask the LLM for evidenced_findings schema."""
        from deep_research.pipeline.analyzer import ANALYZE_PROMPT
        assert "evidenced_findings" in ANALYZE_PROMPT
        assert "verbatim" in ANALYZE_PROMPT.lower()
        assert "confidence" in ANALYZE_PROMPT


class TestTemplates:
    def test_list_templates(self):
        from deep_research.pipeline.templates import list_templates
        templates = list_templates()
        assert len(templates) == 4
        keys = {t["key"] for t in templates}
        assert keys == {"technology", "science", "market", "literature"}

    def test_get_template(self):
        from deep_research.pipeline.templates import get_template
        t = get_template("science")
        assert t is not None
        assert t.use_academic is True
        assert t.decompose_guidance


class TestModels:
    def test_depth_presets(self):
        quick = ResearchConfig.from_depth(ResearchDepth.QUICK)
        assert quick.max_sub_questions == 3
        assert quick.max_iterations == 1
        assert quick.use_thinking is False

        deep = ResearchConfig.from_depth(ResearchDepth.DEEP)
        assert deep.max_sub_questions == 7
        assert deep.max_iterations == 3
        assert deep.use_thinking is True
        assert deep.use_academic_search is True
        assert deep.use_extraction is True

    def test_model_router(self):
        config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
        assert config.models.get("decompose") is None
        config.models.decompose = "glm-4.5-air"
        assert config.models.get("decompose") == "glm-4.5-air"

    def test_max_sources(self):
        config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
        assert config.max_sources == 30

    def test_report_style_default(self):
        config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
        assert config.report_style == ReportStyle.STANDARD

    def test_report_style_override(self):
        config = ResearchConfig.from_depth(ResearchDepth.QUICK)
        config.report_style = ReportStyle.ACADEMIC
        assert config.report_style == ReportStyle.ACADEMIC

    def test_report_style_enum_values(self):
        assert ReportStyle.BRIEF == "brief"
        assert ReportStyle.STANDARD == "standard"
        assert ReportStyle.ACADEMIC == "academic"


# --- Integration: research pipeline ---

@skip_no_keys
class TestResearchPipeline:
    @pytest.mark.asyncio
    async def test_decompose(self):
        from deep_research.pipeline.decomposer import adecompose_query
        subs = await adecompose_query("What is quantum computing?", max_q=3)
        assert len(subs) == 3
        assert all(sq.question for sq in subs)

    @pytest.mark.asyncio
    async def test_analyze_source(self):
        from deep_research.pipeline.analyzer import aanalyze_source
        sa = await aanalyze_source(
            query="What is Python?",
            url="https://example.com",
            title="Python Overview",
            content="Python is a high-level programming language created by Guido van Rossum. "
                    "It emphasizes readability and simplicity. Python supports multiple paradigms.",
        )
        assert sa.url == "https://example.com"
        assert len(sa.key_findings) > 0

    @pytest.mark.asyncio
    async def test_quick_research(self):
        """Full pipeline test at quick depth — fast, 1 iteration."""
        from deep_research.pipeline.orchestrator import arun_research
        config = ResearchConfig.from_depth(ResearchDepth.QUICK)
        report = await arun_research("What is the Python GIL?", config)

        assert report.query == "What is the Python GIL?"
        assert len(report.sub_questions) > 0
        assert len(report.sources) > 0
        assert report.executive_summary
        assert report.detailed_findings
        assert len(report.iterations) >= 1


# --- Database ---

class TestDatabase:
    def test_save_and_load(self, tmp_path, monkeypatch):
        test_db = tmp_path / "test.db"
        import deep_research.config as cfg
        import deep_research.storage.db as db_mod
        monkeypatch.setattr(cfg, "DB_PATH", test_db)
        monkeypatch.setattr(db_mod, "DB_PATH", test_db)

        from deep_research.storage.db import init_db, save_report, load_report, list_sessions, delete_session
        from deep_research.models import ResearchReport, SubQuestion, SourceAnalysis

        init_db()

        report = ResearchReport(
            query="Test query",
            sub_questions=[SubQuestion(question="What is X?", reasoning="because")],
            sources=[SourceAnalysis(url="https://example.com", title="Test", key_findings=["finding1"], relevance="high")],
            executive_summary="Summary here",
            detailed_findings="## Findings\nDetails here",
            follow_up_questions=["Follow up?"],
        )

        session_id = save_report(report, depth="quick")
        assert session_id > 0

        loaded = load_report(session_id)
        assert loaded is not None
        assert loaded.query == "Test query"
        assert loaded.executive_summary == "Summary here"
        assert len(loaded.sources) == 1

        sessions = list_sessions()
        assert len(sessions) == 1

        assert delete_session(session_id) is True
        assert load_report(session_id) is None


class TestSessionForking:
    def _fresh_db(self, tmp_path, monkeypatch):
        test_db = tmp_path / "fork.db"
        import deep_research.config as cfg
        import deep_research.storage.db as db_mod
        monkeypatch.setattr(cfg, "DB_PATH", test_db)
        monkeypatch.setattr(db_mod, "DB_PATH", test_db)
        from deep_research.storage.db import init_db
        init_db()

    def test_fork_session_copies_all_artifacts(self, tmp_path, monkeypatch):
        self._fresh_db(tmp_path, monkeypatch)
        from deep_research.storage.db import (
            fork_session, load_report, save_report,
        )
        from deep_research.models import (
            ResearchIteration,
            ResearchReport,
            SourceAnalysis,
            SubQuestion,
        )

        original = ResearchReport(
            query="Original question",
            sub_questions=[SubQuestion(question="Q1", reasoning="r")],
            sources=[SourceAnalysis(
                url="https://a.com", title="A",
                key_findings=["finding"], relevance="high",
                summary="sum",
            )],
            iterations=[ResearchIteration(iteration_number=1)],
            searched_urls=["https://a.com"],
            executive_summary="Exec",
            detailed_findings="## Details",
            follow_up_questions=["Next?"],
        )
        orig_id = save_report(original, depth="standard")

        fork_id = fork_session(orig_id)
        assert fork_id is not None
        assert fork_id != orig_id

        forked = load_report(fork_id)
        assert forked is not None
        assert forked.parent_session_id == orig_id
        assert forked.query == "Original question"
        assert forked.executive_summary == "Exec"
        assert forked.detailed_findings == "## Details"
        assert forked.follow_up_questions == ["Next?"]
        assert len(forked.sources) == 1
        assert forked.sources[0].url == "https://a.com"
        assert forked.sources[0].summary == "sum"
        assert forked.searched_urls == ["https://a.com"]
        assert len(forked.iterations) == 1

    def test_fork_preserves_original(self, tmp_path, monkeypatch):
        self._fresh_db(tmp_path, monkeypatch)
        from deep_research.storage.db import (
            fork_session, load_report, save_report,
        )
        from deep_research.models import ResearchReport, SourceAnalysis

        original = ResearchReport(
            query="Keep me",
            sources=[SourceAnalysis(url="https://a.com", title="A", key_findings=["x"])],
            executive_summary="Keep",
        )
        orig_id = save_report(original, depth="standard")
        fork_session(orig_id)

        still_there = load_report(orig_id)
        assert still_there is not None
        assert still_there.query == "Keep me"
        assert still_there.executive_summary == "Keep"
        assert still_there.parent_session_id is None  # originals have no parent

    def test_fork_with_new_query(self, tmp_path, monkeypatch):
        self._fresh_db(tmp_path, monkeypatch)
        from deep_research.storage.db import (
            fork_session, load_report, save_report,
        )
        from deep_research.models import ResearchReport, SourceAnalysis

        original = ResearchReport(
            query="Original",
            sources=[SourceAnalysis(url="https://a.com", title="A", key_findings=["x"])],
        )
        orig_id = save_report(original, depth="standard")

        fork_id = fork_session(orig_id, new_query="Refined variant")
        forked = load_report(fork_id)
        assert forked.query == "Refined variant"
        assert forked.parent_session_id == orig_id
        # Sources are still copied over
        assert len(forked.sources) == 1

    def test_fork_missing_session_returns_none(self, tmp_path, monkeypatch):
        self._fresh_db(tmp_path, monkeypatch)
        from deep_research.storage.db import fork_session
        assert fork_session(9999) is None

    def test_list_sessions_includes_parent(self, tmp_path, monkeypatch):
        self._fresh_db(tmp_path, monkeypatch)
        from deep_research.storage.db import (
            fork_session, list_sessions, save_report,
        )
        from deep_research.models import ResearchReport

        orig_id = save_report(ResearchReport(query="Q"), depth="quick")
        fork_id = fork_session(orig_id)

        rows = {s["id"]: s for s in list_sessions()}
        assert rows[orig_id]["parent_session_id"] is None
        assert rows[fork_id]["parent_session_id"] == orig_id


# --- Report formatting ---

class TestReport:
    def test_format_markdown(self):
        from deep_research.output.report import format_report_markdown
        from deep_research.models import ResearchReport, SubQuestion, SourceAnalysis

        report = ResearchReport(
            query="Test query",
            sources=[SourceAnalysis(url="https://example.com", title="Test Source", key_findings=["f1"], relevance="high")],
            executive_summary="Executive summary text",
            detailed_findings="## Detailed\nFindings here",
        )

        md = format_report_markdown(report)
        assert "Test query" in md
        assert "Executive summary text" in md
        assert "References" in md
        assert "[1]" in md
        assert "Test Source" in md

    def test_no_duplicate_references(self):
        """When detailed_findings already has a References section, don't add another."""
        from deep_research.output.report import format_report_markdown
        from deep_research.models import ResearchReport, SourceAnalysis

        report = ResearchReport(
            query="Test",
            sources=[SourceAnalysis(url="https://example.com", title="S1", key_findings=["f1"])],
            executive_summary="Summary",
            detailed_findings="## Findings\nSome text [1].\n\n## References\n\n[1] S1. https://example.com",
        )

        md = format_report_markdown(report)
        count = md.lower().count("## references")
        assert count == 1, f"Expected 1 References section, found {count}"


# --- Citations ---

class TestCitations:
    def test_format_web_citation(self):
        from deep_research.output.citations import format_citation
        from deep_research.models import SourceAnalysis, SourceType

        source = SourceAnalysis(
            url="https://example.com/article",
            title="Test Article",
            key_findings=["f1"],
            source_type=SourceType.WEB,
        )
        citation = format_citation(1, source)
        assert citation.startswith("[1]")
        assert "Test Article" in citation
        assert "https://example.com/article" in citation
        assert "retrieved" in citation

    def test_format_arxiv_citation(self):
        from deep_research.output.citations import format_citation
        from deep_research.models import SourceAnalysis, SourceType

        source = SourceAnalysis(
            url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            key_findings=["f1"],
            source_type=SourceType.ARXIV,
            authors=["Vaswani, A.", "Shazeer, N."],
            published_date="2017-06-12",
            extra={"arxiv_id": "1706.03762"},
        )
        citation = format_citation(1, source)
        assert "Vaswani" in citation
        assert "arXiv:1706.03762" in citation
        assert "(2017)" in citation

    def test_format_scholar_citation(self):
        from deep_research.output.citations import format_citation
        from deep_research.models import SourceAnalysis, SourceType

        source = SourceAnalysis(
            url="https://arxiv.org/abs/2301.00001",
            title="Some Paper",
            key_findings=["f1"],
            source_type=SourceType.SEMANTIC_SCHOLAR,
            authors=["Smith, J."],
            published_date="2023",
            extra={"doi": "10.1234/test", "venue": "NeurIPS"},
        )
        citation = format_citation(1, source)
        assert "Smith" in citation
        assert "NeurIPS" in citation
        assert "DOI: 10.1234/test" in citation

    def test_replace_references_in_report(self):
        from deep_research.output.citations import replace_references_in_report
        from deep_research.models import SourceAnalysis, SourceType

        md = "# Report\n\nSome text [1].\n\n## References\n\n[1] Fake reference"
        sources = [SourceAnalysis(
            url="https://example.com", title="Real Source",
            key_findings=["f1"], source_type=SourceType.WEB,
        )]
        result = replace_references_in_report(md, sources)
        assert "Fake reference" not in result
        assert "Real Source" in result
        assert "## References" in result

    def test_build_references_section(self):
        from deep_research.output.citations import build_references_section
        from deep_research.models import SourceAnalysis, SourceType

        sources = [
            SourceAnalysis(url="https://a.com", title="A", key_findings=["f"], source_type=SourceType.WEB),
            SourceAnalysis(url="https://b.com", title="B", key_findings=["f"], source_type=SourceType.ARXIV,
                           extra={"arxiv_id": "2301.00001"}, published_date="2023-01-01"),
        ]
        refs = build_references_section(sources)
        assert "[1]" in refs
        assert "[2]" in refs
        assert "## References" in refs

    def test_validate_citations_valid(self):
        from deep_research.output.citations import validate_citations
        md = "Some claim [1]. Another claim [2, 3].\n\n## References\n[1] ..."
        result = validate_citations(md, 3)
        assert result["valid"] == [1, 2, 3]
        assert result["invalid"] == []
        assert result["unused_sources"] == []

    def test_validate_citations_invalid(self):
        from deep_research.output.citations import validate_citations
        md = "A claim [1]. Bad ref [5]. Also bad [0].\n\n## References\n"
        result = validate_citations(md, 3)
        assert 1 in result["valid"]
        assert 5 in result["invalid"]
        assert 0 in result["invalid"]
        assert 2 in result["unused_sources"]
        assert 3 in result["unused_sources"]

    def test_clean_invalid_citations(self):
        from deep_research.output.citations import clean_invalid_citations
        md = "Claim [1]. Bad [5]. Mixed [1, 5, 2].\n\n## References\n[1] Source"
        result = clean_invalid_citations(md, 3)
        assert "[5]" not in result.split("## References")[0]
        assert "[1]" in result
        assert "[1, 2]" in result
        # References section untouched
        assert "[1] Source" in result

    def test_validate_uncited_paragraphs(self):
        from deep_research.output.citations import validate_citations
        md = (
            "This is a well-cited paragraph with evidence from the source material [1].\n\n"
            "This paragraph has no citation and is long enough to be substantive and should be flagged by the validator.\n\n"
            "## References\n"
        )
        result = validate_citations(md, 2)
        assert result["uncited_paragraphs"] >= 1

    def test_find_all_citations(self):
        from deep_research.output.citations import find_all_citations
        nums = find_all_citations("Text [1] more [2, 3] end [1].")
        assert sorted(set(nums)) == [1, 2, 3]


# --- Database with enriched fields ---

class TestDatabaseEnriched:
    def test_save_and_load_with_metadata(self, tmp_path, monkeypatch):
        test_db = tmp_path / "test_enriched.db"
        import deep_research.config as cfg
        import deep_research.storage.db as db_mod
        monkeypatch.setattr(cfg, "DB_PATH", test_db)
        monkeypatch.setattr(db_mod, "DB_PATH", test_db)

        from deep_research.storage.db import init_db, save_report, load_report
        from deep_research.models import (
            ContentDepth,
            EvidencedFinding,
            ResearchReport,
            SourceAnalysis,
            SourceType,
        )

        init_db()

        report = ResearchReport(
            query="Enriched test",
            sources=[SourceAnalysis(
                url="https://arxiv.org/abs/1706.03762",
                title="Attention Is All You Need",
                key_findings=["transformers"],
                key_evidence=["The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."],
                evidenced_findings=[
                    EvidencedFinding(
                        finding="Transformers replace recurrence with attention.",
                        evidence="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                        confidence="supported",
                    ),
                ],
                content_depth=ContentDepth.ABSTRACT,
                relevance="high",
                summary="Introduces the transformer architecture.",
                source_type=SourceType.ARXIV,
                authors=["Vaswani, A.", "Shazeer, N."],
                published_date="2017-06-12",
                extra={"arxiv_id": "1706.03762", "categories": ["cs.CL"]},
            )],
            executive_summary="Summary",
            detailed_findings="## Findings",
        )

        session_id = save_report(report, depth="standard")
        loaded = load_report(session_id)

        assert loaded is not None
        s = loaded.sources[0]
        assert s.source_type == SourceType.ARXIV
        assert s.summary == "Introduces the transformer architecture."
        assert s.authors == ["Vaswani, A.", "Shazeer, N."]
        assert s.published_date == "2017-06-12"
        assert s.extra["arxiv_id"] == "1706.03762"
        assert len(s.key_evidence) == 1
        assert "dominant sequence" in s.key_evidence[0]
        assert s.content_depth == ContentDepth.ABSTRACT
        assert len(s.evidenced_findings) == 1
        assert s.evidenced_findings[0].finding.startswith("Transformers")
        assert s.evidenced_findings[0].confidence == "supported"
