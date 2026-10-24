"""Offline end-to-end exercise of the research pipeline.

The real "smoke run" in the task list needs network egress + a live API key,
which this environment doesn't have. This test substitutes for it: it drives
`arun_research` end-to-end with the LLM and search/extraction layers mocked, so
we verify the wiring added for rate-limit resilience — configurable LLM
concurrency is applied, and a synthesis failure degrades to a saved partial
result instead of crashing — deterministically and without the network.
"""
from __future__ import annotations

import pytest

from deep_research.models import (
    ResearchConfig,
    ResearchDepth,
    ResearchReport,
    ScrapedContent,
    SearchResult,
    SourceAnalysis,
    SubQuestion,
)


class _FakeAggregator:
    available_sources = ["web", "ddg", "arxiv", "scholar"]

    def __init__(self, *args, **kwargs):
        pass

    async def search(self, queries, sources=None, max_results_per_query=5, max_total=30, academic_weight=0.5):
        return [
            SearchResult(title="Result A", url="https://a.example/1", snippet="snippet a"),
            SearchResult(title="Result B", url="https://b.example/2", snippet="snippet b"),
        ]


class _FakeParser:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def parse_many(self, results, max_concurrent=5):
        return [
            ScrapedContent(url=r.url, title=r.title, content="body " * 50, word_count=100)
            for r in results
        ]


async def _fake_decompose(query, max_q=5, model=None, template_guidance=""):
    return [SubQuestion(question=f"sub of {query}", reasoning="r")]


async def _fake_analyze(query, url, title, content, **kwargs):
    return SourceAnalysis(
        url=url, title=title, key_findings=["finding"], relevance="high",
        summary="a summary",
    )

def _analyze_failing_for(fail_urls: set[str]):
    """Fake aanalyze_source that raises for the given URLs, succeeds for the rest."""
    async def _analyze(query, url, title, content, **kwargs):
        if url in fail_urls:
            raise RuntimeError(f"simulated analysis failure for {url}")
        return SourceAnalysis(
            url=url, title=title, key_findings=["finding"], relevance="high",
            summary="a summary",
        )
    return _analyze


async def _fake_gaps(query, sub_questions, analyses, searched_queries, thinking=False, model=None):
    return [], True  # sufficient → stop after the first iteration


def _wire_pipeline(monkeypatch):
    import deep_research.pipeline.orchestrator as orch
    monkeypatch.setattr(orch, "SearchAggregator", _FakeAggregator)
    monkeypatch.setattr(orch, "ContentParser", _FakeParser)
    monkeypatch.setattr(orch, "adecompose_query", _fake_decompose)
    monkeypatch.setattr(orch, "aanalyze_source", _fake_analyze)
    monkeypatch.setattr(orch, "aidentify_gaps", _fake_gaps)
    return orch


@pytest.mark.asyncio
async def test_full_run_completes_offline(monkeypatch):
    orch = _wire_pipeline(monkeypatch)

    import deep_research.pipeline.synthesizer as synth

    async def _fake_synth(*args, **kwargs):
        return ("## Executive Summary\nKey result [1].", "## Findings\nDetail [1].", ["next?"])

    monkeypatch.setattr(synth, "asynthesize_report", _fake_synth)

    # Spy on the concurrency setter to confirm the config value is applied.
    seen = {}
    import deep_research.pipeline.analyzer as analyzer
    real_set = analyzer.set_llm_concurrency
    monkeypatch.setattr(orch, "set_llm_concurrency", lambda n: seen.setdefault("n", n) or real_set(n))

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.fresh = True  # skip the cross-session URL DB read
    config.max_llm_concurrency = 4

    report = await orch.arun_research("test topic", config)

    assert seen["n"] == 4
    assert len(report.sources) == 2
    assert report.executive_summary.startswith("## Executive Summary")
    assert "Findings" in report.detailed_findings
    assert report.follow_up_questions == ["next?"]


@pytest.mark.asyncio
async def test_synthesis_failure_yields_saved_partial(monkeypatch):
    orch = _wire_pipeline(monkeypatch)

    import deep_research.pipeline.synthesizer as synth

    async def _boom(*args, **kwargs):
        raise RuntimeError("429 Too Many Requests during synthesis")

    monkeypatch.setattr(synth, "asynthesize_report", _boom)

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.fresh = True

    # The run must NOT raise — sources are preserved and the caller can save
    # the session for --resynthesize.
    report = await orch.arun_research("test topic", config)

    assert len(report.sources) == 2          # search/extraction work preserved
    assert report.detailed_findings == ""    # synthesis produced nothing
    assert "resynthesize" in report.executive_summary.lower()


@pytest.mark.asyncio
async def test_partial_analysis_failure_skips_failed_sources(monkeypatch):
    """One 429'd source must not discard its siblings' analyses."""
    orch = _wire_pipeline(monkeypatch)
    monkeypatch.setattr(orch, "aanalyze_source", _analyze_failing_for({"https://a.example/1"}))

    import deep_research.pipeline.synthesizer as synth

    async def _fake_synth(*args, **kwargs):
        return ("## Executive Summary\nKey result [1].", "## Findings\nDetail [1].", ["next?"])

    monkeypatch.setattr(synth, "asynthesize_report", _fake_synth)

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.fresh = True

    report = await orch.arun_research("test topic", config)

    assert [s.url for s in report.sources] == ["https://b.example/2"]


@pytest.mark.asyncio
async def test_all_analysis_failures_abort_run(monkeypatch):
    """Every analysis failing is a systemic outage — fail loudly instead of
    synthesizing a source-less report."""
    orch = _wire_pipeline(monkeypatch)
    monkeypatch.setattr(
        orch, "aanalyze_source",
        _analyze_failing_for({"https://a.example/1", "https://b.example/2"}),
    )

    import deep_research.pipeline.synthesizer as synth

    async def _fake_synth(*args, **kwargs):
        return ("## Executive Summary", "## Findings", [])

    monkeypatch.setattr(synth, "asynthesize_report", _fake_synth)

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.fresh = True

    with pytest.raises(RuntimeError, match="simulated analysis failure"):
        await orch.arun_research("test topic", config)


@pytest.mark.asyncio
async def test_gap_failure_after_sources_yields_partial(monkeypatch):
    """A 429 during gap analysis (iteration 2) must not crash the run: the
    loop stops early and the already-collected sources are synthesized."""
    orch = _wire_pipeline(monkeypatch)

    async def _boom(*args, **kwargs):
        raise RuntimeError("429 during gap analysis")

    monkeypatch.setattr(orch, "aidentify_gaps", _boom)

    import deep_research.pipeline.synthesizer as synth

    async def _fake_synth(*args, **kwargs):
        return ("## Executive Summary\nKey result [1].", "## Findings\nDetail [1].", ["next?"])

    monkeypatch.setattr(synth, "asynthesize_report", _fake_synth)

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.fresh = True

    report = await orch.arun_research("test topic", config)

    assert len(report.sources) == 2                                    # iteration-1 work preserved
    assert report.executive_summary.startswith("## Executive Summary")  # synthesis still ran


@pytest.mark.asyncio
async def test_agent_search_and_analyze_skips_failed_sources(monkeypatch):
    """Agent mode shares the skip-don't-abort policy for failed analyses."""
    import deep_research.pipeline.agent as agent_mod

    monkeypatch.setattr(agent_mod, "ContentParser", _FakeParser)
    monkeypatch.setattr(agent_mod, "aanalyze_source", _analyze_failing_for({"https://a.example/1"}))

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    analyzed, snippets = await agent_mod._agent_search_and_analyze(
        "test topic", ["sub"], set(), config, _FakeAggregator()
    )

    assert [s.url for s in analyzed] == ["https://b.example/2"]
    assert snippets == []


@pytest.mark.asyncio
async def test_agent_search_and_analyze_all_failed_raises(monkeypatch):
    import deep_research.pipeline.agent as agent_mod

    monkeypatch.setattr(agent_mod, "ContentParser", _FakeParser)
    monkeypatch.setattr(
        agent_mod, "aanalyze_source",
        _analyze_failing_for({"https://a.example/1", "https://b.example/2"}),
    )

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    with pytest.raises(RuntimeError, match="simulated analysis failure"):
        await agent_mod._agent_search_and_analyze(
            "test topic", ["sub"], set(), config, _FakeAggregator()
        )


@pytest.mark.asyncio
async def test_scrape_budget_caps_fetched_pages(monkeypatch):
    """max_scrape_pages limits how many results get fetched; the rest become
    snippet-only sources instead of being scraped."""
    orch = _wire_pipeline(monkeypatch)

    import deep_research.pipeline.synthesizer as synth

    async def _fake_synth(*args, **kwargs):
        return ("## Executive Summary\nKey result [1].", "## Findings\nDetail [1].", ["next?"])

    monkeypatch.setattr(synth, "asynthesize_report", _fake_synth)

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.fresh = True
    config.max_scrape_pages = 1

    report = await orch.arun_research("test topic", config)

    analyzed = [s for s in report.sources if s.relevance != "snippet only"]
    snippets = [s for s in report.sources if s.relevance == "snippet only"]
    assert [s.url for s in analyzed] == ["https://a.example/1"]
    assert [s.url for s in snippets] == ["https://b.example/2"]


@pytest.mark.asyncio
async def test_known_urls_are_skipped_and_searched_urls_run_scoped(monkeypatch):
    """URLs known from earlier sessions are filtered (normalized compare),
    and report.searched_urls contains only URLs first seen in this run."""
    orch = _wire_pipeline(monkeypatch)
    monkeypatch.setattr(orch, "get_known_urls", lambda: {"a.example/1"})

    import deep_research.pipeline.synthesizer as synth

    async def _fake_synth(*args, **kwargs):
        return ("## Executive Summary", "## Findings", [])

    monkeypatch.setattr(synth, "asynthesize_report", _fake_synth)

    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.fresh = False  # consult get_known_urls

    report = await orch.arun_research("test topic", config)

    assert [s.url for s in report.sources] == ["https://b.example/2"]
    assert report.searched_urls == ["b.example/2"]


@pytest.mark.asyncio
async def test_continue_research_honors_template(monkeypatch):
    """Continuation passes template guidance + system prompt into synthesis,
    matching arun_research / arun_agent_research / --resynthesize."""
    from types import SimpleNamespace

    orch = _wire_pipeline(monkeypatch)
    tmpl = SimpleNamespace(
        name="T", use_academic=False, use_extraction=False,
        decompose_guidance="", synthesis_guidance="GUIDE", system_prompt="SYS",
    )
    monkeypatch.setattr(orch, "get_template", lambda key: tmpl)

    import deep_research.pipeline.synthesizer as synth

    synth_kwargs = {}

    async def _fake_synth(*args, **kwargs):
        synth_kwargs.update(kwargs)
        return ("## Executive Summary", "## Findings", ["n"])

    monkeypatch.setattr(synth, "asynthesize_report", _fake_synth)

    prev = ResearchReport(
        query="orig",
        sources=[SourceAnalysis(url="https://x.example/1", title="X", key_findings=["f"])],
    )
    config = ResearchConfig.from_depth(ResearchDepth.STANDARD)
    config.template = "science"
    config.fresh = True

    await orch.acontinue_research(prev, config)

    assert synth_kwargs.get("template_guidance") == "GUIDE"
    assert synth_kwargs.get("system_prompt") == "SYS"
