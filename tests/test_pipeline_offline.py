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
