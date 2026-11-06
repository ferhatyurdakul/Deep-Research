"""Tests for rate-limit resilience: backoff/jitter, retry, fallback, bounded
concurrency, and the partial-result synthesis path.

These use mocked clients and a zeroed-out retry wait so they run instantly and
never touch the network.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError
from tenacity import wait_none


# --- helpers ---------------------------------------------------------------

def _rate_limit_error(msg: str = "429 Too Many Requests") -> RateLimitError:
    request = httpx.Request("POST", "https://api.z.ai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(msg, response=response, body=None)


def _plan_lacks_model_error() -> RateLimitError:
    """What Z.AI sends when the plan doesn't include a model: HTTP 429, code 1311."""
    request = httpx.Request("POST", "https://api.z.ai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    body = {"code": "1311", "message": "Your current subscription plan does not yet include access to GLM-5.3-FlashX"}
    return RateLimitError("Error code: 429 - {'error': " + str(body) + "}", response=response, body=body)


def _ok_response(content: str = "ok"):
    """Minimal stand-in for an OpenAI ChatCompletion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _FakeCompletions:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return self.behavior(self.calls, kwargs)


class _FakeClient:
    def __init__(self, completions: _FakeCompletions):
        self.chat = SimpleNamespace(completions=completions)


@pytest.fixture
def fast_retries(monkeypatch):
    """Zero out the backoff sleep so retry tests don't actually wait."""
    import synapse_deep_research.llm as llm
    monkeypatch.setattr(llm._achat_inner.retry, "wait", wait_none())
    return llm


# --- backoff / jitter ------------------------------------------------------

class TestWaitStrategy:
    def test_rate_limit_waits_longer_than_transient(self, monkeypatch):
        import synapse_deep_research.llm as llm
        monkeypatch.setattr(llm.settings, "llm_retry_base_seconds", 2.0)
        monkeypatch.setattr(llm.settings, "llm_retry_max_seconds", 60.0)

        rl_state = SimpleNamespace(
            attempt_number=1,
            outcome=SimpleNamespace(exception=lambda: _rate_limit_error()),
        )
        transient_state = SimpleNamespace(
            attempt_number=1,
            outcome=SimpleNamespace(exception=lambda: TimeoutError("slow")),
        )
        # Sample a few times since jitter is random; rate-limit floor (3*base/2=3)
        # must always exceed the transient ceiling (base=2).
        for _ in range(50):
            assert llm._wait_strategy(rl_state) >= 3.0
            assert llm._wait_strategy(transient_state) <= 2.0

    def test_wait_is_capped(self, monkeypatch):
        import synapse_deep_research.llm as llm
        monkeypatch.setattr(llm.settings, "llm_retry_base_seconds", 2.0)
        monkeypatch.setattr(llm.settings, "llm_retry_max_seconds", 10.0)
        state = SimpleNamespace(
            attempt_number=20,  # huge exponent
            outcome=SimpleNamespace(exception=lambda: _rate_limit_error()),
        )
        for _ in range(50):
            assert llm._wait_strategy(state) <= 10.0


# --- retry / fallback ------------------------------------------------------

class TestRetryAndFallback:
    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, fast_retries, monkeypatch):
        llm = fast_retries

        def behavior(n, kwargs):
            if n < 3:
                raise _rate_limit_error()
            return _ok_response("recovered")

        comp = _FakeCompletions(behavior)
        monkeypatch.setattr(llm, "get_async_client", lambda: _FakeClient(comp))
        monkeypatch.setattr(llm.settings, "llm_fallback_provider", "")

        out = await llm.achat("hello")
        assert out == "recovered"
        assert comp.calls == 3  # two 429s, then success

    @pytest.mark.asyncio
    async def test_raises_provider_limit_when_no_fallback(self, fast_retries, monkeypatch):
        llm = fast_retries

        def behavior(n, kwargs):
            raise _rate_limit_error()

        comp = _FakeCompletions(behavior)
        monkeypatch.setattr(llm, "get_async_client", lambda: _FakeClient(comp))
        monkeypatch.setattr(llm.settings, "llm_fallback_provider", "")

        with pytest.raises(llm.ProviderLimitExceeded):
            await llm.achat("hello")
        # All attempts were exhausted before giving up.
        assert comp.calls == llm._MAX_ATTEMPTS

    @pytest.mark.opencode_go
    @pytest.mark.asyncio
    async def test_swaps_to_fallback_provider(self, fast_retries, monkeypatch):
        llm = fast_retries
        monkeypatch.setattr(llm.settings, "llm_provider", "zai")
        monkeypatch.setattr(llm.settings, "zai_api_key", "primary-key")
        monkeypatch.setattr(llm.settings, "opencode_api_key", "fallback-key")
        monkeypatch.setattr(llm.settings, "llm_fallback_provider", "opencode-go")

        def behavior(n, kwargs):
            # Primary (zai) always rate-limits; once swapped, succeed.
            if llm.settings.llm_provider == "zai":
                raise _rate_limit_error()
            return _ok_response("from-fallback")

        comp = _FakeCompletions(behavior)
        monkeypatch.setattr(llm, "get_async_client", lambda: _FakeClient(comp))

        out = await llm.achat("hello")
        assert out == "from-fallback"
        assert llm.settings.llm_provider == "opencode-go"


class TestPlanLacksModel:
    """A 429 that says the plan doesn't include the model never clears by waiting."""

    def test_is_recognised_from_the_error_body(self):
        import synapse_deep_research.llm as llm
        assert llm._plan_lacks_model(_plan_lacks_model_error())
        # the SDK may hand over the whole {"error": {...}} envelope
        envelope = _plan_lacks_model_error()
        envelope.body = {"error": envelope.body}
        assert llm._plan_lacks_model(envelope)
        assert not llm._plan_lacks_model(_rate_limit_error())
        assert not llm._plan_lacks_model(TimeoutError("slow"))

    def test_is_not_retried_but_real_rate_limits_are(self):
        import synapse_deep_research.llm as llm
        assert llm._should_retry(_rate_limit_error())
        assert not llm._should_retry(_plan_lacks_model_error())

    @pytest.mark.asyncio
    async def test_fails_after_one_call_instead_of_burning_the_retry_budget(self, fast_retries, monkeypatch):
        llm = fast_retries

        def behavior(n, kwargs):
            raise _plan_lacks_model_error()

        comp = _FakeCompletions(behavior)
        monkeypatch.setattr(llm, "get_async_client", lambda: _FakeClient(comp))
        monkeypatch.setattr(llm.settings, "llm_fallback_provider", "")

        with pytest.raises(llm.ProviderLimitExceeded) as caught:
            await llm.achat("hello", model="glm-5.3-flashx")
        assert comp.calls == 1
        assert "does not yet include access" in str(caught.value)


class TestRetryLogNamesTheRequestedModel:
    def test_uses_the_model_argument_not_the_provider_default(self, monkeypatch):
        import synapse_deep_research.llm as llm
        monkeypatch.setattr(llm.settings, "glm_model", "glm-5.3")
        positional = SimpleNamespace(args=("prompt", "system", "glm-5.3-flash", 0.2, False), kwargs={})
        by_keyword = SimpleNamespace(args=("prompt", "system"), kwargs={"model": "glm-5.2"})
        unset = SimpleNamespace(args=("prompt", "system", None, 0.2, False), kwargs={})
        assert llm._requested_model(positional) == "glm-5.3-flash"
        assert llm._requested_model(by_keyword) == "glm-5.2"
        assert llm._requested_model(unset) == llm.settings.active_model


# --- bounded search concurrency -------------------------------------------

class TestSearchConcurrency:
    @pytest.mark.asyncio
    async def test_batch_search_respects_semaphore(self):
        from synapse_deep_research.search.base import SearchProvider
        from synapse_deep_research.models import SearchResult, SourceType

        state = {"current": 0, "peak": 0}

        class _Provider(SearchProvider):
            source_type = SourceType.WEB

            async def search(self, query, max_results=10):
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
                await asyncio.sleep(0.01)
                state["current"] -= 1
                return [SearchResult(title=query, url=f"https://x/{query}", snippet="s")]

        sem = asyncio.Semaphore(2)
        results = await _Provider().batch_search([str(i) for i in range(8)], 1, semaphore=sem)
        assert len(results) == 8
        assert state["peak"] <= 2


# --- configurable analyze concurrency -------------------------------------

class TestLLMConcurrency:
    @pytest.mark.asyncio
    async def test_set_llm_concurrency_resizes_semaphore(self):
        from synapse_deep_research.pipeline import analyzer

        analyzer.set_llm_concurrency(2)
        assert analyzer._get_semaphore()._value == 2

        analyzer.set_llm_concurrency(7)
        assert analyzer._get_semaphore()._value == 7


# --- partial-result synthesis path ----------------------------------------

class TestPartialSynthesis:
    @pytest.mark.asyncio
    async def test_synthesis_failure_returns_partial(self, monkeypatch):
        import synapse_deep_research.pipeline.synthesizer as synth

        async def boom(*args, **kwargs):
            raise RuntimeError("provider rate-limited mid-synthesis")

        monkeypatch.setattr(synth, "asynthesize_report", boom)
        from synapse_deep_research.models import SourceAnalysis
        src = SourceAnalysis(url="https://a.com", title="A", key_findings=["f"])
        executive, detailed, follow_ups, ok = await synth.asynthesize_report_safe(
            "q", [], [src],
        )
        assert ok is False
        assert detailed == ""
        assert follow_ups == []
        assert "resynthesize" in executive.lower()

    @pytest.mark.asyncio
    async def test_empty_sources_skips_llm_entirely(self, monkeypatch):
        import synapse_deep_research.pipeline.synthesizer as synth

        async def boom(*args, **kwargs):
            raise AssertionError("LLM must not be called with zero sources")

        monkeypatch.setattr(synth, "asynthesize_report", boom)
        executive, detailed, follow_ups, ok = await synth.asynthesize_report_safe(
            "q", [], [],
        )
        assert ok is False
        assert detailed == "" and follow_ups == []
        assert "no sources" in executive.lower()

    @pytest.mark.asyncio
    async def test_synthesis_success_passes_through(self, monkeypatch):
        import synapse_deep_research.pipeline.synthesizer as synth

        async def fine(*args, **kwargs):
            return ("## Executive Summary\nGood [1].", "## Findings\nx [1].", ["q1"])

        monkeypatch.setattr(synth, "asynthesize_report", fine)
        from synapse_deep_research.models import SourceAnalysis
        src = SourceAnalysis(url="https://a.com", title="A", key_findings=["f"])
        executive, detailed, follow_ups, ok = await synth.asynthesize_report_safe(
            "q", [], [src],
        )
        assert ok is True
        assert executive.startswith("## Executive Summary")
        assert follow_ups == ["q1"]


# --- env-driven config knobs ----------------------------------------------

class TestConfigLimits:
    def test_apply_limits_overrides_shape(self, monkeypatch):
        from synapse_deep_research.models import ResearchConfig, ResearchDepth
        from synapse_deep_research.config import settings

        monkeypatch.setattr(settings, "max_llm_concurrency", 9)
        monkeypatch.setattr(settings, "max_search_concurrency", 4)
        monkeypatch.setattr(settings, "max_sub_questions_override", 2)
        monkeypatch.setattr(settings, "max_iterations_override", 1)
        monkeypatch.setattr(settings, "max_sources_override", 0)  # unset → keep preset

        cfg = ResearchConfig.from_depth(ResearchDepth.DEEP)
        assert cfg.max_llm_concurrency == 9
        assert cfg.max_search_concurrency == 4
        assert cfg.max_sub_questions == 2      # overridden
        assert cfg.max_iterations == 1         # overridden
        assert cfg.max_sources == 30           # preset preserved

    def test_agent_budget_override(self, monkeypatch):
        from synapse_deep_research.config import settings
        from synapse_deep_research.models import ResearchDepth
        from synapse_deep_research.pipeline.agent import agent_iteration_budget, MAX_AGENT_ITERATIONS

        monkeypatch.setattr(settings, "agent_budget_deep", 4)
        assert agent_iteration_budget(ResearchDepth.DEEP) == 4

        # Override above the hard cap is clamped.
        monkeypatch.setattr(settings, "agent_budget_deep", 99)
        assert agent_iteration_budget(ResearchDepth.DEEP) == MAX_AGENT_ITERATIONS


class TestFallbackModelRouting:
    @pytest.mark.opencode_go
    @pytest.mark.asyncio
    async def test_swapped_provider_drops_foreign_stage_models(self, monkeypatch):
        """After a mid-run opencode-go → zai swap, a stage model registered
        only under opencode-go must fall back to the zai default instead of
        being sent to a provider that will 400 it."""
        import synapse_deep_research.llm as llm

        monkeypatch.setattr(llm.settings, "llm_provider", "opencode-go")
        monkeypatch.setattr(llm.settings, "opencode_api_key", "k1")
        monkeypatch.setattr(llm.settings, "zai_api_key", "k2")
        monkeypatch.setattr(llm.settings, "llm_fallback_provider", "zai")
        monkeypatch.setattr(llm.settings, "glm_model", "glm-5")
        assert llm._swap_to_fallback_provider() is True
        assert llm.settings.llm_provider == "zai"

        captured = {}

        class _Comp:
            async def create(self, **kwargs):
                captured["model"] = kwargs["model"]
                return _ok_response()

        client = SimpleNamespace(chat=SimpleNamespace(completions=_Comp()))
        monkeypatch.setattr(llm, "get_async_client", lambda: client)

        assert await llm.achat("hi", model="deepseek-v4-flash") == "ok"
        assert captured["model"] == "glm-5"  # foreign model dropped

        await llm.achat("hi", model="glm-5.1")  # registered on both
        assert captured["model"] == "glm-5.1"

    @pytest.mark.asyncio
    async def test_swap_rejects_unconfigured_fallback_name(self, monkeypatch):
        """A fallback name that isn't a known provider never mutates state."""
        import synapse_deep_research.llm as llm

        monkeypatch.setattr(llm.settings, "llm_provider", "zai")
        monkeypatch.setattr(llm.settings, "llm_fallback_provider", "not-a-provider")
        assert llm._swap_to_fallback_provider() is False
        assert llm.settings.llm_provider == "zai"
