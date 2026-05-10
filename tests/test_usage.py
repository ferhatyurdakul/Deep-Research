"""
Tests for usage tracking and fallback-provider behavior.

The fallback path is the load-bearing piece for OpenCode Go's $12/5h cap —
when the primary 429s, the run should swap to the configured fallback and
finish, not abort halfway through synthesis.
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research import llm
from deep_research.llm import ProviderLimitExceeded
from deep_research.usage import UsageTracker, get_tracker, new_run_tracker, stage


class TestUsageTracker:
    def test_record_call_increments(self):
        t = UsageTracker()
        t.record_call("opencode-go", "glm-5.1", "decompose")
        t.record_call("opencode-go", "glm-5.1", "analyze")
        t.record_call("opencode-go", "glm-5.1", "analyze")
        snap = t.snapshot()
        assert snap["total_calls"] == 3
        assert snap["by_stage"] == {"analyze": 2, "decompose": 1}
        assert snap["by_model"] == {"opencode-go/glm-5.1": 3}

    def test_fallback_recorded(self):
        t = UsageTracker()
        t.record_fallback("opencode-go", "zai")
        snap = t.snapshot()
        assert snap["fallback_fired"] is True
        assert snap["fallback_from"] == "opencode-go"
        assert snap["fallback_to"] == "zai"

    def test_format_summary_includes_fallback_line(self):
        t = UsageTracker()
        t.record_call("opencode-go", "glm-5.1", "decompose")
        t.record_fallback("opencode-go", "zai")
        text = t.format_summary()
        assert "1 total" in text
        assert "decompose=1" in text
        assert "opencode-go → zai" in text

    def test_concurrent_record_calls_consistent(self):
        # Mutations from many tasks must all land — the analyze stage runs
        # parallel calls via asyncio.
        t = UsageTracker()

        async def hammer():
            for _ in range(50):
                t.record_call("opencode-go", "glm-5.1", "analyze")

        async def main():
            await asyncio.gather(*(hammer() for _ in range(10)))

        asyncio.run(main())
        assert t.snapshot()["total_calls"] == 500


class TestStageContextTagsRecording:
    def test_stage_async_context_visible_to_tracker(self):
        t = new_run_tracker()

        async def fake_call():
            # Simulate llm._chat_inner recording a call: it reads current_stage()
            from deep_research.usage import current_stage
            t.record_call("opencode-go", "glm-5.1", current_stage())

        async def main():
            async with stage("decompose"):
                await fake_call()
            async with stage("synthesize"):
                await fake_call()

        asyncio.run(main())
        assert t.snapshot()["by_stage"] == {"decompose": 1, "synthesize": 1}


class _FakeRateLimit(Exception):
    """Stand-in for openai.RateLimitError, which requires a response object."""


class TestFallbackOnRateLimit:
    def _patch_rate_limit_types(self):
        return patch.object(llm, "_RATE_LIMIT_TYPES", (_FakeRateLimit,))

    def test_swap_records_fallback_and_succeeds(self):
        # Primary raises rate-limit. After swap, secondary returns a normal
        # response. Tracker should record the fallback event AND the
        # successful call (under the new provider).
        new_run_tracker()  # fresh per-test tracker

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = _FakeRateLimit("429")

        secondary_client = MagicMock()
        ok = MagicMock()
        ok.choices = [MagicMock(message=MagicMock(content="from-fallback"))]
        secondary_client.chat.completions.create.return_value = ok

        clients = [primary_client, secondary_client]
        with self._patch_rate_limit_types(), \
             patch.object(llm.settings, "llm_provider", "opencode-go"), \
             patch.object(llm.settings, "opencode_model", "glm-5.1"), \
             patch.object(llm.settings, "opencode_api_key", "primary-key"), \
             patch.object(llm.settings, "zai_api_key", "fallback-key"), \
             patch.object(llm.settings, "glm_model", "glm-5"), \
             patch.object(llm.settings, "llm_fallback_provider", "zai"), \
             patch.object(llm.settings, "llm_endpoint_family", ""), \
             patch.object(llm, "get_client", side_effect=lambda: clients.pop(0)):
            result = llm.chat("hi")

        assert result == "from-fallback"
        snap = get_tracker().snapshot()
        assert snap["fallback_fired"] is True
        assert snap["fallback_from"] == "opencode-go"
        assert snap["fallback_to"] == "zai"
        # Call against the fallback was recorded; primary's 429 was not.
        assert "zai/glm-5" in snap["by_model"]

    def test_no_fallback_raises_provider_limit_exceeded(self):
        new_run_tracker()
        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = _FakeRateLimit("429")

        with self._patch_rate_limit_types(), \
             patch.object(llm.settings, "llm_provider", "opencode-go"), \
             patch.object(llm.settings, "opencode_model", "glm-5.1"), \
             patch.object(llm.settings, "opencode_api_key", "primary-key"), \
             patch.object(llm.settings, "llm_fallback_provider", ""), \
             patch.object(llm.settings, "llm_endpoint_family", ""), \
             patch.object(llm, "get_client", return_value=primary_client):
            with pytest.raises(ProviderLimitExceeded) as ei:
                llm.chat("hi")

        err = ei.value
        assert err.provider == "opencode-go"
        assert err.model == "glm-5.1"
        assert err.fallback_attempted is False
        assert "subscription cap" in str(err)

    def test_fallback_without_credentials_does_not_swap(self):
        # If LLM_FALLBACK_PROVIDER is set but the fallback has no API key,
        # we should NOT silently swap and pretend things are fine — the
        # original error must surface as ProviderLimitExceeded.
        new_run_tracker()
        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = _FakeRateLimit("429")

        with self._patch_rate_limit_types(), \
             patch.object(llm.settings, "llm_provider", "opencode-go"), \
             patch.object(llm.settings, "opencode_model", "glm-5.1"), \
             patch.object(llm.settings, "opencode_api_key", "primary-key"), \
             patch.object(llm.settings, "zai_api_key", ""), \
             patch.object(llm.settings, "llm_fallback_provider", "zai"), \
             patch.object(llm.settings, "llm_endpoint_family", ""), \
             patch.object(llm, "get_client", return_value=primary_client):
            with pytest.raises(ProviderLimitExceeded):
                llm.chat("hi")

        snap = get_tracker().snapshot()
        assert snap["fallback_fired"] is False
