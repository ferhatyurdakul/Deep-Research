"""
Tests for the request-shape compatibility layer in llm.py.

Covers:
- _apply_thinking() injects the right dialect for each (provider, model)
- "none" dialect omits the thinking field entirely (graceful degrade)
- Dialect/family mismatch is logged and omitted, not shipped
- UpstreamRequestError wraps a BadRequestError with route context
"""
from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research import llm
from deep_research.capabilities import thinking_dialect
from deep_research.llm import (
    UpstreamRequestError,
    _apply_thinking,
    _build_anthropic_kwargs,
    _build_openai_kwargs,
    _Route,
)


class TestBuildKwargsAreGeneric:
    """Generic builders should NEVER inject provider-specific extras."""

    def test_openai_kwargs_has_no_extra_body(self):
        kwargs = _build_openai_kwargs("p", "s", "any-model", 0.5)
        assert "extra_body" not in kwargs
        assert "thinking" not in kwargs
        assert kwargs["model"] == "any-model"
        assert kwargs["temperature"] == 0.5

    def test_anthropic_kwargs_has_no_thinking(self):
        kwargs = _build_anthropic_kwargs("p", "s", "any-model", 0.5)
        assert "thinking" not in kwargs
        assert "extra_body" not in kwargs
        assert kwargs["model"] == "any-model"
        assert kwargs["max_tokens"] > 0


class TestApplyThinking:
    def _route(self, provider, model, family, thinking=True):
        return _Route(provider, model, family, thinking, thinking_dialect(provider, model))

    def test_zai_glm_gets_extra_body(self):
        route = self._route("zai", "glm-5.1", "openai")
        kwargs = {"model": "glm-5.1"}
        _apply_thinking(kwargs, route)
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled", "clear_thinking": True}}

    def test_opencode_go_glm_gets_extra_body(self):
        # GLM via OpenCode Go proxies to Z.AI upstream; same dialect applies.
        route = self._route("opencode-go", "glm-5.1", "openai")
        kwargs = {"model": "glm-5.1"}
        _apply_thinking(kwargs, route)
        assert "extra_body" in kwargs
        assert kwargs["extra_body"]["thinking"]["clear_thinking"] is True

    def test_opencode_go_kimi_omits_thinking(self, caplog):
        # Reasoning-native model: don't ship Z.AI-shaped fields.
        route = self._route("opencode-go", "kimi-k2.6", "openai")
        kwargs = {"model": "kimi-k2.6"}
        with caplog.at_level(logging.INFO, logger="deep_research.llm"):
            _apply_thinking(kwargs, route)
        assert "extra_body" not in kwargs
        assert "thinking" not in kwargs
        assert any("native reasoning" in r.message for r in caplog.records)

    def test_opencode_go_deepseek_omits_thinking(self):
        route = self._route("opencode-go", "deepseek-v4-pro", "openai")
        kwargs = {}
        _apply_thinking(kwargs, route)
        assert kwargs == {}

    def test_opencode_go_minimax_gets_anthropic_thinking(self):
        route = self._route("opencode-go", "minimax-m2.7", "anthropic")
        kwargs = {"model": "minimax-m2.7"}
        _apply_thinking(kwargs, route)
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_thinking_disabled_no_op(self):
        route = self._route("zai", "glm-5.1", "openai", thinking=False)
        kwargs = {}
        _apply_thinking(kwargs, route)
        assert kwargs == {}

    def test_unknown_model_omits_thinking(self):
        route = self._route("opencode-go", "future-model", "openai")
        kwargs = {}
        _apply_thinking(kwargs, route)
        # thinking_dialect("opencode-go", "future-model") -> "none"
        assert kwargs == {}

    def test_dialect_family_mismatch_omits_with_warning(self, caplog):
        # Hand-crafted invalid combo: zai-extra-body shape on anthropic family.
        route = _Route("opencode-go", "minimax-m2.7", "anthropic", True, "zai-extra-body")
        kwargs = {}
        with caplog.at_level(logging.WARNING, logger="deep_research.llm"):
            _apply_thinking(kwargs, route)
        assert kwargs == {}
        assert any("not compatible" in r.message for r in caplog.records)


class TestUpstreamRequestErrorWrapping:
    """A 400 from the upstream should surface with the routing context."""

    def test_wrap_includes_provider_model_family_dialect(self):
        # Build a fake BadRequestError-shaped exception. The real openai
        # BadRequestError needs a response object, so we substitute the
        # _BAD_REQUEST_TYPES tuple with our stub for the duration of the test.
        class _FakeBadRequest(Exception):
            pass

        # active_provider/active_model are Pydantic properties; patch the
        # underlying fields that drive them instead.
        with patch.object(llm, "_BAD_REQUEST_TYPES", (_FakeBadRequest,)), \
             patch.object(llm.settings, "llm_provider", "opencode-go"), \
             patch.object(llm.settings, "opencode_model", "kimi-k2.6"), \
             patch.object(llm.settings, "opencode_api_key", "fake"), \
             patch.object(llm.settings, "llm_endpoint_family", ""):
            fake_client = MagicMock()
            fake_client.chat.completions.create.side_effect = _FakeBadRequest(
                "Unknown field 'thinking.clear_thinking'"
            )
            with patch.object(llm, "get_client", return_value=fake_client):
                with pytest.raises(UpstreamRequestError) as ei:
                    llm.chat("hi", thinking=True)
        err = ei.value
        assert err.provider == "opencode-go"
        assert err.model == "kimi-k2.6"
        assert err.endpoint_family == "openai"
        # Kimi has dialect "none" so we weren't shipping the GLM extra_body —
        # the wrap makes that visible without crawling logs.
        assert err.dialect == "none"
        assert "kimi-k2.6" in str(err)
        assert "family=openai" in str(err)


class TestBackwardCompatZaiDefaultPath:
    """The default Z.AI path must still produce extra_body when thinking=True."""

    def test_default_zai_glm5_keeps_extra_body(self):
        route = _Route("zai", "glm-5", "openai", True, thinking_dialect("zai", "glm-5"))
        kwargs = _build_openai_kwargs("p", "s", "glm-5", 0.7)
        _apply_thinking(kwargs, route)
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled", "clear_thinking": True}}
