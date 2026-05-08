"""Unit tests for the model capability registry."""
from __future__ import annotations

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research.capabilities import (
    CapabilityViolation,
    describe_route,
    known_models,
    lookup,
    resolve_endpoint_family,
    resolve_thinking,
    validate_route,
)
from deep_research.config import Settings


class TestRegistry:
    def test_glm_is_openai_only(self):
        cap = lookup("opencode-go", "glm-5.1")
        assert cap is not None
        assert cap.endpoint_families == frozenset({"openai"})

    def test_minimax_is_anthropic_only(self):
        cap = lookup("opencode-go", "minimax-m2.7")
        assert cap is not None
        assert cap.endpoint_families == frozenset({"anthropic"})

    def test_lookup_unknown_returns_none(self):
        assert lookup("opencode-go", "totally-fake-model") is None

    def test_known_models_lists_opencode_go(self):
        models = known_models("opencode-go")
        # 12 models per https://opencode.ai/docs/go/
        assert len(models) == 12
        assert "glm-5.1" in models
        assert "kimi-k2.6" in models
        assert "deepseek-v4-pro" in models
        assert "minimax-m2.7" in models


class TestResolveEndpointFamily:
    def test_explicit_match_passes(self):
        assert resolve_endpoint_family("opencode-go", "glm-5.1", "openai") == "openai"
        assert resolve_endpoint_family("opencode-go", "minimax-m2.7", "anthropic") == "anthropic"

    def test_explicit_mismatch_raises_zai(self):
        with pytest.raises(CapabilityViolation):
            resolve_endpoint_family("zai", "glm-5", "anthropic")

    def test_glm_with_anthropic_raises(self):
        # GLM is openai-only on OpenCode Go.
        with pytest.raises(CapabilityViolation):
            resolve_endpoint_family("opencode-go", "glm-5.1", "anthropic")

    def test_minimax_with_openai_raises(self):
        # MiniMax routes through /messages only.
        with pytest.raises(CapabilityViolation):
            resolve_endpoint_family("opencode-go", "minimax-m2.7", "openai")

    def test_default_glm_picks_openai(self):
        assert resolve_endpoint_family("opencode-go", "glm-5.1", None) == "openai"

    def test_default_minimax_picks_anthropic(self):
        # Single-family auto-routing: no openai available for MiniMax.
        assert resolve_endpoint_family("opencode-go", "minimax-m2.7", None) == "anthropic"

    def test_unknown_model_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="deep_research.capabilities"):
            family = resolve_endpoint_family("opencode-go", "future-model", None)
        assert family == "openai"
        assert any("not in capability registry" in r.message for r in caplog.records)


class TestValidateRoute:
    def test_valid_combo_returns_capability(self):
        cap = validate_route("opencode-go", "kimi-k2.6", "openai")
        assert cap is not None
        assert cap.model_id == "kimi-k2.6"

    def test_invalid_family_for_known_model_raises(self):
        with pytest.raises(CapabilityViolation):
            validate_route("zai", "glm-5", "anthropic")

    def test_unknown_family_raises(self):
        with pytest.raises(CapabilityViolation):
            validate_route("opencode-go", "glm-5.1", "telepathy")

    def test_unknown_model_warns_but_returns_none(self, caplog):
        with caplog.at_level(logging.WARNING, logger="deep_research.capabilities"):
            assert validate_route("opencode-go", "future-model", "openai") is None
        assert any("not registered" in r.message for r in caplog.records)


class TestResolveThinking:
    def test_passes_through_when_not_requested(self):
        assert resolve_thinking("opencode-go", "kimi-k2.6", False) is False

    def test_unknown_support_passes_through(self):
        # kimi-k2.6 has thinking_supported=None in the registry
        assert resolve_thinking("opencode-go", "kimi-k2.6", True) is True

    def test_explicit_support_passes_through(self):
        assert resolve_thinking("opencode-go", "glm-5.1", True) is True

    def test_unsupported_drops_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="deep_research.capabilities"):
            result = resolve_thinking("zai", "glm-4.5-air", True)
        assert result is False
        assert any("does not support thinking" in r.message for r in caplog.records)


class TestDescribeRoute:
    def test_known_model_includes_metadata(self):
        info = describe_route("opencode-go", "glm-5.1", "openai")
        assert info["provider"] == "opencode-go"
        assert info["model"] == "glm-5.1"
        assert info["endpoint_family"] == "openai"
        assert info["registered"] is True
        assert info["supported_families"] == ["openai"]
        assert info["thinking_supported"] is True

    def test_unknown_model_marks_unregistered(self):
        info = describe_route("opencode-go", "future-model", "openai")
        assert info["registered"] is False
        assert "supported_families" not in info

    def test_extra_models_included(self):
        info = describe_route(
            "opencode-go", "glm-5.1", "openai",
            extra_models=["kimi-k2.6", "deepseek-v4-pro"],
        )
        assert info["stage_models"] == ["kimi-k2.6", "deepseek-v4-pro"]


class TestSettingsValidateRoutes:
    def test_zai_default_passes(self):
        s = Settings(
            llm_provider="zai", zai_api_key="fake",
            glm_model="glm-5",
        )
        info = s.validate_routes()
        assert info["provider"] == "zai"
        assert info["endpoint_family"] == "openai"

    def test_opencode_go_default_passes(self):
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="fake",
            opencode_model="glm-5.1",
        )
        info = s.validate_routes()
        assert info["provider"] == "opencode-go"
        assert info["endpoint_family"] == "openai"
        assert info["registered"] is True

    def test_opencode_go_minimax_auto_routes_to_anthropic(self):
        # No explicit family — single-family auto-routing should land on anthropic.
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="fake",
            opencode_model="minimax-m2.7",
        )
        info = s.validate_routes()
        assert info["endpoint_family"] == "anthropic"
        assert info["supported_families"] == ["anthropic"]

    def test_opencode_go_glm_with_explicit_anthropic_rejected(self):
        # GLM is openai-only on OpenCode Go; explicit anthropic must hard-fail.
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="fake",
            opencode_model="glm-5.1", llm_endpoint_family="anthropic",
        )
        with pytest.raises(CapabilityViolation):
            s.validate_routes()

    def test_zai_with_anthropic_family_rejected(self):
        s = Settings(
            llm_provider="zai", zai_api_key="fake",
            glm_model="glm-5", llm_endpoint_family="anthropic",
        )
        with pytest.raises((ValueError, CapabilityViolation)):
            s.validate_routes()

    def test_stage_overrides_validated(self):
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="fake",
            opencode_model="glm-5.1",
            model_decompose="kimi-k2.6",
            model_synthesize="deepseek-v4-pro",
        )
        info = s.validate_routes()
        assert info["stage_models"] == ["kimi-k2.6", "deepseek-v4-pro"]

    def test_unknown_stage_model_warns_but_ok(self, caplog):
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="fake",
            opencode_model="glm-5.1",
            model_analyze="future-mystery-model",
        )
        with caplog.at_level(logging.WARNING, logger="deep_research.capabilities"):
            info = s.validate_routes()
        assert "future-mystery-model" in info["stage_models"]
        assert any("future-mystery-model" in r.message for r in caplog.records)
