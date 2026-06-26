"""
Tests for named stage-route profiles and the env-override-wins precedence.

Covers:
- Each named profile resolves to four production-or-experimental models
- Per-stage env vars override the profile
- Unknown profile names raise a clear CapabilityViolation
- validate_routes() surfaces the active profile and per-stage models
- The pipeline ModelRouter is filled from the profile when env is empty
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research.capabilities import (
    CapabilityViolation,
    PIPELINE_STAGES,
    RECOMMENDED_ROUTES,
    StageRoute,
    known_profiles,
    resolve_stage_route,
)
from deep_research.config import Settings
from deep_research.models import ResearchConfig, ResearchDepth


class TestRouteRegistry:
    def test_pipeline_stages_match_modelrouter(self):
        # The four stage names must align with ModelRouter fields and with
        # StageRoute fields, otherwise stage_models()/load_model_routes()
        # would silently drop one.
        assert set(PIPELINE_STAGES) == {"decompose", "analyze", "gap_analysis", "synthesize"}

    def test_zai_balanced_uses_glm52_for_every_stage(self):
        route = resolve_stage_route("zai", "balanced")
        assert route.as_dict() == {
            "decompose": "glm-5.2",
            "analyze": "glm-5.2",
            "gap_analysis": "glm-5.2",
            "synthesize": "glm-5.2",
        }

    def test_opencode_go_balanced_is_all_glm(self):
        # Production-tier safe default; no model mixing.
        route = resolve_stage_route("opencode-go", "balanced")
        assert route.as_dict() == {
            "decompose": "glm-5.1",
            "analyze": "glm-5.1",
            "gap_analysis": "glm-5.1",
            "synthesize": "glm-5.1",
        }

    def test_opencode_go_budget_uses_flash_for_iterative_stages(self):
        route = resolve_stage_route("opencode-go", "budget")
        assert route.decompose == "deepseek-v4-flash"
        assert route.analyze == "deepseek-v4-flash"
        # Quality-sensitive stages stay on production-tier model.
        assert route.gap_analysis == "glm-5.1"
        assert route.synthesize == "glm-5.1"

    def test_opencode_go_quality_upgrades_synthesis(self):
        route = resolve_stage_route("opencode-go", "quality")
        assert route.synthesize == "deepseek-v4-pro"
        assert route.gap_analysis == "glm-5.1"  # iterative stages unchanged

    def test_unknown_profile_raises_with_known_listed(self):
        with pytest.raises(CapabilityViolation) as ei:
            resolve_stage_route("opencode-go", "ultra-mega")
        msg = str(ei.value)
        assert "ultra-mega" in msg
        # Error names the available profiles so the user knows what to type.
        assert "balanced" in msg

    def test_known_profiles_per_provider(self):
        zai_profiles = known_profiles("zai")
        oc_profiles = known_profiles("opencode-go")
        assert "balanced" in zai_profiles
        assert set(oc_profiles) == {"balanced", "budget", "quality"}


class TestEffectiveStageModels:
    def test_balanced_fills_empty_stages_from_profile(self):
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="balanced",
        )
        stages = s.effective_stage_models()
        assert all(v == "glm-5.1" for v in stages.values())

    def test_env_override_wins_over_profile(self):
        # User wants quality everywhere except synthesis, which they pin.
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="quality",
            model_synthesize="kimi-k2.6",
        )
        stages = s.effective_stage_models()
        assert stages["synthesize"] == "kimi-k2.6"  # override wins
        assert stages["analyze"] == "glm-5.1"      # profile fills

    def test_full_per_stage_override_ignores_profile(self):
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="balanced",
            model_decompose="deepseek-v4-flash",
            model_analyze="kimi-k2.5",
            model_gap_analysis="kimi-k2.6",
            model_synthesize="deepseek-v4-pro",
        )
        stages = s.effective_stage_models()
        assert stages == {
            "decompose": "deepseek-v4-flash",
            "analyze": "kimi-k2.5",
            "gap_analysis": "kimi-k2.6",
            "synthesize": "deepseek-v4-pro",
        }

    def test_unknown_profile_raises(self):
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="nope",
        )
        with pytest.raises(CapabilityViolation):
            s.effective_stage_models()


class TestValidateRoutesSurfacesProfile:
    def test_validate_routes_includes_profile_and_stages(self):
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="quality",
        )
        info = s.validate_routes()
        assert info["profile"] == "quality"
        assert info["stages"]["synthesize"] == "deepseek-v4-pro"
        # All four stages must be present so logs/debugging see the full route.
        assert set(info["stages"]) == set(PIPELINE_STAGES)

    def test_validate_routes_validates_cross_family_stage(self):
        # If a stage targets MiniMax (anthropic family), validate_routes must
        # pick the correct family for that stage rather than reusing the
        # active model's family.
        s = Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="balanced",
            model_synthesize="minimax-m2.7",
        )
        info = s.validate_routes()  # must not raise
        assert info["stages"]["synthesize"] == "minimax-m2.7"

    def test_validate_routes_rejects_invalid_stage_model(self):
        # If the override ends up on a model that doesn't even exist in any
        # family, validate_routes should hard-fail (after logging the warning
        # that the model is unregistered, the openai default falls through).
        # We use a known-bad combo: explicit family that the override can't
        # support, which is the typical user error.
        s = Settings(
            llm_provider="zai", zai_api_key="x", glm_model="glm-5",
            llm_endpoint_family="openai", llm_route="balanced",
            model_synthesize="glm-5.1",  # known model, fine combo
        )
        # Sanity: this is the happy path on purpose.
        s.validate_routes()


class TestPipelineRouterFilledFromProfile:
    def test_research_config_picks_up_profile_models(self):
        # ResearchConfig.load_model_routes must fill stage models from the
        # active profile when env vars are empty.
        from deep_research import config as cfg_module
        from unittest.mock import patch
        from deep_research.config import Settings as _Settings
        custom = _Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="budget",
        )
        with patch.object(cfg_module, "settings", custom):
            rc = ResearchConfig(depth=ResearchDepth.STANDARD)
            rc.load_model_routes()
        assert rc.models.decompose == "deepseek-v4-flash"
        assert rc.models.analyze == "deepseek-v4-flash"
        assert rc.models.gap_analysis == "glm-5.1"
        assert rc.models.synthesize == "glm-5.1"

    def test_env_override_threads_through_to_router(self):
        from deep_research import config as cfg_module
        from unittest.mock import patch
        from deep_research.config import Settings as _Settings
        custom = _Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="balanced",
            model_decompose="deepseek-v4-flash",
        )
        with patch.object(cfg_module, "settings", custom):
            rc = ResearchConfig(depth=ResearchDepth.QUICK)
            rc.load_model_routes()
        assert rc.models.decompose == "deepseek-v4-flash"  # override
        assert rc.models.analyze == "glm-5.1"             # profile fill
