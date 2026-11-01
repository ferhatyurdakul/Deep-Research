"""
Tests for per-stage model selection.

Each pipeline stage (decompose, analyze, gap_analysis, synthesize) has a model
the user chooses — in setup, with /model, or via MODEL_<STAGE> — with the
provider's recommendation as the fallback. Covers:
- the Z.AI model registry and the recommended models
- the saved-choices file (userconfig)
- the precedence: environment > saved > recommended
- validate_routes() and the pipeline's ModelRouter picking all of it up
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synapse_deep_research import capabilities, userconfig
from synapse_deep_research.capabilities import (
    PIPELINE_STAGES,
    CapabilityViolation,
    listed_models,
    lookup,
    recommended_stage_models,
)
from synapse_deep_research.config import Settings
from synapse_deep_research.models import ResearchConfig, ResearchDepth


def _zai(**kw) -> Settings:
    kw.setdefault("zai_api_key", "x")
    return Settings(llm_provider="zai", **kw)


def _opencode(**kw) -> Settings:
    kw.setdefault("opencode_api_key", "x")
    kw.setdefault("opencode_model", "glm-5.1")
    return Settings(llm_provider="opencode-go", **kw)


class TestRegistry:
    def test_pipeline_stages_match_modelrouter(self):
        assert set(PIPELINE_STAGES) == {"decompose", "analyze", "gap_analysis", "synthesize"}

    def test_routes_are_gone(self):
        """Route profiles were replaced by a model per stage."""
        for name in ("RECOMMENDED_ROUTES", "known_profiles", "resolve_stage_route", "DEFAULT_PROFILE", "StageRoute"):
            assert not hasattr(capabilities, name), name
        assert not hasattr(Settings(llm_provider="zai", zai_api_key="x"), "llm_route")

    def test_zai_models_are_the_current_lineup_in_flagship_first_order(self):
        assert listed_models("zai") == ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx", "glm-5.2", "glm-5.1"]

    def test_legacy_zai_models_stay_registered_but_unlisted(self):
        for legacy in ("glm-5", "glm-4.5-air"):
            assert lookup("zai", legacy) is not None, legacy
            assert legacy not in listed_models("zai"), legacy

    def test_zai_models_use_the_openai_endpoint_and_the_zai_thinking_payload(self):
        for model in listed_models("zai"):
            cap = lookup("zai", model)
            assert cap.endpoint_families == frozenset({"openai"}), model
            assert cap.thinking_supported is True and cap.thinking_dialect == "zai-extra-body", model
        assert lookup("zai", "glm-4.5-air").thinking_supported is False

    def test_glm_5_3_is_production_tier(self):
        cap = lookup("zai", "glm-5.3")
        assert cap.tier == "production"

    def test_opencode_go_listing_is_untouched(self):
        assert "glm-5.1" in listed_models("opencode-go") and "kimi-k2.6" in listed_models("opencode-go")


class TestRecommendedModels:
    def test_zai_uses_the_cheap_model_for_the_high_volume_stages(self):
        rec = recommended_stage_models("zai").as_dict()
        assert rec == {
            "decompose": "glm-5.3-flash",
            "analyze": "glm-5.3-flash",
            "gap_analysis": "glm-5.3",
            "synthesize": "glm-5.3",
        }

    def test_opencode_go_keeps_one_production_model_everywhere(self):
        assert set(recommended_stage_models("opencode-go").as_dict().values()) == {"glm-5.1"}

    @pytest.mark.parametrize("provider", ["zai", "opencode-go"])
    def test_every_recommendation_is_a_registered_listed_model(self, provider):
        for stage, model in recommended_stage_models(provider).as_dict().items():
            assert lookup(provider, model) is not None, (provider, stage, model)
            assert model in listed_models(provider), (provider, stage, model)

    def test_unknown_provider_raises(self):
        with pytest.raises(CapabilityViolation, match="No recommended models"):
            recommended_stage_models("nope")


class TestUserConfig:
    def test_round_trip_and_per_provider_separation(self):
        userconfig.save_choices("zai", {"analyze": "glm-5.2"})
        userconfig.save_choices("opencode-go", {"analyze": "kimi-k2.6"})
        assert userconfig.stage_models("zai") == {"analyze": "glm-5.2"}
        assert userconfig.stage_models("opencode-go") == {"analyze": "kimi-k2.6"}
        assert userconfig.provider() == "opencode-go"  # the last one saved

    def test_nothing_saved(self):
        assert not userconfig.exists()
        assert userconfig.load() == {}
        assert userconfig.provider() is None and userconfig.stage_models("zai") == {}

    @pytest.mark.parametrize("garbage", ["{not json", "[1, 2, 3]", '"a string"', ""])
    def test_damaged_files_read_as_nothing_saved(self, garbage):
        path = userconfig.config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(garbage)
        assert userconfig.load() == {} and userconfig.stage_models("zai") == {}

    def test_invalid_entries_are_ignored(self):
        path = userconfig.config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"models": {"zai": {
            "analyze": "glm-5.2", "bogus_stage": "x", "synthesize": "", "decompose": 5,
        }}}))
        assert userconfig.stage_models("zai") == {"analyze": "glm-5.2"}

    def test_set_and_clear_one_stage(self):
        userconfig.set_stage_model("zai", "analyze", "glm-5.2")
        userconfig.set_stage_model("zai", "synthesize", "glm-5.3")
        userconfig.set_stage_model("zai", "analyze", None)
        assert userconfig.stage_models("zai") == {"synthesize": "glm-5.3"}

    def test_set_provider_keeps_every_providers_models(self):
        userconfig.save_choices("zai", {"analyze": "glm-5.2"})
        userconfig.set_provider("opencode-go")
        assert userconfig.provider() == "opencode-go"
        assert userconfig.stage_models("zai") == {"analyze": "glm-5.2"}

    def test_saves_are_atomic_and_leave_no_temp_files(self):
        userconfig.save_choices("zai", {"analyze": "glm-5.2"})
        siblings = list(userconfig.config_file().parent.iterdir())
        assert [p.name for p in siblings] == ["config.json"]
        assert json.loads(userconfig.config_file().read_text())["version"] == userconfig.VERSION

    def test_path_follows_xdg_unless_overridden(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SYNAPSE_DEEP_RESEARCH_CONFIG_FILE")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert userconfig.config_file() == tmp_path / "xdg" / "synapse-deep-research" / "config.json"

    def test_saved_provider_becomes_the_default_provider(self, monkeypatch):
        """Environment wins, then the provider chosen in setup, then Z.AI."""
        import synapse_deep_research.config as cfg

        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        userconfig.save_choices("opencode-go", {})
        try:
            assert importlib.reload(cfg).Settings().llm_provider == "opencode-go"
            monkeypatch.setenv("LLM_PROVIDER", "zai")
            assert importlib.reload(cfg).Settings().llm_provider == "zai"
        finally:
            monkeypatch.delenv("LLM_PROVIDER", raising=False)
            importlib.reload(cfg)


class TestEffectiveStageModels:
    def test_nothing_chosen_means_the_recommendation(self):
        s = _zai()
        assert s.effective_stage_models() == recommended_stage_models("zai").as_dict()
        assert set(s.stage_sources().values()) == {"recommended"}

    def test_saved_choices_beat_the_recommendation(self):
        userconfig.save_choices("zai", {"analyze": "glm-5.2", "synthesize": "glm-5.1"})
        s = _zai()
        stages = s.effective_stage_models()
        assert stages["analyze"] == "glm-5.2" and stages["synthesize"] == "glm-5.1"
        assert stages["decompose"] == "glm-5.3-flash"  # untouched stages keep the recommendation
        assert s.stage_sources() == {
            "decompose": "recommended", "analyze": "saved",
            "gap_analysis": "recommended", "synthesize": "saved",
        }

    def test_environment_beats_saved_and_recommended(self):
        userconfig.save_choices("zai", {"analyze": "glm-5.2"})
        s = _zai(model_analyze="glm-5.3", model_decompose="glm-5.3")
        stages = s.effective_stage_models()
        assert stages["analyze"] == "glm-5.3" and stages["decompose"] == "glm-5.3"
        assert s.stage_sources()["analyze"] == "environment"
        assert s.stage_sources()["decompose"] == "environment"

    def test_a_full_set_of_environment_overrides_ignores_everything_else(self):
        userconfig.save_choices("opencode-go", {"analyze": "glm-5"})
        s = _opencode(
            model_decompose="deepseek-v4-flash", model_analyze="kimi-k2.5",
            model_gap_analysis="kimi-k2.6", model_synthesize="deepseek-v4-pro",
        )
        assert s.effective_stage_models() == {
            "decompose": "deepseek-v4-flash", "analyze": "kimi-k2.5",
            "gap_analysis": "kimi-k2.6", "synthesize": "deepseek-v4-pro",
        }

    def test_each_provider_has_its_own_saved_models(self):
        userconfig.save_choices("zai", {"analyze": "glm-5.2"})
        assert _opencode().effective_stage_models()["analyze"] == "glm-5.1"  # not zai's choice
        assert _zai().effective_stage_models()["analyze"] == "glm-5.2"

    def test_unknown_provider_raises(self):
        with pytest.raises(CapabilityViolation):
            Settings(llm_provider="typo").effective_stage_models()


class TestValidateRoutes:
    def test_includes_all_four_stages(self):
        info = _zai().validate_routes()
        assert set(info["stages"]) == set(PIPELINE_STAGES)
        assert "profile" not in info

    def test_validates_saved_choices_too(self):
        userconfig.save_choices("zai", {"synthesize": "kimi-k2.6"})  # registered only for opencode-go
        # An unregistered-for-zai model is tolerated at this layer (warned, not fatal)...
        assert _zai().validate_routes()["stages"]["synthesize"] == "kimi-k2.6"

    def test_validates_cross_family_stage(self):
        # A stage on MiniMax (anthropic family) must be validated with its own family.
        s = _opencode(model_synthesize="minimax-m2.7")
        assert s.validate_routes()["stages"]["synthesize"] == "minimax-m2.7"

    def test_rejects_unknown_fallback_provider(self):
        s = _zai(llm_fallback_provider="nope")
        with pytest.raises(CapabilityViolation, match="LLM_FALLBACK_PROVIDER"):
            s.validate_routes()


class TestPipelineRouter:
    def _router(self, settings_obj, depth=ResearchDepth.STANDARD):
        from synapse_deep_research import config as cfg_module

        with patch.object(cfg_module, "settings", settings_obj):
            rc = ResearchConfig(depth=depth)
            rc.load_model_routes()
        return rc.models

    def test_recommended_models_reach_the_pipeline(self):
        models = self._router(_zai())
        assert (models.decompose, models.analyze) == ("glm-5.3-flash", "glm-5.3-flash")
        assert (models.gap_analysis, models.synthesize) == ("glm-5.3", "glm-5.3")

    def test_saved_and_environment_choices_reach_the_pipeline(self):
        userconfig.save_choices("zai", {"analyze": "glm-5.2"})
        models = self._router(_zai(model_decompose="glm-5.3"))
        assert models.decompose == "glm-5.3"    # environment
        assert models.analyze == "glm-5.2"      # saved
        assert models.synthesize == "glm-5.3"   # recommended


# --- live account model list -------------------------------------------------

from synapse_deep_research import options as _options  # noqa: E402
from synapse_deep_research.options import fetch_account_models as real_fetch_account_models  # noqa: E402  (conftest stubs the module attribute)


class TestAccountModels:
    """options.fetch_account_models asks the provider's `GET /models`; every
    failure means None (callers fall back to the built-in list)."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        _options._ACCOUNT_CACHE.clear()
        yield
        _options._ACCOUNT_CACHE.clear()

    @pytest.fixture
    def http(self, monkeypatch):
        import httpx

        calls = []
        state = {"reply": httpx.Response(200, json={"data": [{"id": "glm-5.3"}, {"id": "glm-5.3-flash"}]},
                                         request=httpx.Request("GET", "https://x/models")),
                 "error": None}

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers, timeout))
            if state["error"]:
                raise state["error"]
            return state["reply"]

        monkeypatch.setattr(httpx, "get", fake_get)
        return type("Http", (), {"calls": calls, "state": state})

    def _reply(self, status=200, **kw):
        import httpx
        return httpx.Response(status, request=httpx.Request("GET", "https://x/models"), **kw)

    def test_returns_the_ids_and_authenticates(self, http):
        ids = real_fetch_account_models("zai", api_key="sk-test")
        assert ids == ["glm-5.3", "glm-5.3-flash"]
        url, headers, timeout = http.calls[0]
        assert url.endswith("/models") and "coding/paas/v4" in url
        assert headers == {"Authorization": "Bearer sk-test"}
        assert timeout and timeout <= 10                      # never hangs setup

    def test_uses_the_configured_key_when_none_is_passed(self, http, monkeypatch):
        from synapse_deep_research.config import settings
        monkeypatch.setattr(settings, "zai_api_key", "sk-from-settings")
        real_fetch_account_models("zai")
        assert http.calls[0][1] == {"Authorization": "Bearer sk-from-settings"}

    def test_results_are_cached_per_key(self, http):
        real_fetch_account_models("zai", api_key="sk-aaaaaa")
        real_fetch_account_models("zai", api_key="sk-aaaaaa")
        assert len(http.calls) == 1
        real_fetch_account_models("zai", api_key="sk-bbbbbb")   # a different account asks again
        assert len(http.calls) == 2

    @pytest.mark.parametrize("failure", [
        "http_401", "http_500", "network", "not_json", "no_data_key", "empty_data", "wrong_shape",
    ])
    def test_every_failure_is_none_never_an_exception(self, http, failure):
        import httpx

        if failure == "network":
            http.state["error"] = httpx.ConnectError("offline")
        elif failure.startswith("http_"):
            http.state["reply"] = self._reply(int(failure[5:]), json={"error": "x"})
        elif failure == "not_json":
            http.state["reply"] = self._reply(200, text="<html>nope</html>")
        elif failure == "no_data_key":
            http.state["reply"] = self._reply(200, json={"models": []})
        elif failure == "empty_data":
            http.state["reply"] = self._reply(200, json={"data": []})
        else:
            http.state["reply"] = self._reply(200, json={"data": ["glm-5.3", 5, {"name": "x"}]})
        assert real_fetch_account_models("zai", api_key="sk-k") is None

    def test_failures_are_not_cached(self, http):
        import httpx

        http.state["error"] = httpx.ConnectError("offline")
        assert real_fetch_account_models("zai", api_key="sk-k") is None
        http.state["error"] = None
        assert real_fetch_account_models("zai", api_key="sk-k") == ["glm-5.3", "glm-5.3-flash"]

    def test_no_key_or_unknown_provider_makes_no_request(self, http, monkeypatch):
        from synapse_deep_research.config import settings

        monkeypatch.setattr(settings, "zai_api_key", "")
        assert real_fetch_account_models("zai") is None
        assert real_fetch_account_models("nope", api_key="sk-k") is None
        assert http.calls == []


class TestModelChoicesWithAccount:
    ACCOUNT = ["glm-4.5", "glm-4.5-air", "glm-4.7", "glm-5", "glm-5-turbo", "glm-5.1", "glm-5.2",
               "glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"]

    def test_unverified_models_on_the_account_are_offered_after_ours(self):
        rows = dict(_options.model_choices("zai", account=self.ACCOUNT))
        assert list(rows)[:5] == ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx", "glm-5.2", "glm-5.1"]
        for extra in ("glm-4.5", "glm-4.7", "glm-5-turbo"):
            assert rows[extra] == "on your account · not verified here"
        # Registered legacy models are known, so they aren't repeated as "unverified".
        assert "glm-5" not in rows and "glm-4.5-air" not in rows

    def test_registry_models_the_account_lacks_are_tagged(self):
        rows = dict(_options.model_choices("zai", account=["glm-5.3", "glm-5.3-flash"]))
        assert rows["glm-5.3"].startswith("production")
        assert rows["glm-5.2"].startswith("not on your account")
        assert rows["glm-5.1"].startswith("not on your account")

    def test_non_text_models_are_not_offered(self):
        rows = _options.model_choices("zai", account=["glm-5.3", "glm-5v-turbo", "glm-ocr", "embedding-3",
                                                      "cogview-4", "glm-4.6v", "glm-image", "glm-asr"])
        assert [m for m, _ in rows if m not in listed_models("zai")] == []

    def test_without_an_account_list_nothing_is_tagged(self):
        rows = _options.model_choices("zai", account=None)
        assert all("account" not in blurb for _, blurb in rows)


class TestAccountModelsInThePickers:
    def test_setup_offers_the_account_models_in_the_per_stage_picker(self, monkeypatch):
        import synapse_deep_research.onboarding as onboarding

        seen = []
        monkeypatch.setattr(_options, "fetch_account_models", lambda provider, api_key=None, **k: ["glm-5.3", "glm-4.7"])
        monkeypatch.setattr(onboarding, "pick_option", lambda title, choices, current=None, **k: seen.append(choices) or "glm-5.3")
        onboarding._pick_model("zai", "analyze", "glm-5.3-flash", "Setup", ["glm-5.3", "glm-4.7"])
        values = [v for v, _ in seen[0]]
        assert "glm-4.7" in values and values[-1] == "other"
        assert dict(seen[0])["glm-5.2"].startswith("not on your account")

    def test_model_command_offers_them_too(self, monkeypatch):
        import synapse_deep_research.repl as repl_mod
        from synapse_deep_research.config import Settings

        monkeypatch.setattr(repl_mod, "settings", Settings(llm_provider="zai", zai_api_key="x"))
        monkeypatch.setattr(_options, "fetch_account_models", lambda *a, **k: ["glm-5.3", "glm-4.7"])
        seen = []
        answers = iter(["analyze", "glm-4.7"])
        monkeypatch.setattr(repl_mod, "pick_option", lambda t, c, cur=None, **k: seen.append(c) or next(answers))
        assert repl_mod._pick_for_command("/model", repl_mod.ReplState()) == "/model analyze glm-4.7"
        assert "glm-4.7" in [v for v, _ in seen[1]]

    def test_a_failed_lookup_falls_back_to_the_built_in_list(self, monkeypatch):
        import synapse_deep_research.repl as repl_mod
        from synapse_deep_research.config import Settings

        monkeypatch.setattr(repl_mod, "settings", Settings(llm_provider="zai", zai_api_key="x"))

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(_options, "fetch_account_models", boom)
        seen = []
        answers = iter(["analyze", "glm-5.2"])
        monkeypatch.setattr(repl_mod, "pick_option", lambda t, c, cur=None, **k: seen.append(c) or next(answers))
        assert repl_mod._pick_for_command("/model", repl_mod.ReplState()) == "/model analyze glm-5.2"
        assert [v for v, _ in seen[1]][:2] == ["glm-5.3", "glm-5.3-flash"]
