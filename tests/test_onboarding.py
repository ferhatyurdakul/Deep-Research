"""
Tests for the setup screen: provider -> API key -> search key -> model per stage.

The pickers and prompts are scripted (they need a terminal), the keyring is an
in-memory fake, and the connection check is stubbed, so nothing here touches the
real keychain, the network, or an API.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import synapse_deep_research.onboarding as onboarding
from synapse_deep_research import userconfig
from synapse_deep_research.capabilities import PIPELINE_STAGES, recommended_stage_models
from synapse_deep_research.config import Settings


class _FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, key):
        return self.store.get((service, key))

    def set_password(self, service, key, value):
        self.store[(service, key)] = value

    def delete_password(self, service, key):
        self.store.pop((service, key), None)

    def get_keyring(self):
        return self

    name = "FakeKeyring"


@pytest.fixture
def env(monkeypatch):
    """A terminal, an empty in-memory keyring, and a fresh Settings the flow can mutate."""
    from synapse_deep_research import credentials

    fake = _FakeKeyring()
    monkeypatch.setattr(credentials, "_backend", lambda: fake)
    for name in ("ZAI_API_KEY", "OPENCODE_API_KEY", "TAVILY_API_KEY", "MODEL_ANALYZE",
                 "MODEL_DECOMPOSE", "MODEL_GAP_ANALYSIS", "MODEL_SYNTHESIZE"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        llm_provider="zai", zai_api_key="", opencode_api_key="", tavily_api_key="",
        model_decompose="", model_analyze="", model_synthesize="", model_gap_analysis="",
    )
    monkeypatch.setattr(onboarding, "settings", settings)
    monkeypatch.setattr(onboarding, "tty", lambda: True)
    monkeypatch.setattr(onboarding, "_verify", lambda models: {})
    return type("Env", (), {"keyring": fake, "settings": settings})


def script(monkeypatch, picks=(), texts=()):
    """Feed answers to pick_option / ask_text in order; returns the recorded titles."""
    picks, texts = iter(picks), iter(texts)
    seen = {"picks": [], "texts": []}

    def fake_pick(title, choices, current=None, **kw):
        seen["picks"].append((title, [v for v, _ in choices], current))
        return next(picks)

    def fake_ask(title, **kw):
        seen["texts"].append((title, kw))
        return next(texts)

    monkeypatch.setattr(onboarding, "pick_option", fake_pick)
    monkeypatch.setattr(onboarding, "ask_text", fake_ask)
    return seen


class TestNeedsSetup:
    def test_true_without_a_key_for_the_active_provider(self, env):
        assert onboarding.needs_setup() is True

    def test_false_once_the_key_exists(self, env):
        env.settings.zai_api_key = "sk-x"
        assert onboarding.needs_setup() is False

    def test_the_active_providers_key_is_the_one_that_counts(self, env):
        env.settings.opencode_api_key = "sk-x"          # but the provider is zai
        assert onboarding.needs_setup() is True

    def test_an_invalid_provider_is_a_config_error_not_a_setup_case(self, env):
        env.settings.llm_provider = "typo"
        assert onboarding.needs_setup() is False


class TestRunSetup:
    def test_recommended_models_happy_path(self, env, monkeypatch, capsys):
        seen = script(monkeypatch,
                      picks=["zai", "recommended", "skip"],
                      texts=["sk-test-key", ""])           # API key, no Tavily key
        assert onboarding.run_setup(first_run=True) is True

        assert env.keyring.store[("synapse-deep-research", "ZAI_API_KEY")] == "sk-test-key"
        assert ("synapse-deep-research", "TAVILY_API_KEY") not in env.keyring.store
        assert userconfig.provider() == "zai"
        assert userconfig.stage_models("zai") == recommended_stage_models("zai").as_dict()
        assert env.settings.zai_api_key == "sk-test-key"        # live settings see the new key
        titles = [t for t, _, _ in seen["picks"]]
        assert titles[0].startswith("Welcome · 1/5") and titles[2].startswith("Welcome · 5/5")
        assert seen["texts"][0][1]["secret"] is True            # keys are typed hidden
        out = capsys.readouterr().out
        assert "Setup complete" in out and "glm-5.3-flash" in out and "recommended" in out

    def test_choosing_a_model_for_each_stage(self, env, monkeypatch):
        seen = script(monkeypatch,
                      picks=["zai", "custom", "glm-5.3", "glm-5.2", "glm-5.3-flash", "glm-5.1", "skip"],
                      texts=["sk-k", ""])
        assert onboarding.run_setup() is True
        assert userconfig.stage_models("zai") == {
            "decompose": "glm-5.3", "analyze": "glm-5.2",
            "gap_analysis": "glm-5.3-flash", "synthesize": "glm-5.1",
        }
        model_titles = [t for t, _, _ in seen["picks"] if "Model for" in t]
        assert len(model_titles) == 4
        for stage, title in zip(PIPELINE_STAGES, model_titles):
            assert f"Model for {stage}" in title
        # Each model picker offers the current models plus an escape hatch,
        # and starts on the recommendation.
        _, values, current = [p for p in seen["picks"] if "Model for analyze" in p[0]][0]
        assert values[:2] == ["glm-5.3", "glm-5.3-flash"] and values[-1] == "other"
        assert current == "glm-5.3-flash"

    def test_other_lets_you_type_any_model_id(self, env, monkeypatch):
        script(monkeypatch,
               picks=["zai", "custom", "other", "glm-5.3", "glm-5.3", "glm-5.3", "skip"],
               texts=["sk-k", "", "  glm-9.9  "])           # key, no tavily, then the typed ID
        assert onboarding.run_setup() is True
        assert userconfig.stage_models("zai")["decompose"] == "glm-9.9"

    def test_the_search_key_is_stored_when_given(self, env, monkeypatch):
        script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["sk-k", " tvly-abc "])
        onboarding.run_setup()
        assert env.keyring.store[("synapse-deep-research", "TAVILY_API_KEY")] == "tvly-abc"

    def test_esc_on_the_optional_search_key_just_skips_it(self, env, monkeypatch):
        script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["sk-k", None])
        assert onboarding.run_setup() is True
        assert ("synapse-deep-research", "TAVILY_API_KEY") not in env.keyring.store

    @pytest.mark.opencode_go
    def test_picking_the_other_provider(self, env, monkeypatch):
        script(monkeypatch, picks=["opencode-go", "recommended", "skip"], texts=["oc-key", ""])
        assert onboarding.run_setup() is True
        assert env.keyring.store[("synapse-deep-research", "OPENCODE_API_KEY")] == "oc-key"
        assert userconfig.provider() == "opencode-go"
        assert set(userconfig.stage_models("opencode-go").values()) == {"glm-5.1"}

    def test_an_existing_key_is_kept_on_enter(self, env, monkeypatch):
        env.keyring.set_password("synapse-deep-research", "ZAI_API_KEY", "old-key")
        seen = script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["", ""])
        assert onboarding.run_setup() is True
        assert env.keyring.store[("synapse-deep-research", "ZAI_API_KEY")] == "old-key"
        assert "keeps the key" in seen["texts"][0][1]["hint"]

    def test_a_key_is_required_when_there_is_none(self, env, monkeypatch):
        seen = script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["   ", "sk-second", ""])
        assert onboarding.run_setup() is True
        assert env.keyring.store[("synapse-deep-research", "ZAI_API_KEY")] == "sk-second"
        assert "required" in seen["texts"][1][1]["hint"]

    def test_the_search_key_hint_is_hidden_input_too(self, env, monkeypatch):
        seen = script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["k", ""])
        onboarding.run_setup()
        assert seen["texts"][1][1]["secret"] is True

    @pytest.mark.parametrize("cancel_at", ["provider", "key", "models", "custom-model"])
    def test_esc_anywhere_skips_cleanly_and_saves_nothing(self, env, monkeypatch, capsys, cancel_at):
        plans = {
            "provider": dict(picks=[None], texts=[]),
            "key": dict(picks=["zai"], texts=[None]),
            "models": dict(picks=["zai", None], texts=["sk-k", ""]),
            "custom-model": dict(picks=["zai", "custom", None], texts=["sk-k", ""]),
        }
        script(monkeypatch, **plans[cancel_at])
        assert onboarding.run_setup() is False
        assert not userconfig.exists()
        assert env.settings.llm_provider == "zai"               # restored, not left half-switched
        assert "Setup skipped" in capsys.readouterr().out
        # Keys are only written once every step has been answered.
        assert env.keyring.store == {}

    def test_not_in_a_terminal_it_refuses_instead_of_hanging(self, env, monkeypatch, capsys):
        monkeypatch.setattr(onboarding, "tty", lambda: False)
        seen = script(monkeypatch)
        assert onboarding.run_setup() is False
        assert seen == {"picks": [], "texts": []}               # nothing was asked
        assert "interactive terminal" in capsys.readouterr().out

    def test_a_keyring_failure_is_reported_with_the_env_alternative(self, env, monkeypatch, capsys):
        script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["sk-k", ""])

        def broken(*a, **k):
            raise RuntimeError("keychain locked")

        monkeypatch.setattr(onboarding, "set_credential", broken)
        assert onboarding.run_setup() is False
        out = capsys.readouterr().out
        assert "keychain locked" in out and "ZAI_API_KEY" in out
        assert not userconfig.exists()


class TestConnectionCheck:
    def test_results_are_shown_and_a_failure_does_not_lose_the_choices(self, env, monkeypatch, capsys):
        script(monkeypatch, picks=["zai", "recommended", "yes"], texts=["sk-k", ""])
        monkeypatch.setattr(
            onboarding, "_verify",
            lambda models: {"glm-5.3-flash": None, "glm-5.3": "401 plan has no access"},
        )
        assert onboarding.run_setup() is True
        out = capsys.readouterr().out
        assert "reachable" in out and "401 plan has no access" in out
        assert "didn't answer" in out
        assert userconfig.stage_models("zai") == recommended_stage_models("zai").as_dict()

    def test_skipping_the_check_makes_no_requests(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(onboarding, "_verify", lambda models: calls.append(models) or {})
        script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["sk-k", ""])
        onboarding.run_setup()
        assert calls == []

    def test_verify_makes_one_request_per_distinct_model_and_reports_errors(self, monkeypatch):
        import synapse_deep_research.llm as llm

        asked = []

        def fake_chat(prompt, system=None, model=None, temperature=0.7, thinking=False):
            asked.append(model)
            if model == "glm-5.3":
                raise RuntimeError("Error code: 403 - model not available on your plan\nextra line")
            return "ok"

        monkeypatch.setattr(llm, "chat", fake_chat)
        results = onboarding._verify(recommended_stage_models("zai").as_dict())
        assert asked == ["glm-5.3-flash", "glm-5.3"]            # 4 stages, 2 distinct models
        assert results["glm-5.3-flash"] is None
        assert results["glm-5.3"] == "Error code: 403 - model not available on your plan"


class TestSummaryWarnings:
    def test_environment_overrides_are_called_out(self, env, monkeypatch, capsys):
        monkeypatch.setenv("MODEL_ANALYZE", "glm-5.2")
        script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["sk-k", ""])
        onboarding.run_setup()
        assert "MODEL_ANALYZE" in capsys.readouterr().out

    def test_a_key_in_the_environment_is_called_out(self, env, monkeypatch, capsys):
        monkeypatch.setenv("ZAI_API_KEY", "from-env")
        script(monkeypatch, picks=["zai", "recommended", "skip"], texts=["sk-new", ""])
        onboarding.run_setup()
        assert "environment value wins" in capsys.readouterr().out


class TestFirstRunTrigger:
    def _run_repl_with(self, monkeypatch, *, tty, needs):
        import synapse_deep_research.repl as repl_mod

        calls = []

        class _FakeStdin:
            def isatty(self):
                return tty

        monkeypatch.setattr(sys, "stdin", _FakeStdin())
        monkeypatch.setattr(repl_mod, "needs_setup", lambda: needs)
        monkeypatch.setattr(repl_mod, "run_setup", lambda **kw: calls.append(kw) or True)
        monkeypatch.setattr(repl_mod, "_BoxedPrompt", lambda state: (_ for _ in ()).throw(RuntimeError("no terminal")))
        monkeypatch.setattr("builtins.input", lambda prompt="": "/q")
        repl_mod.run_repl(repl_mod.ReplState())
        return calls

    def test_setup_opens_first_when_nothing_is_configured(self, monkeypatch):
        assert self._run_repl_with(monkeypatch, tty=True, needs=True) == [{"first_run": True}]

    def test_no_setup_when_a_key_exists(self, monkeypatch):
        assert self._run_repl_with(monkeypatch, tty=True, needs=False) == []

    def test_never_in_a_pipe_or_script(self, monkeypatch):
        assert self._run_repl_with(monkeypatch, tty=False, needs=True) == []

    def test_setup_runs_before_the_banner(self, monkeypatch, capsys):
        import synapse_deep_research.repl as repl_mod

        order = []
        monkeypatch.setattr(repl_mod, "run_setup", lambda **kw: order.append("setup") or True)
        monkeypatch.setattr(repl_mod, "_print_banner", lambda state: order.append("banner"))
        monkeypatch.setattr(repl_mod, "needs_setup", lambda: True)

        class _TTY:
            def isatty(self):
                return True

        monkeypatch.setattr(sys, "stdin", _TTY())
        monkeypatch.setattr(repl_mod, "_BoxedPrompt", lambda state: (_ for _ in ()).throw(RuntimeError()))
        monkeypatch.setattr("builtins.input", lambda prompt="": "/q")
        repl_mod.run_repl(repl_mod.ReplState())
        assert order == ["setup", "banner"]


class TestOneShotGate:
    def test_without_a_terminal_a_missing_key_exits_with_a_pointer(self, monkeypatch, capsys):
        import synapse_deep_research.main as main_mod

        monkeypatch.setattr(main_mod, "check_config", lambda: False)
        monkeypatch.setattr(main_mod, "tty", lambda: False)
        monkeypatch.setattr(main_mod.settings, "zai_api_key", "")
        with pytest.raises(SystemExit) as ei:
            main_mod._ensure_ready()
        assert ei.value.code == 1
        assert "synapse-deep-research setup" in capsys.readouterr().out

    def test_with_a_terminal_it_opens_setup_and_exits_if_it_was_skipped(self, monkeypatch, capsys):
        import synapse_deep_research.main as main_mod

        opened = []
        monkeypatch.setattr(main_mod, "check_config", lambda: False)
        monkeypatch.setattr(main_mod, "tty", lambda: True)
        monkeypatch.setattr(main_mod.settings, "zai_api_key", "")
        monkeypatch.setattr(onboarding, "run_setup", lambda **kw: opened.append(kw) or False)
        # reload_credentials() reads the real OS keyring; keep this test hermetic.
        monkeypatch.setattr(type(main_mod.settings), "reload_credentials", lambda self: None)
        with pytest.raises(SystemExit) as ei:
            main_mod._ensure_ready()
        assert ei.value.code == 1 and opened == [{"first_run": True}]
        assert "Still no" in capsys.readouterr().out


class TestRetiredRouteSetting:
    def test_llm_route_in_the_environment_gets_a_notice(self, monkeypatch, capsys):
        import synapse_deep_research.main as main_mod

        monkeypatch.setenv("LLM_ROUTE", "quality")
        main_mod._warn_retired_settings()
        out = " ".join(capsys.readouterr().out.split())
        assert "LLM_ROUTE is no longer used" in out and "synapse-deep-research setup" in out

    def test_silent_when_it_is_not_set(self, monkeypatch, capsys):
        import synapse_deep_research.main as main_mod

        monkeypatch.delenv("LLM_ROUTE", raising=False)
        main_mod._warn_retired_settings()
        assert capsys.readouterr().out == ""


class TestAskText:
    def _ask(self, keys, **kw):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from synapse_deep_research.interactive import ask_text

        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            return ask_text("Title", input=pipe, output=DummyOutput(), **kw)

    def test_returns_what_was_typed(self):
        assert self._ask("glm-5.3-flash\r") == "glm-5.3-flash"

    def test_empty_enter_is_an_empty_string_not_a_cancel(self):
        assert self._ask("\r") == ""

    def test_escape_and_ctrl_c_cancel(self):
        assert self._ask("abc\x1b") is None
        assert self._ask("abc\x03") is None

    def test_initial_text_is_prefilled(self):
        assert self._ask("\r", initial="glm-") == "glm-"
        assert self._ask("5\r", initial="glm-") == "glm-5"

    def test_secret_mode_still_returns_the_real_text(self):
        assert self._ask("sk-secret\r", secret=True) == "sk-secret"
