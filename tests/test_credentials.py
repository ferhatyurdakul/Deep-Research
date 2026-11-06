"""
Tests for the credentials module and the setup wizard's plumbing.

Tests use a fake in-memory keyring backend so they never touch the real
OS keychain. Wizard interactivity is exercised via patched prompts.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _FakeKeyring:
    """In-memory keyring stand-in. Keyed by (service, key) -> value."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, key: str):
        return self._store.get((service, key))

    def set_password(self, service: str, key: str, value: str) -> None:
        self._store[(service, key)] = value

    def delete_password(self, service: str, key: str) -> None:
        if (service, key) not in self._store:
            raise Exception("no such entry")
        del self._store[(service, key)]

    # keyring.get_keyring() returns a backend object whose .name is
    # surfaced by credentials.keyring_backend_name().
    def get_keyring(self):
        return self

    name = "FakeKeyring"


_CRED_FIELDS = ("zai_api_key", "opencode_api_key", "tavily_api_key", "semantic_scholar_api_key")


def _install_fake_keyring(monkeypatch):
    """Point credentials at an in-memory keyring, and make sure the process-wide
    settings are put back afterwards."""
    from synapse_deep_research import credentials
    from synapse_deep_research.config import settings

    fake = _FakeKeyring()
    monkeypatch.setattr(credentials, "_backend", lambda: fake)
    # run_login()/run_logout() call the *global* settings.reload_credentials(),
    # which copies this fake keyring's values (empty) into the process-wide
    # settings singleton. Register the real values with monkeypatch so they are
    # restored at teardown and later modules (test_e2e.py's live tests) still
    # see the credentials they were collected with.
    for field in _CRED_FIELDS:
        monkeypatch.setattr(settings, field, getattr(settings, field))
    return fake


@pytest.fixture
def fake_keyring(monkeypatch):
    return _install_fake_keyring(monkeypatch)


def test_logout_does_not_leak_into_global_settings():
    """Regression: run_logout() reloads the *global* settings from the (fake)
    keyring, blanking every credential for the rest of the pytest process. The
    live e2e tests, collected earlier with real keys, then ran with empty ones
    and failed with "Missing credentials".

    Uses its own MonkeyPatch and undoes it explicitly, so the check doesn't
    depend on the order pytest tears fixtures down in."""
    from synapse_deep_research.config import settings
    from synapse_deep_research.setup_wizard import run_logout

    originals = {f: getattr(settings, f) for f in _CRED_FIELDS}
    for f in _CRED_FIELDS:
        object.__setattr__(settings, f, f"sentinel-{f}")
    mp = pytest.MonkeyPatch()
    try:
        _install_fake_keyring(mp)
        with patch("synapse_deep_research.setup_wizard.Confirm.ask", return_value=True):
            run_logout("all")
        mp.undo()  # what the fake_keyring fixture's teardown does
        after = {f: getattr(settings, f) for f in _CRED_FIELDS}
    finally:
        mp.undo()
        for f, value in originals.items():
            object.__setattr__(settings, f, value)
    assert after == {f: f"sentinel-{f}" for f in _CRED_FIELDS}, (
        "run_logout()/run_login() leaked the fake keyring into the global settings"
    )


class TestGetSetDelete:
    def test_set_and_get_round_trip(self, fake_keyring):
        from synapse_deep_research.credentials import get_credential, set_credential
        set_credential("ZAI_API_KEY", "sk-abc")
        assert get_credential("ZAI_API_KEY") == "sk-abc"

    def test_get_missing_returns_none(self, fake_keyring):
        from synapse_deep_research.credentials import get_credential
        assert get_credential("MISSING_KEY") is None

    def test_set_empty_rejected(self, fake_keyring):
        from synapse_deep_research.credentials import set_credential
        with pytest.raises(ValueError):
            set_credential("ZAI_API_KEY", "")

    def test_delete_returns_true_when_present(self, fake_keyring):
        from synapse_deep_research.credentials import delete_credential, set_credential
        set_credential("TAVILY_API_KEY", "tav-x")
        assert delete_credential("TAVILY_API_KEY") is True
        assert delete_credential("TAVILY_API_KEY") is False  # second call: no-op

    def test_delete_returns_false_when_absent(self, fake_keyring):
        from synapse_deep_research.credentials import delete_credential
        assert delete_credential("NEVER_STORED") is False


class TestListCredentials:
    def test_lists_all_managed_keys_with_presence_only(self, fake_keyring):
        # Never leak values — only presence.
        from synapse_deep_research.credentials import (
            MANAGED_KEY_NAMES,
            list_credentials,
            set_credential,
        )
        set_credential("ZAI_API_KEY", "secret-value")
        result = list_credentials()
        # Keys we set show as True; others False; no values appear.
        assert result["ZAI_API_KEY"] is True
        for k in MANAGED_KEY_NAMES:
            assert k in result
        assert "secret-value" not in str(result)


class TestSettingsKeyringFallback:
    def test_empty_field_filled_from_keyring(self, fake_keyring, monkeypatch):
        # Pydantic field defaults are bound at class-definition time (when
        # the config module is imported), so monkeypatching env after that
        # doesn't change the defaults. To simulate "no env value", we pass
        # empty strings explicitly — that's what model_post_init reacts to.
        from synapse_deep_research.credentials import set_credential
        set_credential("ZAI_API_KEY", "key-from-keyring")

        from synapse_deep_research.config import Settings
        s = Settings(
            zai_api_key="", opencode_api_key="",
            tavily_api_key="", semantic_scholar_api_key="",
        )
        assert s.zai_api_key == "key-from-keyring"
        # Other managed keys remain empty.
        assert s.opencode_api_key == ""

    def test_explicit_constructor_value_wins_over_keyring(self, fake_keyring):
        from synapse_deep_research.credentials import set_credential
        set_credential("ZAI_API_KEY", "from-keyring")

        from synapse_deep_research.config import Settings
        s = Settings(zai_api_key="from-constructor")
        assert s.zai_api_key == "from-constructor"

    def test_reload_credentials_updates_field(self, fake_keyring, monkeypatch):
        # Clear the env so reload_credentials doesn't see ZAI_API_KEY and
        # decide to keep the env-sourced value. Then start with an empty
        # field, set keyring later, reload, observe the update.
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        from synapse_deep_research.config import Settings
        from synapse_deep_research.credentials import set_credential
        s = Settings(zai_api_key="")
        assert s.zai_api_key == ""

        set_credential("ZAI_API_KEY", "newly-set")
        s.reload_credentials()
        assert s.zai_api_key == "newly-set"

    def test_reload_does_not_overwrite_env_sourced_values(self, fake_keyring, monkeypatch):
        # If ZAI_API_KEY is in the live environment, reload_credentials
        # must not blow it away even when keyring has a different value.
        monkeypatch.setenv("ZAI_API_KEY", "from-env")
        from synapse_deep_research.config import Settings
        from synapse_deep_research.credentials import set_credential
        s = Settings(zai_api_key="from-env")
        set_credential("ZAI_API_KEY", "from-keyring")
        s.reload_credentials()
        assert s.zai_api_key == "from-env"


class TestSetupWizardSubcommands:
    """High-level: the wizard calls credentials.set_credential and propagates errors."""

    def test_run_login_with_unknown_key_prints_error(self, fake_keyring, capsys):
        from synapse_deep_research.setup_wizard import run_login
        run_login("MADE_UP_KEY")
        # Should warn rather than store
        from synapse_deep_research.credentials import get_credential
        assert get_credential("MADE_UP_KEY") is None

    def test_run_logout_all_clears_everything(self, fake_keyring):
        from synapse_deep_research.credentials import (
            list_credentials,
            set_credential,
        )
        from synapse_deep_research.setup_wizard import run_logout

        set_credential("ZAI_API_KEY", "a")
        set_credential("TAVILY_API_KEY", "b")

        # Bypass the Confirm prompt by patching it to return True.
        with patch("synapse_deep_research.setup_wizard.Confirm.ask", return_value=True):
            run_logout("all")

        result = list_credentials()
        assert not any(result.values())

    def test_run_logout_specific_key(self, fake_keyring):
        from synapse_deep_research.credentials import (
            get_credential,
            set_credential,
        )
        from synapse_deep_research.setup_wizard import run_logout

        set_credential("ZAI_API_KEY", "a")
        set_credential("TAVILY_API_KEY", "b")
        run_logout("ZAI_API_KEY")
        assert get_credential("ZAI_API_KEY") is None
        assert get_credential("TAVILY_API_KEY") == "b"  # untouched


class TestLogoutDefault:
    def test_bare_logout_clears_only_active_provider_key(self, fake_keyring, monkeypatch):
        from synapse_deep_research import credentials, setup_wizard
        from synapse_deep_research.config import settings

        credentials.set_credential("ZAI_API_KEY", "z-secret")
        credentials.set_credential("TAVILY_API_KEY", "t-secret")
        monkeypatch.setattr(settings, "llm_provider", "zai")

        setup_wizard.run_logout(None)

        assert credentials.get_credential("ZAI_API_KEY") is None
        assert credentials.get_credential("TAVILY_API_KEY") == "t-secret"

    def test_logout_all_clears_everything(self, fake_keyring, monkeypatch):
        from types import SimpleNamespace

        from synapse_deep_research import credentials, setup_wizard

        credentials.set_credential("ZAI_API_KEY", "z")
        credentials.set_credential("TAVILY_API_KEY", "t")
        monkeypatch.setattr(
            setup_wizard, "Confirm",
            SimpleNamespace(ask=lambda *a, **k: True),
        )

        setup_wizard.run_logout("all")

        assert credentials.get_credential("ZAI_API_KEY") is None
        assert credentials.get_credential("TAVILY_API_KEY") is None
