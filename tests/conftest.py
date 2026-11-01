"""
Test-session isolation from the developer's real configuration.

`userconfig` stores the provider and per-stage models in a JSON file under the
user's config dir. Point it at a throwaway path before any test (or module
import) can read the real one, and give every test its own empty file.
"""
import os
import tempfile

import pytest

# Module level: Settings reads the saved provider at import time, before any
# fixture has run.
os.environ["SYNAPSE_DEEP_RESEARCH_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(prefix="dr-test-config-"), "config.json")


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DEEP_RESEARCH_CONFIG_FILE", str(tmp_path / "user-config" / "config.json"))


class FakeKeyring:
    """In-memory stand-in for the OS keyring: (service, key) -> value."""

    name = "FakeKeyring"

    def __init__(self) -> None:
        self.store: dict = {}

    def get_password(self, service, key):
        return self.store.get((service, key))

    def set_password(self, service, key, value):
        self.store[(service, key)] = value

    def delete_password(self, service, key):
        if (service, key) not in self.store:
            raise KeyError(key)
        del self.store[(service, key)]

    def get_keyring(self):
        return self


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    """No test reaches the developer's real keyring. Tests that need to see or seed
    it take this fixture; ones that install their own fake just override it."""
    from synapse_deep_research import credentials

    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "_backend", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _json_mode_does_not_leak():
    """--json points the shared console at stderr for the rest of the process;
    put it back so one test's run can't redirect the next one's output."""
    from synapse_deep_research import ui

    yield
    ui.disable_json_output()


@pytest.fixture(autouse=True)
def _no_live_model_lookup(monkeypatch):
    """The setup screen and /model ask the provider which models the account has
    (a network call). Tests never do; the ones covering the lookup itself stub httpx."""
    import synapse_deep_research.options as options

    monkeypatch.setattr(options, "fetch_account_models", lambda *a, **k: None)
