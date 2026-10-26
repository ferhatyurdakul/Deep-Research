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
os.environ["DEEP_RESEARCH_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(prefix="dr-test-config-"), "config.json")


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_CONFIG_FILE", str(tmp_path / "user-config" / "config.json"))


@pytest.fixture(autouse=True)
def _no_live_model_lookup(monkeypatch):
    """The setup screen and /model ask the provider which models the account has
    (a network call). Tests never do; the ones covering the lookup itself stub httpx."""
    import deep_research.options as options

    monkeypatch.setattr(options, "fetch_account_models", lambda *a, **k: None)
