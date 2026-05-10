"""
Tests for filesystem path resolution.

The contract is:
- DATA_DIR defaults to $XDG_DATA_HOME/deep-research (or ~/.local/share/...)
  and is overridable via DEEP_RESEARCH_DATA_DIR.
- OUTPUTS_DIR defaults to cwd/outputs and is overridable via
  DEEP_RESEARCH_OUTPUTS_DIR.
- OUTPUTS_DIR is NOT created at import time — only when save_report runs.
- .env is loaded from cwd (or above) first, falling back to
  $XDG_CONFIG_HOME/deep-research/.env.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _reload_config():
    """Re-import the config module to pick up env-var changes."""
    import deep_research.config as cfg
    return importlib.reload(cfg)


class TestDataDir:
    def test_xdg_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        monkeypatch.delenv("DEEP_RESEARCH_DATA_DIR", raising=False)
        cfg = _reload_config()
        assert cfg.DATA_DIR == tmp_path / "deep-research"
        assert cfg.DATA_DIR.exists()
        assert cfg.DB_PATH == cfg.DATA_DIR / "research.db"

    def test_explicit_override_wins(self, tmp_path, monkeypatch):
        target = tmp_path / "custom-data"
        monkeypatch.setenv("DEEP_RESEARCH_DATA_DIR", str(target))
        cfg = _reload_config()
        assert cfg.DATA_DIR == target
        assert target.exists()

    def test_default_under_home_when_xdg_unset(self, tmp_path, monkeypatch):
        # Simulate a user without XDG_DATA_HOME set; should fall back to
        # ~/.local/share/deep-research/. Use a fake HOME so we don't litter
        # the real one during tests.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.delenv("DEEP_RESEARCH_DATA_DIR", raising=False)
        cfg = _reload_config()
        assert cfg.DATA_DIR == tmp_path / ".local" / "share" / "deep-research"


class TestOutputsDir:
    def test_default_is_cwd_relative(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEP_RESEARCH_OUTPUTS_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        assert cfg.OUTPUTS_DIR == tmp_path / "outputs"

    def test_not_created_at_import(self, tmp_path, monkeypatch):
        # Critical: bare CLI invocations (e.g. `deep-research --history`)
        # in random directories must NOT leave empty `outputs/` dirs behind.
        monkeypatch.delenv("DEEP_RESEARCH_OUTPUTS_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        # The default path now points at tmp_path/outputs, but that dir
        # should not exist yet because we haven't actually saved anything.
        assert not (tmp_path / "outputs").exists()

    def test_explicit_override_wins(self, tmp_path, monkeypatch):
        target = tmp_path / "custom-outputs"
        monkeypatch.setenv("DEEP_RESEARCH_OUTPUTS_DIR", str(target))
        cfg = _reload_config()
        assert cfg.OUTPUTS_DIR == target
        # Still not created at import.
        assert not target.exists()


class TestSaveReportCreatesOutputDir:
    def test_save_report_mkdirs_lazily(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-outputs"
        monkeypatch.setenv("DEEP_RESEARCH_OUTPUTS_DIR", str(target))
        cfg = _reload_config()
        # Reload report module so it picks up the new OUTPUTS_DIR.
        import deep_research.output.report as report_mod
        report_mod = importlib.reload(report_mod)

        from deep_research.models import ResearchReport
        r = ResearchReport(query="test", executive_summary="x", detailed_findings="y")

        assert not target.exists()
        path = report_mod.save_report(r)
        assert target.exists()
        assert path.parent == target
        assert path.exists()
