"""
Tests for the interactive REPL's command grammar and state mutations.

Loop-level run_repl() isn't tested here — it owns input() and the asyncio
event loop. The contract worth testing is dispatch(): given a slash command
line, the right state mutates and exit signaling propagates.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research.models import ReportStyle, ResearchDepth
from deep_research.repl import COMMANDS, ReplState, dispatch


class TestStateMutations:
    def test_depth_changes(self):
        s = ReplState()
        dispatch("/depth deep", s)
        assert s.depth == ResearchDepth.DEEP
        dispatch("/depth quick", s)
        assert s.depth == ResearchDepth.QUICK

    def test_invalid_depth_does_not_mutate(self, capsys):
        s = ReplState()
        dispatch("/depth ultra", s)
        assert s.depth == ResearchDepth.STANDARD  # unchanged

    def test_style_changes(self):
        s = ReplState()
        dispatch("/style brief", s)
        assert s.style == ReportStyle.BRIEF

    def test_template_set_and_clear(self):
        s = ReplState()
        dispatch("/template science", s)
        assert s.template == "science"
        dispatch("/template none", s)
        assert s.template is None

    def test_unknown_template_does_not_mutate(self):
        s = ReplState()
        dispatch("/template fake", s)
        assert s.template is None

    @pytest.mark.parametrize(
        "cmd,attr",
        [
            ("/thinking", "thinking"),
            ("/academic", "academic"),
            ("/agent", "agent"),
            ("/fresh", "fresh"),
        ],
    )
    def test_toggles_flip(self, cmd, attr):
        s = ReplState()
        assert getattr(s, attr) is False
        dispatch(cmd, s)
        assert getattr(s, attr) is True
        dispatch(cmd, s)
        assert getattr(s, attr) is False


class TestQuit:
    def test_quit_returns_exit_signal(self):
        assert dispatch("/quit", ReplState()) is True

    def test_quit_aliases(self):
        for alias in ("/exit", "/q"):
            assert dispatch(alias, ReplState()) is True

    def test_help_does_not_exit(self, capsys):
        assert dispatch("/help", ReplState()) in (None, False)


class TestUnknownCommand:
    def test_unknown_raises_keyerror_with_name(self):
        with pytest.raises(KeyError) as ei:
            dispatch("/notarealcommand", ReplState())
        assert ei.value.args[0] == "/notarealcommand"


class TestEmptyAndWhitespace:
    def test_empty_line_is_noop(self):
        # The REPL filters empty lines before calling dispatch, but dispatch
        # itself should still tolerate one (returning None) rather than
        # raising — defensive against future callers.
        assert dispatch("", ReplState()) is None
        assert dispatch("   ", ReplState()) is None


class TestCommandRegistry:
    def test_every_command_has_handler_and_summary(self):
        for cmd in COMMANDS:
            assert cmd.name.startswith("/"), cmd.name
            assert callable(cmd.handler), cmd.name
            assert cmd.summary, cmd.name

    def test_aliases_unique(self):
        names = []
        for cmd in COMMANDS:
            names.extend(cmd.all_names())
        assert len(names) == len(set(names)), "duplicate command name/alias"


class TestProviderAndRouteCommands:
    def test_route_change_through_dispatch(self):
        # /route mutates settings.llm_route, validates, and rolls back on bad
        # input. Use an opencode-go config so quality/budget/balanced are valid.
        from deep_research import config as cfg_module
        from deep_research.config import Settings as _Settings
        custom = _Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1", llm_route="balanced",
        )
        with patch.object(cfg_module, "settings", custom), \
             patch.object(__import__("deep_research.repl", fromlist=["settings"]),
                          "settings", custom):
            s = ReplState()
            dispatch("/route quality", s)
            assert custom.llm_route == "quality"
            # Invalid profile should roll back
            dispatch("/route nonexistent", s)
            assert custom.llm_route == "quality"  # unchanged

    def test_provider_invalid_rolls_back(self):
        from deep_research import config as cfg_module
        from deep_research.config import Settings as _Settings
        custom = _Settings(
            llm_provider="opencode-go", opencode_api_key="x",
            opencode_model="glm-5.1",
        )
        with patch.object(cfg_module, "settings", custom), \
             patch.object(__import__("deep_research.repl", fromlist=["settings"]),
                          "settings", custom):
            s = ReplState()
            dispatch("/provider made-up-provider", s)
            assert custom.llm_provider == "opencode-go"  # unchanged
