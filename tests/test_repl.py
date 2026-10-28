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
        "cmd,attr,initial",
        [
            ("/academic", "academic", False),
            ("/agent", "agent", False),
            ("/fresh", "fresh", False),
        ],
    )
    def test_toggles_flip(self, cmd, attr, initial):
        s = ReplState()
        assert getattr(s, attr) is initial
        dispatch(cmd, s)
        assert getattr(s, attr) is (not initial)
        dispatch(cmd, s)
        assert getattr(s, attr) is initial


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
    def test_route_command_is_gone(self):
        from deep_research.repl import _COMMAND_INDEX

        assert "/route" not in _COMMAND_INDEX
        with pytest.raises(KeyError):
            dispatch("/route quality", ReplState())

    def test_provider_switch_is_saved(self):
        from deep_research import userconfig
        from deep_research import config as cfg_module
        from deep_research.config import Settings as _Settings
        custom = _Settings(llm_provider="zai", zai_api_key="x", opencode_api_key="y", opencode_model="glm-5.1")
        with patch.object(cfg_module, "settings", custom), \
             patch.object(__import__("deep_research.repl", fromlist=["settings"]),
                          "settings", custom):
            dispatch("/provider opencode-go", ReplState())
            assert custom.llm_provider == "opencode-go"
            assert userconfig.provider() == "opencode-go"   # chosen in the app, so it sticks

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


class TestForkCommand:
    def test_fork_dispatches_with_optional_query(self, monkeypatch, capsys):
        import deep_research.repl as repl_mod

        calls = {}

        def _fake_fork(sid, new_query=None):
            calls["sid"], calls["q"] = sid, new_query
            return 42

        monkeypatch.setattr(repl_mod, "fork_session", _fake_fork)
        dispatch("/fork 7 newer angle", ReplState())

        assert calls == {"sid": 7, "q": "newer angle"}
        assert "42" in capsys.readouterr().out


class TestReplLaunch:
    def test_bare_launch_boots_and_quits(self):
        """Regression: main() once referenced a removed argparse attribute
        (args.thinking) when constructing ReplState, crashing bare
        `deep-research` before the prompt appeared."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "deep_research.main"],
            input="/q\n", capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "Goodbye" in result.stdout



class TestStatusFragments:
    def test_minimal_status_shows_only_non_defaults(self):
        from deep_research.models import ResearchDepth
        from deep_research.repl import ReplState, _status_fragments

        text = "".join(t for _, t in _status_fragments(ReplState()))
        assert text.startswith(" standard")
        assert "zai" in text or "invalid config" in text
        # Defaults are silent; thinking is gone entirely.
        assert "thinking" not in text
        assert "academic" not in text

        state = ReplState(depth=ResearchDepth.DEEP, template="science", agent=True)
        text = "".join(t for _, t in _status_fragments(state))
        assert " deep" in text
        assert "science" in text
        assert "agent" in text

    def test_slash_only_completer_blocks_plain_queries(self):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from deep_research.repl import _command_completer

        c = _command_completer()
        assert list(c.get_completions(Document("/", 1), CompleteEvent()))
        assert not list(c.get_completions(Document("what is CRISPR", 14), CompleteEvent()))


class TestHelpAndPickers:
    def test_help_lists_every_command_under_its_group(self, capsys):
        from deep_research.repl import GROUPS

        dispatch("/help", ReplState())
        out = capsys.readouterr().out
        for group in GROUPS:
            assert group in out, group
        for cmd in COMMANDS:
            assert cmd.name in out, cmd.name

    def test_every_command_belongs_to_a_known_group(self):
        from deep_research.repl import GROUPS
        assert {c.group for c in COMMANDS} <= set(GROUPS)

    def test_help_for_one_command_shows_values_with_meanings(self, capsys):
        dispatch("/help depth", ReplState())
        out = capsys.readouterr().out
        assert "quick" in out and "standard" in out and "deep" in out
        assert "sub-questions" in out

    def test_help_accepts_name_without_slash(self, capsys):
        dispatch("/help style", ReplState())
        assert "academic" in capsys.readouterr().out

    def test_help_for_unknown_command_suggests(self, capsys):
        dispatch("/help histroy", ReplState())
        out = capsys.readouterr().out
        assert "Unknown command" in out and "/history" in out

    def test_bare_choice_command_lists_every_value_and_marks_current(self, capsys):
        s = ReplState(depth=ResearchDepth.DEEP)
        dispatch("/depth", s)
        out = capsys.readouterr().out
        assert "quick" in out and "standard" in out and "deep" in out
        assert s.depth == ResearchDepth.DEEP  # listing never mutates
        deep_line = next(l for l in out.splitlines() if "deep" in l and "Thorough" in l)
        quick_line = next(l for l in out.splitlines() if "quick" in l)
        assert ("●" in deep_line or "*" in deep_line)
        assert ("○" in quick_line or "o" in quick_line.split("quick")[0])

    def test_template_picker_marks_none_when_unset(self, capsys):
        dispatch("/template", ReplState())
        out = capsys.readouterr().out
        assert "technology" in out and "none" in out

    def test_setting_a_value_echoes_its_meaning(self, capsys):
        dispatch("/depth quick", ReplState())
        out = capsys.readouterr().out
        assert "quick" in out and "1 pass" in out

    def test_invalid_value_lists_valid_ones(self, capsys):
        dispatch("/style loud", ReplState())
        out = capsys.readouterr().out
        assert "Unknown style" in out and "brief, standard, academic" in out


class TestUnknownCommandSuggestions:
    def test_suggests_closest_command(self):
        from deep_research.repl import suggest_command
        assert suggest_command("/histroy") == "/history"
        assert suggest_command("/dpth") == "/depth"
        assert suggest_command("/zzzzzzzz") is None

    def test_unknown_command_message_includes_suggestion(self, capsys):
        from deep_research.repl import _unknown_command
        _unknown_command("/histroy")
        out = capsys.readouterr().out
        assert "Did you mean /history" in out


class TestStyleReachesConfig:
    def test_style_setting_is_applied_to_the_research_config(self):
        """Regression: ReplState.to_config() dropped report_style, so /style
        silently had no effect on what got generated."""
        s = ReplState()
        dispatch("/style academic", s)
        assert s.to_config().report_style == ReportStyle.ACADEMIC
        dispatch("/style brief", s)
        assert s.to_config().report_style == ReportStyle.BRIEF


class TestQueryPersistence:
    def test_repl_query_is_saved_to_the_session_db(self, tmp_path, monkeypatch, capsys):
        """Regression: _run_query referenced an unimported save function, so a
        finished REPL run raised NameError before the session was stored."""
        import asyncio

        import deep_research.config as cfg
        import deep_research.main as main_mod
        import deep_research.output.report as report_mod
        import deep_research.storage.db as db_mod
        from deep_research.models import ResearchReport, SourceAnalysis
        from deep_research.repl import _run_query

        db = tmp_path / "research.db"
        monkeypatch.setattr(cfg, "DB_PATH", db)
        monkeypatch.setattr(db_mod, "DB_PATH", db)
        monkeypatch.setattr(report_mod, "OUTPUTS_DIR", tmp_path / "outputs")

        async def fake(query, config, agent):
            return ResearchReport(
                query=query, executive_summary="S", detailed_findings="D",
                sources=[SourceAnalysis(url="https://e.com", title="T", key_findings=["f"])],
            )

        monkeypatch.setattr(main_mod, "research_query", fake)
        monkeypatch.setattr(main_mod, "check_config", lambda: True)
        asyncio.run(_run_query(ReplState(), "what is x"))

        sessions = db_mod.list_sessions()
        assert [s["query"] for s in sessions] == ["what is x"]
        assert "/continue" in capsys.readouterr().out  # REPL-flavoured next steps


class TestQueryPreflight:
    def test_missing_credentials_stop_the_query_with_a_pointer(self, monkeypatch, capsys):
        """A REPL query with no key configured should say so and point at
        /setup rather than starting a run that dies in the SDK."""
        import asyncio

        import deep_research.main as main_mod
        from deep_research.repl import _run_query

        calls = []

        async def fake(query, config, agent):
            calls.append(query)

        def no_key():
            from deep_research.ui import fail
            fail("ZAI_API_KEY is not set")
            return False

        monkeypatch.setattr(main_mod, "research_query", fake)
        monkeypatch.setattr(main_mod, "check_config", no_key)
        asyncio.run(_run_query(ReplState(), "what is x"))

        assert calls == []
        assert "/setup" in capsys.readouterr().out


class TestCompleter:
    def _complete(self, text):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from deep_research.repl import _command_completer

        return list(_command_completer().get_completions(Document(text, len(text)), CompleteEvent()))

    def test_bare_slash_lists_primary_commands_without_aliases(self):
        texts = {c.text for c in self._complete("/")}
        assert "/depth" in texts and "/history" in texts
        assert not ({"/q", "/exit", "/?"} & texts)

    def test_aliases_are_found_once_you_type_past_the_slash(self):
        assert "/exit" in {c.text for c in self._complete("/ex")}

    def test_values_complete_after_the_command(self):
        comps = self._complete("/depth ")
        assert [c.text for c in comps] == ["quick", "standard", "deep"]
        assert all(c.display_meta_text for c in comps)  # each value explains itself

    def test_value_completion_filters_by_prefix(self):
        assert [c.text for c in self._complete("/depth d")] == ["deep"]
        assert [c.text for c in self._complete("/template sc")] == ["science"]

    def test_commands_without_values_complete_nothing_after_the_name(self):
        assert self._complete("/academic ") == []

    def test_only_the_first_argument_completes(self):
        assert self._complete("/depth deep ") == []


class TestRunReplRobustness:
    def test_unbalanced_quote_does_not_crash_the_loop(self, monkeypatch, capsys):
        from deep_research.repl import run_repl

        lines = iter(['/fork 1 "unterminated', "/q"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
        run_repl(ReplState())
        out = capsys.readouterr().out
        assert "Could not parse" in out and "Goodbye" in out

    def test_unknown_command_is_reported_with_a_hint_and_loop_continues(self, monkeypatch, capsys):
        from deep_research.repl import run_repl

        lines = iter(["/histroy", "/q"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
        run_repl(ReplState())
        out = capsys.readouterr().out
        assert "Did you mean /history" in out and "Goodbye" in out


class TestAcademicIsShownWhenImplied:
    """Science/literature templates and deep depth switch academic search on
    inside the pipeline; the status bar must say so rather than only
    reflecting the explicit /academic switch."""

    def _bar(self, state):
        from deep_research.repl import _status_fragments
        return "".join(t for _, t in _status_fragments(state))

    @pytest.mark.parametrize("template", ["science", "literature"])
    def test_academic_templates_show_the_tag(self, template):
        assert "academic" in self._bar(ReplState(template=template))

    @pytest.mark.parametrize("template", ["technology", "market", None])
    def test_other_templates_do_not(self, template):
        assert "academic" not in self._bar(ReplState(template=template))

    def test_deep_depth_shows_the_tag(self):
        assert "academic" in self._bar(ReplState(depth=ResearchDepth.DEEP))
        assert "academic" not in self._bar(ReplState(depth=ResearchDepth.QUICK))

    def test_banner_shape_line_follows_the_same_rule(self):
        assert "academic" in ReplState(template="science").shape()
        assert "academic" not in ReplState(template="market").shape()

    def test_status_names_the_reason(self, capsys):
        dispatch("/template science", ReplState())  # no-op on a throwaway state
        s = ReplState(template="science")
        dispatch("/status", s)
        assert "science template" in capsys.readouterr().out

    def test_switching_the_toggle_off_explains_it_is_still_on(self, capsys):
        s = ReplState(template="science", academic=True)
        dispatch("/academic", s)
        out = capsys.readouterr().out
        assert s.academic is False
        assert "still searches" in out and "science template" in out

    def test_reason_matches_what_the_pipeline_decides(self):
        """Guard against drift: the display rule is derived from the same
        template flag and depth preset the pipeline reads."""
        from deep_research.models import ResearchConfig
        from deep_research.options import academic_reason
        from deep_research.pipeline.templates import TEMPLATES

        for key, tmpl in TEMPLATES.items():
            assert (academic_reason("standard", key, False) is not None) == tmpl.use_academic, key
        for depth in ResearchDepth:
            expected = ResearchConfig.from_depth(depth).use_academic_search
            assert (academic_reason(depth.value, None, False) is not None) == expected, depth


class TestPicker:
    """Enter on a bare setting command (`/depth`) opens an arrow-key list."""

    CHOICES = [("quick", "Fast"), ("standard", "Balanced"), ("deep", "Thorough")]

    def _pick(self, keys: str, current="standard"):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from deep_research.repl import pick_option

        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            return pick_option("Depth", self.CHOICES, current, input=pipe, output=DummyOutput())

    def test_enter_selects_the_highlighted_current_value(self):
        assert self._pick("\r") == "standard"

    def test_arrows_move_the_highlight(self):
        assert self._pick("\x1b[B\r") == "deep"       # down
        assert self._pick("\x1b[A\r") == "quick"      # up
        assert self._pick("\x1b[B\x1b[B\r") == "quick"   # wraps past the end
        assert self._pick("\x1b[A\x1b[A\x1b[A\r") == "standard"  # wraps back

    def test_vim_keys_and_number_jump(self):
        assert self._pick("j\r") == "deep"
        assert self._pick("k\r") == "quick"
        assert self._pick("1") == "quick"
        assert self._pick("3") == "deep"

    def test_highlight_starts_on_the_current_value(self):
        assert self._pick("\r", current="deep") == "deep"
        assert self._pick("\r", current="unknown") == "quick"  # falls back to the first

    def test_cancel_keys_return_none(self):
        assert self._pick("\x03") is None   # Ctrl-C
        assert self._pick("q") is None
        assert self._pick("\x1b") is None   # lone Esc

    def test_pick_for_command_turns_a_bare_command_into_a_choice(self, monkeypatch):
        import deep_research.repl as repl_mod

        seen = {}

        def fake_pick(title, choices, current=None, **kw):
            seen.update(title=title, values=[v for v, _ in choices], current=current)
            return "deep"

        monkeypatch.setattr(repl_mod, "pick_option", fake_pick)
        out = repl_mod._pick_for_command("/depth", ReplState(depth=ResearchDepth.QUICK))
        assert out == "/depth deep"
        assert seen == {"title": "Depth", "values": ["quick", "standard", "deep"], "current": "quick"}

    def test_template_picker_offers_none_and_marks_it_when_unset(self, monkeypatch):
        import deep_research.repl as repl_mod

        seen = {}
        monkeypatch.setattr(
            repl_mod, "pick_option",
            lambda title, choices, current=None, **kw: seen.update(
                values=[v for v, _ in choices], current=current) or "none",
        )
        assert repl_mod._pick_for_command("/template", ReplState()) == "/template none"
        assert "none" in seen["values"] and seen["current"] == "none"

    def test_cancelled_picker_returns_none(self, monkeypatch):
        import deep_research.repl as repl_mod

        monkeypatch.setattr(repl_mod, "pick_option", lambda *a, **k: None)
        assert repl_mod._pick_for_command("/style", ReplState()) is None

    @pytest.mark.parametrize("line", [
        "/depth deep",          # already has a value
        "/academic",            # a toggle: no options to pick from
        "/history",             # takes no choice
        "/continue 3",
        "just a question",
    ])
    def test_everything_else_passes_through_untouched(self, monkeypatch, line):
        import deep_research.repl as repl_mod

        def boom(*a, **k):
            raise AssertionError("picker must not open")

        monkeypatch.setattr(repl_mod, "pick_option", boom)
        assert repl_mod._pick_for_command(line, ReplState()) == line

    def test_every_command_with_choices_can_report_its_current_value(self):
        """The picker marks the current value, so each choice command needs a
        getter that returns one of its own values."""
        for cmd in COMMANDS:
            if cmd.choices is None or cmd.interactive is not None:
                continue  # custom flows (/model, /delete) pick their own highlight
            assert cmd.current is not None, cmd.name
            values = [v for v, _ in cmd.choices()]
            if values:
                assert cmd.current(ReplState()) in values, cmd.name


class TestBoxCommit:
    """Only a submitted research question leaves its input box behind;
    settings changes and other slash commands erase theirs, so the screen
    never accumulates boxes."""

    def test_commits_box_rules(self):
        from deep_research.repl import _commits_box

        assert _commits_box("what is CRISPR") is True
        assert _commits_box("  what is CRISPR  ") is True
        assert _commits_box("/depth deep") is False
        assert _commits_box("/q") is False
        assert _commits_box("") is False
        assert _commits_box("   ") is False

    def _prompt_with(self, keys: str):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from deep_research.repl import _BoxedPrompt

        with create_pipe_input() as pipe:
            box = _BoxedPrompt(ReplState(), input=pipe, output=DummyOutput())
            pipe.send_text(keys)
            text = box.prompt()
            return text, box._app.erase_when_done

    def test_slash_command_box_is_erased(self):
        text, erased = self._prompt_with("/depth deep\r")
        assert text == "/depth deep" and erased is True

    def test_unique_prefix_expansion_is_still_a_command(self):
        text, erased = self._prompt_with("/dep\r")
        assert text == "/depth" and erased is True

    def test_research_question_box_is_kept(self):
        text, erased = self._prompt_with("what is CRISPR\r")
        assert text == "what is CRISPR" and erased is False

    def test_empty_enter_leaves_no_box(self):
        text, erased = self._prompt_with("\r")
        assert text == "" and erased is True

    def test_esc_closes_the_command_menu_and_clears_the_slash(self):
        # "/dep" opens the menu; Esc backs out of it completely, so what is
        # typed next is a fresh line, not "/dep..." with a stale menu.
        text, erased = self._prompt_with("/dep\x1bhello\r")
        assert text == "hello" and erased is False

    def test_esc_closes_a_value_menu_too(self):
        text, erased = self._prompt_with("/depth \x1bhello\r")
        assert text == "hello" and erased is False

    def test_enter_on_a_bare_slash_is_not_an_unknown_command(self):
        text, erased = self._prompt_with("/\r")
        assert text == "" and erased is True


# --- transient output: nothing left behind after a / command ----------------

class TestTransientCommands:
    """In a terminal, settings succeed silently and everything else the user
    must read goes through a viewer that Enter/Esc dismisses."""

    @pytest.fixture
    def viewer(self, monkeypatch):
        import deep_research.repl as repl_mod

        shown = []
        monkeypatch.setattr(repl_mod, "show_view", lambda text, **kw: shown.append(text))
        return shown

    def test_successful_setting_prints_and_shows_nothing(self, viewer, capsys):
        from deep_research.repl import _run_transient

        s = ReplState()
        assert _run_transient("/depth deep", s) is False
        assert s.depth == ResearchDepth.DEEP        # it did apply
        assert viewer == []                          # nothing to dismiss
        assert capsys.readouterr().out == ""         # and nothing printed

    def test_a_problem_is_shown_in_the_viewer(self, viewer, capsys):
        from deep_research.repl import _run_transient

        s = ReplState()
        _run_transient("/depth nope", s)
        assert s.depth == ResearchDepth.STANDARD
        assert len(viewer) == 1 and "Unknown depth" in viewer[0]
        assert capsys.readouterr().out == ""         # not leaked to the scrollback

    def test_a_warning_on_a_successful_toggle_is_shown(self, viewer):
        from deep_research.repl import _run_transient

        s = ReplState(template="science", academic=True)
        _run_transient("/academic", s)
        assert s.academic is False
        assert len(viewer) == 1 and "still searches" in viewer[0]

    def test_read_only_screens_always_go_to_the_viewer(self, viewer, capsys):
        from deep_research.repl import _run_transient

        _run_transient("/help", ReplState())
        assert len(viewer) == 1 and "Research" in viewer[0] and "/depth" in viewer[0]
        assert capsys.readouterr().out == ""

    def test_unknown_command_is_reported_in_the_viewer(self, viewer):
        from deep_research.repl import _run_transient

        _run_transient("/histroy", ReplState())
        assert len(viewer) == 1 and "Did you mean /history" in viewer[0]

    def test_unparseable_line_is_reported_in_the_viewer(self, viewer):
        from deep_research.repl import _run_transient

        _run_transient('/fork 1 "oops', ReplState())
        assert len(viewer) == 1 and "Could not parse" in viewer[0]

    def test_quit_still_exits_and_shows_nothing(self, viewer):
        from deep_research.repl import _run_transient

        assert _run_transient("/q", ReplState()) is True
        assert viewer == []

    def test_live_commands_print_normally(self, viewer, capsys, monkeypatch):
        import deep_research.repl as repl_mod
        from deep_research.repl import Command, _run_transient

        def handler(state, args):
            console.print("live output")

        from deep_research.ui import console
        monkeypatch.setitem(repl_mod._COMMAND_INDEX, "/livetest", Command("/livetest", handler, "x", display="live"))
        _run_transient("/livetest", ReplState())
        assert "live output" in capsys.readouterr().out
        assert viewer == []

    def test_display_modes_by_command(self):
        from deep_research.repl import _COMMAND_INDEX

        for name in ("/depth", "/style", "/template", "/academic", "/agent", "/fresh",
                     "/provider", "/model", "/delete"):
            assert _COMMAND_INDEX[name].display == "silent", name
        for name in ("/help", "/status", "/history", "/inspect", "/fork", "/keys"):
            assert _COMMAND_INDEX[name].display == "view", name
        for name in ("/continue", "/resynthesize", "/setup", "/login", "/logout", "/clear", "/quit"):
            assert _COMMAND_INDEX[name].display == "live", name


class TestDeleteConfirmation:
    def _store(self, tmp_path, monkeypatch):
        import deep_research.config as cfg
        import deep_research.repl as repl_mod
        import deep_research.storage.db as db_mod
        from deep_research.models import ResearchReport

        db = tmp_path / "research.db"
        monkeypatch.setattr(cfg, "DB_PATH", db)
        monkeypatch.setattr(db_mod, "DB_PATH", db)
        monkeypatch.setattr(repl_mod, "db_load", db_mod.load_report)
        monkeypatch.setattr(repl_mod, "delete_session", db_mod.delete_session)
        return db_mod.save_report(ResearchReport(query="q"), depth="standard"), db_mod

    def test_picker_yes_confirms_without_a_typed_prompt(self, tmp_path, monkeypatch, capsys):
        import deep_research.repl as repl_mod

        sid, db_mod = self._store(tmp_path, monkeypatch)
        monkeypatch.setattr(repl_mod, "pick_option", lambda *a, **k: "yes")
        monkeypatch.setattr(repl_mod.Confirm, "ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("typed prompt")))
        line = repl_mod._pick_for_command(f"/delete {sid}", ReplState())
        assert line == f"/delete {sid} --yes"
        dispatch(line, ReplState())
        assert db_mod.load_report(sid) is None

    def test_picker_no_or_esc_cancels_and_keeps_the_session(self, tmp_path, monkeypatch):
        import deep_research.repl as repl_mod

        sid, db_mod = self._store(tmp_path, monkeypatch)
        for answer in ("no", None):
            monkeypatch.setattr(repl_mod, "pick_option", lambda *a, **k: answer)
            assert repl_mod._pick_for_command(f"/delete {sid}", ReplState()) is None
        assert db_mod.load_report(sid) is not None

    def test_missing_session_skips_the_picker(self, tmp_path, monkeypatch, capsys):
        import deep_research.repl as repl_mod

        self._store(tmp_path, monkeypatch)
        monkeypatch.setattr(repl_mod, "pick_option", lambda *a, **k: (_ for _ in ()).throw(AssertionError("picker")))
        assert repl_mod._pick_for_command("/delete 999", ReplState()) == "/delete 999"

    def test_typed_confirmation_still_works_outside_a_terminal(self, tmp_path, monkeypatch, capsys):
        import deep_research.repl as repl_mod

        sid, db_mod = self._store(tmp_path, monkeypatch)
        monkeypatch.setattr(repl_mod.Confirm, "ask", lambda *a, **k: False)
        dispatch(f"/delete {sid}", ReplState())
        assert db_mod.load_report(sid) is not None
        assert "Cancelled" in capsys.readouterr().out


# --- /model ---------------------------------------------------------------

class TestModelCommand:
    """/model chooses the model for each stage and saves it."""

    @pytest.fixture
    def cfg(self, monkeypatch):
        import deep_research.repl as repl_mod
        from deep_research.config import Settings

        custom = Settings(
            llm_provider="zai", zai_api_key="x",
            model_decompose="", model_analyze="", model_synthesize="", model_gap_analysis="",
        )
        monkeypatch.setattr(repl_mod, "settings", custom)
        return custom

    def test_set_one_stage_and_it_is_saved(self, cfg):
        from deep_research import userconfig

        dispatch("/model analyze glm-5.1", ReplState())
        assert userconfig.stage_models("zai") == {"analyze": "glm-5.1"}
        assert cfg.effective_stage_models()["analyze"] == "glm-5.1"
        assert cfg.effective_stage_models()["synthesize"] == "glm-5.3"  # the rest stay recommended
        assert cfg.stage_sources()["analyze"] == "saved"

    def test_set_every_stage(self, cfg):
        dispatch("/model all glm-5.2", ReplState())
        assert set(cfg.effective_stage_models().values()) == {"glm-5.2"}

    def test_default_returns_a_stage_to_the_recommendation(self, cfg):
        from deep_research import userconfig

        dispatch("/model analyze glm-5.2", ReplState())
        dispatch("/model analyze default", ReplState())
        assert userconfig.stage_models("zai") == {}
        assert cfg.effective_stage_models()["analyze"] == "glm-5.3-flash"

    def test_stage_names_accept_dashes(self, cfg):
        dispatch("/model gap-analysis glm-5.2", ReplState())
        assert cfg.effective_stage_models()["gap_analysis"] == "glm-5.2"

    def test_a_model_from_another_provider_is_rejected(self, cfg, capsys):
        from deep_research import userconfig

        dispatch("/model synthesize kimi-k2.6", ReplState())
        out = capsys.readouterr().out
        assert userconfig.stage_models("zai") == {}
        assert "opencode-go" in out and "not zai" in out

    def test_an_unregistered_model_is_allowed_with_a_warning(self, cfg, capsys):
        dispatch("/model analyze glm-9.9", ReplState())
        assert cfg.effective_stage_models()["analyze"] == "glm-9.9"
        assert "not in the capability registry" in capsys.readouterr().out

    def test_a_legacy_registered_model_is_accepted_without_a_warning(self, cfg, capsys):
        dispatch("/model decompose glm-4.5-air", ReplState())
        assert cfg.effective_stage_models()["decompose"] == "glm-4.5-air"
        assert "registry" not in capsys.readouterr().out

    def test_unknown_stage_and_missing_model_are_explained(self, cfg, capsys):
        from deep_research import userconfig

        dispatch("/model nonsense glm-5.2", ReplState())
        dispatch("/model analyze", ReplState())
        out = capsys.readouterr().out
        assert "Unknown stage" in out and "Usage: /model analyze" in out
        assert userconfig.stage_models("zai") == {}

    def test_an_invalid_choice_rolls_back_to_what_was_saved(self, cfg, monkeypatch):
        from deep_research import userconfig
        from deep_research.capabilities import CapabilityViolation

        dispatch("/model analyze glm-5.2", ReplState())

        def boom(self):
            raise CapabilityViolation("nope")

        # pydantic models reject instance-level method patches; patch the class.
        monkeypatch.setattr(type(cfg), "validate_routes", boom)
        dispatch("/model analyze glm-5.1", ReplState())
        assert userconfig.stage_models("zai") == {"analyze": "glm-5.2"}

    def test_an_environment_override_is_called_out(self, cfg, monkeypatch, capsys):
        monkeypatch.setenv("MODEL_ANALYZE", "glm-5.3")
        dispatch("/model analyze glm-5.2", ReplState())
        assert "MODEL_ANALYZE" in capsys.readouterr().out

    def test_bare_command_lists_stage_models_and_where_they_come_from(self, cfg, capsys):
        dispatch("/model analyze glm-5.2", ReplState())
        capsys.readouterr()
        dispatch("/model", ReplState())
        out = capsys.readouterr().out
        for stage in ("decompose", "analyze", "gap_analysis", "synthesize"):
            assert stage in out
        assert "saved" in out and "recommended" in out

    def test_interactive_flow_picks_a_stage_then_a_model(self, cfg, monkeypatch):
        import deep_research.repl as repl_mod

        titles, answers = [], iter(["analyze", "glm-5.2"])

        def fake(title, choices, current=None, **kw):
            titles.append((title, [v for v, _ in choices], current))
            return next(answers)

        monkeypatch.setattr(repl_mod, "pick_option", fake)
        line = repl_mod._pick_for_command("/model", ReplState())
        assert line == "/model analyze glm-5.2"
        assert titles[0][1][0] == "all" and "analyze" in titles[0][1]
        assert titles[1][0] == "Model for analyze"
        assert titles[1][1][:2] == ["glm-5.3", "glm-5.3-flash"] and titles[1][1][-2:] == ["other", "default"]
        assert titles[1][2] == "glm-5.3-flash"      # highlight starts on the stage's current model

    def test_the_stage_picker_says_what_each_stage_does(self, cfg):
        import deep_research.repl as repl_mod

        rows = dict(repl_mod._model_stage_choices())
        assert "reads each source" in rows["analyze"] and "glm-5.3-flash" in rows["analyze"]
        assert "recommended" in rows["analyze"]

    def test_other_lets_the_user_type_a_model_id(self, cfg, monkeypatch):
        import deep_research.repl as repl_mod

        monkeypatch.setattr(repl_mod, "pick_option", lambda *a, **k: "other")
        monkeypatch.setattr(repl_mod, "ask_text", lambda *a, **k: "  glm-9.9  ")
        assert repl_mod._pick_for_command("/model analyze", ReplState()) == "/model analyze glm-9.9"

    def test_cancelling_the_typed_id_cancels(self, cfg, monkeypatch):
        import deep_research.repl as repl_mod

        monkeypatch.setattr(repl_mod, "pick_option", lambda *a, **k: "other")
        monkeypatch.setattr(repl_mod, "ask_text", lambda *a, **k: None)
        assert repl_mod._pick_for_command("/model analyze", ReplState()) is None

    def test_interactive_flow_with_the_stage_typed_asks_only_for_the_model(self, cfg, monkeypatch):
        import deep_research.repl as repl_mod

        calls = []
        monkeypatch.setattr(repl_mod, "pick_option", lambda t, c, cur=None, **k: calls.append(t) or "glm-5.2")
        assert repl_mod._pick_for_command("/model synthesize", ReplState()) == "/model synthesize glm-5.2"
        assert calls == ["Model for synthesize"]

    @pytest.mark.parametrize("cancel_at", [0, 1])
    def test_cancelling_either_step_cancels(self, cfg, monkeypatch, cancel_at):
        import deep_research.repl as repl_mod

        answers = iter(["analyze", "glm-5.2"])
        step = {"n": 0}

        def fake(*a, **k):
            n, step["n"] = step["n"], step["n"] + 1
            return None if n == cancel_at else next(answers)

        monkeypatch.setattr(repl_mod, "pick_option", fake)
        assert repl_mod._pick_for_command("/model", ReplState()) is None

    def test_a_fully_typed_line_skips_the_pickers(self, cfg, monkeypatch):
        import deep_research.repl as repl_mod

        monkeypatch.setattr(repl_mod, "pick_option", lambda *a, **k: (_ for _ in ()).throw(AssertionError("picker")))
        assert repl_mod._pick_for_command("/model analyze glm-5.2", ReplState()) == "/model analyze glm-5.2"

    def test_completion_walks_stage_then_model(self, cfg):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from deep_research.repl import _command_completer

        def complete(text):
            return [c.text for c in _command_completer().get_completions(Document(text, len(text)), CompleteEvent())]

        assert complete("/model ")[0] == "all" and "synthesize" in complete("/model ")
        models = complete("/model analyze ")
        assert models[:2] == ["glm-5.3", "glm-5.3-flash"] and models[-1] == "default" and "other" not in models
        assert complete("/model analyze glm-5.3-f") == ["glm-5.3-flash", "glm-5.3-flashx"]
        assert complete("/model analyze glm-5.3 ") == []
