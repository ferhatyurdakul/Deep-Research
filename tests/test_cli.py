"""
Tests for the CLI surface: registry-driven help, parser errors, shared views,
and the rendering helpers behind them.

Help is generated from the command registry and the argparse parsers, so the
tests here guard the invariants that keep it honest: every command is
documented, every option has help text, every documented value exists.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research import main as main_mod
from deep_research import views
from deep_research.cli_help import COMMANDS, GROUPS, INDEX
from deep_research.models import (
    ResearchIteration, ResearchReport, SourceAnalysis, SubQuestion,
)
from deep_research.options import (
    CHOICE_DOCS, DEPTH_VALUES, STYLE_VALUES, TEMPLATE_VALUES,
)


@pytest.fixture(autouse=True)
def _restore_logging():
    """main() installs the CLI log handler and level on the root logger;
    undo that after every test so it can't leak into the rest of the suite."""
    import logging

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _run(argv, monkeypatch, capsys):
    """Run main() in-process; returns (exit_code, stdout, stderr)."""
    monkeypatch.setattr(sys, "argv", ["deep-research", *argv])
    code = 0
    try:
        main_mod.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the session DB and the markdown output dir at tmp_path."""
    import deep_research.config as cfg
    import deep_research.output.report as report_mod
    import deep_research.storage.db as db_mod

    db = tmp_path / "research.db"
    monkeypatch.setattr(cfg, "DB_PATH", db)
    monkeypatch.setattr(db_mod, "DB_PATH", db)
    monkeypatch.setattr(report_mod, "OUTPUTS_DIR", tmp_path / "outputs")
    return tmp_path


def _report(query="What is X?") -> ResearchReport:
    return ResearchReport(
        query=query,
        sub_questions=[SubQuestion(question="Sub one?")],
        sources=[SourceAnalysis(url="https://e.com/1", title="Source one", key_findings=["f"])],
        iterations=[ResearchIteration(iteration_number=1)],
        executive_summary="Summary.",
        detailed_findings="## Body\n\nText.",
        follow_up_questions=["Next question?"],
    )


# --- registry / help invariants ----------------------------------------------

class TestRegistry:
    def test_every_command_has_a_handler_and_a_known_group(self):
        assert set(main_mod.HANDLERS) == set(INDEX)
        assert {c.group for c in COMMANDS} <= set(GROUPS)

    def test_command_names_are_unique(self):
        names = [c.name for c in COMMANDS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("name", [c.name for c in COMMANDS])
    def test_every_option_and_argument_is_documented(self, name):
        parser = main_mod.build_command_parser(name)
        for action in parser._actions:
            assert action.help, f"{name}: {action.dest or action.option_strings} has no help text"

    def test_top_level_options_are_documented(self):
        for action in main_mod.build_run_parser()._actions:
            assert action.help, action.dest or action.option_strings

    def test_choice_docs_cover_exactly_the_valid_values(self):
        assert [v for v, _ in CHOICE_DOCS["depth"]()] == DEPTH_VALUES
        assert [v for v, _ in CHOICE_DOCS["style"]()] == STYLE_VALUES
        assert [v for v, _ in CHOICE_DOCS["template"]()] == TEMPLATE_VALUES

    def test_documented_choices_match_parser_choices(self):
        parser = main_mod.build_run_parser()
        by_dest = {a.dest: a for a in parser._actions}
        assert list(by_dest["depth"].choices) == DEPTH_VALUES
        assert list(by_dest["style"].choices) == STYLE_VALUES
        assert list(by_dest["template"].choices) == TEMPLATE_VALUES


class TestHelpOutput:
    def test_top_level_help_lists_every_command_and_option(self, monkeypatch, capsys):
        code, out, _ = _run(["--help"], monkeypatch, capsys)
        assert code == 0
        for cmd in COMMANDS:
            assert cmd.name in out, cmd.name
        for flag in ("--depth", "--style", "--template", "--academic", "--agent", "--fresh",
                     "--help", "--version"):
            assert flag in out, flag
        assert "Examples" in out

    def test_top_level_help_explains_option_values(self, monkeypatch, capsys):
        _, out, _ = _run(["--help"], monkeypatch, capsys)
        for value in DEPTH_VALUES + STYLE_VALUES + TEMPLATE_VALUES:
            assert value in out, value
        assert "sub-questions" in out  # the depth blurbs, not just the names

    def test_short_help_flag(self, monkeypatch, capsys):
        code, out, _ = _run(["-h"], monkeypatch, capsys)
        assert code == 0 and "Usage" in out

    @pytest.mark.parametrize("name", [c.name for c in COMMANDS])
    def test_command_help_exits_zero_with_usage(self, name, monkeypatch, capsys):
        code, out, _ = _run([name, "--help"], monkeypatch, capsys)
        assert code == 0
        assert f"deep-research {name}" in out
        assert "Usage" in out

    def test_help_command_matches_flag(self, monkeypatch, capsys):
        _, via_flag, _ = _run(["continue", "--help"], monkeypatch, capsys)
        _, via_cmd, _ = _run(["help", "continue"], monkeypatch, capsys)
        assert via_flag == via_cmd

    def test_help_for_unknown_command_lists_commands(self, monkeypatch, capsys):
        code, out, _ = _run(["help", "bogus"], monkeypatch, capsys)
        assert code == 2
        assert "Unknown command" in out and "history" in out

    def test_version_flag(self, monkeypatch, capsys):
        from deep_research import __version__
        code, out, _ = _run(["--version"], monkeypatch, capsys)
        assert code == 0 and __version__ in out

    def test_shape_flags_appear_on_continue_but_agent_does_not(self, monkeypatch, capsys):
        _, out, _ = _run(["continue", "--help"], monkeypatch, capsys)
        assert "--depth" in out and "--academic" in out
        assert "--agent" not in out  # /continue ignores agent mode


class TestParserErrors:
    def test_bad_choice_exits_2_with_usage_on_stderr(self, monkeypatch, capsys):
        code, out, err = _run(["--depth", "ultra", "q"], monkeypatch, capsys)
        assert code == 2
        assert "invalid choice" in err and "usage:" in err and "--help" in err
        assert out == ""

    def test_non_integer_id(self, monkeypatch, capsys):
        code, _, err = _run(["inspect", "abc"], monkeypatch, capsys)
        assert code == 2
        assert "usage: deep-research inspect ID" in err

    def test_legacy_flags_still_rejected(self, monkeypatch, capsys):
        code, _, err = _run(["--history"], monkeypatch, capsys)
        assert code == 2 and "unrecognized" in err

    def test_error_text_is_not_parsed_as_markup(self, monkeypatch, capsys):
        # A bad value containing rich-markup-looking text must print verbatim.
        code, _, err = _run(["inspect", "[/bold]"], monkeypatch, capsys)
        assert code == 2
        assert "[/bold]" in err


# --- shared views ------------------------------------------------------------

class TestViews:
    def test_history_empty(self, isolated_store, capsys):
        views.print_history()
        out = capsys.readouterr().out
        assert "No research sessions found" in out

    def test_history_shows_fork_lineage_and_survives_markup_in_query(self, isolated_store, capsys):
        from deep_research.storage.db import save_report

        a = save_report(_report("plain query"), depth="quick")
        save_report(_report("has [bold]markup[/] and [/oops]"), depth="deep", parent_session_id=a)
        views.print_history()
        out = capsys.readouterr().out
        assert "plain query" in out
        assert "[bold]markup[/]" in out  # printed literally, not interpreted
        assert "↳" in out or "^" in out  # the fork column

    def test_history_repl_hints_use_slash_commands(self, isolated_store, capsys):
        from deep_research.storage.db import save_report

        save_report(_report(), depth="standard")
        views.print_history(repl=True)
        assert "/inspect" in capsys.readouterr().out

    def test_inspect_session_and_source(self, isolated_store, capsys):
        from deep_research.storage.db import save_report

        sid = save_report(_report(), depth="standard")
        views.print_session(main_mod.db_load(sid), sid)
        out = capsys.readouterr().out
        assert "Session" in out and "Source one" in out and "Next question?" in out

        assert views.print_source(main_mod.db_load(sid), 1) is True
        assert views.print_source(main_mod.db_load(sid), 9) is False
        assert "out of range" in capsys.readouterr().out

    def test_status_renders_without_credentials(self, capsys):
        views.print_status(shape=[("depth", "deep")])
        out = capsys.readouterr().out
        for title in ("Provider", "Stage models", "Credentials", "Files", "This session"):
            assert title in out

    def test_banner_shows_query_and_shape(self, capsys):
        views.print_banner(shape="deep depth", query="my [question]")
        out = capsys.readouterr().out
        assert "Deep Research" in out and "deep depth" in out and "my [question]" in out

    def test_shape_summary_only_lists_enabled_toggles(self):
        assert views.shape_summary("standard", "standard") == "standard depth · standard style"
        text = views.shape_summary("deep", "brief", "science", academic=True, agent=True)
        assert "science template" in text and "academic" in text and "agent" in text
        assert "fresh" not in text

    def test_shape_summary_shows_academic_when_the_template_or_depth_implies_it(self):
        assert "academic" in views.shape_summary("standard", "standard", "science")
        assert "academic" in views.shape_summary("deep", "standard")
        assert "academic" not in views.shape_summary("standard", "standard", "market")

    def test_present_report_saves_and_prints_card(self, isolated_store, capsys):
        from deep_research.storage.db import load_report

        sid = views.present_report(_report(), depth="standard", started=None)
        out = capsys.readouterr().out
        assert load_report(sid) is not None
        assert "Saved" in out and f"continue {sid}" in out
        assert any((isolated_store / "outputs").glob("*.md"))

    def test_present_report_records_parent_and_relation(self, isolated_store, capsys):
        from deep_research.storage.db import load_report, save_report

        parent = save_report(_report(), depth="standard")
        sid = views.present_report(
            _report(), depth="standard", parent_session_id=parent,
            relation=f"continues session {parent}", repl=True,
        )
        out = capsys.readouterr().out
        assert load_report(sid).parent_session_id == parent
        assert f"continues session {parent}" in out
        assert f"/continue {sid}" in out  # REPL-flavoured next steps

    def test_display_report_numbers_follow_ups(self, capsys):
        from deep_research.output.report import display_report

        display_report(_report())
        out = capsys.readouterr().out
        assert "1." in out and "Next question?" in out


# --- main flows --------------------------------------------------------------

class TestMainFlows:
    def test_async_main_runs_saves_and_reports(self, isolated_store, monkeypatch, capsys):
        import asyncio
        from deep_research.models import ResearchConfig, ResearchDepth

        async def fake(query, config, agent):
            return _report(query)

        monkeypatch.setattr(main_mod, "research_query", fake)
        cfg = ResearchConfig.from_depth(ResearchDepth.QUICK)
        asyncio.run(main_mod.async_main("a question", cfg))
        out = capsys.readouterr().out
        assert "Saved" in out
        assert main_mod.db_load(1).query == "a question"

    def test_async_main_failure_prints_error_and_exits_1(self, isolated_store, monkeypatch, capsys):
        import asyncio
        from deep_research.models import ResearchConfig, ResearchDepth

        async def boom(query, config, agent):
            raise RuntimeError("provider said [/nope]")

        monkeypatch.setattr(main_mod, "research_query", boom)
        with pytest.raises(SystemExit) as ei:
            asyncio.run(main_mod.async_main("q", ResearchConfig.from_depth(ResearchDepth.QUICK)))
        assert ei.value.code == 1
        assert "provider said [/nope]" in capsys.readouterr().out

    def test_no_query_after_strip_is_rejected(self, monkeypatch, capsys):
        code, out, _ = _run([""], monkeypatch, capsys)
        assert code == 1 and "No query provided" in out

    def test_fork_missing_session_still_reports_not_found(self, isolated_store, monkeypatch, capsys):
        code, out, _ = _run(["fork", "999"], monkeypatch, capsys)
        assert code == 1 and "not found" in out

    def test_status_command(self, monkeypatch, capsys):
        code, out, _ = _run(["status"], monkeypatch, capsys)
        assert code == 0 and "Stage models" in out


# --- ui helpers --------------------------------------------------------------

class TestUiHelpers:
    def test_fmt_duration(self):
        from deep_research.ui import fmt_duration
        assert fmt_duration(0) == "0s"
        assert fmt_duration(42.4) == "42s"
        assert fmt_duration(102) == "1m 42s"
        assert fmt_duration(3725) == "1h 02m"

    def test_plural(self):
        from deep_research.ui import plural
        assert plural(1, "source") == "1 source"
        assert plural(0, "source") == "0 sources"
        assert plural(3, "gap") == "3 gaps"

    def test_truncate_collapses_whitespace(self):
        from deep_research.ui import truncate
        assert truncate("a\nb   c", 10) == "a b c"
        assert truncate("x" * 20, 10) == "x" * 9 + "…"

    def test_short_path_prefers_cwd_then_home(self, tmp_path, monkeypatch):
        from deep_research.ui import short_path
        monkeypatch.chdir(tmp_path)
        assert short_path(tmp_path / "outputs" / "r.md") == os.path.join("outputs", "r.md")
        elsewhere = os.path.expanduser("~/definitely/elsewhere/r.md")
        monkeypatch.chdir(tmp_path)
        assert short_path(__import__("pathlib").Path(elsewhere)) == "~/definitely/elsewhere/r.md"

    def test_glyphs_fall_back_to_ascii(self, monkeypatch):
        from deep_research import ui
        monkeypatch.setattr(type(ui.console), "encoding", property(lambda self: "ascii"))
        assert ui.glyph("ok") == "+"
        monkeypatch.setattr(type(ui.console), "encoding", property(lambda self: "utf-8"))
        assert ui.glyph("ok") == "✓"


class TestLogging:
    def _fresh_root(self, monkeypatch):
        """Root logger with none of our own handlers (restored by the autouse fixture)."""
        import logging
        root = logging.getLogger()
        root.handlers[:] = [h for h in root.handlers if not getattr(h, "_deep_research_cli", False)]
        return root

    def test_cli_logging_goes_through_the_console_without_markup(self, monkeypatch, capsys):
        import logging

        from deep_research.logging_config import setup_logging
        from deep_research.ui import console

        self._fresh_root(monkeypatch)
        setup_logging("WARNING", console=console)
        logging.getLogger("deep_research.test").warning("fetch failed for [bold]x[/bold]")
        logging.getLogger("deep_research.test").info("hidden at WARNING")
        out = capsys.readouterr().out
        assert "fetch failed for [bold]x[/bold]" in out  # literal, not interpreted
        assert "hidden at WARNING" not in out

    def test_setup_is_idempotent(self, monkeypatch):
        import logging

        from deep_research.logging_config import setup_logging
        from deep_research.ui import console

        root = self._fresh_root(monkeypatch)
        before = len(root.handlers)
        setup_logging("WARNING", console=console)
        setup_logging("INFO", console=console)
        ours = [h for h in root.handlers if getattr(h, "_deep_research_cli", False)]
        assert len(ours) == 1 and len(root.handlers) == before + 1
        assert root.level == logging.INFO

    def test_default_cli_level_is_warning_and_env_overrides(self, monkeypatch):
        import logging

        self._fresh_root(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["deep-research", "--version"])
        monkeypatch.delenv("DEEP_RESEARCH_LOG_LEVEL", raising=False)
        with pytest.raises(SystemExit):
            main_mod.main()
        assert logging.getLogger().level == logging.WARNING

        monkeypatch.setenv("DEEP_RESEARCH_LOG_LEVEL", "DEBUG")
        with pytest.raises(SystemExit):
            main_mod.main()
        assert logging.getLogger().level == logging.DEBUG


class TestStartupCost:
    def test_help_does_not_import_the_llm_stack(self):
        """--help / history / status must stay snappy: the LLM and search
        client libraries are only imported when a run actually starts."""
        import subprocess

        code = (
            "import sys; sys.argv=['deep-research','--help'];"
            "from deep_research import main\n"
            "try:\n    main.main()\n"
            "except SystemExit: pass\n"
            "heavy=[m for m in ('openai','anthropic','tavily','deep_research.pipeline.orchestrator',"
            "'deep_research.pipeline.agent') if m in sys.modules]\n"
            "print('HEAVY:', heavy)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        )
        assert "HEAVY: []" in result.stdout, result.stdout[-400:] + result.stderr[-400:]


# --- viewer / picker plumbing, model helpers -----------------------------------

class TestViewer:
    def test_view_lines_trims_blank_edges_and_keeps_inner_blanks(self):
        from deep_research.interactive import view_lines

        lines = view_lines("\n\nfirst\n\nsecond\n\n")
        assert ["".join(t for _, t in l) for l in lines] == ["first", "", "second"]

    def test_hyperlink_escapes_are_stripped(self):
        from deep_research.interactive import view_lines

        linked = "see \x1b]8;id=1;https://e.com\x1b\\the docs\x1b]8;;\x1b\\ now"
        text = "".join(t for _, t in view_lines(linked)[0])
        assert text == "see the docs now"

    def test_ansi_colors_survive_as_styles(self):
        from deep_research.interactive import view_lines

        frags = view_lines("\x1b[1;36mCyan\x1b[0m plain")[0]
        assert any("ansicyan" in style for style, _ in frags)

    @pytest.mark.parametrize("keys", ["\r", "\x1b", "q", "\x03"])
    def test_every_close_key_dismisses(self, keys):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from deep_research.interactive import show_view

        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            show_view("hello\nworld", input=pipe, output=DummyOutput())  # returns = closed

    def test_scrolling_a_tall_view_then_closing(self):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from deep_research.interactive import show_view

        text = "\n".join(f"line {i}" for i in range(60))
        with create_pipe_input() as pipe:
            pipe.send_text("jjj\x1b[B\x1b[6~\x1b[5~G g\r")  # down, PgDn, PgUp, end, home, then close
            show_view(text, rows=10, input=pipe, output=DummyOutput())

    def test_empty_output_is_not_shown(self):
        from deep_research.interactive import show_view
        show_view("   \n\n")  # would hang waiting for a key if it opened


class TestProblemCounter:
    def test_only_warnings_and_failures_count(self):
        from deep_research import ui

        before = ui.problem_count()
        ui.ok("fine")
        ui.note("fyi")
        assert ui.problem_count() == before
        ui.warn("hmm")
        ui.fail("no")
        assert ui.problem_count() == before + 2


class TestModelHelpers:
    def test_model_choices_list_current_models_flagship_first_and_describe_each(self):
        from deep_research.options import model_choices

        rows = model_choices("zai")
        assert [m for m, _ in rows] == ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx", "glm-5.2", "glm-5.1"]
        blurbs = dict(rows)
        assert "production" in blurbs["glm-5.3"] and "thinking" in blurbs["glm-5.3-flash"]
        assert "glm-4.5-air" not in blurbs  # legacy models are registered but not offered

    def test_model_choices_tag_the_recommendation_for_a_stage(self):
        from deep_research.options import model_choices

        analyze = dict(model_choices("zai", "analyze"))
        assert analyze["glm-5.3-flash"].startswith("recommended")
        assert not analyze["glm-5.3"].startswith("recommended")
        synth = dict(model_choices("zai", "synthesize"))
        assert synth["glm-5.3"].startswith("recommended")

    def test_stage_summary(self):
        from deep_research.options import stage_summary

        same = {"decompose": "a", "analyze": "a", "gap_analysis": "a", "synthesize": "a"}
        assert stage_summary(same) == "a"
        mixed = {"decompose": "air", "analyze": "big", "gap_analysis": "big", "synthesize": "big"}
        assert stage_summary(mixed) == "big · air for decompose"
        two = {"decompose": "air", "analyze": "air", "gap_analysis": "big", "synthesize": "big"}
        assert stage_summary(two) == "air · big for gap+synthesize"   # tie -> first stage's model leads
        assert stage_summary({}) == "?"

    def test_banner_shows_what_each_stage_uses(self, monkeypatch, capsys):
        from deep_research.config import Settings

        monkeypatch.setattr(views, "settings", Settings(llm_provider="zai", zai_api_key="x"))
        views.print_banner(shape="standard depth")
        out = " ".join(capsys.readouterr().out.split())
        assert "glm-5.3-flash · glm-5.3 for gap+synthesize via zai" in out
