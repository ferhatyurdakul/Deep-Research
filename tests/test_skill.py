"""
The bundled agent skill (src/synapse_deep_research/skill/SKILL.md).

A skill is prose that tells another agent which flags to type and what comes back.
It rots silently when the CLI changes, so these tests read the skill text and
hold it against the real parsers, the real option values and the real JSON.
"""
from __future__ import annotations

import json
import re
from importlib import resources

import pytest

from synapse_deep_research import main as main_mod
from synapse_deep_research import views
from synapse_deep_research.cli_help import INDEX
from synapse_deep_research.models import ResearchConfig, ResearchDepth, ResearchReport
from synapse_deep_research.options import DEPTH_VALUES, STYLE_VALUES, TEMPLATE_VALUES

PROG = "synapse-deep-research"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return resources.files("synapse_deep_research").joinpath("skill").joinpath("SKILL.md").read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with a --- frontmatter block"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


class TestFrontmatter:
    """The open Agent Skills format: name matches the folder it is installed into,
    lowercase letters, digits and hyphens, and a description that says when to use it."""

    def test_name_is_the_folder_name_and_well_formed(self, skill_text):
        name = _frontmatter(skill_text)["name"]
        assert name == PROG
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64

    def test_description_says_when_to_use_it_and_when_not_to(self, skill_text):
        description = _frontmatter(skill_text)["description"]
        assert 0 < len(description) <= 1024
        assert "Use it when" in description and "Do not use it" in description


class TestAgainstTheCli:
    def test_every_flag_it_mentions_exists(self, skill_text):
        known = {"-h", "--help", "--version"}
        for parser in [main_mod.build_run_parser()] + [main_mod.build_command_parser(n) for n in INDEX]:
            for action in parser._actions:
                known.update(action.option_strings)
        mentioned = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", skill_text))
        assert mentioned, "the skill should show flags"
        assert mentioned <= known, f"flags in SKILL.md the CLI does not accept: {sorted(mentioned - known)}"

    def test_every_command_it_runs_exists(self, skill_text):
        commands = set(re.findall(rf"`{PROG} ([a-z]+)[ `]", skill_text)) - {"--json"}
        assert {"setup", "inspect", "continue", "resynthesize", "history"} <= commands
        assert commands <= set(INDEX), f"commands in SKILL.md that do not exist: {sorted(commands - set(INDEX))}"

    def test_the_flags_it_puts_on_a_command_are_accepted_by_that_command(self, skill_text):
        for command in ("continue", "resynthesize"):
            line = next(l for l in skill_text.splitlines() if f"`{PROG} {command} " in l)
            argv = re.search(rf"`{PROG} {command} ([^`]+)`", line).group(1)
            argv = re.sub(r"<[^>]+>", "3", argv).split()  # <session_id> -> a number
            main_mod.build_command_parser(command).parse_args(argv)  # exits (SystemExit) on a bad flag

    def test_it_lists_exactly_the_values_the_cli_has(self, skill_text):
        for value in DEPTH_VALUES + STYLE_VALUES + TEMPLATE_VALUES:
            assert f"`{value}`" in skill_text, f"SKILL.md does not mention `{value}`"

    @pytest.mark.parametrize("depth", list(ResearchDepth))
    def test_the_depth_numbers_are_the_real_presets(self, skill_text, depth):
        cfg = ResearchConfig.from_depth(depth)
        match = re.search(rf"{depth.value}: (\d+)(?: sub-questions)?, (\d+) pass", skill_text)
        assert match, f"SKILL.md does not describe the {depth.value} preset"
        assert (int(match.group(1)), int(match.group(2))) == (cfg.max_sub_questions, cfg.max_iterations)

    def test_the_json_it_shows_has_the_keys_a_run_really_prints(self, skill_text, tmp_path):
        block = re.search(r"Success:\n\n```json\n(.*?)\n```", skill_text, re.DOTALL).group(1)
        shown = json.loads(block)
        real = views.result_payload(
            ResearchReport(query="q"), session_id=1, path=tmp_path / "r.md", depth="quick",
        )
        assert set(shown) == set(real)
