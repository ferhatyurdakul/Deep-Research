"""
`synapse-deep-research skills`: copying the bundled skill into other agents.

Every test uses stand-in agents and a temp home/project, so nothing here can
touch a real agent's skills folder.
"""
from __future__ import annotations

import sys

import pytest

from synapse_deep_research import main as main_mod
from synapse_deep_research import skills_install as skills
from synapse_deep_research.skills_install import SKILL_NAME, Agent, SkillsError, Target

REAL_AGENTS = skills.AGENTS  # captured before the autouse fixture swaps in stand-ins
ALPHA = Agent("alpha", "Alpha", ".alpha/skills", ".alpha/skills", ".alpha")
BETA = Agent("beta", "Beta Agent", ".config/beta/skills", ".beta/skills", ".config/beta")


@pytest.fixture(autouse=True)
def _two_agents(monkeypatch):
    monkeypatch.setattr(skills, "AGENTS", (ALPHA, BETA))


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def project(tmp_path, monkeypatch):
    path = tmp_path / "project"
    path.mkdir()
    monkeypatch.chdir(path)
    return path


def _target(tmp_path, name="agent") -> Target:
    return Target("Agent", tmp_path / name / "skills" / SKILL_NAME)


class TestTheRealAgentTable:
    """The paths below are what each agent's own documentation says (checked 2026-11):
    Claude Code reads .claude/skills, Hermes ~/.hermes/skills, and Codex (like Gemini CLI,
    Cursor, Copilot and OpenCode) the shared ~/.agents/skills."""

    @pytest.fixture(autouse=True)
    def _real(self, monkeypatch):
        monkeypatch.setattr(skills, "AGENTS", REAL_AGENTS)

    def test_the_documented_user_folders(self, home, project):
        def folder(key):
            return skills.choose_targets([key], project=False, home=home, cwd=project)[0].path

        assert folder("claude") == home / ".claude/skills" / SKILL_NAME
        assert folder("hermes") == home / ".hermes/skills" / SKILL_NAME
        for key in ("codex", "gemini", "cursor", "copilot", "opencode"):
            assert folder(key) == home / ".agents/skills" / SKILL_NAME

    def test_the_documented_project_folders(self, home, project):
        def folder(key):
            return skills.choose_targets([key], project=True, home=home, cwd=project)[0].path

        assert folder("claude") == project / ".claude/skills" / SKILL_NAME
        assert folder("hermes") == project / ".hermes/skills" / SKILL_NAME
        assert folder("codex") == project / ".agents/skills" / SKILL_NAME

    def test_agents_that_read_the_same_folder_share_one_copy(self, home, project):
        targets = skills.choose_targets(["codex", "gemini", "claude"], project=False, home=home, cwd=project)
        assert {t.label for t in targets} == {"Codex, Gemini CLI", "Claude Code"}
        assert len(targets) == 2

    def test_all_means_one_copy_per_distinct_folder(self, home, project):
        targets = skills.choose_targets(["all"], project=False, home=home, cwd=project)
        assert len(targets) == 3  # claude, hermes and the shared .agents folder

    def test_the_list_has_one_row_per_folder_with_every_agent_that_reads_it(self, home, project, monkeypatch, capsys):
        (home / ".codex").mkdir()
        code, out, _ = _run(["skills"], monkeypatch, capsys)
        out = " ".join(out.replace("│", " ").split())  # long rows wrap inside the panel at test width
        assert code == 0
        assert out.count("not installed") == 3  # Claude Code, the shared folder, Hermes
        assert "Codex, Gemini CLI, Cursor, GitHub Copilot, OpenCode not installed" in out
        assert "Claude Code not installed (agent not found)" in out
        assert "Codex, Gemini CLI, Cursor, GitHub Copilot, OpenCode not installed (agent not found)" not in out  # Codex is there

    def test_keys_are_unique_and_the_skill_name_is_the_folder_name(self):
        assert len({a.key for a in REAL_AGENTS}) == len(REAL_AGENTS)
        assert {"claude", "codex", "hermes"} <= {a.key for a in REAL_AGENTS}


class TestBundledSkill:
    def test_the_skill_ships_with_the_package(self):
        files = skills.bundled_files()
        assert list(files) == ["SKILL.md"]
        assert files["SKILL.md"].startswith(b"---\nname: " + SKILL_NAME.encode())


class TestChoosingTargets:
    def test_defaults_to_the_agents_found_in_the_home_folder(self, home, project):
        (home / ".alpha").mkdir()
        targets = skills.choose_targets([], project=False, home=home, cwd=project)
        assert targets == [Target("Alpha", home / ".alpha/skills" / SKILL_NAME)]

    def test_nothing_found_says_how_to_name_an_agent(self, home, project):
        with pytest.raises(SkillsError) as err:
            skills.choose_targets([], project=False, home=home, cwd=project)
        assert "--agent" in str(err.value) and "--dir" in str(err.value) and "alpha, beta" in str(err.value)

    def test_naming_an_agent_does_not_need_it_to_be_found(self, home, project):
        targets = skills.choose_targets(["beta"], project=False, home=home, cwd=project)
        assert targets == [Target("Beta Agent", home / ".config/beta/skills" / SKILL_NAME)]

    def test_all_means_every_known_agent(self, home, project):
        targets = skills.choose_targets(["all"], project=False, home=home, cwd=project)
        assert [t.label for t in targets] == ["Alpha", "Beta Agent"]

    def test_project_scope_uses_the_project_folders(self, home, project):
        targets = skills.choose_targets(["beta"], project=True, home=home, cwd=project)
        assert targets == [Target("Beta Agent", project / ".beta/skills" / SKILL_NAME)]

    def test_an_explicit_folder_wins_over_everything(self, home, project, tmp_path):
        targets = skills.choose_targets(["alpha"], project=True, home=home, cwd=project, directory=tmp_path / "mine")
        assert targets == [Target("custom folder", tmp_path / "mine" / SKILL_NAME)]

    def test_an_unknown_agent_is_an_error_that_lists_the_known_ones(self, home, project):
        with pytest.raises(SkillsError) as err:
            skills.choose_targets(["nope"], project=False, home=home, cwd=project)
        assert "alpha, beta" in str(err.value)


class TestInstall:
    def test_a_fresh_install_writes_the_skill(self, tmp_path):
        target, files = _target(tmp_path), skills.bundled_files()
        outcome = skills.install(target, files)
        assert outcome.status == "installed"
        assert (target.path / "SKILL.md").read_bytes() == files["SKILL.md"]

    def test_installing_again_changes_nothing(self, tmp_path):
        target, files = _target(tmp_path), skills.bundled_files()
        skills.install(target, files)
        before = (target.path / "SKILL.md").stat().st_mtime_ns
        assert skills.install(target, files).status == "unchanged"
        assert (target.path / "SKILL.md").stat().st_mtime_ns == before

    def test_an_older_copy_is_updated(self, tmp_path):
        target, files = _target(tmp_path), skills.bundled_files()
        skills.install(target, files)
        (target.path / "SKILL.md").write_text("an older version")
        assert skills.install(target, files).status == "updated"
        assert (target.path / "SKILL.md").read_bytes() == files["SKILL.md"]

    def test_files_the_user_added_beside_it_are_kept(self, tmp_path):
        target, files = _target(tmp_path), skills.bundled_files()
        skills.install(target, files)
        (target.path / "notes.txt").write_text("mine")
        skills.install(target, {"SKILL.md": b"newer"})
        assert (target.path / "notes.txt").read_text() == "mine"

    def test_a_dry_run_reports_without_writing(self, tmp_path):
        target = _target(tmp_path)
        outcome = skills.install(target, skills.bundled_files(), dry_run=True)
        assert outcome.status == "installed" and not target.path.exists()

    def test_a_symlinked_folder_is_left_alone(self, tmp_path):
        checkout = tmp_path / "my-checkout"
        checkout.mkdir()
        (checkout / "SKILL.md").write_text("under development")
        target = _target(tmp_path)
        target.path.parent.mkdir(parents=True)
        target.path.symlink_to(checkout)

        outcome = skills.install(target, skills.bundled_files())

        assert outcome.status == "skipped" and "symlink" in outcome.detail
        assert (checkout / "SKILL.md").read_text() == "under development"


class TestUninstall:
    def test_removes_only_its_own_folder(self, tmp_path):
        target, files = _target(tmp_path), skills.bundled_files()
        skills.install(target, files)
        sibling = target.path.parent / "another-skill"
        sibling.mkdir()
        (sibling / "SKILL.md").write_text("---\nname: another-skill\n---\n")

        assert skills.uninstall(target).status == "removed"

        assert not target.path.exists()
        assert (sibling / "SKILL.md").exists()

    def test_nothing_to_remove(self, tmp_path):
        assert skills.uninstall(_target(tmp_path)).status == "absent"

    def test_a_folder_with_that_name_that_is_not_this_skill_is_kept(self, tmp_path):
        target = _target(tmp_path)
        target.path.mkdir(parents=True)
        (target.path / "SKILL.md").write_text("---\nname: something-else\n---\n")

        outcome = skills.uninstall(target)

        assert outcome.status == "skipped" and (target.path / "SKILL.md").exists()

    def test_a_dry_run_keeps_it(self, tmp_path):
        target = _target(tmp_path)
        skills.install(target, skills.bundled_files())
        assert skills.uninstall(target, dry_run=True).status == "removed"
        assert target.path.exists()

    def test_a_symlinked_folder_is_left_alone(self, tmp_path):
        checkout = tmp_path / "my-checkout"
        checkout.mkdir()
        (checkout / "SKILL.md").write_text(f"---\nname: {SKILL_NAME}\n---\n")
        target = _target(tmp_path)
        target.path.parent.mkdir(parents=True)
        target.path.symlink_to(checkout)

        assert skills.uninstall(target).status == "skipped"
        assert (checkout / "SKILL.md").exists()


class TestState:
    def test_not_installed_installed_outdated(self, tmp_path):
        target, files = _target(tmp_path), skills.bundled_files()
        assert skills.state(target, files) == "not installed"
        skills.install(target, files)
        assert skills.state(target, files) == "installed"
        (target.path / "SKILL.md").write_text("old")
        assert skills.state(target, files) == "outdated"


def _run(argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["synapse-deep-research", *argv])
    code = 0
    try:
        main_mod.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    captured = capsys.readouterr()
    return code, " ".join(captured.out.split()), " ".join(captured.err.split())


class TestSkillsCommand:
    def test_with_no_action_it_lists_every_known_agent(self, home, project, monkeypatch, capsys):
        (home / ".alpha").mkdir()
        code, out, _ = _run(["skills"], monkeypatch, capsys)
        assert code == 0
        assert "Alpha not installed" in out and "Beta Agent not installed (agent not found)" in out

    def test_install_goes_to_the_agents_found(self, home, project, monkeypatch, capsys):
        (home / ".alpha").mkdir()
        code, out, _ = _run(["skills", "install"], monkeypatch, capsys)
        assert code == 0 and "Alpha: installed" in out and "Beta" not in out
        assert (home / ".alpha/skills" / SKILL_NAME / "SKILL.md").is_file()
        assert not (home / ".config/beta").exists()  # an agent that is not there gets no folders

        code, out, _ = _run(["skills", "install"], monkeypatch, capsys)
        assert "Alpha: already up to date" in out

        code, out, _ = _run(["skills"], monkeypatch, capsys)
        assert "Alpha installed" in out

    def test_install_for_a_named_agent_and_the_project(self, home, project, monkeypatch, capsys):
        code, out, _ = _run(["skills", "install", "--agent", "beta", "--project"], monkeypatch, capsys)
        assert code == 0 and "Beta Agent: installed" in out
        assert (project / ".beta/skills" / SKILL_NAME / "SKILL.md").is_file()
        assert not (home / ".config").exists()

    def test_dry_run_says_what_would_happen_and_writes_nothing(self, home, project, monkeypatch, capsys):
        (home / ".alpha").mkdir()
        code, out, _ = _run(["skills", "install", "--dry-run"], monkeypatch, capsys)
        assert code == 0 and "Alpha: would be installed" in out
        assert not (home / ".alpha/skills").exists()

    def test_uninstall(self, home, project, monkeypatch, capsys):
        (home / ".alpha").mkdir()
        _run(["skills", "install"], monkeypatch, capsys)
        code, out, _ = _run(["skills", "uninstall"], monkeypatch, capsys)
        assert code == 0 and "Alpha: removed" in out
        assert not (home / ".alpha/skills" / SKILL_NAME).exists()

    def test_into_a_folder_for_an_agent_that_is_not_listed(self, home, project, tmp_path, monkeypatch, capsys):
        folder = tmp_path / "other-agent" / "skills"
        code, out, _ = _run(["skills", "install", "--dir", str(folder)], monkeypatch, capsys)
        assert code == 0 and "custom folder: installed" in out
        assert (folder / SKILL_NAME / "SKILL.md").is_file()

    def test_no_agent_found_is_an_error_that_says_what_to_do(self, home, project, monkeypatch, capsys):
        code, out, _ = _run(["skills", "install"], monkeypatch, capsys)
        assert code == 1 and "--agent" in out and "--dir" in out

    def test_an_unknown_agent_is_a_usage_error(self, home, project, monkeypatch, capsys):
        code, _, err = _run(["skills", "install", "--agent", "nope"], monkeypatch, capsys)
        assert code == 2 and "invalid choice" in err

    def test_an_unknown_action_is_a_usage_error(self, home, project, monkeypatch, capsys):
        code, _, err = _run(["skills", "frobnicate"], monkeypatch, capsys)
        assert code == 2 and "invalid choice" in err
