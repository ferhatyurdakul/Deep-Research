"""
Put the bundled agent skill where other coding agents will find it.

The skill is a small folder (SKILL.md) in the open Agent Skills format, so an
agent such as Claude Code or Codex can run this tool for its user without anyone
opening it separately. `skills install` copies the folder into each agent's
skills directory; the skill ships inside this package, so what gets installed
always matches the installed CLI.

Only the folder named after the skill is ever written or removed, and a symlink
at that spot is left alone (it is probably a checkout someone is developing).
This module only does the filesystem work; main.py does the printing.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

SKILL_NAME = "synapse-deep-research"


@dataclass(frozen=True)
class Agent:
    key: str          # the --agent value
    label: str        # how the agent is named to the user
    user_dir: str     # its skills folder, relative to the home directory
    project_dir: str  # its skills folder, relative to a project
    detect: str       # a folder under home that exists when the agent is installed


# Where each agent looks for skills, from its own documentation (checked 2026-11).
# Several agents read the shared ~/.agents/skills folder; Claude Code reads only
# .claude/skills, and Hermes only ~/.hermes/skills (project skills there load
# only after `hermes skills trust`). Agents that share a folder get one copy.
AGENTS: tuple = (
    Agent("claude", "Claude Code", ".claude/skills", ".claude/skills", ".claude"),
    Agent("codex", "Codex", ".agents/skills", ".agents/skills", ".codex"),
    Agent("hermes", "Hermes", ".hermes/skills", ".hermes/skills", ".hermes"),
    Agent("gemini", "Gemini CLI", ".agents/skills", ".agents/skills", ".gemini"),
    Agent("cursor", "Cursor", ".agents/skills", ".agents/skills", ".cursor"),
    Agent("copilot", "GitHub Copilot", ".agents/skills", ".agents/skills", ".copilot"),
    Agent("opencode", "OpenCode", ".agents/skills", ".agents/skills", ".config/opencode"),
)


@dataclass(frozen=True)
class Target:
    label: str
    path: Path  # the skill's own folder: <skills dir>/synapse-deep-research


@dataclass(frozen=True)
class Outcome:
    target: Target
    status: str  # installed | updated | unchanged | removed | absent | skipped
    detail: str = ""


class SkillsError(Exception):
    """Something the user can fix by changing the command line."""


def bundled_files() -> dict:
    """{file name: bytes} of the skill that ships in this package."""
    root = resources.files("synapse_deep_research").joinpath("skill")
    return {entry.name: entry.read_bytes() for entry in root.iterdir() if entry.is_file()}


def agent_by_key(key: str) -> Agent:
    for agent in AGENTS:
        if agent.key == key:
            return agent
    raise SkillsError(f"Unknown agent {key!r}. Known: {', '.join(a.key for a in AGENTS)}.")


def detected_agents(home: Path) -> list:
    return [a for a in AGENTS if (home / a.detect).exists()]


def choose_targets(
    agent_keys: list, *, project: bool, home: Path, cwd: Path, directory: Optional[Path] = None,
) -> list:
    """Where to install. Explicit `--dir` wins, then `--agent` (or `all`); with neither,
    every agent found in the home directory."""
    if directory is not None:
        return [Target("custom folder", Path(directory).expanduser() / SKILL_NAME)]
    if "all" in agent_keys:
        agents = list(AGENTS)
    elif agent_keys:
        agents = [agent_by_key(k) for k in agent_keys]
    else:
        agents = detected_agents(home)
        if not agents:
            raise SkillsError(
                "No supported agent found in your home folder. Name one with --agent "
                f"({', '.join(a.key for a in AGENTS)}), or give a skills folder with --dir."
            )
    by_folder: dict = {}  # agents that read the same folder share one copy
    for a in agents:
        folder = (cwd / a.project_dir if project else home / a.user_dir) / SKILL_NAME
        by_folder.setdefault(folder, []).append(a.label)
    return [Target(", ".join(labels), folder) for folder, labels in by_folder.items()]


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".skill-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def install(target: Target, files: dict, *, dry_run: bool = False) -> Outcome:
    path = target.path
    if path.is_symlink():
        return Outcome(target, "skipped", "it is a symlink, so it was left alone")
    existed = path.is_dir()
    if existed and all((path / n).is_file() and (path / n).read_bytes() == d for n, d in files.items()):
        return Outcome(target, "unchanged")
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            _write_atomic(path / name, data)
    return Outcome(target, "updated" if existed else "installed")


def _is_ours(path: Path) -> bool:
    skill_file = path / "SKILL.md"
    try:
        head = skill_file.read_text(encoding="utf-8")[:2000]
    except OSError:
        return False
    return re.search(rf"^name:\s*{re.escape(SKILL_NAME)}\s*$", head, re.MULTILINE) is not None


def uninstall(target: Target, *, dry_run: bool = False) -> Outcome:
    path = target.path
    if path.is_symlink():
        return Outcome(target, "skipped", "it is a symlink, so it was left alone")
    if not path.exists():
        return Outcome(target, "absent")
    if not path.is_dir() or not _is_ours(path):
        return Outcome(target, "skipped", f"{path.name} there is not this skill, so it was left alone")
    if not dry_run:
        shutil.rmtree(path)
    return Outcome(target, "removed")


def state(target: Target, files: dict) -> str:
    """For `skills list`: installed | outdated | not installed."""
    if not target.path.is_dir():
        return "not installed"
    same = all((target.path / n).is_file() and (target.path / n).read_bytes() == d for n, d in files.items())
    return "installed" if same else "outdated"
