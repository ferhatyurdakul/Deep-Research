"""
One-time carry-over from this tool's former name.

Before it was Synapse Deep Research the tool was called `deep-research` and kept
everything under that name: ~/.config/deep-research (settings, global .env, REPL
history), ~/.local/share/deep-research (the session database) and API keys in the
OS keyring under the service "deep-research". The first launch under the new name
copies what is there, so a rename doesn't look like lost history or lost keys.

Copy, never move: the old files and keyring entries stay exactly as they were, so
the old install keeps working until it is removed and an interrupted copy loses
nothing. Nothing already present under the new name is overwritten, and a marker
file records that this ran, so it happens once.

Imports nothing heavy and not `config`: it has to run before config.py reads the
global .env, the saved settings and the data dir.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import credentials

logger = logging.getLogger(__name__)

OLD_NAME = "deep-research"
NEW_NAME = "synapse-deep-research"
MARKER = ".carried-over-from-deep-research"

_CONFIG_FILES = ("config.json", ".env", "repl_history")
_DB_FILES = ("research.db", "research.db-wal")  # the -wal holds writes not yet in the .db; the -shm is rebuilt from it
# Set when something outside the default locations decides where state lives
# (tests, CI, a custom setup). Then "the old place" isn't ours to guess.
_OVERRIDES = ("SYNAPSE_DEEP_RESEARCH_CONFIG_FILE", "SYNAPSE_DEEP_RESEARCH_DATA_DIR")


@dataclass
class CarriedOver:
    files: list = field(default_factory=list)   # destination paths of the files copied
    keys: list = field(default_factory=list)    # names of the credentials copied (never values)
    old_dirs: list = field(default_factory=list)  # old directories something was copied from

    def __bool__(self) -> bool:
        return bool(self.files or self.keys)


def _xdg(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, os.path.expanduser(default)))


def _copy_missing(source_dir: Path, target_dir: Path, names) -> list:
    """Copy each of `names` that exists in `source_dir` and not yet in `target_dir`."""
    copied = []
    for name in names:
        source, target = source_dir / name, target_dir / name
        if not source.is_file() or target.exists():
            continue
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            logger.warning("Could not copy %s to %s: %s", source, target, exc)
            continue
        copied.append(target)
    return copied


def _copy_keys() -> list:
    """Copy the managed credentials the old service holds and the new one lacks."""
    carried = []
    for key in credentials.MANAGED_KEY_NAMES:
        try:
            if credentials.get_credential(key):
                continue
            value = credentials.get_credential(key, service=OLD_NAME)
            if value:
                credentials.set_credential(key, value)
                carried.append(key)
        except Exception as exc:  # a locked or missing keyring just means there is nothing to carry
            logger.debug("Could not carry over %s: %s", key, exc)
    return carried


def carry_over(config_home: Optional[Path] = None, data_home: Optional[Path] = None) -> CarriedOver:
    """Copy settings, session history and API keys from the `deep-research` install.

    Falsy when there was nothing to copy (or it already ran). `config_home` and
    `data_home` default to the XDG locations; tests pass their own."""
    result = CarriedOver()
    if any(os.environ.get(name) for name in _OVERRIDES):
        return result
    config_home = config_home or _xdg("XDG_CONFIG_HOME", "~/.config")
    data_home = data_home or _xdg("XDG_DATA_HOME", "~/.local/share")
    new_config = config_home / NEW_NAME
    marker = new_config / MARKER
    if marker.exists():
        return result

    old_config, old_data = config_home / OLD_NAME, data_home / OLD_NAME
    new_data = data_home / NEW_NAME
    copied = _copy_missing(old_config, new_config, _CONFIG_FILES)
    if copied:
        result.files += copied
        result.old_dirs.append(old_config)
    if not (new_data / "research.db").exists():  # never splice an old database into a live one
        copied = _copy_missing(old_data, new_data, _DB_FILES)
        if copied:
            result.files += copied
            result.old_dirs.append(old_data)
    result.keys = _copy_keys()

    try:
        new_config.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"Copied from {OLD_NAME} on first launch as {NEW_NAME}.\n", encoding="utf-8")
    except OSError:
        pass  # worst case it is tried again next launch, and copies nothing
    return result


def describe(carried: CarriedOver) -> str:
    """What was copied, in the user's terms (no paths, no values)."""
    names = {f.name for f in carried.files}
    parts = []
    if "config.json" in names:
        parts.append("settings")
    if "research.db" in names:
        parts.append("session history")
    if ".env" in names:
        parts.append("the global .env")
    if carried.keys:
        parts.append(f"{len(carried.keys)} API key{'s' if len(carried.keys) != 1 else ''}")
    if not parts:
        parts.append("REPL history")
    return ", ".join(parts)
