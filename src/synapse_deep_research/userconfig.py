"""
Saved user choices: which provider, and which model each pipeline stage uses.

Written by the setup screen and `/model`. API keys never go here (they live in
the OS keyring); this file only holds non-secret preferences, as one small JSON
file under the XDG config dir. JSON rather than TOML because the package still
supports Python 3.9, which has no built-in TOML reader.

    {"version": 1,
     "provider": "zai",
     "models": {"zai": {"decompose": "glm-5.3-flash", "analyze": "glm-5.3-flash",
                        "gap_analysis": "glm-5.3", "synthesize": "glm-5.3"}}}

Models are stored per provider, so switching provider and back keeps each one's
choices. Reads are forgiving (a missing or damaged file is "nothing saved");
writes are atomic.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from .capabilities import PIPELINE_STAGES

VERSION = 1


def config_file() -> Path:
    """Where the file lives. SYNAPSE_DEEP_RESEARCH_CONFIG_FILE overrides it (tests, CI)."""
    override = os.environ.get("SYNAPSE_DEEP_RESEARCH_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(base) / "synapse-deep-research" / "config.json"


def exists() -> bool:
    return config_file().is_file()


def load() -> dict:
    """The saved data, or {} when nothing usable is saved."""
    try:
        data = json.loads(config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(data: dict) -> Path:
    """Write the file atomically (temp file + rename) and return its path."""
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": VERSION, **{k: v for k, v in data.items() if k != "version"}}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def provider() -> Optional[str]:
    value = load().get("provider")
    return value if isinstance(value, str) and value else None


def stage_models(provider_name: str) -> dict:
    """The saved model per stage for `provider_name` (only valid, non-empty entries)."""
    models = load().get("models")
    saved = models.get(provider_name) if isinstance(models, dict) else None
    if not isinstance(saved, dict):
        return {}
    return {
        stage: model for stage, model in saved.items()
        if stage in PIPELINE_STAGES and isinstance(model, str) and model.strip()
    }


def save_choices(provider_name: str, models: dict) -> Path:
    """Remember `provider_name` as the provider and replace its saved stage models."""
    data = load()
    all_models = data.get("models") if isinstance(data.get("models"), dict) else {}
    all_models[provider_name] = {s: m for s, m in models.items() if s in PIPELINE_STAGES and m}
    data["provider"] = provider_name
    data["models"] = all_models
    return save(data)


def set_stage_model(provider_name: str, stage: str, model: Optional[str]) -> Path:
    """Save one stage's model for `provider_name`; `None` clears it (back to the recommendation)."""
    current = dict(stage_models(provider_name))
    if model:
        current[stage] = model
    else:
        current.pop(stage, None)
    return save_choices(provider_name, current)


def set_provider(provider_name: str) -> Path:
    """Remember `provider_name` as the provider, keeping every provider's saved models."""
    data = load()
    data["provider"] = provider_name
    return save(data)
