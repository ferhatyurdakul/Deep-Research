"""
The setup screen: provider -> API key -> search key -> a model for each stage.

Shown automatically in a terminal when no provider is usable yet (no API key),
and again on demand through `/setup` and `deep-research setup`. Every step is a
transient picker or prompt that erases itself; only the final summary stays on
screen. Esc skips the whole thing at any step. Nothing here runs outside a
terminal, so scripts and pipes are never blocked.

API keys go to the OS keyring; the provider and the per-stage models go to the
saved-choices file (see userconfig).
"""
from __future__ import annotations

import os
from typing import Optional

from rich.markup import escape
from rich.text import Text

from . import userconfig
from .capabilities import PIPELINE_STAGES, recommended_stage_models
from .config import _PROVIDER_PROTOCOLS, settings
from .credentials import get_credential, keyring_backend_name, set_credential
from .interactive import ask_text, pick_option
from . import options as _options
from .options import STAGE_BLURBS, model_choices, stage_summary
from .ui import console, fail, glyph, kv_grid, note, panel, short_path, tty, warn

PROVIDER_CHOICES = [
    ("zai", "Z.AI — GLM models"),
    ("opencode-go", "OpenCode Go — GLM, Kimi, DeepSeek, Qwen (model list not refreshed yet)"),
]
_PROVIDER_NAMES = {"zai": "Z.AI", "opencode-go": "OpenCode Go"}
_KEY_ENV = {"zai": "ZAI_API_KEY", "opencode-go": "OPENCODE_API_KEY"}
STEPS = 5


def needs_setup() -> bool:
    """True when there is no API key for the active provider — the app has
    nothing to run on yet. An invalid provider name is a configuration error
    for check_config to report, not a reason to open setup."""
    try:
        return not settings.active_api_key
    except Exception:
        return False


def _title(label: str, step: int, text: str) -> str:
    return f"{label} · {step}/{STEPS} · {text}"


def _skipped() -> bool:
    note("Setup skipped — run /setup (or `deep-research setup`) whenever you're ready.")
    return False


def _ask_key(label: str, step: int, name: str, env_name: str) -> Optional[str]:
    """Ask for a secret. Returns the new value, "" to keep the existing one,
    or None if the user cancelled."""
    has_key = bool(os.getenv(env_name) or get_credential(env_name))
    hint = "Enter keeps the key you already have" if has_key else "Paste your key — it is hidden"
    while True:
        value = ask_text(_title(label, step, f"{name} API key"), hint=hint, secret=True, esc="skip setup")
        if value is None:
            return None
        value = value.strip()
        if value or has_key:
            return value
        hint = "A key is required to continue"


def _pick_model(provider: str, stage: str, current: str, label: str, account: Optional[list] = None) -> Optional[str]:
    """Choose one stage's model from the provider's list, or type another."""
    choices = model_choices(provider, stage, account) + [("other", "Type another model ID")]
    title = _title(label, 4, f"Model for {stage} — {STAGE_BLURBS[stage]}")
    while True:
        picked = pick_option(title, choices, current)
        if picked != "other":
            return picked
        typed = ask_text(f"Model ID for {stage}", hint="Any ID your provider serves, e.g. glm-5.3-flash")
        if typed is None:
            return None
        if typed.strip():
            return typed.strip()


def _verify(models: dict) -> dict:
    """One tiny request per distinct model. Returns {model: None | error text}."""
    from .llm import chat

    results: dict = {}
    for model in dict.fromkeys(models.values()):
        with console.status(f"[muted]Checking {escape(model)}…[/muted]"):
            try:
                chat("Reply with the single word: ok", model=model, temperature=0.0, thinking=False)
                results[model] = None
            except Exception as e:  # noqa: BLE001 — report whatever the provider said
                text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                results[model] = text[:140]
    return results


def run_setup(*, first_run: bool = False) -> bool:
    """Walk through setup. True when it finished and saved; False if it was
    skipped, cancelled, or can't run here (no terminal)."""
    if not tty():
        fail(
            "Setup needs an interactive terminal.",
            hint="Set ZAI_API_KEY (or OPENCODE_API_KEY) in the environment or a .env file instead.",
        )
        return False

    label = "Welcome" if first_run else "Setup"
    original_provider = settings.llm_provider

    # 1/5 — provider
    default = original_provider if original_provider in _PROVIDER_PROTOCOLS else "zai"
    provider = pick_option(_title(label, 1, "Choose a provider"), PROVIDER_CHOICES, default)
    if provider is None:
        return _skipped()
    settings.llm_provider = provider
    env_name, name = _KEY_ENV[provider], _PROVIDER_NAMES[provider]

    # 2/5 — API key (stored in the OS keyring)
    key = _ask_key(label, 2, name, env_name)
    if key is None:
        settings.llm_provider = original_provider
        return _skipped()

    # 3/5 — optional search key; Enter or Esc skips
    tavily = ask_text(
        _title(label, 3, "Tavily search key (optional)"),
        hint="Better web search than DuckDuckGo alone", secret=True, esc="skip",
    )
    tavily = (tavily or "").strip()

    # 4/5 — a model for each stage
    recommended = recommended_stage_models(provider).as_dict()
    saved = userconfig.stage_models(provider)
    current = {s: saved.get(s) or recommended[s] for s in PIPELINE_STAGES}
    mode = pick_option(
        _title(label, 4, "Models"),
        [("recommended", stage_summary(recommended)), ("custom", "Choose a model for each stage")],
        "recommended",
    )
    if mode is None:
        settings.llm_provider = original_provider
        return _skipped()
    models = dict(recommended)
    if mode == "custom":
        with console.status("[muted]Checking which models your account has…[/muted]"):
            account = _options.fetch_account_models(provider, api_key=key or None)
        for stage in PIPELINE_STAGES:
            chosen = _pick_model(provider, stage, current[stage], label, account)
            if chosen is None:
                settings.llm_provider = original_provider
                return _skipped()
            models[stage] = chosen

    # 5/5 — optional connection check
    check = pick_option(
        _title(label, 5, "Test the connection?"),
        [("yes", "Send one tiny request per model"), ("skip", "Skip")], "yes",
    )

    # Save everything, then check — a failed check must not lose the choices.
    try:
        if key:
            set_credential(env_name, key)
        if tavily:
            set_credential("TAVILY_API_KEY", tavily)
    except Exception as e:  # noqa: BLE001 — keyring backends fail in many ways
        settings.llm_provider = original_provider
        fail(f"Could not store the key in the OS keyring: {escape(str(e))}",
             hint=f"Set {env_name} in your environment or a .env file instead.")
        return False
    settings.reload_credentials()
    userconfig.save_choices(provider, models)
    results = _verify(models) if check == "yes" else {}

    _print_summary(provider, models, recommended, results, key_changed=bool(key))
    return True


def _print_summary(provider: str, models: dict, recommended: dict, results: dict, *, key_changed: bool) -> None:
    rows: list = [
        ("provider", Text(f"{_PROVIDER_NAMES[provider]} ({provider})")),
        ("api key", Text(f"{glyph('on')} {'saved' if key_changed else 'kept'} in your OS keyring ({keyring_backend_name()})")),
    ]
    for stage in PIPELINE_STAGES:
        model = models[stage]
        line = Text(model, style="bold")
        if model == recommended[stage]:
            line.append("  recommended", style="muted")
        if model in results:
            if results[model] is None:
                line.append(f"  {glyph('ok')} reachable", style="ok")
            else:
                line.append(f"  {glyph('fail')} {results[model]}", style="err")
        rows.append((stage, line))
    rows.append(("saved to", Text(short_path(userconfig.config_file()), overflow="fold")))
    console.print()
    console.print(panel(kv_grid(rows), title=f"{glyph('ok')} Setup complete", border="green"))

    failed = [m for m, err in results.items() if err]
    if failed:
        warn("Some models didn't answer. Check the message above, then change them with /model "
             "(your plan may not include every model).")
    overridden = [s for s in PIPELINE_STAGES if os.getenv(f"MODEL_{s.upper()}")]
    if overridden:
        warn("Environment variables override your choice for: "
             + ", ".join(f"{s} (MODEL_{s.upper()})" for s in overridden) + ".")
    if os.getenv(_KEY_ENV[provider]) and key_changed:
        warn(f"{_KEY_ENV[provider]} is also set in your environment, and the environment value wins.")
    note("Change models any time with /model, or re-run setup with /setup.")
