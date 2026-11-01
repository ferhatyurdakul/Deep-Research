"""
Interactive credential setup wizard.

Shared between the `synapse-deep-research setup` CLI subcommand and the REPL
/setup command. Walks the user through picking a provider, entering the
required API key, and (optionally) Tavily / Semantic Scholar keys.
Stores everything in the OS keyring via credentials.py.
"""
from __future__ import annotations

from getpass import getpass
from typing import Iterable, Optional

from rich import box
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from .credentials import (
    MANAGED_KEYS,
    delete_credential,
    get_credential,
    keyring_backend_name,
    list_credentials,
    set_credential,
)
from .ui import console, fail, glyph, note, ok, panel, warn


def _prompt_secret(label: str, current_set: bool, step: str = "") -> Optional[str]:
    """
    Prompt for a secret without echoing it to the terminal.

    Returns:
      - the new value if entered,
      - "" if the user explicitly wants to clear it ("-" sentinel),
      - None if the user pressed Enter to keep the existing value.
    """
    status = f"  [ok]{glyph('on')} set[/ok]" if current_set else f"  [muted]{glyph('off')} not set[/muted]"
    console.print(f"\n[muted]{step}[/muted] [bold]{escape(label)}[/bold]{status}")
    note("  Enter to keep the current value, '-' to clear it." if current_set
         else "  Enter to skip.")
    raw = getpass("  > ")
    if raw == "":
        return None
    if raw == "-":
        return ""
    return raw


def _show_current_state() -> None:
    state = list_credentials()
    table = Table(box=box.SIMPLE_HEAD, header_style="muted", pad_edge=False)
    table.add_column("Key", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("For")
    for env_name, label, required in MANAGED_KEYS:
        is_set = state.get(env_name, False)
        status = Text(f"{glyph('on')} set", style="ok") if is_set else Text(f"{glyph('off')} not set", style="muted")
        purpose = Text(label)
        if required and not is_set:
            purpose.append("  (needed for the default provider)", style="warn")
        table.add_row(Text(env_name, style="cmd"), status, purpose)
    console.print(panel(table, title="Stored credentials"))
    note(f"Stored in your OS keyring ({keyring_backend_name()}). Values are never displayed.")


def run_setup_wizard(keys: Optional[Iterable[str]] = None) -> None:
    """
    Walk the user through entering credentials.

    Pass `keys` to limit the wizard to a subset (e.g. ["ZAI_API_KEY"] from
    /login ZAI_API_KEY). With no `keys` this is the full setup screen
    (provider, keys, models), which lives in onboarding.
    """
    if keys is None:
        from .onboarding import run_setup
        run_setup()
        return
    console.print()
    console.print(panel(
        "Store API keys in your OS keyring (Keychain on macOS, Secret Service on Linux,\n"
        "Credential Manager on Windows) instead of a plain-text .env file.",
        title=f"{glyph('mark')} Credential setup",
    ))
    _show_current_state()

    target_keys = set(keys) if keys else {k for k, _, _ in MANAGED_KEYS}
    todo = [(k, label) for k, label, _ in MANAGED_KEYS if k in target_keys]
    updated: list[str] = []
    cleared: list[str] = []

    for i, (env_name, label) in enumerate(todo, 1):
        currently = bool(get_credential(env_name))
        step = f"[{i}/{len(todo)}]" if len(todo) > 1 else ""
        value = _prompt_secret(f"{label} ({env_name})", currently, step)
        if value is None:
            continue
        if value == "":
            if delete_credential(env_name):
                cleared.append(env_name)
                warn(f"Cleared {env_name}.")
            continue
        try:
            set_credential(env_name, value)
            updated.append(env_name)
            ok(f"Saved {env_name}.")
        except Exception as e:
            fail(f"Could not store {env_name}: {escape(str(e))}")

    if updated or cleared:
        console.print()
        ok(f"Done. Updated {len(updated)}, cleared {len(cleared)}.")
        # Hand the active settings object a chance to re-read so the running
        # process sees the new credentials immediately.
        try:
            from .config import settings
            settings.reload_credentials()
        except Exception:
            pass
    else:
        console.print()
        note("No changes.")


def run_login(key_name: Optional[str] = None) -> None:
    """
    Set a single credential.

    If `key_name` is None, default to the active provider's API key so
    `/login` does the right thing without a second prompt.
    """
    if key_name is None:
        from .config import settings
        key_name = settings.active_api_key_env_name
    if key_name not in [k for k, _, _ in MANAGED_KEYS]:
        fail(
            f"Unknown credential {escape(repr(key_name))}.",
            hint=f"Valid keys: {', '.join(k for k, _, _ in MANAGED_KEYS)}",
        )
        return
    run_setup_wizard(keys=[key_name])


def run_logout(key_name: Optional[str] = None) -> None:
    """
    Clear one stored credential (or all of them with `all`).
    """
    if key_name in ("all", "ALL"):
        targets = [k for k, _, _ in MANAGED_KEYS]
    else:
        if key_name is None:
            # Mirror run_login: bare `logout` (or /logout) clears exactly the
            # active provider's key — not every stored credential.
            from .config import settings
            try:
                key_name = settings.active_api_key_env_name
            except Exception as e:
                fail(f"Cannot resolve the active provider: {escape(str(e))}")
                return
        if key_name not in [k for k, _, _ in MANAGED_KEYS]:
            fail(
                f"Unknown credential {escape(repr(key_name))}.",
                hint=f"Use 'all' to clear everything, or one of: {', '.join(k for k, _, _ in MANAGED_KEYS)}",
            )
            return
        targets = [key_name]

    if len(targets) > 1:
        if not Confirm.ask(
            f"[yellow]Clear {len(targets)} stored credentials from the keyring?[/yellow]",
            default=False,
        ):
            return

    cleared = 0
    for k in targets:
        if delete_credential(k):
            cleared += 1
            warn(f"Cleared {k}.")
    console.print()
    ok(f"Done. Cleared {cleared}/{len(targets)}.")
    try:
        from .config import settings
        settings.reload_credentials()
    except Exception:
        pass


def show_keys_status() -> None:
    """Print which managed keys are present in the keyring. Never prints values."""
    _show_current_state()
