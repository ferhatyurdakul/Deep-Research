"""
Interactive credential setup wizard.

Shared between the `deep-research setup` CLI subcommand and the REPL
/setup command. Walks the user through picking a provider, entering the
required API key, and (optionally) Tavily / Semantic Scholar keys.
Stores everything in the OS keyring via credentials.py.
"""
from __future__ import annotations

from getpass import getpass
from typing import Iterable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from .credentials import (
    MANAGED_KEYS,
    delete_credential,
    get_credential,
    keyring_backend_name,
    list_credentials,
    set_credential,
)

console = Console()


def _prompt_secret(label: str, current_set: bool) -> Optional[str]:
    """
    Prompt for a secret without echoing it to the terminal.

    Returns:
      - the new value if entered,
      - "" if the user explicitly wants to clear it ("-" sentinel),
      - None if the user pressed Enter to keep the existing value.
    """
    hint = "  [dim]press Enter to keep existing, '-' to clear[/dim]" if current_set else "  [dim]press Enter to skip[/dim]"
    console.print(f"\n  [bold]{label}[/bold]" + (" [green](currently set)[/green]" if current_set else ""))
    console.print(hint)
    raw = getpass("  > ")
    if raw == "":
        return None
    if raw == "-":
        return ""
    return raw


def _show_current_state() -> None:
    state = list_credentials()
    console.print(Panel(
        "\n".join(
            f"  {key:30} {'[green]set[/green]' if is_set else '[dim]not set[/dim]'}"
            for key, is_set in state.items()
        ) + f"\n\n  [dim]keyring backend: {keyring_backend_name()}[/dim]",
        title="Stored credentials",
        border_style="cyan",
    ))


def run_setup_wizard(keys: Optional[Iterable[str]] = None) -> None:
    """
    Walk the user through entering credentials.

    By default prompts for every managed key. Pass `keys` to limit the
    wizard to a subset (e.g. ["ZAI_API_KEY"] from /login ZAI_API_KEY).
    """
    console.print(Panel(
        "[bold]Deep Research credential setup[/bold]\n"
        "[dim]Keys are stored in your OS keyring (Keychain on macOS, "
        "Secret Service on Linux, Credential Manager on Windows).[/dim]",
        border_style="blue",
    ))
    _show_current_state()

    target_keys = set(keys) if keys else {k for k, _, _ in MANAGED_KEYS}
    updated: list[str] = []
    cleared: list[str] = []

    for env_name, label, _required in MANAGED_KEYS:
        if env_name not in target_keys:
            continue
        currently = bool(get_credential(env_name))
        value = _prompt_secret(f"{label} ({env_name})", currently)
        if value is None:
            continue
        if value == "":
            if delete_credential(env_name):
                cleared.append(env_name)
                console.print(f"  [yellow]Cleared {env_name}.[/yellow]")
            continue
        try:
            set_credential(env_name, value)
            updated.append(env_name)
            console.print(f"  [green]Saved {env_name}.[/green]")
        except Exception as e:
            console.print(f"  [red]Could not store {env_name}: {e}[/red]")

    if updated or cleared:
        console.print(
            f"\n[green]Done.[/green] Updated: {len(updated)}; cleared: {len(cleared)}."
        )
        # Hand the active settings object a chance to re-read so the running
        # process sees the new credentials immediately.
        try:
            from .config import settings
            settings.reload_credentials()
        except Exception:
            pass
    else:
        console.print("\n[dim]No changes.[/dim]")


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
        console.print(
            f"[red]Unknown credential {key_name!r}.[/red] "
            f"Valid keys: {', '.join(k for k, _, _ in MANAGED_KEYS)}"
        )
        return
    run_setup_wizard(keys=[key_name])


def run_logout(key_name: Optional[str] = None) -> None:
    """
    Clear one stored credential (or all of them with `all`).
    """
    if key_name in (None, "all", "ALL"):
        targets = [k for k, _, _ in MANAGED_KEYS]
    else:
        if key_name not in [k for k, _, _ in MANAGED_KEYS]:
            console.print(
                f"[red]Unknown credential {key_name!r}.[/red] "
                f"Use 'all' to clear everything, or one of: "
                f"{', '.join(k for k, _, _ in MANAGED_KEYS)}"
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
            console.print(f"  [yellow]Cleared {k}.[/yellow]")
    console.print(f"\n[green]Done.[/green] Cleared {cleared}/{len(targets)}.")
    try:
        from .config import settings
        settings.reload_credentials()
    except Exception:
        pass


def show_keys_status() -> None:
    """Print which managed keys are present in the keyring. Never prints values."""
    _show_current_state()
