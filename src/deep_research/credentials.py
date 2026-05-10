"""
OS-native credential storage for Deep Research API keys.

Uses the `keyring` library, which delegates to:
  - macOS:   Keychain
  - Linux:   Secret Service (GNOME Keyring, KWallet) — fallback to encrypted file
  - Windows: Credential Manager

Service name is "deep-research"; the key argument is the same env-var name
the rest of the codebase already uses (`ZAI_API_KEY`, `OPENCODE_API_KEY`,
`TAVILY_API_KEY`, etc.) so resolution logic in config.py just substitutes
this backend for the .env-file lookup.

Plain-text storage in .env is still supported for backward compat and
CI/scripting workflows — see config.py for the full resolution order.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SERVICE_NAME = "deep-research"

# The canonical list of keys we manage. Adding a key here adds it to the
# setup wizard, /keys list output, and /logout all behavior. Display name
# is what the user sees in prompts; the key is the env-var name used
# everywhere else.
MANAGED_KEYS: tuple[tuple[str, str, bool], ...] = (
    # (env_var_name, friendly_label, required_for_default_provider)
    ("ZAI_API_KEY",              "Z.AI API key",              True),
    ("OPENCODE_API_KEY",         "OpenCode Go API key",       False),
    ("TAVILY_API_KEY",           "Tavily search API key",     False),
    ("SEMANTIC_SCHOLAR_API_KEY", "Semantic Scholar API key",  False),
)

MANAGED_KEY_NAMES: tuple[str, ...] = tuple(k for k, _, _ in MANAGED_KEYS)


def _backend():
    """Lazy-import keyring so test code can stub it before module load."""
    import keyring
    return keyring


def get_credential(key: str) -> Optional[str]:
    """Read a credential from the OS keyring. Returns None if missing or backend errors."""
    try:
        return _backend().get_password(SERVICE_NAME, key)
    except Exception as e:  # keyring backends can fail in many ways on locked sessions, headless boxes, etc.
        logger.debug("keyring read for %s failed: %s", key, e)
        return None


def set_credential(key: str, value: str) -> None:
    """Store a credential in the OS keyring. Raises on backend failure."""
    if not value:
        raise ValueError("Refusing to store an empty credential.")
    _backend().set_password(SERVICE_NAME, key, value)


def delete_credential(key: str) -> bool:
    """Remove a credential. Returns True if a credential existed and was deleted."""
    try:
        existing = _backend().get_password(SERVICE_NAME, key)
    except Exception:
        return False
    if not existing:
        return False
    try:
        _backend().delete_password(SERVICE_NAME, key)
        return True
    except Exception as e:
        logger.warning("Failed to delete keyring entry for %s: %s", key, e)
        return False


def list_credentials() -> dict[str, bool]:
    """
    Return {key_name: is_set} for every managed key. Never returns values —
    only presence — so logs / UI can show status without leaking secrets.
    """
    out: dict[str, bool] = {}
    for key in MANAGED_KEY_NAMES:
        out[key] = bool(get_credential(key))
    return out


def keyring_backend_name() -> str:
    """Diagnostic helper: name of the active keyring backend."""
    try:
        return _backend().get_keyring().name
    except Exception:
        return "(unavailable)"
