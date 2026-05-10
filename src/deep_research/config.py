from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- .env discovery -----------------------------------------------------
# Prefer a project-local .env (cwd or above) so dev workflow is unchanged.
# Fall back to ~/.config/deep-research/.env so a pipx-installed CLI works
# from any directory without exporting credentials in the shell profile.
def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")))


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")))


_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path)
else:
    _global_env = _xdg_config_home() / "deep-research" / ".env"
    if _global_env.exists():
        load_dotenv(_global_env)


# --- Filesystem paths ---------------------------------------------------
# DATA_DIR holds the SQLite session DB. Lives under XDG_DATA_HOME by default
# so a pipx-installed CLI doesn't lose state when the venv is rebuilt.
# Override with DEEP_RESEARCH_DATA_DIR to keep state next to the source
# (handy during development).
DATA_DIR = Path(
    os.environ.get(
        "DEEP_RESEARCH_DATA_DIR",
        str(_xdg_data_home() / "deep-research"),
    )
)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "research.db"

# OUTPUTS_DIR is where saved markdown reports land. Defaults to cwd so the
# user can `cd ~/work && deep-research "..."` and find the report next to
# their other files. Override with DEEP_RESEARCH_OUTPUTS_DIR. The directory
# is *not* created at import time (we don't want every CLI invocation in
# every directory to leave an empty `outputs/` behind) — see
# output/report.py:save_report which mkdirs on demand.
OUTPUTS_DIR = Path(
    os.environ.get(
        "DEEP_RESEARCH_OUTPUTS_DIR",
        str(Path.cwd() / "outputs"),
    )
)


# --- Migration notice ---------------------------------------------------
# v0.2.0 moved DATA_DIR from <repo>/data/ to ~/.local/share/deep-research/.
# If the old DB exists and the new one doesn't, point users at the move
# so they don't think their history vanished.
_legacy_db = PROJECT_ROOT / "data" / "research.db"
if _legacy_db.exists() and not DB_PATH.exists() and _legacy_db != DB_PATH:
    warnings.warn(
        f"Legacy session DB found at {_legacy_db}. v0.2.0+ stores sessions at {DB_PATH}. "
        f"Move the file to keep your history, or set DEEP_RESEARCH_DATA_DIR={_legacy_db.parent} "
        f"to keep using the old location.",
        stacklevel=2,
    )

# Supported provider families and their endpoint protocols.
# A provider family is a vendor; an endpoint family is a wire protocol.
# OpenCode Go publishes both an OpenAI-compatible and an Anthropic-compatible
# endpoint, so it can be paired with either family at runtime.
_PROVIDER_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "zai": ("openai",),
    "opencode-go": ("openai", "anthropic"),
}


class Settings(BaseModel):
    # --- Provider selection ---
    # "zai" (default) | "opencode-go"
    llm_provider: str = os.getenv("LLM_PROVIDER", "zai")
    # "" lets each provider pick its default. Override with "openai" or "anthropic".
    llm_endpoint_family: str = os.getenv("LLM_ENDPOINT_FAMILY", "")

    # --- Z.AI provider (existing) ---
    zai_api_key: str = os.getenv("ZAI_API_KEY", "")
    zai_base_url: str = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    glm_model: str = os.getenv("GLM_MODEL", "glm-5")

    # --- OpenCode Go provider ---
    opencode_api_key: str = os.getenv("OPENCODE_API_KEY", "")
    opencode_base_url: str = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
    opencode_model: str = os.getenv("OPENCODE_MODEL", "glm-5.1")

    # --- Other settings ---
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    default_depth: str = os.getenv("RESEARCH_DEPTH", "standard")
    max_sub_questions: int = 5
    max_search_results: int = 5
    max_scrape_pages: int = 3

    # Named stage-routing profile: "balanced" (default) | "budget" | "quality"
    # See capabilities.RECOMMENDED_ROUTES for the per-provider catalog.
    llm_route: str = os.getenv("LLM_ROUTE", "balanced")

    # Fallback provider when the primary trips a rate-limit / subscription cap.
    # Empty = no fallback (raise a clear error and abort the run).
    # Setting it to another configured provider (e.g. "zai" while primary is
    # "opencode-go") makes Deep Research swap providers for the rest of the
    # process after a 429 is observed.
    llm_fallback_provider: str = os.getenv("LLM_FALLBACK_PROVIDER", "")

    # Multi-model routing: override per pipeline stage. Empty = use the
    # active profile's recommended model for that stage.
    model_decompose: str = os.getenv("MODEL_DECOMPOSE", "")
    model_analyze: str = os.getenv("MODEL_ANALYZE", "")
    model_synthesize: str = os.getenv("MODEL_SYNTHESIZE", "")
    model_gap_analysis: str = os.getenv("MODEL_GAP_ANALYSIS", "")

    @property
    def active_provider(self) -> str:
        if self.llm_provider not in _PROVIDER_PROTOCOLS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. "
                f"Supported: {', '.join(_PROVIDER_PROTOCOLS)}"
            )
        return self.llm_provider

    @property
    def active_endpoint_family(self) -> str:
        supported = _PROVIDER_PROTOCOLS[self.active_provider]
        if self.llm_endpoint_family:
            if self.llm_endpoint_family not in supported:
                raise ValueError(
                    f"Provider '{self.active_provider}' does not support endpoint family "
                    f"'{self.llm_endpoint_family}'. Supported: {', '.join(supported)}"
                )
            return self.llm_endpoint_family
        return supported[0]

    @property
    def active_api_key(self) -> str:
        return {
            "zai": self.zai_api_key,
            "opencode-go": self.opencode_api_key,
        }[self.active_provider]

    @property
    def active_base_url(self) -> str:
        return {
            "zai": self.zai_base_url,
            "opencode-go": self.opencode_base_url,
        }[self.active_provider]

    @property
    def active_model(self) -> str:
        return {
            "zai": self.glm_model,
            "opencode-go": self.opencode_model,
        }[self.active_provider]

    @property
    def active_api_key_env_name(self) -> str:
        return {
            "zai": "ZAI_API_KEY",
            "opencode-go": "OPENCODE_API_KEY",
        }[self.active_provider]

    @property
    def effective_endpoint_family(self) -> str:
        """
        The endpoint family that will actually be used at call time.

        Differs from `active_endpoint_family` (which is just the user override
        or the provider-level default) by consulting per-model constraints from
        the capability map. For example, OpenCode Go's MiniMax models are
        Anthropic-only even though the provider supports both protocols.

        Falls back to `active_endpoint_family` on any unexpected error so the
        property never hard-fails just to populate a banner.
        """
        try:
            from .capabilities import resolve_endpoint_family
            return resolve_endpoint_family(
                self.active_provider,
                self.active_model,
                self.llm_endpoint_family or None,
            )
        except Exception:
            return self.active_endpoint_family

    def model_post_init(self, __context) -> None:
        """
        Fill empty managed credentials from the OS keyring.

        Resolution priority (highest → lowest):
          1. Environment variable / project .env / global .env  (already in the
             field at construction time because load_dotenv populates os.environ).
          2. OS keyring (populated by `deep-research setup` or REPL /login).

        Both options stay available so CI/scripts can override per-run and
        keyring is the unattended default for installed CLI use.
        """
        from .credentials import get_credential
        # Iterate over (field_name, keyring_key) pairs. Only fill when empty
        # so explicit constructor args (used by tests) win.
        pairs = (
            ("zai_api_key",               "ZAI_API_KEY"),
            ("opencode_api_key",          "OPENCODE_API_KEY"),
            ("tavily_api_key",            "TAVILY_API_KEY"),
            ("semantic_scholar_api_key",  "SEMANTIC_SCHOLAR_API_KEY"),
        )
        for field, key in pairs:
            if not getattr(self, field, ""):
                stored = get_credential(key)
                if stored:
                    object.__setattr__(self, field, stored)

    def reload_credentials(self) -> None:
        """
        Re-read managed credentials from the keyring. Call after `/login` /
        `/logout` so a running REPL session picks up the change without
        restarting. Does not touch fields whose values came from env/.env —
        we only refresh from the keyring source.
        """
        from .credentials import get_credential, MANAGED_KEY_NAMES
        env_name_to_field = {
            "ZAI_API_KEY": "zai_api_key",
            "OPENCODE_API_KEY": "opencode_api_key",
            "TAVILY_API_KEY": "tavily_api_key",
            "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
        }
        for key in MANAGED_KEY_NAMES:
            field = env_name_to_field.get(key)
            if field is None:
                continue
            # Only overwrite from keyring if the current value isn't from env.
            # Env values stick (highest priority); keyring fills/refreshes the rest.
            if os.getenv(key):
                continue
            object.__setattr__(self, field, get_credential(key) or "")

    def stage_models(self) -> dict[str, str]:
        """Return non-empty per-stage env overrides only (no profile fill-in)."""
        return {
            stage: model
            for stage, model in (
                ("decompose", self.model_decompose),
                ("analyze", self.model_analyze),
                ("synthesize", self.model_synthesize),
                ("gap_analysis", self.model_gap_analysis),
            )
            if model
        }

    def effective_stage_models(self) -> dict[str, str]:
        """
        Resolve all four pipeline stages: per-stage env override wins, then
        the active LLM_ROUTE profile fills the rest. Raises CapabilityViolation
        if the profile is unknown for the active provider.
        """
        from .capabilities import resolve_stage_route

        route = resolve_stage_route(self.active_provider, self.llm_route)
        return {
            "decompose": self.model_decompose or route.decompose,
            "analyze": self.model_analyze or route.analyze,
            "gap_analysis": self.model_gap_analysis or route.gap_analysis,
            "synthesize": self.model_synthesize or route.synthesize,
        }

    def validate_routes(self) -> dict[str, object]:
        """
        Verify every configured (provider, model, endpoint_family) combination.

        Called at startup before expensive research runs. Raises
        capabilities.CapabilityViolation on a hard incompatibility; warnings
        for unknown-but-not-forbidden models are emitted via the logger.

        Returns a debug dict describing the routing decision for logging.
        """
        # Imported lazily to avoid a circular import: capabilities.py is a leaf
        # module that does not depend on config.
        from .capabilities import (
            describe_route, resolve_endpoint_family, validate_route,
        )

        provider = self.active_provider
        family = resolve_endpoint_family(
            provider, self.active_model, self.llm_endpoint_family or None,
        )
        # Ensure the resolved family is also one this provider supports
        # at the wire-config level (the provider/protocol matrix).
        supported = _PROVIDER_PROTOCOLS[provider]
        if family not in supported:
            from .capabilities import CapabilityViolation
            raise CapabilityViolation(
                f"Resolved endpoint family {family!r} is not enabled for provider "
                f"{provider!r} in the protocol matrix. Enabled: {list(supported)}"
            )

        validate_route(provider, self.active_model, family)
        # Validate every stage that will actually run (profile + overrides),
        # not just the explicit env overrides.
        stages = self.effective_stage_models()
        for stage_model in stages.values():
            # Stage models can route through their own family if it differs
            # from the active model's (e.g. minimax-m2.7 → anthropic), so
            # resolve their family individually before validating.
            from .capabilities import resolve_endpoint_family
            stage_family = resolve_endpoint_family(provider, stage_model, None)
            validate_route(provider, stage_model, stage_family)

        info = describe_route(
            provider, self.active_model, family,
            extra_models=stages.values(),
        )
        info["profile"] = self.llm_route
        info["stages"] = stages
        return info


settings = Settings()
