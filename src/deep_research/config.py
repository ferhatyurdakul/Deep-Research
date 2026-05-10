from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "research.db"

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
