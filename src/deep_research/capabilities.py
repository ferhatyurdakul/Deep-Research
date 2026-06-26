"""
Model capability registry.

Tracks per-model metadata so the pipeline can validate provider/model/endpoint
combinations and pick the right client path before doing expensive work.

Fields are deliberately optional: if upstream metadata is unverified we leave
it as None and fall back to permissive behavior with a warning, rather than
fabricating numbers.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Endpoint families this codebase knows how to speak.
KNOWN_ENDPOINT_FAMILIES: frozenset[str] = frozenset({"openai", "anthropic"})

# Thinking-payload dialects we know how to emit. Each model is mapped to one
# of these so we never ship a Z.AI-shaped `clear_thinking` flag at a non-Z.AI
# upstream that doesn't recognize it.
#   "zai-extra-body":      Z.AI/GLM extension via OpenAI extra_body
#                          ({"thinking": {"type": "enabled", "clear_thinking": True}})
#   "anthropic-thinking":  Anthropic Messages API extended-thinking
#                          ({"type": "enabled", "budget_tokens": N})
#   "none":                No verified dialect — omit the thinking payload.
#                          Reasoning-native models (DeepSeek, Kimi) still think
#                          internally; we just don't request a specific shape.
KNOWN_THINKING_DIALECTS: frozenset[str] = frozenset({
    "zai-extra-body", "anthropic-thinking", "none",
})

# Maturity tiers that drive what shows up in the recommended routes and
# what gets warned about at startup. The bar is "have we exercised this
# end-to-end against a real upstream," not "does the model exist."
#   "production":   verified path; safe default.
#   "experimental": code path is correct but the model has not been
#                   exercised live from this codebase. Use at your own risk.
#   "unsupported":  do not route to this model; entry exists only so the
#                   registry can warn instead of silently working.
KNOWN_TIERS: frozenset[str] = frozenset({"production", "experimental", "unsupported"})


class ModelCapability(BaseModel):
    provider: str
    model_id: str
    # Wire protocols this model is reachable through. OpenCode Go exposes both
    # an OpenAI-compatible and an Anthropic-compatible endpoint that both route
    # to the same upstream model, so most entries list both.
    endpoint_families: frozenset
    # None = unknown / not verified. Treated as permissive at runtime.
    thinking_supported: Optional[bool] = None
    # How to spell the thinking-enabled request for this model. "none" means
    # we don't add any thinking-specific field — the model handles reasoning
    # natively if it has it.
    thinking_dialect: str = "none"
    context_window: Optional[int] = None
    tier: str = "experimental"
    notes: str = ""

    def supports_family(self, family: str) -> bool:
        return family in self.endpoint_families


class CapabilityViolation(ValueError):
    """Raised when a configured provider/model/family combination is invalid."""


# Per the official OpenCode Go docs (https://opencode.ai/docs/go/), models are
# split by which endpoint they're reachable through:
#   - /chat/completions (OpenAI-compat) handles GLM, Kimi, MiMo, Qwen, DeepSeek
#   - /messages         (Anthropic-compat) handles MiniMax only
# The capability map encodes this so a single-family model auto-routes to the
# correct endpoint and incompatible LLM_ENDPOINT_FAMILY overrides hard-fail.
_OPENAI_ONLY = frozenset({"openai"})
_ANTHROPIC_ONLY = frozenset({"anthropic"})

_REGISTRY: dict[tuple[str, str], ModelCapability] = {}


def _register(cap: ModelCapability) -> None:
    _REGISTRY[(cap.provider, cap.model_id)] = cap


# --- Z.AI ---
# Existing models the project has been running on. Endpoint family is openai-only
# because Z.AI's coding paas v4 is the OpenAI-compatible endpoint.
_register(ModelCapability(
    provider="zai", model_id="glm-5",
    endpoint_families=frozenset({"openai"}),
    thinking_supported=True,
    thinking_dialect="zai-extra-body",
    tier="production",
))
_register(ModelCapability(
    provider="zai", model_id="glm-5.1",
    endpoint_families=frozenset({"openai"}),
    thinking_supported=True,
    thinking_dialect="zai-extra-body",
    tier="production",
))
_register(ModelCapability(
    provider="zai", model_id="glm-5.2",
    endpoint_families=frozenset({"openai"}),
    thinking_supported=True,
    thinking_dialect="zai-extra-body",
    tier="production",
    notes="GLM-5.2 direct API model. The Claude Code-only [1m] suffix is not accepted by this endpoint.",
))
_register(ModelCapability(
    provider="zai", model_id="glm-4.5-air",
    endpoint_families=frozenset({"openai"}),
    thinking_supported=False,
    thinking_dialect="none",
    tier="production",
    notes="Lighter/faster GLM used for decomposition stage in some configs.",
))


# --- OpenCode Go ---
# Model list and per-model endpoint family per https://opencode.ai/docs/go/
# (12 models). Context windows are not published per model in the OpenCode
# docs, so we leave context_window=None. thinking_supported is set to True
# only where the underlying model is known to expose extended thinking.
_OPENCODE_GO_MODELS: tuple[tuple[str, "frozenset", "Optional[bool]", str, str, str], ...] = (
    # (model_id, endpoint_families, thinking_supported, thinking_dialect, tier, notes)
    # GLM on OpenCode Go proxies to Z.AI, so the Z.AI extra_body shape works.
    # Tier: production — same code path and dialect we already use against Z.AI.
    ("glm-5",             _OPENAI_ONLY,    True,  "zai-extra-body", "production",
        "GLM-5 family exposes extended thinking via Z.AI upstream."),
    ("glm-5.1",           _OPENAI_ONLY,    True,  "zai-extra-body", "production",
        "GLM-5.1 family exposes extended thinking via Z.AI upstream."),
    # Kimi K2 / DeepSeek / MiMo / Qwen are reasoning-native — they think
    # internally without needing a specific request flag. We omit the
    # thinking payload to avoid shipping Z.AI-shaped fields they don't accept.
    # Tier: experimental — wire format is correct but not exercised live yet.
    ("kimi-k2.5",         _OPENAI_ONLY,    None,  "none", "experimental", ""),
    ("kimi-k2.6",         _OPENAI_ONLY,    None,  "none", "experimental", ""),
    ("mimo-v2.5",         _OPENAI_ONLY,    None,  "none", "experimental", ""),
    ("mimo-v2.5-pro",     _OPENAI_ONLY,    None,  "none", "experimental", ""),
    ("qwen3.5-plus",      _OPENAI_ONLY,    None,  "none", "experimental", ""),
    ("qwen3.6-plus",      _OPENAI_ONLY,    None,  "none", "experimental", ""),
    ("deepseek-v4-pro",   _OPENAI_ONLY,    None,  "none", "experimental", ""),
    ("deepseek-v4-flash", _OPENAI_ONLY,    None,  "none", "experimental", ""),
    # MiniMax routes through /messages so the standard Anthropic
    # extended-thinking shape is the right payload.
    # Tier: experimental — Anthropic-compat path is wired but not exercised live.
    ("minimax-m2.5",      _ANTHROPIC_ONLY, None,  "anthropic-thinking", "experimental",
        "Routed through OpenCode Go /messages (Anthropic-compat)."),
    ("minimax-m2.7",      _ANTHROPIC_ONLY, None,  "anthropic-thinking", "experimental",
        "Routed through OpenCode Go /messages (Anthropic-compat)."),
)

for _mid, _families, _thinking, _dialect, _tier, _notes in _OPENCODE_GO_MODELS:
    _register(ModelCapability(
        provider="opencode-go",
        model_id=_mid,
        endpoint_families=_families,
        thinking_supported=_thinking,
        thinking_dialect=_dialect,
        tier=_tier,
        notes=_notes,
    ))


# --- Recommended stage routes ---
#
# Named profiles that fill the four pipeline stages with sensible defaults.
# Per-stage env vars (MODEL_DECOMPOSE / MODEL_ANALYZE / MODEL_SYNTHESIZE /
# MODEL_GAP_ANALYSIS) still win when set; the profile is the fallback.
#
# Conservative bias: the default profile uses production-tier models only.
# "quality" and "budget" profiles deliberately reach into experimental tier;
# we surface that at startup so the user knows what they signed up for.

PIPELINE_STAGES: tuple[str, ...] = ("decompose", "analyze", "gap_analysis", "synthesize")


class StageRoute(BaseModel):
    decompose: str
    analyze: str
    gap_analysis: str
    synthesize: str

    def as_dict(self) -> dict[str, str]:
        return {
            "decompose": self.decompose,
            "analyze": self.analyze,
            "gap_analysis": self.gap_analysis,
            "synthesize": self.synthesize,
        }


# Keyed by (provider, profile_name).
RECOMMENDED_ROUTES: dict[tuple[str, str], StageRoute] = {
    # Z.AI: use GLM-5.2 for every stage by default.
    ("zai", "balanced"): StageRoute(
        decompose="glm-5.2",
        analyze="glm-5.2",
        gap_analysis="glm-5.2",
        synthesize="glm-5.2",
    ),
    # OpenCode Go: only the GLMs are production-tier here, so the safe
    # default keeps every stage on glm-5.1. No model mixing, no surprises.
    ("opencode-go", "balanced"): StageRoute(
        decompose="glm-5.1",
        analyze="glm-5.1",
        gap_analysis="glm-5.1",
        synthesize="glm-5.1",
    ),
    # Cheaper iterative loop with experimental models for decomposition and
    # source analysis (high volume), keeping GLM-5.1 for the
    # quality-sensitive gap-analysis and synthesis stages.
    ("opencode-go", "budget"): StageRoute(
        decompose="deepseek-v4-flash",
        analyze="deepseek-v4-flash",
        gap_analysis="glm-5.1",
        synthesize="glm-5.1",
    ),
    # Bias toward the strongest reasoning model we have wired in for the
    # synthesis stage; iterative stages stay on GLM-5.1.
    ("opencode-go", "quality"): StageRoute(
        decompose="glm-5.1",
        analyze="glm-5.1",
        gap_analysis="glm-5.1",
        synthesize="deepseek-v4-pro",
    ),
}

DEFAULT_PROFILE: str = "balanced"


def known_profiles(provider: str) -> list[str]:
    return sorted(name for (p, name) in RECOMMENDED_ROUTES if p == provider)


def resolve_stage_route(provider: str, profile: str) -> StageRoute:
    """Look up the (provider, profile) route or raise CapabilityViolation."""
    route = RECOMMENDED_ROUTES.get((provider, profile))
    if route is None:
        available = known_profiles(provider) or ["(none)"]
        raise CapabilityViolation(
            f"No route profile {profile!r} defined for provider {provider!r}. "
            f"Available: {', '.join(available)}"
        )
    return route


# --- Public API ---

def lookup(provider: str, model_id: str) -> ModelCapability | None:
    return _REGISTRY.get((provider, model_id))


def known_models(provider: str) -> list[str]:
    return sorted(m for (p, m) in _REGISTRY if p == provider)


def resolve_endpoint_family(
    provider: str,
    model_id: str,
    requested: str | None,
) -> str:
    """
    Pick the endpoint family to use for (provider, model_id).

    - If the model's capability entry is single-family, return that family.
      If `requested` was given and conflicts, raise CapabilityViolation.
    - If multi-family, honor `requested` (or default to first listed).
    - If the model is unknown, fall back to `requested` or "openai" with a warning.
    """
    cap = lookup(provider, model_id)
    if cap is None:
        family = requested or "openai"
        logger.warning(
            "Model %s/%s not in capability registry; using endpoint family %r without validation.",
            provider, model_id, family,
        )
        return family

    if requested:
        if not cap.supports_family(requested):
            raise CapabilityViolation(
                f"Model {provider}/{model_id} does not support endpoint family "
                f"{requested!r}. Supported: {sorted(cap.endpoint_families)}"
            )
        return requested

    # No explicit family requested — pick the first listed.
    # frozenset has no order, so sort to stay deterministic, but prefer
    # "openai" when available since it's the more universally tested path here.
    if "openai" in cap.endpoint_families:
        return "openai"
    return sorted(cap.endpoint_families)[0]


def validate_route(provider: str, model_id: str, endpoint_family: str) -> ModelCapability | None:
    """
    Verify (provider, model_id, endpoint_family) is a valid combination.

    Returns the capability entry if known. Returns None and emits a warning if
    the model is not in the registry. Raises CapabilityViolation if the model
    is known but the requested family is unsupported, or if the family is not
    one this codebase implements at all.
    """
    if endpoint_family not in KNOWN_ENDPOINT_FAMILIES:
        raise CapabilityViolation(
            f"Endpoint family {endpoint_family!r} is not implemented. "
            f"Known: {sorted(KNOWN_ENDPOINT_FAMILIES)}"
        )

    cap = lookup(provider, model_id)
    if cap is None:
        logger.warning(
            "Model %s/%s is not registered; skipping capability validation. "
            "Known models for %s: %s",
            provider, model_id, provider, ", ".join(known_models(provider)) or "(none)",
        )
        return None

    if not cap.supports_family(endpoint_family):
        raise CapabilityViolation(
            f"Model {provider}/{model_id} does not support endpoint family "
            f"{endpoint_family!r}. Supported: {sorted(cap.endpoint_families)}"
        )
    return cap


def thinking_dialect(provider: str, model_id: str) -> str:
    """Return the model's thinking dialect, defaulting to 'none' when unknown."""
    cap = lookup(provider, model_id)
    if cap is None:
        return "none"
    return cap.thinking_dialect


def resolve_thinking(
    provider: str,
    model_id: str,
    requested: bool,
) -> bool:
    """
    Gate the `thinking` flag against per-model support.

    Returns False (with a warning) when the caller asked for thinking but the
    model is registered as not supporting it. Unknown/None capability entries
    pass the flag through unchanged.
    """
    if not requested:
        return False
    cap = lookup(provider, model_id)
    if cap is None or cap.thinking_supported is None:
        return True
    if not cap.thinking_supported:
        logger.warning(
            "Model %s/%s does not support thinking mode; ignoring thinking=True.",
            provider, model_id,
        )
        return False
    return True


def describe_route(
    provider: str,
    model_id: str,
    endpoint_family: str,
    *,
    extra_models: Iterable[str] = (),
) -> dict[str, object]:
    """Build a debug-friendly dict describing the routing decision."""
    cap = lookup(provider, model_id)
    info: dict[str, object] = {
        "provider": provider,
        "model": model_id,
        "endpoint_family": endpoint_family,
        "registered": cap is not None,
    }
    if cap is not None:
        info["supported_families"] = sorted(cap.endpoint_families)
        info["thinking_supported"] = cap.thinking_supported
        info["thinking_dialect"] = cap.thinking_dialect
        info["context_window"] = cap.context_window
        info["tier"] = cap.tier
    extras = [m for m in extra_models if m]
    if extras:
        info["stage_models"] = extras
    return info
