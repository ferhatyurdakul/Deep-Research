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
    # Shown in model pickers. Older models stay registered (so existing configs
    # validate and keep their capability metadata) but are hidden from the lists.
    listed: bool = True

    def supports_family(self, family: str) -> bool:
        return family in self.endpoint_families


class CapabilityViolation(ValueError):
    """Raised when a configured provider/model/family combination is invalid."""


# OpenCode Go is switched off for now: Z.AI is the only provider we ship and
# OpenCode Go's model list is out of date. Everything specific to it (models,
# recommended models, the provider itself, its managed API key, the setup
# choice) is gated on this one flag, and the tests that need it skip while it is
# False. To bring it back, refresh the model list below, set this to True and
# restore its docs (README, .env.example) from git history.
OPENCODE_GO_ENABLED = False

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
# Current lineup per https://docs.z.ai/guides/overview/pricing and the model
# pages (checked 2026-10-25). All of them are reached through the
# OpenAI-compatible endpoint, with thinking requested via the zai-extra-body
# payload. glm-5.3 and glm-5.3-flash force thinking on (it cannot be disabled).
#
# Tier: "production" only for models exercised live from this codebase; the
# rest are "experimental" until someone runs them through the pipeline.
# `listed=False` keeps a legacy model registered but out of the pickers.
_ZAI_MODELS: tuple[tuple[str, "Optional[bool]", str, bool, str], ...] = (
    # (model_id, thinking_supported, tier, listed, notes)
    ("glm-5.3",        True,  "production",   True,
        "Flagship. 1M context. Thinking is always on."),
    ("glm-5.3-flash",  True,  "production",   True,
        "Low-cost tier, stronger than GLM-5.2. Coding Plan: 3x the quota of glm-5.3."),
    ("glm-5.3-flashx", True,  "experimental", True,
        "Speed tier. Rejected on the Coding Plan (error 1311) even though it is listed."),
    ("glm-5.2",        True,  "production",   True,
        "Previous generation."),
    ("glm-5.1",        True,  "production",   True,
        "Previous generation."),
    ("glm-5",          True,  "production",   False, "Legacy."),
    ("glm-4.5-air",    False, "production",   False, "Legacy light model; no thinking mode."),
)

for _mid, _thinking, _tier, _listed, _notes in _ZAI_MODELS:
    _register(ModelCapability(
        provider="zai",
        model_id=_mid,
        endpoint_families=frozenset({"openai"}),
        thinking_supported=_thinking,
        thinking_dialect="zai-extra-body" if _thinking else "none",
        tier=_tier,
        listed=_listed,
        notes=_notes,
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

for _mid, _families, _thinking, _dialect, _tier, _notes in (_OPENCODE_GO_MODELS if OPENCODE_GO_ENABLED else ()):
    _register(ModelCapability(
        provider="opencode-go",
        model_id=_mid,
        endpoint_families=_families,
        thinking_supported=_thinking,
        thinking_dialect=_dialect,
        tier=_tier,
        notes=_notes,
    ))


# --- Pipeline stages and recommended models ---
#
# The pipeline has four stages that call the LLM. Which model each one uses is
# the user's choice (setup, `/model`, or MODEL_<STAGE> in the environment); the
# table below is only what setup pre-selects and what a run falls back to when
# nothing has been chosen yet.
#
# Z.AI: the many `analyze` calls (one per source) and the simple `decompose`
# call use the low-cost flash model; the rare, quality-critical `gap_analysis`
# and `synthesize` calls use the flagship.
# OpenCode Go: only the GLMs are production-tier there, so every stage uses
# glm-5.1 (no model mixing, no surprises); not revisited yet.

PIPELINE_STAGES: tuple[str, ...] = ("decompose", "analyze", "gap_analysis", "synthesize")


class StageModels(BaseModel):
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


RECOMMENDED_STAGE_MODELS: dict[str, StageModels] = {
    "zai": StageModels(
        decompose="glm-5.3-flash",
        analyze="glm-5.3-flash",
        gap_analysis="glm-5.3",
        synthesize="glm-5.3",
    ),
}
if OPENCODE_GO_ENABLED:
    RECOMMENDED_STAGE_MODELS["opencode-go"] = StageModels(
        decompose="glm-5.1",
        analyze="glm-5.1",
        gap_analysis="glm-5.1",
        synthesize="glm-5.1",
    )


def recommended_stage_models(provider: str) -> StageModels:
    """The recommended model for each stage, or CapabilityViolation for an unknown provider."""
    rec = RECOMMENDED_STAGE_MODELS.get(provider)
    if rec is None:
        raise CapabilityViolation(
            f"No recommended models for provider {provider!r}. "
            f"Known: {', '.join(sorted(RECOMMENDED_STAGE_MODELS))}"
        )
    return rec


def lookup(provider: str, model_id: str) -> ModelCapability | None:
    return _REGISTRY.get((provider, model_id))


def known_models(provider: str) -> list[str]:
    return sorted(m for (p, m) in _REGISTRY if p == provider)


def listed_models(provider: str) -> list[str]:
    """Models to offer in pickers, in registration order (flagship first). Legacy
    entries stay registered, so existing configs validate, but are hidden here."""
    return [m for (p, m), cap in _REGISTRY.items() if p == provider and cap.listed]


def registered_providers_for(model_id: str) -> list[str]:
    """All providers that have `model_id` registered (any tier)."""
    return sorted(p for (p, m) in _REGISTRY if m == model_id)


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
