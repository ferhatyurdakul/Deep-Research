"""
Option metadata shared by `--help`, the REPL's `/help` and tab completion.

The "what does this value actually do" strings live here once, derived from
the real presets where possible, so the CLI help and the REPL can't drift
apart from each other or from the pipeline.
"""
from __future__ import annotations

import re
from typing import Optional

from .models import ReportStyle, ResearchConfig, ResearchDepth
from .pipeline.templates import TEMPLATES

Choice = tuple  # (value, description)

_DEPTH_LEAD = {
    ResearchDepth.QUICK: "Fast scan",
    ResearchDepth.STANDARD: "Balanced",
    ResearchDepth.DEEP: "Thorough",
}

_STYLE_BLURB = {
    ReportStyle.BRIEF: "~500-1000 words: summary and key findings",
    ReportStyle.STANDARD: "~2000-4000 words, themed sections",
    ReportStyle.ACADEMIC: "~3000-5000 words: abstract, methodology, discussion",
}


def depth_choices() -> list[Choice]:
    out = []
    for depth in ResearchDepth:
        cfg = ResearchConfig.from_depth(depth)
        passes = f"{cfg.max_iterations} pass" + ("" if cfg.max_iterations == 1 else "es")
        extra = ", research papers" if cfg.use_academic_search else ""
        out.append((
            depth.value,
            f"{_DEPTH_LEAD[depth]}: {cfg.max_sub_questions} sub-questions, {passes}{extra}",
        ))
    return out


def academic_choices(depth: Optional[str] = None, template: Optional[str] = None) -> list[Choice]:
    """The three modes of the papers setting. With `depth` (and `template`), `auto` also says
    what it amounts to right now — the part that is otherwise invisible."""
    auto = "Follow the template and depth"
    if depth is not None:
        reason = academic_reason(depth, template, None)
        auto += f" — now {'on (' + reason + ')' if reason else 'off'}"
    return [
        ("auto", auto),
        ("on", "Always search research papers too (arXiv, Semantic Scholar)"),
        ("off", "Web search only, even on deep depth or the science template"),
    ]


def style_choices() -> list[Choice]:
    return [(s.value, _STYLE_BLURB[s]) for s in ReportStyle]


def template_choices() -> list[Choice]:
    return [(key, t.description) for key, t in TEMPLATES.items()]


ACADEMIC_MODES = ("auto", "on", "off")


def academic_mode_name(mode: Optional[bool]) -> str:
    return "auto" if mode is None else ("on" if mode else "off")


def academic_mode_value(name: str) -> Optional[bool]:
    return {"auto": None, "on": True, "off": False}[name]


def academic_reason(depth: str, template: Optional[str], mode: Optional[bool] = None) -> Optional[str]:
    """Why arXiv + Semantic Scholar will be searched, or None when they won't be.

    `mode` is the user's say: True (on) and False (off) are final; None (auto)
    leaves it to what the pipeline does by itself — a template that needs
    academic sources (science, literature) or the deep depth preset. Status
    displays use this so they show what a run will actually do.
    """
    if mode is not None:
        return "on" if mode else None
    tmpl = TEMPLATES.get(template) if template else None
    if tmpl is not None and tmpl.use_academic:
        return f"{template} template"
    if ResearchConfig.from_depth(ResearchDepth(depth)).use_academic_search:
        return f"{depth} depth"
    return None


_NOT_TEXT_MODELS = ("embed", "ocr", "image", "video", "vision", "voice", "asr", "tts", "rerank", "cogview")
_ACCOUNT_CACHE: dict = {}
_ACCOUNT_CACHE_SECONDS = 300


def _looks_like_a_text_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(tag in lowered for tag in _NOT_TEXT_MODELS) and not re.search(r"\dv(-|$)", lowered)


def fetch_account_models(provider: str, api_key: Optional[str] = None, timeout: float = 4.0) -> Optional[list]:
    """The model IDs the provider says this account can use (the OpenAI-compatible
    `GET /models`), or None if that isn't available — no network, no key, an
    endpoint without the route, an unexpected reply. Callers fall back to the
    built-in list, so this never raises. Results are cached for a few minutes."""
    import time

    import httpx

    from .config import settings

    try:
        base = {"zai": settings.zai_base_url, "opencode-go": settings.opencode_base_url}[provider].rstrip("/")
        key = api_key or {"zai": settings.zai_api_key, "opencode-go": settings.opencode_api_key}[provider]
    except (KeyError, AttributeError):
        return None
    if not key:
        return None
    cache_key = (provider, base, key[-6:])
    cached = _ACCOUNT_CACHE.get(cache_key)
    if cached and time.monotonic() - cached[0] < _ACCOUNT_CACHE_SECONDS:
        return cached[1]
    try:
        reply = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        reply.raise_for_status()
        items = reply.json().get("data")
        ids = [i["id"] for i in items if isinstance(i, dict) and isinstance(i.get("id"), str)]
    except Exception:  # noqa: BLE001 — any failure just means "use the built-in list"
        return None
    if not ids:
        return None
    _ACCOUNT_CACHE[cache_key] = (time.monotonic(), ids)
    return ids


def model_choices(provider: str, stage: Optional[str] = None, account: Optional[list] = None) -> list:
    """The provider's current models with what is known about each. With
    `stage`, the model recommended for that stage is tagged. With `account` (the
    IDs from fetch_account_models), registry models the account lacks are tagged
    and models it has that we haven't verified are appended."""
    from .capabilities import listed_models, lookup, recommended_stage_models

    try:
        recommended = recommended_stage_models(provider).as_dict().get(stage) if stage else None
    except Exception:
        recommended = None
    out = []
    for model_id in listed_models(provider):
        cap = lookup(provider, model_id)
        bits = [cap.tier]
        if cap.thinking_supported is not None:
            bits.append("thinking" if cap.thinking_supported else "no thinking")
        blurb = ", ".join(bits) + (f" — {cap.notes}" if cap.notes else "")
        if model_id == recommended:
            blurb = f"recommended · {blurb}"
        if account is not None and model_id not in account:
            blurb = f"not on your account · {blurb}"
        out.append((model_id, blurb))
    if account is not None:
        for model_id in account:
            if lookup(provider, model_id) is None and _looks_like_a_text_model(model_id):
                out.append((model_id, "on your account · not verified here"))
    return out


def stage_summary(stages: dict) -> str:
    """One line for what a run uses: `glm-5.3` when every stage shares a model,
    else the most common one plus the exceptions —
    `glm-5.3 · glm-4.5-air for decompose`."""
    if not stages:
        return "?"
    order = list(stages.values())
    primary = max(dict.fromkeys(order), key=lambda m: (order.count(m), -order.index(m)))
    others: dict = {}
    for stage, model in stages.items():
        if model != primary:
            others.setdefault(model, []).append("gap" if stage == "gap_analysis" else stage)
    if not others:
        return primary
    extras = ", ".join(f"{model} for {'+'.join(names)}" for model, names in others.items())
    return f"{primary} · {extras}"


# Keyed by argparse `dest` / REPL command name.
CHOICE_DOCS = {
    "depth": depth_choices,
    "style": style_choices,
    "template": template_choices,
}

DEPTH_VALUES = [d.value for d in ResearchDepth]
STYLE_VALUES = [s.value for s in ReportStyle]
TEMPLATE_VALUES = list(TEMPLATES)


# What each pipeline stage does, in the user's terms — shown wherever a model
# is chosen, so the choice isn't blind.
STAGE_BLURBS = {
    "decompose": "splits your question into sub-questions (1 call)",
    "analyze": "reads each source and extracts findings (most calls)",
    "gap_analysis": "decides what is still missing, or when to stop",
    "synthesize": "writes the final cited report (1 long call)",
}
