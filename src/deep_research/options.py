"""
Option metadata shared by `--help`, the REPL's `/help` and tab completion.

The "what does this value actually do" strings live here once, derived from
the real presets where possible, so the CLI help and the REPL can't drift
apart from each other or from the pipeline.
"""
from __future__ import annotations

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
        extra = ", academic sources" if cfg.use_academic_search else ""
        out.append((
            depth.value,
            f"{_DEPTH_LEAD[depth]}: {cfg.max_sub_questions} sub-questions, {passes}{extra}",
        ))
    return out


def style_choices() -> list[Choice]:
    return [(s.value, _STYLE_BLURB[s]) for s in ReportStyle]


def template_choices() -> list[Choice]:
    return [(key, t.description) for key, t in TEMPLATES.items()]


def academic_reason(depth: str, template: Optional[str], explicit: bool) -> Optional[str]:
    """Why arXiv + Semantic Scholar will be searched, or None when they won't be.

    Mirrors what the pipeline does: the --academic / /academic switch, a
    template that needs academic sources (science, literature), or the deep
    depth preset all turn academic search on. Status displays use this so they
    show what a run will actually do, not just the explicit switch.
    """
    if explicit:
        return "on"
    tmpl = TEMPLATES.get(template) if template else None
    if tmpl is not None and tmpl.use_academic:
        return f"{template} template"
    if ResearchConfig.from_depth(ResearchDepth(depth)).use_academic_search:
        return f"{depth} depth"
    return None


def model_choices(provider: str, preferred: Optional[str] = None) -> list:
    """The provider's registered models with what is known about each,
    `preferred` (the default model) first."""
    from .capabilities import known_models, lookup

    ids = known_models(provider)
    if preferred in ids:
        ids.remove(preferred)
        ids.insert(0, preferred)
    out = []
    for model_id in ids:
        cap = lookup(provider, model_id)
        bits = [cap.tier]
        if cap.thinking_supported is not None:
            bits.append("thinking" if cap.thinking_supported else "no thinking")
        blurb = ", ".join(bits) + (f" — {cap.notes}" if cap.notes else "")
        out.append((model_id, blurb))
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
