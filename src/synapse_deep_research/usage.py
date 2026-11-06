"""
Per-run usage tracker.

OpenCode Go enforces a $12/5h, $30/week, $60/month dollar cap on subscription
plans, but exposes no public usage-query endpoint. We can't tell the user
"you have $X left." What we *can* do: count our own calls and surface them
at the end of a run so the user can correlate "this query took N analyze
calls + 1 synthesize call" with what their dashboard shows later.

Tokens are not tracked here — not all upstreams populate usage on the
response, and getting consistent token counts across OpenAI-compat and
Anthropic-compat would require per-family parsing. Call counts are
sufficient for the volume-mapping use case the limits docs ask for.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)


class UsageTracker:
    """
    Thread- and asyncio-safe in-memory counter.

    The pipeline runs analyze calls in parallel via asyncio (semaphore-limited
    to 5 concurrent). Counter mutations are guarded by a Lock so the recorded
    totals match the actual call count.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Keys: (provider, model, stage)
        self._calls: Counter[tuple[str, str, str]] = Counter()
        self._fallback_fired: bool = False
        self._fallback_from: Optional[str] = None
        self._fallback_to: Optional[str] = None

    def record_call(self, provider: str, model: str, stage: str) -> None:
        with self._lock:
            self._calls[(provider, model, stage)] += 1

    def record_fallback(self, from_provider: str, to_provider: str) -> None:
        with self._lock:
            self._fallback_fired = True
            self._fallback_from = from_provider
            self._fallback_to = to_provider

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
            self._fallback_fired = False
            self._fallback_from = None
            self._fallback_to = None

    def snapshot(self) -> dict:
        """Return a serializable view of the current totals."""
        with self._lock:
            by_stage: dict[str, int] = {}
            by_model: dict[str, int] = {}
            for (provider, model, stage), n in self._calls.items():
                by_stage[stage] = by_stage.get(stage, 0) + n
                key = f"{provider}/{model}"
                by_model[key] = by_model.get(key, 0) + n
            return {
                "total_calls": sum(self._calls.values()),
                "by_stage": dict(sorted(by_stage.items())),
                "by_model": dict(sorted(by_model.items())),
                "fallback_fired": self._fallback_fired,
                "fallback_from": self._fallback_from,
                "fallback_to": self._fallback_to,
            }

    def format_summary(self) -> str:
        snap = self.snapshot()
        lines = [f"LLM calls: {snap['total_calls']} total"]
        if snap["by_stage"]:
            stage_str = ", ".join(f"{k}={v}" for k, v in snap["by_stage"].items())
            lines.append(f"  by stage: {stage_str}")
        if snap["by_model"]:
            model_str = ", ".join(f"{k}={v}" for k, v in snap["by_model"].items())
            lines.append(f"  by model: {model_str}")
        if snap["fallback_fired"]:
            lines.append(
                f"  fallback: {snap['fallback_from']} → {snap['fallback_to']} (rate-limit)"
            )
        return "\n".join(lines)


# A run-scoped tracker. The default tracker is used when no run context is
# active; the orchestrator pushes a fresh tracker per research run via
# `set_tracker()` and reads it back at the end.
_GLOBAL_TRACKER = UsageTracker()
_TRACKER_VAR: ContextVar[Optional[UsageTracker]] = ContextVar("usage_tracker", default=None)


def get_tracker() -> UsageTracker:
    return _TRACKER_VAR.get() or _GLOBAL_TRACKER


def set_tracker(tracker: UsageTracker) -> None:
    """Bind a per-run tracker. Call once at the start of a research run."""
    _TRACKER_VAR.set(tracker)


def new_run_tracker() -> UsageTracker:
    """Allocate and bind a fresh tracker for the current run."""
    t = UsageTracker()
    set_tracker(t)
    return t


# --- Stage stack (so llm.py can record without every call site passing it) ---
#
# The pipeline calls chat()/achat() inside stage handlers. We want to record
# the stage for usage analytics without changing the public chat() signature.
# A ContextVar holds the current stage; pipeline modules push/pop via the
# `stage()` context manager.

_STAGE_VAR: ContextVar[str] = ContextVar("usage_stage", default="other")


def current_stage() -> str:
    return _STAGE_VAR.get()


class stage:
    """
    Context manager that tags the active pipeline stage for usage tracking.

        async with stage("analyze"):
            await aanalyze_source(...)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._token = None

    def __enter__(self) -> "stage":
        self._token = _STAGE_VAR.set(self._name)
        return self

    def __exit__(self, *exc) -> None:
        _STAGE_VAR.reset(self._token)

    async def __aenter__(self) -> "stage":
        self._token = _STAGE_VAR.set(self._name)
        return self

    async def __aexit__(self, *exc) -> None:
        _STAGE_VAR.reset(self._token)
