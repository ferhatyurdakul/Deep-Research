from __future__ import annotations

import json
import logging

from openai import (
    OpenAI, AsyncOpenAI,
    APITimeoutError, APIConnectionError, RateLimitError,
    BadRequestError as _OpenAIBadRequestError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .capabilities import (
    describe_route,
    resolve_endpoint_family,
    resolve_thinking,
    thinking_dialect,
    validate_route,
)
from .config import settings

logger = logging.getLogger(__name__)

_RETRYABLE: tuple[type[BaseException], ...] = (APITimeoutError, APIConnectionError, RateLimitError)

_BAD_REQUEST_TYPES: tuple[type[BaseException], ...] = (_OpenAIBadRequestError,)

try:
    from anthropic import (
        Anthropic, AsyncAnthropic,
        APITimeoutError as _AnthropicTimeout,
        APIConnectionError as _AnthropicConnection,
        RateLimitError as _AnthropicRate,
        BadRequestError as _AnthropicBadRequestError,
    )
    _ANTHROPIC_AVAILABLE = True
    _RETRYABLE = _RETRYABLE + (_AnthropicTimeout, _AnthropicConnection, _AnthropicRate)
    _BAD_REQUEST_TYPES = _BAD_REQUEST_TYPES + (_AnthropicBadRequestError,)
except ImportError:
    _ANTHROPIC_AVAILABLE = False


class UpstreamRequestError(RuntimeError):
    """
    Wraps a 400-class error from the upstream provider with the routing
    context so the cause is visible in one log line instead of three.

    Attributes:
        provider, model, endpoint_family, dialect: the resolved route
        original: the underlying SDK exception (also chained via __cause__)
    """

    def __init__(
        self,
        provider: str,
        model: str,
        endpoint_family: str,
        dialect: str,
        original: BaseException,
    ) -> None:
        self.provider = provider
        self.model = model
        self.endpoint_family = endpoint_family
        self.dialect = dialect
        self.original = original
        super().__init__(
            f"Upstream rejected request for {provider}/{model} "
            f"(family={endpoint_family}, thinking_dialect={dialect}): {original}"
        )

_RETRY_KWARGS = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE),
    before_sleep=lambda rs: logger.warning(f"LLM call failed, retrying (attempt {rs.attempt_number})..."),
)

_ANTHROPIC_MAX_TOKENS = 8192


def _require_anthropic() -> None:
    if not _ANTHROPIC_AVAILABLE:
        raise RuntimeError(
            "LLM_ENDPOINT_FAMILY=anthropic requires the 'anthropic' package. "
            "Install with: pip install anthropic"
        )


# --- Client factories ---

def get_client() -> OpenAI:
    return OpenAI(api_key=settings.active_api_key, base_url=settings.active_base_url)


def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.active_api_key, base_url=settings.active_base_url)


def get_anthropic_client() -> "Anthropic":
    _require_anthropic()
    return Anthropic(api_key=settings.active_api_key, base_url=settings.active_base_url)


def get_async_anthropic_client() -> "AsyncAnthropic":
    _require_anthropic()
    return AsyncAnthropic(api_key=settings.active_api_key, base_url=settings.active_base_url)


# --- Per-call routing ---

# Sentinel: log a one-line routing summary the first time chat()/achat() runs,
# so post-hoc debugging can recover the resolved provider+model+family combo
# without crawling every call site.
_ROUTING_LOGGED: set[tuple[str, str, str]] = set()


class _Route:
    __slots__ = ("provider", "model", "family", "thinking", "dialect")

    def __init__(self, provider: str, model: str, family: str, thinking: bool, dialect: str):
        self.provider = provider
        self.model = model
        self.family = family
        self.thinking = thinking
        self.dialect = dialect


def _resolve_route(model: str | None, thinking: bool) -> _Route:
    """
    Resolve the routing for one call.

    Consults the capability map so a single-family model auto-routes to its
    only supported family, thinking is gated by per-model support, and the
    thinking-payload dialect is selected to match the upstream model.
    """
    provider = settings.active_provider
    resolved_model = model or settings.active_model
    requested_family = settings.llm_endpoint_family or None
    family = resolve_endpoint_family(provider, resolved_model, requested_family)
    validate_route(provider, resolved_model, family)
    effective_thinking = resolve_thinking(provider, resolved_model, thinking)
    dialect = thinking_dialect(provider, resolved_model)

    key = (provider, resolved_model, family)
    if key not in _ROUTING_LOGGED:
        _ROUTING_LOGGED.add(key)
        logger.info("LLM route: %s", describe_route(provider, resolved_model, family))
    return _Route(provider, resolved_model, family, effective_thinking, dialect)


def _apply_thinking(kwargs: dict, route: _Route) -> None:
    """
    Inject the thinking payload using the dialect appropriate for the model.

    Generic call options (model, messages, system, temperature, max_tokens)
    are constructed in the _build_*_kwargs helpers; this function is the only
    place where provider-specific extras land in the kwargs dict.

    No-ops when thinking is disabled, when the dialect is "none", or when the
    dialect doesn't match the wire family (logs a warning and degrades to
    the model's native reasoning rather than shipping a payload it can't parse).
    """
    if not route.thinking:
        return
    if route.dialect == "none":
        logger.info(
            "Thinking requested for %s/%s but no verified dialect; relying on the model's native reasoning.",
            route.provider, route.model,
        )
        return
    if route.dialect == "zai-extra-body" and route.family == "openai":
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "clear_thinking": True}}
        return
    if route.dialect == "anthropic-thinking" and route.family == "anthropic":
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        return
    logger.warning(
        "Thinking dialect %r is not compatible with endpoint family %r for %s/%s; "
        "omitting the thinking payload.",
        route.dialect, route.family, route.provider, route.model,
    )


# --- Helpers ---

def _build_openai_kwargs(
    prompt: str,
    system: str,
    model: str,
    temperature: float,
) -> dict:
    """Generic OpenAI Chat Completions kwargs — provider-agnostic fields only."""
    return dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )


def _build_anthropic_kwargs(
    prompt: str,
    system: str,
    model: str,
    temperature: float,
) -> dict:
    """Generic Anthropic Messages kwargs — provider-agnostic fields only."""
    return dict(
        model=model,
        max_tokens=_ANTHROPIC_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )


def _extract_anthropic_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _clean_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned


# --- Sync ---

@retry(**_RETRY_KWARGS)
def chat(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    model: str | None = None,
    temperature: float = 0.7,
    thinking: bool = False,
) -> str:
    route = _resolve_route(model, thinking)
    if route.family == "openai":
        kwargs = _build_openai_kwargs(prompt, system, route.model, temperature)
        _apply_thinking(kwargs, route)
        try:
            response = get_client().chat.completions.create(**kwargs)
        except _BAD_REQUEST_TYPES as e:
            raise UpstreamRequestError(route.provider, route.model, route.family, route.dialect, e) from e
        return response.choices[0].message.content or ""
    if route.family == "anthropic":
        kwargs = _build_anthropic_kwargs(prompt, system, route.model, temperature)
        _apply_thinking(kwargs, route)
        try:
            response = get_anthropic_client().messages.create(**kwargs)
        except _BAD_REQUEST_TYPES as e:
            raise UpstreamRequestError(route.provider, route.model, route.family, route.dialect, e) from e
        return _extract_anthropic_text(response)
    raise ValueError(f"Unsupported endpoint family: {route.family}")


def chat_json(
    prompt: str,
    system: str = "You are a helpful research assistant. Always respond with valid JSON.",
    model: str | None = None,
    max_retries: int = 2,
    thinking: bool = False,
) -> dict | list:
    for attempt in range(max_retries + 1):
        raw = chat(prompt, system=system, model=model, temperature=0.3, thinking=thinking)
        cleaned = _clean_json(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if attempt < max_retries:
                logger.warning(f"JSON parse failed (attempt {attempt + 1}), retrying LLM call...")
            else:
                logger.error(f"JSON parse failed after {max_retries + 1} attempts. Raw: {raw[:200]}")
                raise


# --- Async ---

@retry(**_RETRY_KWARGS)
async def achat(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    model: str | None = None,
    temperature: float = 0.7,
    thinking: bool = False,
) -> str:
    route = _resolve_route(model, thinking)
    if route.family == "openai":
        kwargs = _build_openai_kwargs(prompt, system, route.model, temperature)
        _apply_thinking(kwargs, route)
        try:
            response = await get_async_client().chat.completions.create(**kwargs)
        except _BAD_REQUEST_TYPES as e:
            raise UpstreamRequestError(route.provider, route.model, route.family, route.dialect, e) from e
        return response.choices[0].message.content or ""
    if route.family == "anthropic":
        kwargs = _build_anthropic_kwargs(prompt, system, route.model, temperature)
        _apply_thinking(kwargs, route)
        try:
            response = await get_async_anthropic_client().messages.create(**kwargs)
        except _BAD_REQUEST_TYPES as e:
            raise UpstreamRequestError(route.provider, route.model, route.family, route.dialect, e) from e
        return _extract_anthropic_text(response)
    raise ValueError(f"Unsupported endpoint family: {route.family}")


async def achat_json(
    prompt: str,
    system: str = "You are a helpful research assistant. Always respond with valid JSON.",
    model: str | None = None,
    max_retries: int = 2,
    thinking: bool = False,
) -> dict | list:
    for attempt in range(max_retries + 1):
        raw = await achat(prompt, system=system, model=model, temperature=0.3, thinking=thinking)
        cleaned = _clean_json(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if attempt < max_retries:
                logger.warning(f"JSON parse failed (attempt {attempt + 1}), retrying LLM call...")
            else:
                logger.error(f"JSON parse failed after {max_retries + 1} attempts. Raw: {raw[:200]}")
                raise
