from __future__ import annotations

import json
import logging

from openai import (
    OpenAI, AsyncOpenAI,
    APITimeoutError, APIConnectionError, RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import settings

logger = logging.getLogger(__name__)

_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)
_RETRY_KWARGS = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE),
    before_sleep=lambda rs: logger.warning(f"LLM call failed, retrying (attempt {rs.attempt_number})..."),
)


def get_client() -> OpenAI:
    return OpenAI(api_key=settings.zai_api_key, base_url=settings.zai_base_url)


def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.zai_api_key, base_url=settings.zai_base_url)


def _build_kwargs(
    prompt: str,
    system: str,
    model: str | None,
    temperature: float,
    thinking: bool,
) -> dict:
    kwargs = dict(
        model=model or settings.glm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    if thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "clear_thinking": True}}
    return kwargs


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
    client = get_client()
    kwargs = _build_kwargs(prompt, system, model, temperature, thinking)
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


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
    client = get_async_client()
    kwargs = _build_kwargs(prompt, system, model, temperature, thinking)
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


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
