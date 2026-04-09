from __future__ import annotations

import asyncio
import logging

import httpx

from deep_research.models import SearchResult, SourceType

from .base import SearchProvider

logger = logging.getLogger(__name__)


class TavilySearchProvider(SearchProvider):
    """Web search via Tavily API (higher quality, requires API key)."""

    source_type = SourceType.WEB

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        from tavily import AsyncTavilyClient

        try:
            client = AsyncTavilyClient(api_key=self.api_key)
            response = await client.search(
                query=query, max_results=max_results, include_answer=False
            )
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return []

        results: list[SearchResult] = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score", 0.0),
                    source_type=SourceType.WEB,
                )
            )
        return results


class DuckDuckGoSearchProvider(SearchProvider):
    """Free web search via DuckDuckGo (no API key required)."""

    source_type = SourceType.WEB

    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        loop = asyncio.get_event_loop()
        try:
            from ddgs import DDGS

            raw = await loop.run_in_executor(
                None, lambda: DDGS().text(query, max_results=max_results)
            )
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []

        results: list[SearchResult] = []
        for item in raw:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    source_type=SourceType.WEB,
                )
            )
        return results
