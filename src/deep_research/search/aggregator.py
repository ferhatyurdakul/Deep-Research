from __future__ import annotations

import asyncio
import logging
import re

from deep_research.models import SearchResult, SourceType

from .arxiv_search import ArxivSearchProvider
from .base import SearchProvider
from .semantic_scholar import SemanticScholarProvider
from .web_search import DuckDuckGoSearchProvider, TavilySearchProvider

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Normalize a URL for dedup comparisons."""
    url = url.rstrip("/")
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url.lower()


def _title_similarity(a: str, b: str) -> float:
    """Rough title overlap ratio for dedup."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(len(a_words), len(b_words))


def _source_type_priority(source_type: SourceType) -> int:
    """Priority ranking for keeping duplicates (lower = higher priority)."""
    return {
        SourceType.ARXIV: 0,
        SourceType.SEMANTIC_SCHOLAR: 1,
        SourceType.WEB: 2,
    }.get(source_type, 3)


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicates by URL and title similarity. Academic sources take priority."""
    seen_urls: dict[str, SearchResult] = {}
    output: list[SearchResult] = []

    for result in results:
        norm_url = _normalize_url(result.url)

        if norm_url in seen_urls:
            existing = seen_urls[norm_url]
            if _source_type_priority(result.source_type) < _source_type_priority(existing.source_type):
                idx = output.index(existing)
                output[idx] = result
                seen_urls[norm_url] = result
            continue

        is_dup = False
        for existing in output:
            if _title_similarity(result.title, existing.title) > 0.8:
                is_dup = True
                break

        if not is_dup:
            seen_urls[norm_url] = result
            output.append(result)

    output.sort(key=lambda r: (
        _source_type_priority(r.source_type),
        r.published_date or "0000",
    ))
    return output


class SearchAggregator:
    """Orchestrates multiple search providers and deduplicates results."""

    def __init__(
        self,
        tavily_api_key: str = "",
        semantic_scholar_api_key: str = "",
    ) -> None:
        self._providers: dict[str, SearchProvider] = {}
        self._init_providers(tavily_api_key, semantic_scholar_api_key)

    def _init_providers(
        self, tavily_api_key: str, semantic_scholar_api_key: str
    ) -> None:
        # Academic providers (always available, free APIs)
        self._providers["arxiv"] = ArxivSearchProvider()
        self._providers["scholar"] = SemanticScholarProvider(
            api_key=semantic_scholar_api_key
        )

        # Web search: Tavily if key available, else DuckDuckGo as free fallback
        if tavily_api_key:
            self._providers["web"] = TavilySearchProvider(tavily_api_key)
        else:
            self._providers["web"] = DuckDuckGoSearchProvider()

        # DuckDuckGo as additional web source if Tavily is primary
        if tavily_api_key:
            self._providers["ddg"] = DuckDuckGoSearchProvider()

    @property
    def available_sources(self) -> list[str]:
        return list(self._providers.keys())

    async def search(
        self,
        queries: list[str],
        sources: list[str] | None = None,
        max_results_per_query: int = 5,
        max_total: int = 30,
    ) -> list[SearchResult]:
        """Run searches across selected providers and return deduplicated results."""
        active_sources = sources or list(self._providers.keys())

        tasks = []
        for name in active_sources:
            provider = self._providers.get(name)
            if not provider:
                continue
            tasks.append(provider.batch_search(queries, max_results_per_query))

        if not tasks:
            return []

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[SearchResult] = []
        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"Search provider failed: {resp}")
                continue
            all_results.extend(resp)

        deduped = deduplicate_results(all_results)
        return deduped[:max_total]
