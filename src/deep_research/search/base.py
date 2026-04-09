from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from deep_research.models import SearchResult, SourceType


class SearchProvider(ABC):
    """Abstract base for all search providers."""

    source_type: SourceType

    @abstractmethod
    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        """Search for results matching the query."""
        ...

    async def batch_search(
        self, queries: list[str], max_results_per_query: int = 5
    ) -> list[SearchResult]:
        """Run multiple searches and flatten results."""
        tasks = [
            self.search(q, max_results=max_results_per_query) for q in queries
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[SearchResult] = []
        for resp in responses:
            if isinstance(resp, Exception):
                continue
            results.extend(resp)

        return results
