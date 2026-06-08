from __future__ import annotations

import asyncio
import logging

import httpx

from deep_research.models import SearchResult, SourceType

from .base import SearchProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,url,abstract,authors,year,externalIds,citationCount,venue,openAccessPdf"
_MAX_RETRIES = 3


class SemanticScholarProvider(SearchProvider):
    """Search academic papers via the Semantic Scholar API."""

    source_type = SourceType.SEMANTIC_SCHOLAR

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "User-Agent": "deep-research/0.3 (https://github.com/ferhatyurdakul/Deep-Research)"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        params = {"query": query, "limit": max_results, "fields": _FIELDS}

        async with httpx.AsyncClient(timeout=15.0) as client:
            data = {}
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.get(
                        f"{_BASE_URL}/paper/search",
                        params=params,
                        headers=self._headers(),
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if attempt == _MAX_RETRIES - 1:
                        logger.warning(f"Semantic Scholar search failed: {e}")
                        return []
                    await asyncio.sleep(2 ** attempt)

        results: list[SearchResult] = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in (paper.get("authors") or [])]
            year = paper.get("year")
            ext_ids = paper.get("externalIds") or {}
            abstract = paper.get("abstract") or ""
            pdf_info = paper.get("openAccessPdf") or {}

            url = paper.get("url", "")
            arxiv_id = ext_ids.get("ArXiv", "")
            if arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"

            doi = ext_ids.get("DOI", "")

            if paper.get("title") and url:
                results.append(SearchResult(
                    title=paper["title"],
                    url=url,
                    snippet=abstract[:500].strip(),
                    score=0.0,
                    source_type=SourceType.SEMANTIC_SCHOLAR,
                    authors=authors,
                    published_date=str(year) if year else "",
                    extra={
                        "doi": doi,
                        "arxiv_id": arxiv_id,
                        "venue": paper.get("venue", "") or "",
                        "citation_count": paper.get("citationCount", 0),
                        "pdf_url": pdf_info.get("url", ""),
                    },
                ))

        logger.info(f"Semantic Scholar: found {len(results)} results for '{query}'")
        return results
