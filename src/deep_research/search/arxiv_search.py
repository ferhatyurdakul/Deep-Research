from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET

import httpx

from deep_research.models import SearchResult, SourceType

from .base import SearchProvider

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"

# arXiv's export API rejects requests without a descriptive User-Agent.
_HEADERS = {"User-Agent": "deep-research/0.3 (https://github.com/ferhatyurdakul/Deep-Research)"}


def _parse_arxiv_entry(entry: ET.Element, ns: dict) -> SearchResult | None:
    title_el = entry.find("atom:title", ns)
    summary_el = entry.find("atom:summary", ns)
    link_el = entry.find("atom:id", ns)
    if title_el is None or link_el is None:
        return None

    title = " ".join((title_el.text or "").split())
    snippet = " ".join((summary_el.text or "").split())[:500]
    url = (link_el.text or "").strip()
    url = url.replace("http://arxiv.org/abs/", "https://arxiv.org/abs/")

    authors = []
    for author_el in entry.findall("atom:author", ns):
        name_el = author_el.find("atom:name", ns)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    published_el = entry.find("atom:published", ns)
    published_date = ""
    if published_el is not None and published_el.text:
        published_date = published_el.text[:10]

    arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else ""

    categories = []
    for cat_el in entry.findall("atom:category", ns):
        term = cat_el.get("term", "")
        if term:
            categories.append(term)

    return SearchResult(
        title=title, url=url, snippet=snippet, score=0.0,
        source_type=SourceType.ARXIV,
        authors=authors,
        published_date=published_date,
        extra={"arxiv_id": arxiv_id, "categories": categories},
    )


class ArxivSearchProvider(SearchProvider):
    """Search arXiv preprints via the arXiv API."""

    source_type = SourceType.ARXIV

    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=_HEADERS) as client:
                response = await client.get(ARXIV_API, params=params)
                response.raise_for_status()

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(response.text)
            results = []
            for entry in root.findall("atom:entry", ns):
                result = _parse_arxiv_entry(entry, ns)
                if result:
                    results.append(result)
            logger.info(f"arXiv: found {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
            return []
