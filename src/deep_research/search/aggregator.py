from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

from deep_research.models import SearchResult, SourceType

from .arxiv_search import ArxivSearchProvider
from .base import SearchProvider
from .semantic_scholar import SemanticScholarProvider
from .web_search import DuckDuckGoSearchProvider, TavilySearchProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL / ID helpers
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """Normalize a URL for dedup comparisons."""
    url = url.rstrip("/")
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url.lower()


def _extract_arxiv_id(result: SearchResult) -> str:
    """Return an arXiv ID from the result's extra dict or URL."""
    aid = (result.extra.get("arxiv_id") or "").strip()
    if aid:
        return aid.lower()
    m = re.search(r"arxiv\.org/abs/([^\s?#]+)", result.url)
    return m.group(1).lower() if m else ""


def _extract_doi(result: SearchResult) -> str:
    """Return a DOI from the result's extra dict."""
    doi = (result.extra.get("doi") or "").strip()
    return doi.lower() if doi else ""


# ---------------------------------------------------------------------------
# Title similarity
# ---------------------------------------------------------------------------

def _title_similarity(a: str, b: str) -> float:
    """Rough title overlap ratio for dedup."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(len(a_words), len(b_words))


# ---------------------------------------------------------------------------
# Source-type priority (for picking the winner among duplicates)
# ---------------------------------------------------------------------------

def _source_type_priority(source_type: SourceType) -> int:
    """Priority ranking for keeping duplicates (lower = higher priority)."""
    return {
        SourceType.ARXIV: 0,
        SourceType.SEMANTIC_SCHOLAR: 1,
        SourceType.WEB: 2,
    }.get(source_type, 3)


# ---------------------------------------------------------------------------
# Domain reliability
# ---------------------------------------------------------------------------

_HIGH_RELIABILITY_DOMAINS = {
    # Government
    "gov", "edu", "ac.uk", "gov.uk", "europa.eu",
    # Major publishers / knowledge bases
    "nature.com", "science.org", "sciencedirect.com", "springer.com",
    "wiley.com", "ieee.org", "acm.org", "plos.org", "bmj.com",
    "thelancet.com", "nejm.org", "pubmed.ncbi.nlm.nih.gov",
    "nih.gov", "who.int", "cdc.gov",
    # Established tech/news
    "arxiv.org", "semanticscholar.org", "wikipedia.org",
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
}

_LOW_RELIABILITY_PATTERNS = re.compile(
    r"(pinterest\.|quora\.com|yahoo\.com/answers|ehow\.|wikihow\."
    r"|medium\.com/@|blogspot\.|wordpress\.com(?!/)|tumblr\.com)",
    re.I,
)


def domain_reliability_score(url: str) -> float:
    """Return a reliability score in [0, 1] based on domain heuristics.

    1.0 = highly reliable, 0.5 = neutral, 0.2 = low-quality.
    """
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return 0.5
    hostname = hostname.lower()

    # Check high-reliability domains / TLDs
    for domain in _HIGH_RELIABILITY_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return 1.0

    # Check low-reliability patterns
    if _LOW_RELIABILITY_PATTERNS.search(url):
        return 0.2

    # .org and .edu subdomains not in explicit list still get a small bump
    if hostname.endswith(".org") or hostname.endswith(".edu"):
        return 0.8

    return 0.5


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_source_score(result: SearchResult) -> float:
    """Compute a composite relevance/quality score for ranking.

    Components:
    - provider_score: raw relevance score from the search provider (0-1)
    - source_type_bonus: academic sources get a bump
    - recency_bonus: newer sources score higher
    - citation_bonus: highly-cited papers score higher
    - domain_reliability: domain trustworthiness
    """
    # Provider relevance score (Tavily provides 0-1, others default to 0)
    provider = min(result.score, 1.0)

    # Source type bonus
    type_bonus = {
        SourceType.ARXIV: 0.15,
        SourceType.SEMANTIC_SCHOLAR: 0.15,
        SourceType.WEB: 0.0,
    }.get(result.source_type, 0.0)

    # Recency bonus: prefer newer sources (max 0.15 for 2024+, decays)
    recency = 0.0
    if result.published_date:
        try:
            year = int(result.published_date[:4])
            if year >= 2024:
                recency = 0.15
            elif year >= 2022:
                recency = 0.10
            elif year >= 2020:
                recency = 0.05
        except (ValueError, IndexError):
            pass

    # Citation count bonus (from Semantic Scholar)
    cite_count = result.extra.get("citation_count", 0)
    if isinstance(cite_count, (int, float)) and cite_count > 0:
        # Log scale: 10 cites ≈ 0.05, 100 ≈ 0.10, 1000 ≈ 0.15
        import math
        citation_bonus = min(0.15, 0.05 * math.log10(max(cite_count, 1)))
    else:
        citation_bonus = 0.0

    # Domain reliability
    reliability = domain_reliability_score(result.url) * 0.2  # max 0.2

    # Has snippet/abstract content (better than title-only)
    content_bonus = 0.05 if result.snippet and len(result.snippet) > 50 else 0.0

    return provider + type_bonus + recency + citation_bonus + reliability + content_bonus


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicates by URL, DOI, arXiv ID, and title similarity.

    When duplicates are found, the academic source version is kept.
    """
    seen_urls: dict[str, SearchResult] = {}
    seen_arxiv_ids: dict[str, SearchResult] = {}
    seen_dois: dict[str, SearchResult] = {}
    output: list[SearchResult] = []

    def _maybe_replace(existing: SearchResult, candidate: SearchResult) -> None:
        """Replace existing with candidate if candidate has higher source priority."""
        if _source_type_priority(candidate.source_type) < _source_type_priority(existing.source_type):
            try:
                idx = output.index(existing)
            except ValueError:
                # `existing` was already evicted by an earlier replacement but a
                # stale dict entry still points at it. Nothing to swap in-place;
                # just refresh the index maps below so future lookups resolve.
                idx = None
            if idx is not None:
                output[idx] = candidate
            # Update all index dicts to point to new winner. Re-map the *old*
            # URL too, so a later result matching it resolves to the live
            # winner instead of the evicted object (which would raise above).
            seen_urls[_normalize_url(existing.url)] = candidate
            norm = _normalize_url(candidate.url)
            seen_urls[norm] = candidate
            aid = _extract_arxiv_id(candidate) or _extract_arxiv_id(existing)
            if aid:
                seen_arxiv_ids[aid] = candidate
            doi = _extract_doi(candidate) or _extract_doi(existing)
            if doi:
                seen_dois[doi] = candidate

    for result in results:
        norm_url = _normalize_url(result.url)
        arxiv_id = _extract_arxiv_id(result)
        doi = _extract_doi(result)

        # Check URL match
        if norm_url in seen_urls:
            _maybe_replace(seen_urls[norm_url], result)
            continue

        # Check arXiv ID match (catches arxiv.org vs semanticscholar.org duplicates)
        if arxiv_id and arxiv_id in seen_arxiv_ids:
            _maybe_replace(seen_arxiv_ids[arxiv_id], result)
            continue

        # Check DOI match
        if doi and doi in seen_dois:
            _maybe_replace(seen_dois[doi], result)
            continue

        # Check title similarity
        is_dup = False
        for existing in output:
            if _title_similarity(result.title, existing.title) > 0.8:
                _maybe_replace(existing, result)
                is_dup = True
                break

        if not is_dup:
            seen_urls[norm_url] = result
            if arxiv_id:
                seen_arxiv_ids[arxiv_id] = result
            if doi:
                seen_dois[doi] = result
            output.append(result)

    return output


_ACADEMIC_PROVIDERS = {"arxiv", "scholar"}
_WEB_PROVIDERS = {"web", "ddg"}


class SearchAggregator:
    """Orchestrates multiple search providers and deduplicates results."""

    def __init__(
        self,
        tavily_api_key: str = "",
        semantic_scholar_api_key: str = "",
        max_concurrency: int = 5,
    ) -> None:
        self._providers: dict[str, SearchProvider] = {}
        self._max_concurrency = max(1, max_concurrency)
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
        academic_weight: float = 0.5,
    ) -> list[SearchResult]:
        """Run searches across selected providers and return deduplicated, ranked results.

        Args:
            academic_weight: Fraction of max_total reserved for academic sources
                when both academic and web providers are active (0.0-1.0).
                Default 0.5 gives equal allocation. Set higher (e.g. 0.7) for
                academic-heavy research.
        """
        active_sources = sources or list(self._providers.keys())

        # Determine per-provider result limits for breadth control
        has_academic = any(s in _ACADEMIC_PROVIDERS for s in active_sources)
        has_web = any(s in _WEB_PROVIDERS for s in active_sources)

        # One semaphore shared across every provider's batch_search so the
        # total concurrent search requests stay under max_concurrency, no
        # matter how many providers × queries fan out.
        search_sem = asyncio.Semaphore(self._max_concurrency)

        tasks = []
        task_names = []
        for name in active_sources:
            provider = self._providers.get(name)
            if not provider:
                continue
            # When both academic and web are active, give academic providers
            # more results per query to improve breadth on scholarly topics
            per_query = max_results_per_query
            if has_academic and has_web and name in _ACADEMIC_PROVIDERS:
                per_query = max(max_results_per_query, max_results_per_query + 2)
            tasks.append(provider.batch_search(queries, per_query, semaphore=search_sem))
            task_names.append(name)

        if not tasks:
            return []

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[SearchResult] = []
        for name, resp in zip(task_names, responses):
            if isinstance(resp, Exception):
                logger.warning(f"Search provider '{name}' failed: {resp}")
                continue
            all_results.extend(resp)

        deduped = deduplicate_results(all_results)

        # Score and rank
        scored = [(compute_source_score(r), r) for r in deduped]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # When both academic and web sources are present, ensure balanced allocation
        if has_academic and has_web and len(scored) > max_total:
            academic_budget = max(1, int(max_total * academic_weight))
            web_budget = max_total - academic_budget

            academic_results = [(s, r) for s, r in scored if r.source_type in (SourceType.ARXIV, SourceType.SEMANTIC_SCHOLAR)]
            web_results = [(s, r) for s, r in scored if r.source_type == SourceType.WEB]

            # Fill each bucket, then redistribute unused slots
            picked_academic = academic_results[:academic_budget]
            picked_web = web_results[:web_budget]

            # If one bucket underflows, give remaining slots to the other
            remaining = max_total - len(picked_academic) - len(picked_web)
            if remaining > 0:
                used = {id(r) for _, r in picked_academic + picked_web}
                extras = [(s, r) for s, r in scored if id(r) not in used]
                picked_academic += extras[:remaining]  # type doesn't matter, just fill

            final = picked_academic + picked_web
            final.sort(key=lambda pair: pair[0], reverse=True)
            return [r for _, r in final][:max_total]

        return [r for _, r in scored][:max_total]
