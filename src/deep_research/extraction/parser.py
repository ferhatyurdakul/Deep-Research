from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from deep_research.models import ScrapedContent, SearchResult, SourceType

from .cleaner import clean_text, truncate_text
from .fetcher import AsyncFetcher

logger = logging.getLogger(__name__)

_NOISE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}
_MIN_CONTENT_LENGTH = 100
_MAX_WORDS = 3000


def _is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return True
    if "arxiv.org/pdf/" in url:
        return True
    return False


def _convert_arxiv_to_pdf(url: str) -> str:
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/") + ".pdf"
    return url


def _extract_main_text(html: str) -> str:
    """Extract the main textual content from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(role="main")
        or soup.find(class_=re.compile(r"content|article|post|entry", re.I))
        or soup.find(id=re.compile(r"content|article|post|entry|body", re.I))
        or soup.find("body")
        or soup
    )

    return main.get_text(separator="\n", strip=True)


def _get_html_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.title.string.strip() if soup.title and soup.title.string else ""


def _parse_pdf_bytes(pdf_bytes: bytes, url: str) -> ScrapedContent | None:
    """Extract text from PDF bytes using pymupdf."""
    try:
        import pymupdf

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        title = doc.metadata.get("title", "") or ""

        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_text.append(text)
            if page_num >= 14:
                pages_text.append("\n[... remaining pages truncated]")
                break

        if not title:
            try:
                first_page = doc[0]
                blocks = first_page.get_text("blocks")
                _skip = ["permission", "copyright", "licensed", "provided",
                         "preprint", "arxiv", "under review", "abstract"]
                for b in blocks[:8]:
                    text = b[4].strip().replace("\n", " ")
                    if len(text) < 10 or len(text) > 200:
                        continue
                    if any(s in text.lower() for s in _skip):
                        continue
                    if "@" in text:
                        continue
                    title = text
                    break
            except Exception:
                pass

        doc.close()
        full_text = "\n\n".join(pages_text)
        full_text = truncate_text(full_text, _MAX_WORDS)
        word_count = len(full_text.split())

        return ScrapedContent(url=url, title=title, content=full_text, word_count=word_count)
    except Exception as e:
        logger.warning(f"PDF extraction failed for {url}: {e}")
        return None


class ContentParser:
    """Fetches URLs and extracts clean text content."""

    def __init__(self, fetcher: AsyncFetcher | None = None) -> None:
        self._fetcher = fetcher or AsyncFetcher()
        self._owns_fetcher = fetcher is None

    async def parse_url(self, url: str) -> ScrapedContent | None:
        """Fetch and parse a single URL into ScrapedContent."""
        fetch_url = _convert_arxiv_to_pdf(url)
        is_pdf = _is_pdf_url(fetch_url)

        if is_pdf:
            status, content_bytes, content_type = await self._fetcher.fetch_bytes(fetch_url)
            if status < 200 or status >= 300:
                return None
            return _parse_pdf_bytes(content_bytes, url)

        status, html, content_type = await self._fetcher.fetch(fetch_url)
        if status < 200 or status >= 300:
            return None

        if "application/pdf" in content_type:
            status, content_bytes, _ = await self._fetcher.fetch_bytes(fetch_url)
            return _parse_pdf_bytes(content_bytes, url)

        title = _get_html_title(html)
        raw_text = _extract_main_text(html)
        content = clean_text(raw_text)
        content = truncate_text(content, _MAX_WORDS)

        if len(content) < _MIN_CONTENT_LENGTH:
            return None

        word_count = len(content.split())
        return ScrapedContent(url=url, title=title, content=content, word_count=word_count)

    async def parse_search_result(self, result: SearchResult) -> ScrapedContent | None:
        """Fetch and parse a SearchResult.

        For arXiv papers, uses the abstract directly to avoid fetching the full PDF.
        """
        if result.source_type == SourceType.ARXIV and result.snippet:
            return ScrapedContent(
                url=result.url,
                title=result.title,
                content=result.snippet,
                word_count=len(result.snippet.split()),
            )

        scraped = await self.parse_url(result.url)
        if scraped is None and result.snippet:
            return ScrapedContent(
                url=result.url,
                title=result.title,
                content=result.snippet,
                word_count=len(result.snippet.split()),
            )
        return scraped

    async def parse_many(
        self, results: list[SearchResult], max_concurrent: int = 5
    ) -> list[ScrapedContent]:
        """Fetch and parse multiple SearchResults concurrently."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _bounded_parse(result: SearchResult) -> ScrapedContent | None:
            async with sem:
                return await self.parse_search_result(result)

        tasks = [_bounded_parse(r) for r in results]
        scraped = await asyncio.gather(*tasks)
        return [s for s in scraped if s is not None and s.word_count > 20]

    async def close(self) -> None:
        if self._owns_fetcher:
            await self._fetcher.close()

    async def __aenter__(self) -> ContentParser:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
