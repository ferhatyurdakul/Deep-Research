from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_MAX_RETRIES = 2


class AsyncFetcher:
    """Concurrent HTTP fetcher with rate limiting and retries."""

    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: float = 15.0,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0),
                follow_redirects=True,
                headers=_HEADERS,
            )
        return self._client

    async def fetch(self, url: str) -> tuple[int, str, str]:
        """Fetch a single URL. Returns (status, content, content_type)."""
        async with self._semaphore:
            return await self._fetch_with_retry(url)

    async def _fetch_with_retry(self, url: str) -> tuple[int, str, str]:
        client = await self._get_client()

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.get(url)
                content_type = resp.headers.get("content-type", "")

                if resp.status_code == 429 and attempt < _MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue

                return resp.status_code, resp.text, content_type
            except httpx.TimeoutException:
                if attempt == _MAX_RETRIES:
                    return 0, "", ""
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Fetch failed for {url}: {e}")
                return 0, "", ""

        return 0, "", ""

    async def fetch_bytes(self, url: str) -> tuple[int, bytes, str]:
        """Fetch raw bytes (for PDF). Returns (status, bytes, content_type)."""
        async with self._semaphore:
            client = await self._get_client()
            try:
                resp = await client.get(url)
                content_type = resp.headers.get("content-type", "")
                return resp.status_code, resp.content, content_type
            except Exception as e:
                logger.warning(f"Fetch bytes failed for {url}: {e}")
                return 0, b"", ""

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncFetcher:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
