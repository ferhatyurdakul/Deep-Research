"""Shared URL normalization.

One canonical form for dedup comparisons across layers: within-search dedup
(aggregator), cross-session filtering (seen/known URLs in the pipelines), and
storage keys (analyzed_urls). Without this, `https://example.com/page` and
`http://www.example.com/page/` dedup inside one search but re-fetch and
re-analyze across sessions.

Semantics intentionally identical to the aggregator's original helper so
existing dedup behavior is unchanged: strip trailing slashes, strip a leading
scheme and optional `www.`, lowercase. Query strings are preserved.
"""
from __future__ import annotations

import re


def normalize_url(url: str) -> str:
    """Normalize a URL for dedup comparisons."""
    url = url.rstrip("/")
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url.lower()
