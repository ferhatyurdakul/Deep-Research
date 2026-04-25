"""Benchmark query set for Deep Research evaluation.

Each entry has:
  id       — stable slug used for output directories
  query    — the research question
  depth    — quick | standard | deep
  template — technology | science | market | literature | None
  style    — brief | standard | academic
  agent    — bool, whether to use agent mode
  tags     — subsets (e.g. "smoke" for fast sanity checks, "full" for everything)
"""
from __future__ import annotations

BENCHMARKS: list[dict] = [
    {
        "id": "wasm-vs-js",
        "query": "How does WebAssembly compare to JavaScript for performance-critical web applications?",
        "depth": "quick",
        "template": "technology",
        "style": "standard",
        "agent": False,
        "tags": ["smoke", "full"],
    },
    {
        "id": "rust-backend-adoption",
        "query": "What is the current state of Rust adoption in production backend systems?",
        "depth": "standard",
        "template": "technology",
        "style": "standard",
        "agent": False,
        "tags": ["smoke", "full"],
    },
    {
        "id": "mrna-variants",
        "query": "How effective are mRNA vaccines against emerging viral variants?",
        "depth": "standard",
        "template": "science",
        "style": "academic",
        "agent": False,
        "tags": ["full"],
    },
    {
        "id": "ev-charging-landscape",
        "query": "What is the competitive landscape of EV charging infrastructure in 2026?",
        "depth": "standard",
        "template": "market",
        "style": "standard",
        "agent": False,
        "tags": ["full"],
    },
    {
        "id": "transformer-long-context",
        "query": "What does the literature say about transformer architectures' limitations in long-context reasoning?",
        "depth": "deep",
        "template": "literature",
        "style": "academic",
        "agent": False,
        "tags": ["full"],
    },
    {
        "id": "remote-work-productivity",
        "query": "Does remote work increase or decrease software engineering productivity?",
        "depth": "standard",
        "template": None,
        "style": "standard",
        "agent": False,
        "tags": ["full", "contested"],
    },
]


def select(tag: str | None = None, ids: list[str] | None = None) -> list[dict]:
    if ids:
        wanted = set(ids)
        return [b for b in BENCHMARKS if b["id"] in wanted]
    if tag:
        return [b for b in BENCHMARKS if tag in b.get("tags", [])]
    return list(BENCHMARKS)
