from __future__ import annotations

import re


_BOILERPLATE_RE = re.compile(
    r"|".join([
        r"accept (all )?cookies",
        r"cookie (policy|settings|preferences)",
        r"we use cookies",
        r"subscribe to (our )?newsletter",
        r"sign up for",
        r"enable javascript",
        r"ad ?blocker",
        r"privacy policy",
        r"terms of (service|use)",
        r"©\s*\d{4}",
    ]),
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Remove boilerplate lines, normalize whitespace, and clean extracted text."""
    if not text:
        return ""

    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize unicode dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        if _BOILERPLATE_RE.search(line):
            continue
        lines.append(line)

    return "\n".join(lines)


def truncate_text(text: str, max_words: int = 3000) -> str:
    """Truncate text to max_words."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "\n\n[... truncated]"
    return text
