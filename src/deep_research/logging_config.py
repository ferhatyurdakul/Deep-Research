from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO", console: Optional[object] = None) -> None:
    """
    Configure root logging.

    With no `console` this is the plain stderr format used by the eval runner.
    The CLI passes its shared rich Console instead: records are then printed
    through it, so a warning appears cleanly above a live progress spinner
    rather than tearing it, and message text is never interpreted as markup.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    if console is None:
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        return

    from rich.highlighter import NullHighlighter
    from rich.logging import RichHandler

    root = logging.getLogger()
    root.setLevel(numeric)
    # Idempotent: replace our own handler from an earlier call, leave anyone
    # else's (pytest's capture handlers, for one) alone.
    for existing in list(root.handlers):
        if getattr(existing, "_deep_research_cli", False):
            root.removeHandler(existing)
    handler = RichHandler(
        console=console, show_time=False, show_path=False, markup=False,
        highlighter=NullHighlighter(),
    )
    handler._deep_research_cli = True  # type: ignore[attr-defined]
    # Logger names only help when the user asked for diagnostics.
    handler.setFormatter(logging.Formatter(
        "%(name)s: %(message)s" if numeric <= logging.INFO else "%(message)s"
    ))
    root.addHandler(handler)
