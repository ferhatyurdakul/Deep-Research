"""Synapse Deep Research - AI-powered research framework using Z.AI GLM models."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("synapse-deep-research")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "1.0.0"
