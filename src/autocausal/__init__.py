"""autocausal — auto-impute and exploratory causal discovery for tabular data."""

from __future__ import annotations

from autocausal.api import AutoCausal
from autocausal.results import DiscoveryResult
from autocausal.__version__ import __version__

__all__ = ["AutoCausal", "DiscoveryResult", "__version__"]
