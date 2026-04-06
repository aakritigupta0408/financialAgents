"""
src.data — market-data layer for the paper-trading research system.

Public API
----------
get_provider() -> CachedProvider
    Single entry point for all downstream modules.
    Returns a CachedProvider wrapping AlphaVantageProvider.
    Cache directory is read from config.settings.CACHE_DIR.

Example
-------
    from src.data import get_provider

    provider = get_provider()
    series = provider.fetch_ohlcv("AAPL", "5m", limit=100)
    df = series.to_dataframe()
"""

from __future__ import annotations

import os

from src.data.alpha_vantage import AlphaVantageProvider
from src.data.cache import CachedProvider
from src.data.provider import DataFetchError, MarketDataProvider

__all__ = [
    "get_provider",
    "AlphaVantageProvider",
    "CachedProvider",
    "MarketDataProvider",
    "DataFetchError",
]

_provider_singleton: CachedProvider | None = None


def get_provider(
    api_key: str | None = None,
    force_new: bool = False,
) -> CachedProvider:
    """
    Return a CachedProvider wrapping AlphaVantageProvider.

    The instance is cached as a module-level singleton so repeated calls
    within the same process reuse the same diskcache connection.

    Parameters
    ----------
    api_key : str | None
        Alpha Vantage API key.  Falls back to ALPHA_VANTAGE_API_KEY env var.
    force_new : bool
        If True, discard the singleton and create a fresh provider.

    Returns
    -------
    CachedProvider
    """
    global _provider_singleton

    if _provider_singleton is None or force_new:
        from config.settings import CACHE_DIR  # lazy import avoids circular deps

        av = AlphaVantageProvider(api_key=api_key)
        _provider_singleton = CachedProvider(provider=av, cache_dir=CACHE_DIR)

    return _provider_singleton
