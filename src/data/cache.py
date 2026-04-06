"""
DiskCache-backed caching layer for any MarketDataProvider.

CachedProvider wraps a concrete MarketDataProvider and transparently
caches OHLCVSeries results to local disk using the `diskcache` library.

Cache key schema
----------------
    "<ticker>:<timeframe>:<bucket>"

where <bucket> is:
    - Intraday (1m, 5m, 15m, 30m, 1h):  "YYYY-MM-DD-HH"  (hourly buckets)
    - Daily (1d, 4h):                    "YYYY-MM-DD"      (daily buckets)

TTL
---
    Intraday data:  5 minutes   (300 seconds)
    Daily data:     6 hours     (21600 seconds)

Force-refresh
-------------
    Pass force_refresh=True to fetch_ohlcv / fetch_snapshot to bypass
    the cache and write a fresh result back.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import diskcache
import structlog

from schemas.market_data import MarketSnapshot, OHLCVSeries
from src.data.provider import DataFetchError, MarketDataProvider

log = structlog.get_logger(__name__)

# ── TTL constants ──────────────────────────────────────────────────────────
_TTL_INTRADAY_S: int = 300      # 5 minutes
_TTL_DAILY_S: int = 21_600      # 6 hours

_INTRADAY_TIMEFRAMES: frozenset[str] = frozenset({"1m", "5m", "15m", "30m", "1h", "4h"})


def _ttl_for(timeframe: str) -> int:
    """Return the TTL in seconds for a given timeframe."""
    return _TTL_INTRADAY_S if timeframe in _INTRADAY_TIMEFRAMES else _TTL_DAILY_S


def _bucket_for(timeframe: str, now: datetime | None = None) -> str:
    """
    Return the cache bucket string for a given timeframe.

    Intraday → 'YYYY-MM-DD-HH'  (changes every hour)
    Daily    → 'YYYY-MM-DD'     (changes every day)
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)
    if timeframe in _INTRADAY_TIMEFRAMES:
        return now.strftime("%Y-%m-%d-%H")
    return now.strftime("%Y-%m-%d")


def _cache_key(ticker: str, timeframe: str, limit: int, bucket: str) -> str:
    """
    Build a cache key string.  Includes *limit* so different limit values
    produce separate cache entries (avoids serving fewer bars than requested).
    """
    raw = f"{ticker.upper()}:{timeframe}:{limit}:{bucket}"
    # Hash to keep key length manageable and filesystem-safe.
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{ticker.upper()}_{timeframe}_{digest}"


class CachedProvider(MarketDataProvider):
    """
    Caching wrapper around any MarketDataProvider.

    Parameters
    ----------
    provider : MarketDataProvider
        The underlying data provider (e.g. AlphaVantageProvider).
    cache_dir : str | Path
        Directory used by diskcache for persistent storage.
    size_limit : int
        Maximum cache size in bytes (default 512 MB).
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        cache_dir: Any,  # str | Path
        size_limit: int = 512 * 1024 * 1024,
    ) -> None:
        self._provider = provider
        self._cache = diskcache.Cache(
            str(cache_dir),
            size_limit=size_limit,
            eviction_policy="least-recently-used",
        )
        log.info(
            "cached_provider.init",
            provider=provider.name,
            cache_dir=str(cache_dir),
        )

    # ── MarketDataProvider interface ───────────────────────────────────────

    @property
    def name(self) -> str:
        return f"cached({self._provider.name})"

    @property
    def supported_timeframes(self) -> list[str]:
        return self._provider.supported_timeframes

    def fetch_ohlcv(
        self,
        ticker: str,
        timeframe: str,
        limit: int = 100,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> OHLCVSeries:
        """
        Return an OHLCVSeries, served from cache when available.

        Parameters
        ----------
        ticker : str
        timeframe : str
        limit : int
        force_refresh : bool
            Bypass the cache and fetch fresh data.
        **kwargs
            Passed through to the underlying provider.
        """
        bucket = _bucket_for(timeframe)
        key = _cache_key(ticker, timeframe, limit, bucket)
        ttl = _ttl_for(timeframe)

        if not force_refresh:
            cached: OHLCVSeries | None = self._cache.get(key)
            if cached is not None:
                log.debug(
                    "cached_provider.cache_hit",
                    ticker=ticker,
                    timeframe=timeframe,
                    key=key,
                )
                return cached

        log.debug(
            "cached_provider.cache_miss",
            ticker=ticker,
            timeframe=timeframe,
            key=key,
            force_refresh=force_refresh,
        )

        series = self._provider.fetch_ohlcv(
            ticker=ticker,
            timeframe=timeframe,
            limit=limit,
            **kwargs,
        )
        self._cache.set(key, series, expire=ttl)
        return series

    def fetch_snapshot(
        self,
        ticker: str,
        timeframes: list[str] | None = None,
        limit: int = 100,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> MarketSnapshot:
        """
        Return a MarketSnapshot, fetching each timeframe through the cache.

        Parameters
        ----------
        ticker : str
        timeframes : list[str] | None
        limit : int
        force_refresh : bool
            Bypass cache for all timeframes in this snapshot.
        **kwargs
            Passed through to the underlying provider.
        """
        requested = timeframes if timeframes is not None else self.supported_timeframes
        snapshot_time = datetime.now(tz=timezone.utc)

        tf_series: dict[str, OHLCVSeries | None] = {}
        for tf in requested:
            try:
                tf_series[tf] = self.fetch_ohlcv(
                    ticker=ticker,
                    timeframe=tf,
                    limit=limit,
                    force_refresh=force_refresh,
                    **kwargs,
                )
            except DataFetchError as exc:
                log.warning(
                    "cached_provider.fetch_snapshot.tf_failed",
                    ticker=ticker,
                    timeframe=tf,
                    error=str(exc),
                )
                tf_series[tf] = None

        return MarketSnapshot(
            ticker=ticker,
            snapshot_time=snapshot_time,
            tf_1m=tf_series.get("1m"),
            tf_5m=tf_series.get("5m"),
            tf_15m=tf_series.get("15m"),
            tf_1h=tf_series.get("1h"),
            tf_4h=tf_series.get("4h"),
            tf_1d=tf_series.get("1d"),
        )

    # ── Cache management ───────────────────────────────────────────────────

    def clear(self) -> None:
        """Evict all entries from the disk cache."""
        self._cache.clear()
        log.info("cached_provider.cache_cleared")

    def close(self) -> None:
        """Close the underlying diskcache connection."""
        self._cache.close()

    def __enter__(self) -> "CachedProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Return basic cache statistics for diagnostics."""
        return {
            "volume": self._cache.volume(),
            "size_limit": self._cache.size_limit,
            "directory": str(self._cache.directory),
        }
