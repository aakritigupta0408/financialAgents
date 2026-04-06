"""
ReplayLoader: loads historical data from local store into OHLCVSeries
compatible with the existing backtest and live-paper pipeline.

Priority order:
1. Local data store (real data, any timeframe)
2. Real daily fixture files (tests/fixtures/real_data/{ticker}.csv) for "1d"
3. Structured synthetic 1h series for "1h" when no real data exists
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from schemas.market_data import OHLCVSeries
from src.data_store.store import DataStore
from src.data_store.paths import DATA_STORE_DIR


class ReplayLoader:
    """
    Unified loader for replay into backtest/live-paper pipeline.
    """

    def __init__(self, store: DataStore | None = None):
        self._store = store or DataStore()

    def load(
        self,
        ticker: str,
        timeframe: str = "1h",
        start: datetime | None = None,
        end: datetime | None = None,
        min_bars: int = 100,
    ) -> OHLCVSeries:
        """
        Load series for replay. Falls back through priority chain.
        Raises ValueError if no data found and all fallbacks fail.
        """
        # 1. Try local data store
        series = self._store.load(ticker, timeframe, start, end)
        if len(series.bars) >= min_bars:
            return series

        # 2. Try real daily fixture for 1d timeframe
        if timeframe == "1d":
            try:
                from src.validation.real_data import load_ticker
                series = load_ticker(ticker, timeframe="1d")
                if len(series.bars) > 0:
                    return series
            except (FileNotFoundError, ImportError):
                pass

        # 3. Synthetic 1h fallback
        if timeframe == "1h":
            try:
                from src.validation.intraday_synthetic import (
                    load_or_generate_1h_series,
                    make_structured_1h_series,
                    TICKER_PROXIES,
                )
                # Check if ticker is in TICKER_PROXIES (using ticker or ticker_1H variant)
                proxy_tickers = {p["ticker"] for p in TICKER_PROXIES}
                if ticker in proxy_tickers or f"{ticker}_1H" in proxy_tickers:
                    lookup = ticker if ticker in proxy_tickers else f"{ticker}_1H"
                    return load_or_generate_1h_series(lookup, n_bars=max(600, min_bars))
                else:
                    # Unknown ticker: use structured synthetic with hash-based seed
                    seed = hash(ticker) % 1000
                    if seed < 0:
                        seed = seed + 1000
                    return make_structured_1h_series(
                        ticker=ticker,
                        n_bars=max(600, min_bars),
                        seed=seed,
                    )
            except ImportError:
                pass

        # If we got here with some bars (less than min_bars), return what we have
        if series.bars:
            return series

        raise ValueError(
            f"ReplayLoader: no data found for {ticker}/{timeframe} "
            f"after exhausting all fallbacks (min_bars={min_bars})"
        )

    def load_multi(
        self,
        tickers: list[str],
        timeframe: str = "1h",
        start: datetime | None = None,
        end: datetime | None = None,
        min_bars: int = 100,
    ) -> dict[str, OHLCVSeries]:
        """Load multiple tickers. Skips tickers that fail all fallbacks."""
        result: dict[str, OHLCVSeries] = {}
        for ticker in tickers:
            try:
                result[ticker] = self.load(ticker, timeframe, start, end, min_bars)
            except (ValueError, Exception):
                pass
        return result

    def describe(self, ticker: str, timeframe: str = "1h") -> dict:
        """Return metadata about available data for ticker/timeframe."""
        ranges = self._store.available_ranges(ticker, timeframe)
        if not ranges:
            return {
                "ticker": ticker,
                "timeframe": timeframe,
                "source": "none",
                "first_ts": None,
                "last_ts": None,
                "row_count": 0,
                "partitions": 0,
            }

        first_ts = min(r.first_ts for r in ranges)
        last_ts = max(r.last_ts for r in ranges)
        row_count = sum(r.row_count for r in ranges)
        return {
            "ticker": ticker,
            "timeframe": timeframe,
            "source": "local_store",
            "first_ts": first_ts,
            "last_ts": last_ts,
            "row_count": row_count,
            "partitions": len(ranges),
        }
