"""
Alpha Vantage concrete market-data provider.

Uses the Alpha Vantage REST API directly via the `requests` library.
API key is read from the ALPHA_VANTAGE_API_KEY environment variable.

Supported timeframes (verified against AV documentation and live endpoints):
  - 1m   → TIME_SERIES_INTRADAY  interval=1min
  - 5m   → TIME_SERIES_INTRADAY  interval=5min
  - 15m  → TIME_SERIES_INTRADAY  interval=15min
  - 30m  → TIME_SERIES_INTRADAY  interval=30min
  - 1h   → TIME_SERIES_INTRADAY  interval=60min
  - 1d   → TIME_SERIES_DAILY

NOTE: Alpha Vantage free tier returns at most ~100 intraday bars per call
      (one trading session) and 20 years of daily data.
      The `outputsize` parameter controls compact (100 bars) vs. full.

TODO: 4h timeframe is NOT natively supported by Alpha Vantage.
      Downstream consumers requesting '4h' will receive an empty OHLCVSeries.
      To support 4h in future: resample 1h bars after fetching.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests
import structlog

from schemas.market_data import MarketSnapshot, OHLCVBar, OHLCVSeries
from src.data.provider import DataFetchError, MarketDataProvider
from src.data.retry import with_retry

log = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_AV_BASE_URL = "https://www.alphavantage.co/query"

# Mapping from our internal timeframe keys to AV interval strings.
# 4h is intentionally absent — AV does not support it natively.
_INTRADAY_INTERVAL_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
}

# 4h is included so that the provider does not reject callers who request it
# (the schema supports it).  fetch_ohlcv returns an empty OHLCVSeries for 4h
# because AV has no native 4h endpoint.
# TODO: Implement 4h resampling from 1h bars once the 1h endpoint is validated.
_SUPPORTED_TIMEFRAMES: list[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# AV daily series key inside the JSON response.
_DAILY_SERIES_KEY = "Time Series (Daily)"

# AV bar field names (consistent across intraday and daily).
_FIELD_MAP = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume",
}


class AlphaVantageProvider(MarketDataProvider):
    """
    Fetches OHLCV data from the Alpha Vantage REST API.

    Parameters
    ----------
    api_key : str | None
        Alpha Vantage API key. If None, falls back to the
        ALPHA_VANTAGE_API_KEY environment variable.
    session : requests.Session | None
        Optional pre-configured requests Session (useful for testing).
    request_timeout : float
        HTTP request timeout in seconds (default 15).
    """

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        request_timeout: float = 15.0,
    ) -> None:
        self._api_key: str = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self._session: requests.Session = session or requests.Session()
        self._timeout: float = request_timeout

        if not self._api_key:
            log.warning(
                "alpha_vantage.no_api_key",
                msg="ALPHA_VANTAGE_API_KEY is not set; live fetches will fail",
            )

    # ── MarketDataProvider interface ───────────────────────────────────────

    @property
    def name(self) -> str:
        return "alpha_vantage"

    @property
    def supported_timeframes(self) -> list[str]:
        return list(_SUPPORTED_TIMEFRAMES)

    def fetch_ohlcv(
        self,
        ticker: str,
        timeframe: str,
        limit: int = 100,
        force_full: bool = False,
        **kwargs: Any,
    ) -> OHLCVSeries:
        """
        Fetch OHLCV bars for *ticker* at *timeframe*.

        Parameters
        ----------
        ticker : str
            Equity symbol (e.g. 'AAPL').
        timeframe : str
            Must be one of supported_timeframes.
        limit : int
            Return at most this many bars (newest first after sorting).
        force_full : bool
            If True, request the full history from AV (slower, more data).

        Returns
        -------
        OHLCVSeries
            Empty bars list if AV returns no data — never raises on empty.

        Raises
        ------
        DataFetchError
            On HTTP errors or malformed AV responses.
        ValueError
            If timeframe is unsupported.
        """
        self._validate_timeframe(timeframe)

        # TODO: 4h resampling — not implemented; AV has no 4h endpoint.
        if timeframe == "4h":
            log.warning(
                "alpha_vantage.unsupported_4h",
                ticker=ticker,
                msg="4h timeframe is not supported; returning empty OHLCVSeries",
            )
            return OHLCVSeries(ticker=ticker, timeframe="4h")  # type: ignore[arg-type]

        try:
            raw = self._fetch_with_retry(ticker=ticker, timeframe=timeframe, force_full=force_full)
            bars = self._parse_bars(raw=raw, ticker=ticker, timeframe=timeframe)
        except DataFetchError:
            raise
        except Exception as exc:
            raise DataFetchError(
                f"Unexpected error fetching {ticker}/{timeframe}: {exc}",
                ticker=ticker,
                timeframe=timeframe,
                cause=exc,
            ) from exc

        # Sort ascending by timestamp, then take the newest *limit* bars.
        bars.sort(key=lambda b: b.timestamp)
        if limit > 0:
            bars = bars[-limit:]

        series = OHLCVSeries(ticker=ticker, timeframe=timeframe, bars=bars)  # type: ignore[arg-type]
        log.info(
            "alpha_vantage.fetch_ohlcv.done",
            ticker=ticker,
            timeframe=timeframe,
            bar_count=len(bars),
        )
        return series

    def fetch_snapshot(
        self,
        ticker: str,
        timeframes: list[str] | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> MarketSnapshot:
        """
        Fetch a MarketSnapshot for *ticker* across the requested *timeframes*.

        Missing or failed timeframes are silently set to None and a warning
        is logged — they do NOT cause an exception.
        """
        requested = timeframes if timeframes is not None else self.supported_timeframes
        snapshot_time = datetime.now(tz=timezone.utc)

        tf_series: dict[str, OHLCVSeries | None] = {}
        for tf in requested:
            try:
                tf_series[tf] = self.fetch_ohlcv(ticker=ticker, timeframe=tf, limit=limit)
            except DataFetchError as exc:
                log.warning(
                    "alpha_vantage.fetch_snapshot.tf_failed",
                    ticker=ticker,
                    timeframe=tf,
                    error=str(exc),
                )
                tf_series[tf] = None

        snapshot = MarketSnapshot(
            ticker=ticker,
            snapshot_time=snapshot_time,
            tf_1m=tf_series.get("1m"),
            tf_5m=tf_series.get("5m"),
            tf_15m=tf_series.get("15m"),
            tf_1h=tf_series.get("1h"),
            tf_4h=tf_series.get("4h"),
            tf_1d=tf_series.get("1d"),
        )
        log.info(
            "alpha_vantage.fetch_snapshot.done",
            ticker=ticker,
            timeframes=requested,
        )
        return snapshot

    # ── Internal helpers ───────────────────────────────────────────────────

    @with_retry(max_retries=3, base_delay=1.0, max_delay=15.0)
    def _fetch_with_retry(
        self,
        ticker: str,
        timeframe: str,
        force_full: bool = False,
    ) -> dict[str, Any]:
        """
        Make the raw HTTP request to Alpha Vantage.
        Decorated with retry/backoff for transient failures.
        """
        params = self._build_params(ticker=ticker, timeframe=timeframe, force_full=force_full)
        log.debug("alpha_vantage.request", ticker=ticker, timeframe=timeframe, params=params)

        try:
            resp = self._session.get(_AV_BASE_URL, params=params, timeout=self._timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            # Map to DataFetchError so retry logic recognises them as retryable.
            raise DataFetchError(
                f"Network error fetching {ticker}/{timeframe}: {exc}",
                ticker=ticker,
                timeframe=timeframe,
                cause=exc,
            ) from exc

        if resp.status_code == 429:
            raise DataFetchError(
                "Alpha Vantage rate limit exceeded (HTTP 429). Back off and retry.",
                ticker=ticker,
                timeframe=timeframe,
                status_code=429,
            )
        if resp.status_code == 401:
            raise DataFetchError(
                "Alpha Vantage authentication failed (HTTP 401). Check your API key.",
                ticker=ticker,
                timeframe=timeframe,
                status_code=401,
            )
        if resp.status_code >= 400:
            raise DataFetchError(
                f"Alpha Vantage HTTP {resp.status_code} for {ticker}/{timeframe}.",
                ticker=ticker,
                timeframe=timeframe,
                status_code=resp.status_code,
            )
        if resp.status_code >= 500:
            raise DataFetchError(
                f"Alpha Vantage server error HTTP {resp.status_code} for {ticker}/{timeframe}.",
                ticker=ticker,
                timeframe=timeframe,
                status_code=resp.status_code,
            )

        try:
            payload: dict[str, Any] = resp.json()
        except Exception as exc:
            raise DataFetchError(
                f"Alpha Vantage returned non-JSON for {ticker}/{timeframe}: {exc}",
                ticker=ticker,
                timeframe=timeframe,
                cause=exc,
            ) from exc

        # AV signals errors in the JSON body even with HTTP 200.
        if "Error Message" in payload:
            raise DataFetchError(
                f"Alpha Vantage error: {payload['Error Message']}",
                ticker=ticker,
                timeframe=timeframe,
                status_code=400,  # treat as 4xx — bad symbol / bad params
            )
        if "Note" in payload:
            # "Thank you for using Alpha Vantage! ..." — free-tier call limit.
            log.warning(
                "alpha_vantage.rate_limit_note",
                ticker=ticker,
                note=payload["Note"][:120],
            )
            raise DataFetchError(
                f"Alpha Vantage call-frequency limit: {payload['Note'][:80]}",
                ticker=ticker,
                timeframe=timeframe,
                status_code=429,
            )
        if "Information" in payload:
            # Similar to "Note" — premium endpoint or rate-limit message.
            log.warning(
                "alpha_vantage.information_message",
                ticker=ticker,
                info=payload["Information"][:120],
            )
            raise DataFetchError(
                f"Alpha Vantage information: {payload['Information'][:80]}",
                ticker=ticker,
                timeframe=timeframe,
                status_code=429,
            )

        return payload

    def _build_params(
        self,
        ticker: str,
        timeframe: str,
        force_full: bool = False,
    ) -> dict[str, str]:
        """Build the query-string parameters for the AV endpoint."""
        outputsize = "full" if force_full else "compact"

        if timeframe == "1d":
            return {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": outputsize,
                "apikey": self._api_key,
            }

        interval = _INTRADAY_INTERVAL_MAP[timeframe]
        return {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": interval,
            "outputsize": outputsize,
            "adjusted": "true",
            "extended_hours": "false",
            "apikey": self._api_key,
        }

    def _parse_bars(
        self,
        raw: dict[str, Any],
        ticker: str,
        timeframe: str,
    ) -> list[OHLCVBar]:
        """
        Parse a raw AV JSON response into a list of OHLCVBar objects.

        Returns an empty list if the expected series key is missing.
        Skips individual bars that fail Pydantic validation (logs a warning).
        """
        series_key = self._find_series_key(raw)
        if series_key is None:
            log.warning(
                "alpha_vantage.parse_bars.no_series_key",
                ticker=ticker,
                timeframe=timeframe,
                available_keys=list(raw.keys()),
            )
            return []

        series: dict[str, dict[str, str]] = raw[series_key]
        bars: list[OHLCVBar] = []

        for ts_str, fields in series.items():
            try:
                ts = self._parse_timestamp(ts_str)
                bar = OHLCVBar(
                    timestamp=ts,
                    open=float(fields["1. open"]),
                    high=float(fields["2. high"]),
                    low=float(fields["3. low"]),
                    close=float(fields["4. close"]),
                    volume=float(fields["5. volume"]),
                    ticker=ticker,
                    timeframe=timeframe,  # type: ignore[arg-type]
                )
                bars.append(bar)
            except (KeyError, ValueError, Exception) as exc:
                log.warning(
                    "alpha_vantage.parse_bars.skipped_bar",
                    ticker=ticker,
                    timeframe=timeframe,
                    timestamp=ts_str,
                    error=str(exc),
                )

        return bars

    @staticmethod
    def _find_series_key(payload: dict[str, Any]) -> str | None:
        """
        Return the first key in *payload* that contains the OHLCV time series.
        AV names keys like 'Time Series (5min)' or 'Time Series (Daily)'.
        """
        for key in payload:
            if "Time Series" in key:
                return key
        return None

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        """
        Parse an AV timestamp string into a timezone-aware datetime (UTC).
        AV returns naive strings in US/Eastern but without tz label.
        We store them as-is (naive) and attach UTC for consistency.

        TODO: Properly localise AV intraday timestamps from US/Eastern to UTC
              if exact timezone handling is required downstream.
              For now we mark them as UTC to satisfy the Pydantic schema.
        """
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(ts_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse AV timestamp: {ts_str!r}")
