"""
Abstract base class for market-data providers.

All concrete providers must subclass MarketDataProvider and return
types defined in schemas.market_data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataFetchError(Exception):
    """
    Raised when a market-data fetch fails in an unrecoverable way.

    Attributes
    ----------
    message : str
        Human-readable description of the failure.
    ticker : str | None
        Ticker symbol being fetched, if known.
    timeframe : str | None
        Timeframe being fetched, if known.
    status_code : int | None
        HTTP status code, if the failure originated from an HTTP response.
        4xx codes indicate auth / bad-request errors that should NOT be retried.
        5xx codes indicate server-side errors that MAY be retried.
    """

    def __init__(
        self,
        message: str,
        ticker: str | None = None,
        timeframe: str | None = None,
        status_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.ticker = ticker
        self.timeframe = timeframe
        self.status_code = status_code
        self.cause = cause

    def is_retryable(self) -> bool:
        """Return True if this error is safe to retry (i.e. not a 4xx)."""
        if self.status_code is not None and 400 <= self.status_code < 500:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"DataFetchError(ticker={self.ticker!r}, timeframe={self.timeframe!r}, "
            f"status_code={self.status_code!r}, message={str(self)!r})"
        )


class MarketDataProvider(ABC):
    """
    Abstract market-data provider.

    All concrete implementations must:
    - Return OHLCVSeries from fetch_ohlcv
    - Return MarketSnapshot from fetch_snapshot
    - Raise DataFetchError on unrecoverable failures
    """

    # ── Identity ───────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable name for this provider (e.g. 'alpha_vantage')."""
        ...

    @property
    @abstractmethod
    def supported_timeframes(self) -> list[str]:
        """
        Ordered list of timeframe strings this provider can return.
        Values must be a subset of OHLCVBar.timeframe Literal choices:
        '1m', '5m', '15m', '30m', '1h', '4h', '1d'.
        """
        ...

    # ── Core fetch methods ─────────────────────────────────────────────────

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        timeframe: str,
        limit: int = 100,
        **kwargs: Any,
    ):  # -> OHLCVSeries
        """
        Fetch up to *limit* bars of OHLCV data for *ticker* at *timeframe*.

        Parameters
        ----------
        ticker : str
            Equity symbol, e.g. 'AAPL'.
        timeframe : str
            One of self.supported_timeframes.
        limit : int
            Maximum number of bars to return (newest *limit* bars).
        **kwargs
            Provider-specific pass-through options.

        Returns
        -------
        OHLCVSeries
            Validated series; bars may be empty if no data is available.

        Raises
        ------
        DataFetchError
            On unrecoverable failure.
        ValueError
            If timeframe is not in supported_timeframes.
        """
        ...

    @abstractmethod
    def fetch_snapshot(
        self,
        ticker: str,
        timeframes: list[str] | None = None,
        limit: int = 100,
        **kwargs: Any,
    ):  # -> MarketSnapshot
        """
        Fetch a MarketSnapshot for *ticker* across all requested timeframes.

        Parameters
        ----------
        ticker : str
            Equity symbol.
        timeframes : list[str] | None
            Subset of self.supported_timeframes to populate.
            If None, all supported timeframes are fetched.
        limit : int
            Maximum bars per timeframe.
        **kwargs
            Provider-specific pass-through options.

        Returns
        -------
        MarketSnapshot
            Snapshot with populated timeframe fields.
            Unavailable timeframes are set to None.

        Raises
        ------
        DataFetchError
            On unrecoverable failure for any required timeframe.
        """
        ...

    # ── Utility ────────────────────────────────────────────────────────────

    def _validate_timeframe(self, timeframe: str) -> None:
        if timeframe not in self.supported_timeframes:
            raise ValueError(
                f"Timeframe {timeframe!r} not supported by {self.name}. "
                f"Supported: {self.supported_timeframes}"
            )
