"""
src.timesfm.base — Abstract base class for all forecasters.

Rules:
- All input is past data only (series sorted ascending, no future leakage).
- Minimum context length is configurable (default 32 bars).
- Insufficient data returns a low-confidence ForecastOutput rather than crashing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from schemas.forecast import ForecastOutput
from schemas.market_data import OHLCVSeries

logger = logging.getLogger(__name__)

# Returned when the series is too short to forecast meaningfully.
_LOW_CONFIDENCE_DIRECTION = "up"
_LOW_CONFIDENCE_RETURN = 0.0
_LOW_CONFIDENCE_CONFIDENCE = 0.05


class BaseForecaster(ABC):
    """
    Abstract base for TimesFM and statistical fallback forecasters.

    Subclasses must implement:
      - forecast()
      - is_available() classmethod
      - name property
    """

    DEFAULT_MIN_CONTEXT_LEN: int = 32

    def __init__(self, min_context_len: int = DEFAULT_MIN_CONTEXT_LEN) -> None:
        self.min_context_len = min_context_len

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def forecast(
        self,
        series: OHLCVSeries,
        horizon: int,
        ticker: str,
        timeframe: str,
    ) -> ForecastOutput:
        """
        Generate a forecast from the given series.

        Parameters
        ----------
        series    : OHLCVSeries — past OHLCV bars (ascending).
        horizon   : int — number of bars ahead to forecast.
        ticker    : str — ticker symbol (passed through to ForecastOutput).
        timeframe : str — bar timeframe label (passed through to ForecastOutput).

        Returns
        -------
        ForecastOutput — direction, expected_return, confidence, horizon, quantile paths.
        """

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this forecaster's backend is installed and ready."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this forecaster."""

    # ── Shared helpers ────────────────────────────────────────────────────

    def _prepare_prices(
        self,
        series: OHLCVSeries,
    ) -> tuple[np.ndarray | None, str | None]:
        """
        Convert OHLCVSeries to a sorted, validated numpy close-price array.

        Returns
        -------
        (prices, None)  if data is valid and meets minimum length.
        (None, reason)  if data is invalid or too short.
        """
        df = series.to_dataframe()

        if df.empty:
            return None, "empty series"

        # Enforce ascending sort — no future leakage through ordering.
        if not df.index.is_monotonic_increasing:
            logger.warning("Series index is not ascending; sorting now.")
            df = df.sort_index()

        if "close" not in df.columns:
            return None, "no 'close' column in dataframe"

        prices: np.ndarray = df["close"].values.astype(np.float64)

        # Drop any NaN prices.
        prices = prices[np.isfinite(prices)]

        if len(prices) < self.min_context_len:
            logger.warning(
                "Insufficient data for %s: got %d bars, need %d.",
                series.ticker,
                len(prices),
                self.min_context_len,
            )
            return None, f"need {self.min_context_len} bars, got {len(prices)}"

        return prices, None

    def _low_confidence_output(
        self,
        ticker: str,
        timeframe: str,
        horizon: int,
        reason: str,
    ) -> ForecastOutput:
        """
        Return a minimal ForecastOutput signalling "no information".
        Used when data is insufficient or backend is unavailable.
        """
        logger.debug(
            "Returning low-confidence forecast for %s %s (reason: %s).",
            ticker,
            timeframe,
            reason,
        )
        return ForecastOutput(
            ticker=ticker,
            timeframe=timeframe,
            direction=_LOW_CONFIDENCE_DIRECTION,
            expected_return=_LOW_CONFIDENCE_RETURN,
            confidence=_LOW_CONFIDENCE_CONFIDENCE,
            horizon=horizon,
            quantile_50=[],
            quantile_10=[],
            quantile_90=[],
        )
