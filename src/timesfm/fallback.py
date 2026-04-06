"""
src.timesfm.fallback — Statistical fallback forecaster.

Used when TimesFM is not installed (Python 3.12 incompatibility as of 2026-04).

Two methods are combined:

  Method A — EWM Momentum
    Compute log-returns, apply EWM with configurable span, project over horizon.

  Method B — Linear Regression Trend
    Fit OLS on (bar_index, log_price) over the last `context_len` bars,
    extrapolate slope over horizon.

Combined:
  combined_return = 0.5 * ewm_return + 0.5 * lr_return

Confidence heuristic:
  Base = |combined_return| / atr_pct, clamped to [0.10, 0.85].
  When the two methods disagree (opposite signs) → multiply by 0.6.

No lookahead: only prices up to and including the last bar are used.
"""

from __future__ import annotations

import logging

import numpy as np

from schemas.forecast import ForecastOutput
from schemas.market_data import OHLCVSeries
from src.timesfm.base import BaseForecaster

logger = logging.getLogger(__name__)

# ── sklearn import (soft dependency) ─────────────────────────────────────────
try:
    from sklearn.linear_model import LinearRegression as _LinearRegression
    _SKLEARN_AVAILABLE = True
except (ImportError, AttributeError, ValueError):
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available; linear-regression component will be disabled. "
        "Install with: pip install scikit-learn"
    )


class StatisticalForecaster(BaseForecaster):
    """
    Pure-Python/NumPy/pandas statistical forecaster.

    Parameters
    ----------
    ewm_span      : int   — span for EWM return smoothing (default 12).
    context_len   : int   — number of bars to use for LR trend (default 32).
    min_context_len: int  — minimum bars required (default 32).
    """

    def __init__(
        self,
        ewm_span: int = 12,
        context_len: int = 32,
        min_context_len: int = 32,
    ) -> None:
        super().__init__(min_context_len=min_context_len)
        self.ewm_span = ewm_span
        self.context_len = context_len

    # ── Abstract interface ────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        return True  # Pure Python/NumPy — always available.

    @property
    def name(self) -> str:
        return "StatisticalForecaster"

    # ── Forecast ──────────────────────────────────────────────────────────

    def forecast(
        self,
        series: OHLCVSeries,
        horizon: int,
        ticker: str,
        timeframe: str,
    ) -> ForecastOutput:
        """
        Produce a forecast using EWM + Linear Regression combination.
        Uses only past data (no lookahead).
        """
        prices, err = self._prepare_prices(series)
        if prices is None:
            return self._low_confidence_output(ticker, timeframe, horizon, err)

        # Restrict to the last `context_len` prices (no future leakage).
        prices = prices[-self.context_len :]

        log_returns = np.log(prices[1:] / prices[:-1])

        ewm_ret = self._ewm_projected_return(log_returns, horizon)
        lr_ret = self._lr_projected_return(prices, horizon)

        combined_return = 0.5 * ewm_ret + 0.5 * lr_ret
        direction = "up" if combined_return > 0 else "down"

        atr_pct = self._atr_pct(prices)
        confidence = self._confidence(ewm_ret, lr_ret, combined_return, atr_pct)

        q50, q10, q90 = self._quantile_paths(prices, log_returns, horizon)

        logger.debug(
            "%s | %s %s | ewm=%.4f lr=%.4f combined=%.4f conf=%.3f dir=%s",
            self.name, ticker, timeframe,
            ewm_ret, lr_ret, combined_return, confidence, direction,
        )

        return ForecastOutput(
            ticker=ticker,
            timeframe=timeframe,
            direction=direction,
            expected_return=float(combined_return),
            confidence=float(confidence),
            horizon=horizon,
            quantile_50=[float(v) for v in q50],
            quantile_10=[float(v) for v in q10],
            quantile_90=[float(v) for v in q90],
        )

    # ── Method A: EWM Momentum ────────────────────────────────────────────

    def _ewm_projected_return(
        self,
        log_returns: np.ndarray,
        horizon: int,
    ) -> float:
        """
        EWM-smoothed return * horizon.
        Uses only historical returns — no future data.
        """
        if len(log_returns) == 0:
            return 0.0

        # EWM: alpha = 2 / (span + 1), applied iteratively (no lookahead).
        alpha = 2.0 / (self.ewm_span + 1)
        ewm = float(log_returns[0])
        for r in log_returns[1:]:
            ewm = alpha * r + (1 - alpha) * ewm

        return ewm * horizon

    # ── Method B: Linear Regression Trend ────────────────────────────────

    def _lr_projected_return(
        self,
        prices: np.ndarray,
        horizon: int,
    ) -> float:
        """
        Fit OLS on (bar_index, log_price); extrapolate slope over horizon bars.
        Falls back to a NumPy polyfit when sklearn is unavailable.
        """
        n = len(prices)
        log_prices = np.log(prices)
        x = np.arange(n, dtype=np.float64)

        if _SKLEARN_AVAILABLE:
            reg = _LinearRegression()
            reg.fit(x.reshape(-1, 1), log_prices)
            slope = float(reg.coef_[0])
        else:
            # numpy polyfit fallback (degree-1 polynomial = linear regression).
            coeffs = np.polyfit(x, log_prices, deg=1)
            slope = float(coeffs[0])

        return slope * horizon

    # ── Confidence heuristic ──────────────────────────────────────────────

    def _confidence(
        self,
        ewm_ret: float,
        lr_ret: float,
        combined_return: float,
        atr_pct: float,
    ) -> float:
        """
        Heuristic confidence in [0.10, 0.85].

        Base: |combined_return| / atr_pct.
        Disagreement penalty when EWM and LR point in opposite directions.
        """
        denominator = atr_pct if atr_pct > 1e-8 else 0.01
        base = abs(combined_return) / denominator
        confidence = max(0.10, min(0.85, base))

        # Disagreement: the two methods point in opposite directions.
        if ewm_ret * lr_ret < 0:
            confidence *= 0.6

        return max(0.10, min(0.85, confidence))

    # ── Quantile paths ────────────────────────────────────────────────────

    def _quantile_paths(
        self,
        prices: np.ndarray,
        log_returns: np.ndarray,
        horizon: int,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Build quantile_50, quantile_10, quantile_90 price paths.

          q50: EWM-smoothed log-return applied step-by-step from last price.
          q10/q90: ±1 sigma of recent log-returns applied symmetrically.
        """
        last_price = float(prices[-1])

        if len(log_returns) == 0:
            flat = [last_price] * horizon
            return flat, flat, flat

        sigma = float(np.std(log_returns)) if len(log_returns) > 1 else 0.0

        alpha = 2.0 / (self.ewm_span + 1)
        ewm = float(log_returns[0])
        for r in log_returns[1:]:
            ewm = alpha * r + (1 - alpha) * ewm

        q50, q10, q90 = [], [], []
        price_50 = last_price
        price_10 = last_price
        price_90 = last_price

        for _ in range(horizon):
            price_50 *= np.exp(ewm)
            price_10 *= np.exp(ewm - sigma)
            price_90 *= np.exp(ewm + sigma)
            q50.append(float(price_50))
            q10.append(float(price_10))
            q90.append(float(price_90))

        return q50, q10, q90

    # ── ATR proxy ─────────────────────────────────────────────────────────

    @staticmethod
    def _atr_pct(prices: np.ndarray) -> float:
        """
        Approximate ATR% as std of log-returns.
        Used only as a denominator for the confidence heuristic.
        """
        if len(prices) < 2:
            return 0.01
        log_rets = np.log(prices[1:] / prices[:-1])
        std = float(np.std(log_rets))
        return std if std > 1e-8 else 0.01
