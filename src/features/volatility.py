"""
src.features.volatility — ATR and volatility regime computation.

Uses Wilder's smoothed ATR (the same method as tradingview / standard TA).

Wilder's ATR smoothing:
  ATR[0] = mean(TR[0:window])
  ATR[i] = (ATR[i-1] * (window - 1) + TR[i]) / window

Regime classification:
  Compute a rolling 50-bar history of ATR values.
  Classify the current ATR into:
    low:     below 25th percentile
    normal:  25th–75th percentile
    high:    75th–90th percentile
    extreme: above 90th percentile
  Falls back to "normal" when fewer than 50 bars are available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from schemas.features import VolatilityFeatures

# Number of historical ATR bars used for percentile-based regime classification.
_REGIME_HISTORY = 50

# Percentile thresholds for regime classification.
_PCT_LOW = 25.0
_PCT_HIGH = 75.0
_PCT_EXTREME = 90.0

# Number of bars to look back when checking if ATR is expanding.
_EXPANDING_LOOKBACK = 5


def compute_volatility(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    atr_window: int = 14,
) -> VolatilityFeatures:
    """
    Compute ATR and volatility regime features.

    Parameters
    ----------
    df : pd.DataFrame
        Timestamp-indexed OHLCV DataFrame (ascending).
    ticker : str
    timeframe : str
    atr_window : int
        ATR smoothing window (default 14, Wilder's standard).

    Returns
    -------
    VolatilityFeatures
    """
    if df.empty or len(df) < 2:
        return VolatilityFeatures(
            ticker=ticker,
            timeframe=timeframe,
            atr=0.0,
            atr_pct=0.0,
            volatility_regime="normal",
            is_expanding=False,
        )

    atr_series = _compute_atr(df, atr_window)

    current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    current_close = float(df["close"].iloc[-1])
    atr_pct = current_atr / current_close if current_close != 0.0 else 0.0

    regime = _classify_regime(atr_series)
    is_expanding = _check_expanding(atr_series)

    return VolatilityFeatures(
        ticker=ticker,
        timeframe=timeframe,
        atr=round(current_atr, 6),
        atr_pct=round(atr_pct, 6),
        volatility_regime=regime,
        is_expanding=is_expanding,
    )


# ── ATR computation ────────────────────────────────────────────────────────


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute Wilder's ATR for the full DataFrame.

    Returns a pd.Series indexed the same as df, with NaN for the first bar
    (no previous close available) and the first window-1 bars (insufficient
    history for the seed average).
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    tr = np.empty(n)
    tr[0] = high[0] - low[0]  # No previous close for bar 0.

    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    # Wilder's smoothing: seed with simple mean over first `window` bars.
    atr = np.full(n, np.nan)
    if n < window:
        return pd.Series(atr, index=df.index)

    atr[window - 1] = float(np.mean(tr[:window]))
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window

    return pd.Series(atr, index=df.index)


# ── Regime classification ──────────────────────────────────────────────────


def _classify_regime(atr_series: pd.Series) -> str:
    """
    Classify current ATR into a volatility regime.

    Uses the last _REGIME_HISTORY valid (non-NaN) ATR values as the
    reference distribution for percentile computation.
    """
    valid = atr_series.dropna()
    if len(valid) < _REGIME_HISTORY:
        return "normal"  # Insufficient history; fallback.

    history = valid.values[-_REGIME_HISTORY:]
    current = float(valid.iloc[-1])

    p25 = float(np.percentile(history, _PCT_LOW))
    p75 = float(np.percentile(history, _PCT_HIGH))
    p90 = float(np.percentile(history, _PCT_EXTREME))

    if current >= p90:
        return "extreme"
    if current >= p75:
        return "high"
    if current >= p25:
        return "normal"
    return "low"


# ── Expansion check ────────────────────────────────────────────────────────


def _check_expanding(atr_series: pd.Series) -> bool:
    """
    Return True if the current ATR is strictly greater than the ATR
    _EXPANDING_LOOKBACK bars ago.
    """
    valid = atr_series.dropna()
    if len(valid) <= _EXPANDING_LOOKBACK:
        return False
    current = float(valid.iloc[-1])
    past = float(valid.iloc[-1 - _EXPANDING_LOOKBACK])
    return current > past
