"""
src.features.liquidity — volume and spread-estimate computation.

Computes simple liquidity proxies from OHLCV data.

spread_estimate uses the high-low range as a rough proxy for bid/ask spread.
TODO: Replace spread_estimate with real bid/ask spread data once a tick-data
      or Level 2 provider is integrated.  The current proxy is only useful
      for relative comparison across instruments, not absolute spread values.
"""

from __future__ import annotations

import pandas as pd

from schemas.features import LiquidityFeatures


def compute_liquidity(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    vol_window: int = 20,
) -> LiquidityFeatures:
    """
    Compute liquidity features from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Timestamp-indexed OHLCV DataFrame (ascending).
    ticker : str
    timeframe : str
    vol_window : int
        Rolling window for average volume computation (default 20).

    Returns
    -------
    LiquidityFeatures
    """
    if df.empty:
        return LiquidityFeatures(
            ticker=ticker,
            timeframe=timeframe,
            avg_volume=0.0,
            relative_volume=0.0,
            spread_estimate=0.0,
        )

    volume = df["volume"]

    # avg_volume: rolling mean over the last vol_window bars (inclusive of current).
    # Use min_periods=1 so we get a value even for short series.
    rolling_mean = volume.rolling(window=vol_window, min_periods=1).mean()
    avg_vol = float(rolling_mean.iloc[-1])

    current_vol = float(volume.iloc[-1])

    # relative_volume: current bar volume / rolling average.
    if avg_vol > 0.0:
        rel_vol = current_vol / avg_vol
    else:
        rel_vol = 0.0

    # spread_estimate: rough proxy using 10% of the high-low range.
    # Returns 0.0 when volume is zero (no trading on that bar).
    # TODO: replace with real bid/ask spread when Level 2 data is available.
    if current_vol == 0.0:
        spread = 0.0
    else:
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        spread = (high - low) * 0.1

    return LiquidityFeatures(
        ticker=ticker,
        timeframe=timeframe,
        avg_volume=round(avg_vol, 4),
        relative_volume=round(rel_vol, 4),
        spread_estimate=round(spread, 6),
    )
