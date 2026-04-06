"""
src.features.pipeline — master feature computation pipeline.

Ties together structure, levels, volatility, and liquidity into a single
call that operates on a MarketSnapshot.

4h resampling
-------------
Alpha Vantage has no native 4h endpoint.  When the caller requests 4h
features, this pipeline fetches the 1h OHLCVSeries from the snapshot
and resamples it to 4h using src.features.resampling.resample_ohlcv().
If tf_1h is None or empty, 4h features cannot be produced and the
corresponding entries will be absent from the output dict.

Timezone note
-------------
Timestamps are taken as-is from the snapshot (US/Eastern stored as
UTC-labelled).  No conversion is attempted.  See resampling.py for details.
"""

from __future__ import annotations

import pandas as pd

from schemas.features import (
    LevelFeatures,
    LiquidityFeatures,
    StructureFeatures,
    VolatilityFeatures,
)
from schemas.market_data import MarketSnapshot, OHLCVSeries
from src.features.levels import compute_levels
from src.features.liquidity import compute_liquidity
from src.features.resampling import resample_ohlcv
from src.features.structure import compute_structure
from src.features.volatility import compute_volatility


def compute_all_features(
    snapshot: MarketSnapshot,
    primary_tf: str = "1h",
) -> dict[str, StructureFeatures | LevelFeatures | VolatilityFeatures | LiquidityFeatures]:
    """
    Run the full feature engineering pipeline for a single MarketSnapshot.

    Parameters
    ----------
    snapshot : MarketSnapshot
        Must contain at least the series for *primary_tf* (or 1h for 4h).
    primary_tf : str
        The timeframe on which to run all features (default "1h").
        If "4h" is requested but tf_4h is None/empty, falls back to
        resampling from tf_1h.

    Returns
    -------
    dict with keys: "structure", "levels", "volatility", "liquidity"
    Each value is the corresponding Pydantic model.
    Returns partial results (empty schemas) for any component that fails.
    """
    ticker = snapshot.ticker
    series = _resolve_series(snapshot, primary_tf, ticker)

    if series is None or not series.bars:
        # Return empty feature shells so callers always get the full dict.
        return _empty_features(ticker, primary_tf)

    df = series.to_dataframe()
    tf = series.timeframe

    structure = _safe_compute(
        lambda: compute_structure(df, ticker, tf),
        StructureFeatures(ticker=ticker, timeframe=tf, trend_state="unknown"),
    )

    levels = _safe_compute(
        lambda: compute_levels(df, structure, ticker, tf),
        LevelFeatures(ticker=ticker, timeframe=tf),
    )

    volatility = _safe_compute(
        lambda: compute_volatility(df, ticker, tf),
        VolatilityFeatures(ticker=ticker, timeframe=tf, atr=0.0, atr_pct=0.0),
    )

    liquidity = _safe_compute(
        lambda: compute_liquidity(df, ticker, tf),
        LiquidityFeatures(ticker=ticker, timeframe=tf, avg_volume=0.0, relative_volume=0.0, spread_estimate=0.0),
    )

    return {
        "structure": structure,
        "levels": levels,
        "volatility": volatility,
        "liquidity": liquidity,
    }


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_series(
    snapshot: MarketSnapshot,
    primary_tf: str,
    ticker: str,
) -> OHLCVSeries | None:
    """
    Return the OHLCVSeries to use for feature computation.

    For "4h": try tf_4h first; if missing or empty, resample from tf_1h.
    For all others: return directly from snapshot.get().
    """
    if primary_tf == "4h":
        tf4h = snapshot.get("4h")
        if tf4h and tf4h.bars:
            return tf4h
        # Fall back: resample from 1h.
        tf1h = snapshot.get("1h")
        if tf1h and tf1h.bars:
            return resample_ohlcv(tf1h, "4h")
        return None

    return snapshot.get(primary_tf)


def _safe_compute(fn, fallback):
    """
    Execute *fn()* and return the result.
    On any exception, return *fallback* and do not raise.
    """
    try:
        return fn()
    except Exception:
        return fallback


def _empty_features(
    ticker: str,
    timeframe: str,
) -> dict:
    return {
        "structure": StructureFeatures(ticker=ticker, timeframe=timeframe, trend_state="unknown"),
        "levels": LevelFeatures(ticker=ticker, timeframe=timeframe),
        "volatility": VolatilityFeatures(ticker=ticker, timeframe=timeframe, atr=0.0, atr_pct=0.0),
        "liquidity": LiquidityFeatures(
            ticker=ticker,
            timeframe=timeframe,
            avg_volume=0.0,
            relative_volume=0.0,
            spread_estimate=0.0,
        ),
    }
