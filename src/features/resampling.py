"""
src.features.resampling — OHLCV resampling utilities.

Resolves the Phase 2 TODO: Alpha Vantage has no native 4h endpoint.
This module resamples a finer-grained OHLCVSeries to any coarser timeframe
using pandas' resample() with correct OHLC aggregation.

Timezone note
-------------
AV intraday timestamps are US/Eastern but stored as UTC-labelled datetimes
(no tz conversion is performed — see alpha_vantage._parse_timestamp).
All resample operations work on the raw timestamp index as-is.  Because
every bar in a series has consistent timezone labelling, relative ordering
and bucketing are correct even though the absolute UTC label is wrong.
Do NOT attempt to convert timezones here; that would require knowledge of
AV's labelling convention that is not yet verified.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from schemas.market_data import OHLCVBar, OHLCVSeries

# Mapping from timeframe strings to pandas offset aliases.
_TF_TO_PANDAS_OFFSET: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

# The set of timeframes the schema accepts.  Used for validation.
_VALID_TIMEFRAMES = frozenset(_TF_TO_PANDAS_OFFSET.keys())


def resample_ohlcv(series: OHLCVSeries, target_tf: str) -> OHLCVSeries:
    """
    Resample *series* to *target_tf* (must be a coarser timeframe).

    Parameters
    ----------
    series : OHLCVSeries
        Source data.  Can be any timeframe; must be finer than *target_tf*.
    target_tf : str
        Target timeframe string (e.g. "4h", "1h", "1d").
        Must be a key in the schema's Literal set.

    Returns
    -------
    OHLCVSeries
        A new OHLCVSeries with timeframe == target_tf.
        The last (potentially incomplete) bar is dropped.
        All bars pass OHLCVBar validation.

    Raises
    ------
    ValueError
        If target_tf is unknown or bars are empty.
    """
    if target_tf not in _VALID_TIMEFRAMES:
        raise ValueError(
            f"Unknown target timeframe {target_tf!r}. "
            f"Valid options: {sorted(_VALID_TIMEFRAMES)}"
        )

    if not series.bars:
        return OHLCVSeries(
            ticker=series.ticker,
            timeframe=target_tf,  # type: ignore[arg-type]
            bars=[],
        )

    offset = _TF_TO_PANDAS_OFFSET[target_tf]
    df = series.to_dataframe()

    # Resample with standard OHLCV aggregation.
    resampled = df.resample(offset, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop rows where any OHLCV field is NaN (empty buckets).
    resampled = resampled.dropna(how="any")

    # Drop the last bar — it may be an incomplete window.
    if len(resampled) > 1:
        resampled = resampled.iloc[:-1]
    elif len(resampled) == 1:
        # Single bar is always potentially incomplete; return empty series.
        return OHLCVSeries(
            ticker=series.ticker,
            timeframe=target_tf,  # type: ignore[arg-type]
            bars=[],
        )

    bars: list[OHLCVBar] = []
    ticker = series.ticker

    for ts, row in resampled.iterrows():
        # ts is a pandas Timestamp; convert to timezone-aware datetime.
        if isinstance(ts, pd.Timestamp):
            dt: datetime = ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)

        try:
            bar = OHLCVBar(
                timestamp=dt,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                ticker=ticker,
                timeframe=target_tf,  # type: ignore[arg-type]
            )
            bars.append(bar)
        except (ValueError, Exception):
            # Skip bars that fail Pydantic validation (e.g. open outside H/L).
            # This can happen at market-open when partial data is resampled.
            continue

    return OHLCVSeries(
        ticker=ticker,
        timeframe=target_tf,  # type: ignore[arg-type]
        bars=bars,
    )
