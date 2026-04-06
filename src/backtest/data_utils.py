"""
src.backtest.data_utils — Synthetic and file-based historical data for backtesting.

No API calls are made here. All data is generated deterministically from a
random seed or loaded from local files.

No-lookahead contract
---------------------
build_snapshot_from_series() is the single enforcement point for the no-lookahead
guarantee. It accepts a t_idx and slices series.bars[:t_idx+1] (inclusive of the
current bar). The caller must never pass a future bar into this function.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from schemas.market_data import MarketSnapshot, OHLCVBar, OHLCVSeries

_UTC = timezone.utc


def make_synthetic_ohlcv(
    n_bars: int = 300,
    start_price: float = 100.0,
    ticker: str = "SYN",
    timeframe: str = "1h",
    seed: int = 42,
    trend: float = 0.0002,
    volatility: float = 0.008,
) -> OHLCVSeries:
    """
    Generate a synthetic OHLCV series using a log-normal random walk with drift.

    Parameters
    ----------
    n_bars      : Number of bars to generate.
    start_price : Starting close price.
    ticker      : Ticker symbol for the series.
    timeframe   : Timeframe label (must be one of the literals in OHLCVBar).
    seed        : Random seed for reproducibility. Same seed → same series.
    trend       : Per-bar drift (log-return mean). Positive = uptrend.
    volatility  : Per-bar standard deviation of log-returns.

    Returns
    -------
    OHLCVSeries with n_bars validated OHLCVBar objects.

    Bar construction
    ----------------
    log_return ~ Normal(trend, volatility)
    close[t]   = close[t-1] * exp(log_return)
    open[t]    = close[t-1]
    high[t]    = close[t] * (1 + abs(log_return) * 0.5)   — always >= close
    low[t]     = close[t] * (1 - abs(log_return) * 0.5)   — always <= close
    volume[t]  = random int in [50_000, 500_000]
    """
    rng = random.Random(seed)

    # Box-Muller transform for normal samples without numpy dependency at this level.
    def _normal(mu: float, sigma: float) -> float:
        u1 = rng.random()
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(max(u1, 1e-300))) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    # Start timestamps at a round UTC hour.
    base_ts = datetime(2024, 1, 2, 9, 0, 0, tzinfo=_UTC)
    # Determine bar duration based on timeframe string for timestamp stepping.
    _tf_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440,
    }
    bar_minutes = _tf_minutes.get(timeframe, 60)

    bars: list[OHLCVBar] = []
    prev_close = start_price

    for i in range(n_bars):
        log_ret = _normal(trend, volatility)
        close = prev_close * math.exp(log_ret)
        open_ = prev_close
        abs_lr = abs(log_ret)
        high = close * (1.0 + abs_lr * 0.5)
        low = close * (1.0 - abs_lr * 0.5)

        # Clamp: high must be >= both open and close; low must be <= both.
        high = max(high, open_, close)
        low = min(low, open_, close)

        volume = float(rng.randint(50_000, 500_000))
        ts = base_ts + timedelta(minutes=bar_minutes * i)

        bar = OHLCVBar(
            timestamp=ts,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            ticker=ticker,
            timeframe=timeframe,  # type: ignore[arg-type]
        )
        bars.append(bar)
        prev_close = close

    return OHLCVSeries(
        ticker=ticker,
        timeframe=timeframe,  # type: ignore[arg-type]
        bars=bars,
        fetched_at=datetime.now(_UTC),
    )


def build_snapshot_from_series(
    series_1h: OHLCVSeries,
    t_idx: int,
    context_bars: int = 100,
) -> MarketSnapshot:
    """
    Build a no-lookahead MarketSnapshot for timestep t_idx.

    CRITICAL — No-lookahead guarantee
    ----------------------------------
    Only bars 0..t_idx (inclusive) are visible. The slice is further limited
    to the last `context_bars` bars so the feature pipeline never exceeds the
    intended window size. Bars at t_idx+1 and beyond are never included.

    Parameters
    ----------
    series_1h   : Full OHLCVSeries (all bars, including future bars).
    t_idx       : Current timestep index (0-based). This bar is the newest
                  bar the backtest loop is allowed to see.
    context_bars: Maximum number of recent bars to include in the snapshot.

    Returns
    -------
    MarketSnapshot with tf_1h populated; all other timeframes are None.
    """
    if t_idx < 0:
        raise ValueError(f"t_idx must be >= 0, got {t_idx}")

    # Slice: bars[0..t_idx] inclusive, then take the last context_bars.
    visible_bars = series_1h.bars[: t_idx + 1]
    context_slice = visible_bars[-context_bars:]

    truncated = OHLCVSeries(
        ticker=series_1h.ticker,
        timeframe=series_1h.timeframe,
        bars=list(context_slice),
        fetched_at=series_1h.fetched_at,
    )

    snapshot_time = context_slice[-1].timestamp if context_slice else datetime.now(_UTC)

    return MarketSnapshot(
        ticker=series_1h.ticker,
        snapshot_time=snapshot_time,
        tf_1h=truncated,
    )
