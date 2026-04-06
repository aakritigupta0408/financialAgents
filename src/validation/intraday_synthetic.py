"""
src.validation.intraday_synthetic — Structure-rich 1h synthetic OHLCV generator.

Designed specifically for FTA validation. The wave skeleton produces genuine
swing highs and lows that compute_structure(swing_window=5) can detect reliably,
while ATR stays in the 0.5-0.8% range typical for 1h equities.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from schemas.market_data import OHLCVBar, OHLCVSeries

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Wave skeleton definition
# ---------------------------------------------------------------------------

# Each segment: (direction, n_bars, move_pct, noise_pct)
# direction +1=up, -1=down (informational; move_pct sign drives the direction)
# move_pct: total fractional price move over n_bars (positive or negative)
# noise_pct: per-bar noise amplitude (applied as ± std of normal draw)
#
# Design goals:
# - Clear uptrend phases so compute_structure sees uptrend within 100-bar windows
# - Resistance zones >= 3% above entry so TimesFM forecast can clear them
# - Short pullbacks that don't consume too much of the 100-bar context window
# - Net positive drift so S/R zones are well-separated from current price
_WAVE_SKELETON: list[tuple[int, int, float, float]] = [
    # Phase A — Strong bull run (creates clear HH/HL structure)
    (+1, 50, +0.12, 0.0005),   # wave1_up: clean 12% run over 50 bars
    (-1, 15, -0.03, 0.0005),   # pullback1: shallow 3% dip
    (+1, 55, +0.10, 0.0005),   # wave2_up: 10% continuation
    (-1, 15, -0.025, 0.0005),  # pullback2: shallow
    (+1, 50, +0.08, 0.0005),   # wave3_up: 8% extension
    # Phase B — Mild range/consolidation
    (+1, 20, +0.01, 0.001),    # range: sideways with slight up bias
    # Phase C — Bear correction (moderate, not severe)
    (-1, 25, -0.05, 0.0005),   # bear wave: 5% down
    (+1, 10, +0.02, 0.0005),   # small bounce
    (-1, 20, -0.04, 0.0005),   # second bear leg
    # Phase D — Re-accumulation base
    (+1, 25, +0.03, 0.0008),   # base: 3% recovery
]
# One full cycle = 285 bars

# ---------------------------------------------------------------------------
# Ticker proxies
# ---------------------------------------------------------------------------

TICKER_PROXIES: list[dict] = [
    {"ticker": "AAPL_1H", "seed": 1, "base_price": 255.0},
    {"ticker": "MSFT_1H", "seed": 2, "base_price": 400.0},
    {"ticker": "NVDA_1H", "seed": 3, "base_price": 185.0},
    {"ticker": "TSLA_1H", "seed": 4, "base_price": 420.0},
    {"ticker": "SPY_1H",  "seed": 5, "base_price": 680.0},
    {"ticker": "AMD_1H",  "seed": 6, "base_price": 215.0},
]

_PROXY_MAP: dict[str, dict] = {p["ticker"]: p for p in TICKER_PROXIES}

# Trading hours: 09:00–15:00 ET = 14:00–20:00 UTC inclusive
# 7 bars per day (hours 14, 15, 16, 17, 18, 19, 20 UTC)
_SESSION_HOURS_UTC = set(range(14, 21))  # 14 inclusive, 21 exclusive → 14..20


def _generate_session_timestamps(n_bars: int) -> list[datetime]:
    """Generate n_bars timestamps at hourly intervals within US session hours."""
    timestamps: list[datetime] = []
    current = datetime(2025, 1, 2, 14, 0, 0, tzinfo=_UTC)
    while len(timestamps) < n_bars:
        if current.hour in _SESSION_HOURS_UTC:
            timestamps.append(current)
        current += timedelta(hours=1)
        # If we moved past 20:00 UTC, skip to next day 14:00
        if current.hour > 20:
            current = current.replace(
                hour=14, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
    return timestamps[:n_bars]


def _build_close_series(
    n_bars: int,
    base_price: float,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """
    Build the close price series and per-bar noise scalar.

    Returns (closes, noise_scalars).
    """
    closes: list[float] = []
    noise_scalars: list[float] = []

    price = base_price

    # Build a flat list of per-bar (drift, noise_pct) from the repeating skeleton
    bar_specs: list[tuple[float, float]] = []
    skeleton_total = sum(seg[1] for seg in _WAVE_SKELETON)
    cycles_needed = math.ceil(n_bars / skeleton_total) + 1

    for _ in range(cycles_needed):
        for direction, seg_bars, move_pct, noise_pct in _WAVE_SKELETON:
            drift_per_bar = move_pct / seg_bars
            for k in range(seg_bars):
                # Smooth portion: first 80% of segment uses tiny noise
                smooth_noise = noise_pct * 0.5 if k < int(seg_bars * 0.8) else noise_pct
                bar_specs.append((drift_per_bar, smooth_noise))

    bar_specs = bar_specs[:n_bars]

    for drift, noise_pct in bar_specs:
        noise = rng.standard_normal() * noise_pct
        noise_scalars.append(noise)
        price = price * (1.0 + drift + noise)
        price = max(price, 1.0)
        closes.append(price)

    return closes, noise_scalars


def make_structured_1h_series(
    ticker: str = "SIM",
    n_bars: int = 600,
    seed: int = 42,
    base_price: float = 200.0,
) -> OHLCVSeries:
    """
    Generate a structure-rich 1h synthetic series designed for FTA validation.

    The wave skeleton ensures that:
    - Bull runs (40-45 bar segments) create clear swing highs detectable by
      compute_structure(swing_window=5).
    - Pullbacks (20 bar segments) create clear swing lows.
    - ATR per bar is ~0.6% of price (atr_factor=0.006), realistic for 1h equities.
    - Minimum 600 bars ensures enough data for the full feature pipeline.

    Parameters
    ----------
    ticker    : Ticker symbol string.
    n_bars    : Number of bars to generate (must be >= 12 for swing detection).
    seed      : Random seed for reproducibility.
    base_price: Starting price for the series.

    Returns
    -------
    OHLCVSeries with n_bars OHLCVBar objects in ascending timestamp order.
    """
    rng = np.random.default_rng(seed)
    atr_factor = 0.005  # ~0.5% of price per bar (realistic 1h equity)

    closes, noise_scalars = _build_close_series(n_bars, base_price, rng)
    timestamps = _generate_session_timestamps(n_bars)

    bars: list[OHLCVBar] = []
    for i in range(n_bars):
        close = closes[i]
        ns = noise_scalars[i]
        bar_range = close * atr_factor

        # High and low spread asymmetrically around close using noise scalar
        high = close + bar_range * abs(ns) * 1.2 + bar_range * 0.3
        low = close - bar_range * abs(ns) * 1.2 - bar_range * 0.3

        # Open is previous close (or slightly below first close)
        if i > 0:
            open_ = closes[i - 1]
        else:
            open_ = close * (1.0 - 0.001)

        # Ensure OHLCV validity: high >= open,close >= low
        high = max(high, open_, close)
        low = min(low, open_, close)
        open_ = max(low, min(high, open_))

        volume = int(1_000_000 * (1.0 + 0.5 * abs(ns)))

        bar = OHLCVBar(
            timestamp=timestamps[i],
            open=round(open_, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(close, 4),
            volume=float(volume),
            ticker=ticker,
            timeframe="1h",
        )
        bars.append(bar)

    return OHLCVSeries(
        ticker=ticker,
        timeframe="1h",
        bars=bars,
        fetched_at=datetime.now(_UTC),
    )


def load_or_generate_1h_series(
    ticker: str,
    n_bars: int = 600,
) -> OHLCVSeries:
    """
    Load a real CSV fixture if available, otherwise generate synthetic data.

    CSV fixture path: tests/fixtures/real_data_1h/{ticker}.csv
    CSV format: timestamp,open,high,low,close,volume (ISO datetime strings).

    Falls back to make_structured_1h_series() using TICKER_PROXIES config.
    If ticker is not in TICKER_PROXIES, uses seed=42 and base_price=200.0.

    Parameters
    ----------
    ticker : Ticker string (e.g. "AAPL_1H").
    n_bars : Number of bars to generate if no fixture exists.

    Returns
    -------
    OHLCVSeries for the requested ticker.
    """
    # Check for real fixture first
    root = Path(__file__).parent.parent.parent  # repo root
    fixture_path = root / "tests" / "fixtures" / "real_data_1h" / f"{ticker}.csv"

    if fixture_path.exists():
        return _load_csv_fixture(fixture_path, ticker)

    # Generate synthetic
    proxy = _PROXY_MAP.get(ticker, {"seed": 42, "base_price": 200.0})
    return make_structured_1h_series(
        ticker=ticker,
        n_bars=n_bars,
        seed=proxy["seed"],
        base_price=proxy["base_price"],
    )


def _load_csv_fixture(path: Path, ticker: str) -> OHLCVSeries:
    """Load an OHLCVSeries from a CSV file in timestamp,open,high,low,close,volume format."""
    bars: list[OHLCVBar] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts_raw = row["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_UTC)

            bar = OHLCVBar(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                ticker=ticker,
                timeframe="1h",
            )
            bars.append(bar)

    bars.sort(key=lambda b: b.timestamp)
    return OHLCVSeries(
        ticker=ticker,
        timeframe="1h",
        bars=bars,
        fetched_at=datetime.now(_UTC),
    )
