"""
scripts/smoke_timesfm.py — Phase 4 smoke test for the TimesFM wrapper.

Runs without an API key and without TimesFM installed.
Exercises the statistical fallback forecaster on a 100-bar synthetic series
with three phases: uptrend, flat, then downtrend.
"""

from __future__ import annotations

import sys
import os

# Ensure repo root is on sys.path regardless of working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from schemas.market_data import OHLCVBar, OHLCVSeries
from src.timesfm import get_forecaster, run_forecast


def _build_synthetic_series(ticker: str = "SYNTH", timeframe: str = "1h") -> OHLCVSeries:
    """
    Build a 100-bar synthetic 1h series:
      - Bars 0–39   : uptrend (100 → 120)
      - Bars 40–59  : flat (120)
      - Bars 60–99  : downtrend (120 → 100)
    """
    base_time = datetime(2024, 6, 1, 9, 0, 0)
    prices: list[float] = []

    # Uptrend: 40 bars, +0.50 per bar.
    for i in range(40):
        prices.append(100.0 + i * 0.50)

    # Flat: 20 bars at 120.
    for _ in range(20):
        prices.append(120.0)

    # Downtrend: 40 bars, -0.50 per bar.
    for i in range(40):
        prices.append(120.0 - i * 0.50)

    bars = []
    for i, close in enumerate(prices):
        bars.append(
            OHLCVBar(
                timestamp=base_time + timedelta(hours=i),
                open=close,
                high=close + 0.25,
                low=max(close - 0.25, 0.01),
                close=close,
                volume=50_000.0 + i * 100,
                ticker=ticker,
                timeframe=timeframe,
            )
        )

    return OHLCVSeries(ticker=ticker, timeframe=timeframe, bars=bars)


def main() -> None:
    print("=" * 60)
    print("  Phase 4 — TimesFM Wrapper Smoke Test")
    print("=" * 60)

    series = _build_synthetic_series()
    horizon = 10

    forecaster = get_forecaster(prefer_timesfm=True)
    print(f"\nForecaster name : {forecaster.name}")
    print(f"TimesFM available: {type(forecaster).__name__ == 'TimesFMForecaster'}")

    print(f"\nRunning forecast on '{series.ticker}' ({series.timeframe}) "
          f"— {len(series.bars)} bars, horizon={horizon}.\n")

    out = run_forecast(series, horizon=horizon)

    print(f"  Direction       : {out.direction}")
    print(f"  Expected return : {out.expected_return * 100:.4f}%")
    print(f"  Confidence      : {out.confidence:.4f}")
    print(f"  Horizon         : {out.horizon} bars")

    print(f"\n  quantile_50 path (all {horizon} values):")
    for i, price in enumerate(out.quantile_50):
        bar_label = f"  bar+{i+1:02d}"
        print(f"    {bar_label}: {price:.4f}")

    print()
    print("  quantile_10 (pessimistic)  q50 (central)  quantile_90 (optimistic)")
    for i in range(horizon):
        q10 = out.quantile_10[i] if i < len(out.quantile_10) else float("nan")
        q50 = out.quantile_50[i] if i < len(out.quantile_50) else float("nan")
        q90 = out.quantile_90[i] if i < len(out.quantile_90) else float("nan")
        print(f"    bar+{i+1:02d}: {q10:9.4f}  {q50:9.4f}  {q90:9.4f}")

    # Sanity checks.
    print()
    assert out.direction in ("up", "down"), "FAIL: direction must be 'up' or 'down'"
    assert 0.0 <= out.confidence <= 1.0, f"FAIL: confidence {out.confidence} out of range"
    assert len(out.quantile_50) == horizon, f"FAIL: q50 len {len(out.quantile_50)} != horizon"
    assert any(v != 0.0 for v in out.quantile_50), "FAIL: quantile_50 is all zeros — fallback is not computing"

    print("All sanity checks passed.")
    print("\nSmoke test COMPLETE — no errors.\n")


if __name__ == "__main__":
    main()
