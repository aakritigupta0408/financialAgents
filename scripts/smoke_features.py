"""
scripts/smoke_features.py — Phase 3 feature engineering smoke test.

Generates a synthetic 200-bar 1h OHLCV DataFrame, runs compute_all_features
through the pipeline, and prints a human-readable summary.

Runs without requiring an API key.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the project root is on the path when run directly.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from schemas.market_data import MarketSnapshot, OHLCVBar, OHLCVSeries
from src.features import compute_all_features

# ── Synthetic data generation ──────────────────────────────────────────────

EPOCH = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
TICKER = "SYNTHETIC"
N_BARS = 200
rng = np.random.default_rng(seed=42)


def _build_synthetic_series(n: int = N_BARS) -> OHLCVSeries:
    """
    Build a realistic-looking synthetic 1h OHLCV series with trending structure.
    Price follows a slow uptrend with noise, producing genuine swing highs/lows.
    """
    # Slow upward drift with noise.
    returns = rng.normal(loc=0.0005, scale=0.008, size=n)
    closes = 150.0 * np.exp(np.cumsum(returns))

    # Bar-level OHLCV generation.
    bars: list[OHLCVBar] = []
    for i in range(n):
        close = float(closes[i])
        prev_close = float(closes[i - 1]) if i > 0 else close
        open_ = prev_close + rng.normal(0, 0.002 * close)
        half_range = abs(rng.normal(0, 0.005 * close))
        high = max(open_, close) + abs(rng.normal(0, half_range * 0.5))
        low = min(open_, close) - abs(rng.normal(0, half_range * 0.5))

        # Clamp open/close to [low, high].
        open_ = float(np.clip(open_, low, high))
        close = float(np.clip(close, low, high))
        volume = float(rng.lognormal(mean=10.0, sigma=0.5))

        bars.append(
            OHLCVBar(
                timestamp=EPOCH + timedelta(hours=i),
                open=open_,
                high=float(high),
                low=float(low),
                close=close,
                volume=volume,
                ticker=TICKER,
                timeframe="1h",  # type: ignore[arg-type]
            )
        )

    return OHLCVSeries(ticker=TICKER, timeframe="1h", bars=bars)  # type: ignore[arg-type]


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Phase 3 Feature Engineering Smoke Test")
    print("=" * 60)

    print(f"\nGenerating {N_BARS}-bar synthetic 1h OHLCV series for {TICKER}...")
    series_1h = _build_synthetic_series(N_BARS)
    print(f"  Bars generated: {len(series_1h.bars)}")
    print(f"  Price range: {min(b.close for b in series_1h.bars):.2f} - {max(b.close for b in series_1h.bars):.2f}")

    snapshot = MarketSnapshot(
        ticker=TICKER,
        snapshot_time=EPOCH + timedelta(hours=N_BARS),
        tf_1h=series_1h,
    )

    print("\nRunning compute_all_features (primary_tf=1h)...")
    features = compute_all_features(snapshot, primary_tf="1h")

    # ── Structure ──────────────────────────────────────────────────────────
    structure = features["structure"]
    print("\n--- Structure Features ---")
    print(f"  Swing highs : {len(structure.swing_highs)}")
    print(f"  Swing lows  : {len(structure.swing_lows)}")
    print(f"  Trend state : {structure.trend_state}")
    print(f"  Trend strength: {structure.trend_strength:.3f}")
    print(f"  BOS events  : {len(structure.bos_events)}")
    print(f"  CHoCH events: {len(structure.choch_events)}")

    if structure.swing_highs:
        last_high = structure.swing_highs[-1]
        print(f"  Last swing high: {last_high.price:.4f} at {last_high.timestamp}")
    if structure.swing_lows:
        last_low = structure.swing_lows[-1]
        print(f"  Last swing low : {last_low.price:.4f} at {last_low.timestamp}")

    # ── Levels ─────────────────────────────────────────────────────────────
    levels = features["levels"]
    print("\n--- Level Features ---")
    print(f"  Resistance zones: {len(levels.resistance_zones)}")
    print(f"  Support zones   : {len(levels.support_zones)}")

    if levels.resistance_zones:
        top_res = levels.resistance_zones[-1]
        print(f"  Nearest resistance: [{top_res.low:.4f}, {top_res.high:.4f}]  strength={top_res.strength:.3f}")
    if levels.support_zones:
        top_sup = levels.support_zones[0]
        print(f"  Nearest support:    [{top_sup.low:.4f}, {top_sup.high:.4f}]  strength={top_sup.strength:.3f}")

    # ── Volatility ─────────────────────────────────────────────────────────
    vol = features["volatility"]
    print("\n--- Volatility Features ---")
    print(f"  ATR            : {vol.atr:.4f}")
    print(f"  ATR %          : {vol.atr_pct * 100:.3f}%")
    print(f"  Regime         : {vol.volatility_regime}")
    print(f"  Is expanding   : {vol.is_expanding}")

    # ── Liquidity ──────────────────────────────────────────────────────────
    liq = features["liquidity"]
    print("\n--- Liquidity Features ---")
    print(f"  Avg volume (20-bar): {liq.avg_volume:,.1f}")
    print(f"  Relative volume    : {liq.relative_volume:.3f}x")
    print(f"  Spread estimate    : {liq.spread_estimate:.6f}")

    # ── 4h resample check ──────────────────────────────────────────────────
    print("\n--- 4h Resample Check (from 1h, no tf_4h in snapshot) ---")
    features_4h = compute_all_features(snapshot, primary_tf="4h")
    struct_4h = features_4h["structure"]
    print(f"  4h swing highs : {len(struct_4h.swing_highs)}")
    print(f"  4h trend state : {struct_4h.trend_state}")

    print("\n" + "=" * 60)
    print("Smoke test PASSED — all features computed without errors.")
    print("=" * 60)


if __name__ == "__main__":
    main()
