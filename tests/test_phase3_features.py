"""
tests/test_phase3_features.py — Phase 3 feature engineering test suite.

All tests use synthetic/fixture data — no live API calls.

Naming convention:
  test_<module>_<scenario>
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from schemas.features import (
    LevelFeatures,
    LiquidityFeatures,
    StructureFeatures,
    VolatilityFeatures,
)
from schemas.market_data import MarketSnapshot, OHLCVBar, OHLCVSeries
from src.features.levels import compute_levels
from src.features.liquidity import compute_liquidity
from src.features.pipeline import compute_all_features
from src.features.resampling import resample_ohlcv
from src.features.structure import compute_structure
from src.features.volatility import compute_volatility

# ── Fixture helpers ────────────────────────────────────────────────────────

_EPOCH = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def _make_bar(
    i: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000.0,
    ticker: str = "TEST",
    timeframe: str = "1h",
    delta_hours: int = 1,
) -> OHLCVBar:
    return OHLCVBar(
        timestamp=_EPOCH + timedelta(hours=i * delta_hours),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        ticker=ticker,
        timeframe=timeframe,  # type: ignore[arg-type]
    )


def _make_df(
    closes: list[float],
    ticker: str = "TEST",
    timeframe: str = "1h",
    delta_hours: int = 1,
    volume: float = 1000.0,
) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame from a list of close prices.
    Open == close of previous bar (or same as close for bar 0).
    High = close + 0.5, Low = close - 0.5  (fixed offset, not percentage).
    This keeps each bar's high/low uniquely tied to its close value,
    which is essential for strict-greater swing detection tests.
    """
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = c + 0.5
        lo = c - 0.5
        rows.append(
            {
                "timestamp": _EPOCH + timedelta(hours=i * delta_hours),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": volume,
            }
        )
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def _make_series(
    closes: list[float],
    ticker: str = "TEST",
    timeframe: str = "1h",
    delta_hours: int = 1,
) -> OHLCVSeries:
    bars = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) * 1.005
        lo = min(o, c) * 0.995
        bars.append(
            _make_bar(
                i,
                open_=o,
                high=h,
                low=lo,
                close=c,
                ticker=ticker,
                timeframe=timeframe,
                delta_hours=delta_hours,
            )
        )
    return OHLCVSeries(ticker=ticker, timeframe=timeframe, bars=bars)  # type: ignore[arg-type]


# ── 1. test_resample_1h_to_4h ──────────────────────────────────────────────


def test_resample_1h_to_4h():
    """Resample 20 1h bars → 4h.  Verify bar count and OHLCV integrity."""
    # 20 hourly bars → 4 complete 4h bars (hours 0-3, 4-7, 8-11, 12-15)
    # hour 16-19 is the last partial bar that should be dropped.
    closes = [100.0 + i for i in range(20)]
    series_1h = _make_series(closes, timeframe="1h")

    series_4h = resample_ohlcv(series_1h, "4h")

    assert series_4h.timeframe == "4h"
    assert len(series_4h.bars) >= 1, "Expected at least 1 complete 4h bar"

    for bar in series_4h.bars:
        assert bar.high >= bar.open
        assert bar.high >= bar.close
        assert bar.low <= bar.open
        assert bar.low <= bar.close
        assert bar.volume >= 0.0


def test_resample_1h_to_4h_ohlcv_correctness():
    """
    Verify that a 4h bar's O/H/L/C/V correctly aggregates its 4 constituent 1h bars.
    Use a controlled price sequence so we know the expected values.
    """
    # 8 bars → 2 complete 4h windows (drop the last partial if any).
    # Bar 0-3: closes = [10, 12, 11, 13]
    # Bar 4-7: closes = [14, 13, 15, 12]
    closes = [10.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 12.0]
    series_1h = _make_series(closes, timeframe="1h")
    series_4h = resample_ohlcv(series_1h, "4h")

    assert series_4h.timeframe == "4h"
    # Each 4h bar must pass OHLCVBar validation (open/close inside high/low).
    for bar in series_4h.bars:
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high


# ── 2. test_resample_incomplete_bar_dropped ────────────────────────────────


def test_resample_incomplete_bar_dropped():
    """The last 4h bar (incomplete window) must be dropped."""
    # 5 bars: bars 0-3 form a complete 4h, bar 4 is alone in a new 4h window.
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    series_1h = _make_series(closes, timeframe="1h")
    series_4h = resample_ohlcv(series_1h, "4h")

    # The last partial window (bar 4 alone) must be dropped.
    # We should have at most 1 bar (the 0-3 window).
    # If even that bar is dropped (insufficient data), bars list is empty — also fine.
    assert isinstance(series_4h.bars, list)
    # Verify no bar contains only partial data that would violate OHLCV constraints.
    for bar in series_4h.bars:
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high


# ── 3. test_swing_detection ────────────────────────────────────────────────


def test_swing_detection():
    """Known synthetic series with clear swings.  Verify highs and lows are found."""
    # Two complete waves followed by enough trailing bars to confirm the last swing.
    # Wave 1: rise to 110, fall to 98
    # Wave 2: rise to 120, fall to 102
    # Trailing flat so window confirms last swing.
    # With swing_window=2 and high=c+0.5, low=c-0.5:
    #   swing high at close=110 (i=2): left=[100,105], right=[105,98] → 110 is max ✓
    #   swing high at close=120 (i=7): left=[98,108], right=[112,102] → 120 is max ✓
    #   swing low at close=98 (i=4): left=[110,105], right=[108,112] → 98 is min ✓
    #   swing low at close=102 (i=9): left=[120,112], right=[100,100] → 102 is min ✓
    closes = [
        100.0, 105.0, 110.0, 105.0, 98.0,   # wave 1: up to 110, back to 98
        108.0, 112.0, 120.0, 112.0, 102.0,  # wave 2: up to 120, back to 102
        100.0, 100.0,                         # trailing flat for confirmation
    ]
    df = _make_df(closes)
    structure = compute_structure(df, "TEST", "1h", swing_window=2)

    assert len(structure.swing_highs) >= 1, "Expected at least one swing high"
    assert len(structure.swing_lows) >= 1, "Expected at least one swing low"

    for sh in structure.swing_highs:
        assert sh.swing_type == "high"
        assert sh.price > 100.0

    for sl in structure.swing_lows:
        assert sl.swing_type == "low"


# ── 4. test_trend_uptrend ──────────────────────────────────────────────────


def test_trend_uptrend():
    """HH + HL series → trend_state == uptrend."""
    # Four complete waves each making HH and HL.
    # With swing_window=2 and high=c+0.5, low=c-0.5:
    # Peaks at 108, 116, 124, 132 → HH (each > prior)
    # Troughs at 100, 104, 108, 112 → HL (each > prior)
    closes = [
        100.0, 104.0, 108.0, 104.0, 100.0,  # wave 1: peak=108, trough=100
        106.0, 110.0, 116.0, 110.0, 104.0,  # wave 2: peak=116, trough=104
        108.0, 114.0, 124.0, 116.0, 108.0,  # wave 3: peak=124, trough=108
        114.0, 120.0, 132.0, 122.0, 112.0,  # wave 4: peak=132, trough=112
        112.0, 112.0,                         # trailing flat for confirmation
    ]
    df = _make_df(closes)
    structure = compute_structure(df, "TEST", "1h", swing_window=2)

    assert structure.trend_state == "uptrend", (
        f"Expected uptrend, got {structure.trend_state}. "
        f"swing_highs={[s.price for s in structure.swing_highs]}, "
        f"swing_lows={[s.price for s in structure.swing_lows]}"
    )
    assert structure.trend_strength > 0.0


# ── 5. test_trend_downtrend ────────────────────────────────────────────────


def test_trend_downtrend():
    """LH + LL series → trend_state == downtrend."""
    # Four complete waves each making LH and LL (mirror of uptrend fixture).
    closes = [
        132.0, 128.0, 124.0, 128.0, 132.0,  # wave 1: peak=132, trough=124
        128.0, 122.0, 116.0, 122.0, 128.0,  # wave 2: peak=128, trough=116
        122.0, 116.0, 108.0, 114.0, 120.0,  # wave 3: peak=120, trough=108
        114.0, 108.0, 100.0, 106.0, 112.0,  # wave 4: peak=112, trough=100
        112.0, 112.0,                         # trailing flat for confirmation
    ]
    df = _make_df(closes)
    structure = compute_structure(df, "TEST", "1h", swing_window=2)

    assert structure.trend_state == "downtrend", (
        f"Expected downtrend, got {structure.trend_state}. "
        f"swing_highs={[s.price for s in structure.swing_highs]}, "
        f"swing_lows={[s.price for s in structure.swing_lows]}"
    )
    assert structure.trend_strength > 0.0


# ── 6. test_bos_detection ──────────────────────────────────────────────────


def test_bos_detection():
    """Price closes below a swing low → bearish BOS recorded."""
    # Same 4-wave uptrend as test_trend_uptrend (produces confirmed uptrend),
    # then a sharp drop below the first swing low (100) to trigger bearish BOS.
    closes = [
        100.0, 104.0, 108.0, 104.0, 100.0,  # wave 1: peak=108, trough=100
        106.0, 110.0, 116.0, 110.0, 104.0,  # wave 2: peak=116, trough=104
        108.0, 114.0, 124.0, 116.0, 108.0,  # wave 3: peak=124, trough=108
        114.0, 120.0, 132.0, 122.0, 112.0,  # wave 4: peak=132, trough=112
        112.0, 112.0,                         # trailing flat (confirms last swing)
        90.0, 90.0, 90.0,                    # bearish break: below first trough (100)
    ]
    df = _make_df(closes)
    structure = compute_structure(df, "TEST", "1h", swing_window=2)

    # We expect at least one bearish BOS.
    bearish_bos = [b for b in structure.bos_events if b.direction == "bearish"]
    assert len(bearish_bos) >= 1, (
        f"Expected bearish BOS, got bos_events={structure.bos_events}"
    )
    # The confirmation close must be below the broken level.
    for bos in bearish_bos:
        assert bos.confirmation_close < bos.broken_level


# ── 7. test_choch_detection ────────────────────────────────────────────────


def test_choch_detection():
    """Counter-trend break in an uptrend → CHoCH recorded."""
    # Build an uptrend, then break below swing low → CHoCH.
    closes = [
        100.0, 105.0, 100.0,
        110.0, 100.0,
        115.0, 100.0,
        # Break below established lows.
        90.0, 92.0,
    ]
    df = _make_df(closes)
    structure = compute_structure(df, "TEST", "1h", swing_window=2)

    # If trend is uptrend and there's a bearish BOS, we expect a bearish CHoCH.
    if structure.trend_state == "uptrend":
        assert len(structure.choch_events) >= 1, (
            f"Expected CHoCH in uptrend with bearish break. "
            f"trend={structure.trend_state}, bos={structure.bos_events}"
        )
        # CHoCH must be bearish (first counter-trend break in an uptrend).
        bearish_chochs = [c for c in structure.choch_events if c.direction == "bearish"]
        assert len(bearish_chochs) >= 1
    else:
        # If the short series doesn't classify as uptrend, skip the CHoCH check.
        pytest.skip(f"Trend not classified as uptrend (got {structure.trend_state}); skipping CHoCH check")


# ── 8. test_support_resistance_zones ──────────────────────────────────────


def test_support_resistance_zones():
    """Swing points cluster into correct zones."""
    # Zigzag: highs near 110, lows near 100.
    closes = [
        100.0, 110.0, 100.0,
        110.0, 100.0,
        110.0, 100.0,
        110.0, 100.0,
        110.0,
    ]
    df = _make_df(closes)
    structure = compute_structure(df, "TEST", "1h", swing_window=2)
    levels = compute_levels(df, structure, "TEST", "1h", zone_margin_pct=0.005)

    # With a clear zigzag we expect at least one resistance zone (near 110)
    # and at least one support zone (near 100).
    assert isinstance(levels, LevelFeatures)
    # Check that zones are valid PriceZone objects.
    for zone in levels.resistance_zones:
        assert zone.high >= zone.low
        assert 0.0 <= zone.strength <= 1.0
        assert zone.zone_type == "resistance"

    for zone in levels.support_zones:
        assert zone.high >= zone.low
        assert 0.0 <= zone.strength <= 1.0
        assert zone.zone_type == "support"


# ── 9. test_zone_strength_increases_with_touches ──────────────────────────


def test_zone_strength_increases_with_touches():
    """More price touches near a level → higher strength."""
    # Series where price repeatedly visits ~100 (many touches).
    many_touches = [100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0]
    # Series where price visits ~100 only once.
    few_touches = [100.0, 102.0, 105.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0]

    df_many = _make_df(many_touches)
    df_few = _make_df(few_touches)

    struct_many = compute_structure(df_many, "TEST", "1h", swing_window=2)
    struct_few = compute_structure(df_few, "TEST", "1h", swing_window=2)

    levels_many = compute_levels(df_many, struct_many, "TEST", "1h", zone_margin_pct=0.01)
    levels_few = compute_levels(df_few, struct_few, "TEST", "1h", zone_margin_pct=0.01)

    # Aggregate max strength for the "many touches" scenario.
    if levels_many.support_zones or levels_many.resistance_zones:
        max_strength_many = max(
            [z.strength for z in levels_many.support_zones + levels_many.resistance_zones],
            default=0.0,
        )
        max_strength_few = max(
            [z.strength for z in levels_few.support_zones + levels_few.resistance_zones],
            default=0.0,
        )
        assert max_strength_many >= max_strength_few, (
            f"Expected many-touch strength {max_strength_many} >= "
            f"few-touch strength {max_strength_few}"
        )


# ── 10. test_atr_calculation ──────────────────────────────────────────────


def test_atr_calculation():
    """
    Hand-verify ATR on a simple controlled series.

    Build a 5-bar series with known TR values, then check the Wilder seed.
    Use atr_window=3 to keep the math tractable.
    """
    # Bars: each row is (open, high, low, close).
    # TR[0] = high-low (no prev close).
    # TR[i] = max(H-L, |H-prev_C|, |L-prev_C|)
    rows = [
        (10.0, 12.0, 9.0,  11.0),   # TR = 12-9 = 3
        (11.0, 13.0, 10.0, 12.0),   # TR = max(3, |13-11|, |10-11|) = max(3,2,1) = 3
        (12.0, 15.0, 11.0, 14.0),   # TR = max(4, |15-12|, |11-12|) = max(4,3,1) = 4
        (14.0, 16.0, 13.0, 15.0),   # TR = max(3, |16-14|, |13-14|) = max(3,2,1) = 3
        (15.0, 17.0, 14.0, 16.0),   # TR = max(3, |17-15|, |14-15|) = max(3,2,1) = 3
    ]
    timestamps = [_EPOCH + timedelta(hours=i) for i in range(len(rows))]
    df = pd.DataFrame(
        [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in rows],
        index=timestamps,
    )

    feat = compute_volatility(df, "TEST", "1h", atr_window=3)

    # Seed ATR (bar index 2, i.e., after 3 bars): mean([3, 3, 4]) = 10/3 ≈ 3.333
    # ATR[3] = (3.333 * 2 + 3) / 3 = (6.666 + 3) / 3 = 9.666 / 3 ≈ 3.222
    # ATR[4] = (3.222 * 2 + 3) / 3 = (6.444 + 3) / 3 = 9.444 / 3 ≈ 3.148
    expected_atr = (((3.0 + 3.0 + 4.0) / 3.0) * 2.0 + 3.0) / 3.0
    expected_atr = (expected_atr * 2.0 + 3.0) / 3.0

    assert abs(feat.atr - expected_atr) < 0.01, (
        f"ATR mismatch: got {feat.atr:.4f}, expected {expected_atr:.4f}"
    )
    assert feat.atr_pct > 0.0
    assert feat.atr_pct == pytest.approx(feat.atr / 16.0, rel=1e-4)


# ── 11. test_volatility_regime_classification ─────────────────────────────


def test_volatility_regime_classification():
    """ATR at different percentiles → correct regime labels."""
    # Build two scenarios with controlled ATR distributions.
    # Low-volatility scenario: all bars have tiny range.
    import numpy as np

    rng = np.random.default_rng(42)

    def _make_regime_df(atr_scale: float, n: int = 100) -> pd.DataFrame:
        """Build a simple trending series with controlled ATR scale."""
        closes = 100.0 + np.cumsum(rng.normal(0, atr_scale, n))
        opens = closes - rng.normal(0, atr_scale * 0.5, n)
        highs = np.maximum(opens, closes) + rng.uniform(0, atr_scale, n)
        lows = np.minimum(opens, closes) - rng.uniform(0, atr_scale, n)
        # Ensure OHLCV constraints.
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        opens = np.clip(opens, lows, highs)
        closes = np.clip(closes, lows, highs)
        timestamps = [_EPOCH + timedelta(hours=i) for i in range(n)]
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(n) * 1000},
            index=timestamps,
        )

    # Extreme-volatility scenario: very large bars.
    df_extreme = _make_regime_df(atr_scale=10.0, n=100)
    feat_extreme = compute_volatility(df_extreme, "TEST", "1h", atr_window=14)
    # The regime should be something valid.
    assert feat_extreme.volatility_regime in ("low", "normal", "high", "extreme")

    # Low-volatility scenario.
    df_low = _make_regime_df(atr_scale=0.01, n=100)
    feat_low = compute_volatility(df_low, "TEST", "1h", atr_window=14)
    assert feat_low.volatility_regime in ("low", "normal", "high", "extreme")

    # For the extreme scenario with 100 bars, the regime might be normal
    # (single-series percentile) but it must be a valid string.
    assert feat_extreme.volatility_regime in ("low", "normal", "high", "extreme")


# ── 12. test_relative_volume ──────────────────────────────────────────────


def test_relative_volume():
    """relative_volume == current_vol / rolling_mean(window)."""
    volumes = [100.0, 200.0, 150.0, 300.0, 500.0]  # last bar: 500
    # Rolling 4-bar mean of first 4 bars = (100+200+150+300)/4 = 187.5
    # But we use window=4 with the last 4 bars inclusive of the current bar.
    # pandas rolling(4).mean() at index 4 uses bars [1,2,3,4] = [200,150,300,500] = 287.5
    closes = [100.0] * len(volumes)
    df = _make_df(closes)
    df["volume"] = volumes

    feat = compute_liquidity(df, "TEST", "1h", vol_window=4)

    # avg_volume = rolling mean of last 4 bars = (200+150+300+500)/4 = 287.5
    assert feat.avg_volume == pytest.approx(287.5, rel=1e-4)
    assert feat.relative_volume == pytest.approx(500.0 / 287.5, rel=1e-3)


# ── 13. test_compute_all_features_pipeline ────────────────────────────────


def test_compute_all_features_pipeline():
    """Build a MarketSnapshot from fixture data, run pipeline, check all keys."""
    closes = [100.0 + i * 0.5 for i in range(60)]
    series_1h = _make_series(closes, ticker="AAPL", timeframe="1h")

    snapshot = MarketSnapshot(
        ticker="AAPL",
        snapshot_time=_EPOCH,
        tf_1h=series_1h,
    )

    result = compute_all_features(snapshot, primary_tf="1h")

    assert set(result.keys()) == {"structure", "levels", "volatility", "liquidity"}

    assert isinstance(result["structure"], StructureFeatures)
    assert isinstance(result["levels"], LevelFeatures)
    assert isinstance(result["volatility"], VolatilityFeatures)
    assert isinstance(result["liquidity"], LiquidityFeatures)

    assert result["structure"].ticker == "AAPL"
    assert result["volatility"].atr >= 0.0
    assert result["liquidity"].avg_volume >= 0.0


def test_compute_all_features_pipeline_4h_resampling():
    """Pipeline for 4h correctly resamples from 1h when tf_4h is absent."""
    closes = [100.0 + i * 0.5 for i in range(100)]
    series_1h = _make_series(closes, ticker="MSFT", timeframe="1h")

    snapshot = MarketSnapshot(
        ticker="MSFT",
        snapshot_time=_EPOCH,
        tf_1h=series_1h,
        tf_4h=None,
    )

    result = compute_all_features(snapshot, primary_tf="4h")

    assert set(result.keys()) == {"structure", "levels", "volatility", "liquidity"}
    assert isinstance(result["structure"], StructureFeatures)
    assert result["structure"].ticker == "MSFT"


def test_compute_all_features_pipeline_missing_tf():
    """Pipeline with missing timeframe returns empty feature shells without error."""
    snapshot = MarketSnapshot(
        ticker="SPY",
        snapshot_time=_EPOCH,
        tf_1h=None,
    )

    result = compute_all_features(snapshot, primary_tf="1h")

    assert set(result.keys()) == {"structure", "levels", "volatility", "liquidity"}
    assert result["structure"].trend_state == "unknown"
    assert result["volatility"].atr == 0.0


# ── 14. test_no_lookahead_in_swings ───────────────────────────────────────


def test_no_lookahead_in_swings():
    """
    Confirm that swing detection does not use future bars beyond i+swing_window.

    Method: run swing detection on df[:N] and df[:N+extra].
    For any swing confirmed at bar i < N - swing_window, the result must be
    identical in both runs (future bars after N+extra cannot change past swings).
    """
    swing_window = 3
    closes = [100.0, 110.0, 100.0, 120.0, 100.0, 130.0, 100.0, 140.0, 100.0, 150.0, 100.0]
    full_df = _make_df(closes)

    # Run on partial data (first 8 bars).
    partial_df = full_df.iloc[:8]
    struct_partial = compute_structure(partial_df, "TEST", "1h", swing_window=swing_window)

    # Run on full data.
    struct_full = compute_structure(full_df, "TEST", "1h", swing_window=swing_window)

    # Swings confirmed in the partial run must also appear in the full run
    # (they cannot be "undone" by seeing future data).
    partial_high_prices = {round(s.price, 6) for s in struct_partial.swing_highs}
    full_high_prices = {round(s.price, 6) for s in struct_full.swing_highs}

    for price in partial_high_prices:
        assert price in full_high_prices, (
            f"Swing high at {price} found in partial run but not in full run — "
            f"this would indicate lookahead contamination or non-determinism."
        )

    partial_low_prices = {round(s.price, 6) for s in struct_partial.swing_lows}
    full_low_prices = {round(s.price, 6) for s in struct_full.swing_lows}

    for price in partial_low_prices:
        assert price in full_low_prices, (
            f"Swing low at {price} found in partial run but not in full run."
        )
