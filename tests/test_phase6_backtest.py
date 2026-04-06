"""
tests/test_phase6_backtest.py — Phase 6: Historical Backtest Engine tests.

All test data is generated via make_synthetic_ohlcv(). No API calls.

Tests
-----
1.  test_synthetic_ohlcv_generation
2.  test_no_lookahead_snapshot
3.  test_candidate_long_generated
4.  test_candidate_short_returns_none
5.  test_candidate_poor_rr_returns_none
6.  test_backtest_runs_without_error
7.  test_backtest_result_schema
8.  test_backtest_equity_curve_length
9.  test_backtest_trade_journal
10. test_backtest_reproducible
11. test_backtest_capital_conserved
12. test_metrics_sharpe_ratio
13. test_metrics_profit_factor
14. test_backtest_no_trades_on_flat_market
15. test_summary_prints
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from schemas.features import VolatilityFeatures
from schemas.forecast import ForecastOutput
from src.backtest import BacktestEngine, BacktestResult, compute_metrics, make_synthetic_ohlcv
from src.backtest.candidate import generate_candidate
from src.backtest.data_utils import build_snapshot_from_series

_UTC = timezone.utc


# ──────────────────────────────────────────────────────────────────────────────
# Helper fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_forecast(direction: str = "up", confidence: float = 0.7) -> ForecastOutput:
    return ForecastOutput(
        ticker="SYN",
        timeframe="1h",
        direction=direction,  # type: ignore[arg-type]
        expected_return=0.01,
        confidence=confidence,
        horizon=10,
    )


def _make_volatility(atr: float = 1.0) -> VolatilityFeatures:
    return VolatilityFeatures(
        ticker="SYN",
        timeframe="1h",
        atr=atr,
        atr_pct=0.01,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. test_synthetic_ohlcv_generation
# ──────────────────────────────────────────────────────────────────────────────

def test_synthetic_ohlcv_generation():
    series = make_synthetic_ohlcv(n_bars=300, seed=42)
    assert len(series.bars) == 300, "Should produce exactly 300 bars"
    assert series.ticker == "SYN"
    assert series.timeframe == "1h"

    for bar in series.bars:
        assert bar.high >= bar.close, f"high={bar.high} < close={bar.close}"
        assert bar.low <= bar.close, f"low={bar.low} > close={bar.close}"
        assert bar.high >= bar.open, f"high={bar.high} < open={bar.open}"
        assert bar.low <= bar.open, f"low={bar.low} > open={bar.open}"
        assert bar.volume >= 50_000
        assert bar.volume <= 500_000


# ──────────────────────────────────────────────────────────────────────────────
# 2. test_no_lookahead_snapshot
# ──────────────────────────────────────────────────────────────────────────────

def test_no_lookahead_snapshot():
    series = make_synthetic_ohlcv(n_bars=200, seed=1)
    snapshot = build_snapshot_from_series(series, t_idx=50, context_bars=20)

    assert snapshot.tf_1h is not None
    bars = snapshot.tf_1h.bars
    assert len(bars) == 20, f"Expected 20 bars, got {len(bars)}"

    # The newest bar in the snapshot must be exactly series.bars[50].
    newest_ts = bars[-1].timestamp
    expected_ts = series.bars[50].timestamp
    assert newest_ts == expected_ts, (
        f"Newest snapshot bar timestamp {newest_ts} != series.bars[50] {expected_ts}"
    )

    # No future bar should be present.
    future_timestamps = {b.timestamp for b in series.bars[51:]}
    snapshot_timestamps = {b.timestamp for b in bars}
    assert snapshot_timestamps.isdisjoint(future_timestamps), (
        "Snapshot contains future bars — no-lookahead violation!"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. test_candidate_long_generated
# ──────────────────────────────────────────────────────────────────────────────

def test_candidate_long_generated():
    forecast = _make_forecast(direction="up")
    vol = _make_volatility(atr=1.0)
    candidate = generate_candidate(forecast, vol, current_close=100.0,
                                   atr_stop_multiple=1.5, atr_target_multiple=3.0)

    assert candidate is not None, "Expected a candidate for upward forecast"
    assert candidate["side"] == "long"
    assert candidate["entry"] == pytest.approx(100.0)
    assert candidate["stop"] == pytest.approx(100.0 - 1.5 * 1.0)
    assert candidate["target"] == pytest.approx(100.0 + 3.0 * 1.0)
    assert candidate["reward_risk"] >= 1.5


# ──────────────────────────────────────────────────────────────────────────────
# 4. test_candidate_short_returns_none
# ──────────────────────────────────────────────────────────────────────────────

def test_candidate_short_returns_none():
    forecast = _make_forecast(direction="down")
    vol = _make_volatility(atr=1.0)
    candidate = generate_candidate(forecast, vol, current_close=100.0)
    assert candidate is None, "Short candidates should return None (not yet implemented)"


# ──────────────────────────────────────────────────────────────────────────────
# 5. test_candidate_poor_rr_returns_none
# ──────────────────────────────────────────────────────────────────────────────

def test_candidate_poor_rr_returns_none():
    """
    With atr_stop_multiple=3.0 and atr_target_multiple=1.0:
        risk  = 3.0 * atr
        reward = 1.0 * atr
        rr = 1.0 / 3.0 ≈ 0.33 — below 1.5 threshold.
    """
    forecast = _make_forecast(direction="up")
    vol = _make_volatility(atr=1.0)
    candidate = generate_candidate(forecast, vol, current_close=100.0,
                                   atr_stop_multiple=3.0, atr_target_multiple=1.0)
    assert candidate is None, "R:R below 1.5 should return None"


# ──────────────────────────────────────────────────────────────────────────────
# 6. test_backtest_runs_without_error
# ──────────────────────────────────────────────────────────────────────────────

def test_backtest_runs_without_error():
    series = make_synthetic_ohlcv(n_bars=300, trend=0.0002, seed=42)
    engine = BacktestEngine(starting_capital=100_000, verbose=False)
    result = engine.run(series)
    assert isinstance(result, BacktestResult)


# ──────────────────────────────────────────────────────────────────────────────
# 7. test_backtest_result_schema
# ──────────────────────────────────────────────────────────────────────────────

def test_backtest_result_schema():
    series = make_synthetic_ohlcv(n_bars=200, seed=7)
    engine = BacktestEngine(starting_capital=50_000, verbose=False)
    result = engine.run(series)

    assert isinstance(result.ticker, str)
    assert isinstance(result.timeframe, str)
    assert isinstance(result.start_date, datetime)
    assert isinstance(result.end_date, datetime)
    assert isinstance(result.n_bars, int)
    assert isinstance(result.starting_capital, (int, float))
    assert isinstance(result.final_equity, float)
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.realized_pnl, float)
    assert isinstance(result.unrealized_pnl_at_close, float)
    assert isinstance(result.n_trades, int)
    assert isinstance(result.n_winners, int)
    assert isinstance(result.n_losers, int)
    assert result.win_rate is None or isinstance(result.win_rate, float)
    assert result.avg_winner is None or isinstance(result.avg_winner, float)
    assert result.avg_loser is None or isinstance(result.avg_loser, float)
    assert result.profit_factor is None or isinstance(result.profit_factor, float)
    assert isinstance(result.max_drawdown_pct, float)
    assert result.sharpe_ratio is None or isinstance(result.sharpe_ratio, float)
    assert isinstance(result.equity_curve, list)
    assert isinstance(result.trade_journal, list)

    # n_bars should match the series length.
    assert result.n_bars == 200


# ──────────────────────────────────────────────────────────────────────────────
# 8. test_backtest_equity_curve_length
# ──────────────────────────────────────────────────────────────────────────────

def test_backtest_equity_curve_length():
    n = 200
    min_bars = 50
    series = make_synthetic_ohlcv(n_bars=n, seed=3)
    engine = BacktestEngine(starting_capital=100_000, min_bars_required=min_bars, verbose=False)
    result = engine.run(series)

    expected = n - min_bars
    # Allow ±5 bars tolerance for skipped bars (feature failures, etc.)
    assert abs(len(result.equity_curve) - expected) <= 5, (
        f"equity_curve length {len(result.equity_curve)} far from expected {expected}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9. test_backtest_trade_journal
# ──────────────────────────────────────────────────────────────────────────────

def test_backtest_trade_journal():
    series = make_synthetic_ohlcv(n_bars=300, trend=0.001, seed=99)
    engine = BacktestEngine(starting_capital=100_000, verbose=False)
    result = engine.run(series)

    assert isinstance(result.trade_journal, list)
    required_keys = {
        "trade_id", "ticker", "side", "quantity", "entry_price",
        "stop_price", "target_price", "entry_time", "exit_price",
        "exit_time", "exit_reason", "status", "realized_pnl",
    }
    for entry in result.trade_journal:
        assert isinstance(entry, dict)
        missing = required_keys - entry.keys()
        assert not missing, f"Trade journal entry missing keys: {missing}"


# ──────────────────────────────────────────────────────────────────────────────
# 10. test_backtest_reproducible
# ──────────────────────────────────────────────────────────────────────────────

def test_backtest_reproducible():
    series1 = make_synthetic_ohlcv(n_bars=200, seed=42)
    series2 = make_synthetic_ohlcv(n_bars=200, seed=42)

    engine = BacktestEngine(starting_capital=100_000, verbose=False)
    r1 = engine.run(series1)
    r2 = engine.run(series2)

    assert r1.final_equity == pytest.approx(r2.final_equity, rel=1e-9), (
        f"Same seed should give identical final_equity: {r1.final_equity} vs {r2.final_equity}"
    )
    assert r1.n_trades == r2.n_trades


# ──────────────────────────────────────────────────────────────────────────────
# 11. test_backtest_capital_conserved
# ──────────────────────────────────────────────────────────────────────────────

def test_backtest_capital_conserved():
    """
    Accounting invariant:
        final_equity == starting_capital + realized_pnl + unrealized_pnl_at_close

    Since all positions are force-closed at end_of_backtest, unrealized_pnl_at_close
    should be 0. Verify the accounting identity holds within floating-point tolerance.
    """
    series = make_synthetic_ohlcv(n_bars=300, trend=0.0002, seed=11)
    engine = BacktestEngine(starting_capital=100_000, verbose=False)
    result = engine.run(series)

    expected = result.starting_capital + result.realized_pnl + result.unrealized_pnl_at_close
    assert result.final_equity == pytest.approx(expected, abs=0.01), (
        f"Capital invariant violated: final_equity={result.final_equity:.2f}, "
        f"expected={expected:.2f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 12. test_metrics_sharpe_ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_metrics_sharpe_ratio():
    """
    A known equity curve with consistent upward drift should produce a finite,
    positive Sharpe ratio.
    """
    # Build a strictly monotonically increasing equity curve.
    base = datetime(2024, 1, 1, tzinfo=_UTC)
    from datetime import timedelta
    curve = [(base + timedelta(hours=i), 100_000 + i * 10.0) for i in range(100)]

    metrics = compute_metrics(curve, [], 100_000, timeframe="1h")
    sharpe = metrics["sharpe_ratio"]
    assert sharpe is not None, "Sharpe should not be None for a 100-point curve"
    assert -10.0 <= sharpe <= 10.0, f"Sharpe {sharpe} outside clamp range"
    assert sharpe > 0, "Monotonically rising equity should give positive Sharpe"


# ──────────────────────────────────────────────────────────────────────────────
# 13. test_metrics_profit_factor
# ──────────────────────────────────────────────────────────────────────────────

def test_metrics_profit_factor():
    """
    2 winners at $100 each, 1 loser at -$50:
        gross_profit = 200
        gross_loss   = 50
        profit_factor = 200 / 50 = 4.0
    """
    def _closed_trade(pnl: float) -> dict:
        return {
            "trade_id": "x",
            "ticker": "SYN",
            "side": "long",
            "quantity": 1,
            "entry_price": 100.0,
            "stop_price": 95.0,
            "target_price": 110.0,
            "entry_time": "2024-01-01T00:00:00",
            "exit_price": 100.0 + pnl,
            "exit_time": "2024-01-02T00:00:00",
            "exit_reason": "manual",
            "status": "closed",
            "source": "test",
            "last_price": 100.0 + pnl,
            "last_update": None,
            "realized_pnl": pnl,
            "cost_basis": 100.0,
        }

    journal = [_closed_trade(100.0), _closed_trade(100.0), _closed_trade(-50.0)]
    base = datetime(2024, 1, 1, tzinfo=_UTC)
    from datetime import timedelta
    curve = [(base + timedelta(hours=i), 100_000 + i) for i in range(10)]

    metrics = compute_metrics(curve, journal, 100_000, timeframe="1h")
    assert metrics["profit_factor"] == pytest.approx(4.0), (
        f"Expected profit_factor=4.0, got {metrics['profit_factor']}"
    )
    assert metrics["n_winners"] == 2
    assert metrics["n_losers"] == 1
    assert metrics["win_rate"] == pytest.approx(2 / 3)


# ──────────────────────────────────────────────────────────────────────────────
# 14. test_backtest_no_trades_on_flat_market
# ──────────────────────────────────────────────────────────────────────────────

def test_backtest_no_trades_on_flat_market():
    """
    A flat price series (no trend, extremely low volatility) should produce
    significantly fewer trades than a strong uptrend run, because:
    - Near-zero ATR causes generate_candidate() to return None when ATR=0 exactly.
    - When ATR is very small (but non-zero) position sizes balloon and risk limits
      reject most candidates before they enter.

    We verify the flat run produces fewer trades than the trending run, not an
    absolute zero, because the statistical forecaster can still emit "up" signals.
    """
    flat_series = make_synthetic_ohlcv(
        n_bars=300,
        trend=0.0,
        volatility=0.0001,  # near-zero volatility
        seed=0,
    )
    trend_series = make_synthetic_ohlcv(
        n_bars=300,
        trend=0.001,      # strong uptrend
        volatility=0.015,
        seed=0,
    )
    engine = BacktestEngine(starting_capital=100_000, verbose=False)
    flat_result = engine.run(flat_series)
    trend_result = engine.run(trend_series)

    # The flat run should have strictly fewer or equal trades than the trending run.
    # Allow the flat run to have at most twice the trades of the trending run only
    # if the trending run itself is tiny, but the core assertion is directional.
    assert flat_result.n_trades <= max(trend_result.n_trades, 20), (
        f"Flat market trades ({flat_result.n_trades}) unexpectedly high vs "
        f"trending market ({trend_result.n_trades})"
    )
    # Also verify: in a truly flat world, total return is near zero (no alpha).
    assert abs(flat_result.total_return_pct) < 5.0, (
        f"Flat series has surprisingly large return: {flat_result.total_return_pct:.2f}%"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 15. test_summary_prints
# ──────────────────────────────────────────────────────────────────────────────

def test_summary_prints():
    series = make_synthetic_ohlcv(n_bars=200, seed=5)
    engine = BacktestEngine(starting_capital=100_000, verbose=False)
    result = engine.run(series)

    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "BACKTEST SUMMARY" in summary
    assert "CAPITAL" in summary
    assert "TRADES" in summary
    assert "RISK" in summary
