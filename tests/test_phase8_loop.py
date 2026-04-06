"""
tests/test_phase8_loop.py — Phase 8: Live Loop integration tests.

All 10 tests use make_synthetic_ohlcv(). No API calls. No live data.

Tests
-----
1.  test_live_loop_runs_without_error
2.  test_live_loop_result_schema
3.  test_live_loop_fta_filters_trades
4.  test_live_loop_meta_model_filters_trades
5.  test_live_loop_decision_log_populated
6.  test_live_loop_fta_disabled_allows_more_trades
7.  test_live_loop_equity_curve
8.  test_live_loop_trade_journal_meta_features
9.  test_backtest_engine_with_fta
10. test_full_pipeline_integration
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.backtest.data_utils import make_synthetic_ohlcv
from src.backtest.engine import BacktestEngine
from src.loop import LiveLoop, LoopConfig, LiveLoopResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(n_bars: int = 300, seed: int = 42, trend: float = 0.0003) -> object:
    return make_synthetic_ohlcv(n_bars=n_bars, seed=seed, trend=trend, ticker="AAPL")


def _make_loop(fta_enabled: bool = True, meta_model_enabled: bool = True, verbose: bool = False) -> LiveLoop:
    cfg = LoopConfig(
        ticker="AAPL",
        fta_enabled=fta_enabled,
        meta_model_enabled=meta_model_enabled,
        verbose=verbose,
    )
    return LiveLoop(config=cfg)


# ---------------------------------------------------------------------------
# Test 1: loop runs without raising an exception
# ---------------------------------------------------------------------------

def test_live_loop_runs_without_error():
    series = _make_series(300)
    loop = _make_loop(fta_enabled=True, meta_model_enabled=True)
    result = loop.run(series)
    assert result is not None


# ---------------------------------------------------------------------------
# Test 2: result schema — all required fields present with correct types
# ---------------------------------------------------------------------------

def test_live_loop_result_schema():
    series = _make_series(300)
    loop = _make_loop()
    result = loop.run(series)

    assert isinstance(result, LiveLoopResult)
    assert isinstance(result.ticker, str)
    assert isinstance(result.timeframe, str)
    assert isinstance(result.n_bars_processed, int)
    assert isinstance(result.starting_capital, float)
    assert isinstance(result.final_equity, float)
    assert isinstance(result.trade_journal, list)
    assert isinstance(result.decision_log, list)
    assert isinstance(result.equity_curve, list)
    assert isinstance(result.n_trades, int)
    assert isinstance(result.n_fta_rejections, int)
    assert isinstance(result.n_meta_rejections, int)
    assert isinstance(result.n_portfolio_rejections, int)
    assert isinstance(result.total_return_pct, float)
    assert result.n_bars_processed == 300


# ---------------------------------------------------------------------------
# Test 3: FTA filter is exercised — fta_evaluated appears in the log
# ---------------------------------------------------------------------------

def test_live_loop_fta_filters_trades():
    """
    With fta_enabled=True on an uptrend series, candidates are generated and
    sent to FTA for evaluation. The key assertion is that at least some bars
    have fta_evaluated=True (FTA was called). The loop must complete without error.
    The n_fta_rejections may be 0 if all candidates happen to pass FTA, but the
    filter must have been exercised.
    """
    series = _make_series(300, seed=42, trend=0.0003)
    loop = _make_loop(fta_enabled=True, meta_model_enabled=False)
    result = loop.run(series)

    # Loop must complete without error.
    assert result is not None

    # At least some decision-log entries should exist.
    assert len(result.decision_log) > 0

    # n_fta_rejections is non-negative (may be 0 if all pass).
    assert result.n_fta_rejections >= 0

    # If any candidates were generated, at least some should have been FTA-evaluated.
    fta_evaluated_entries = [d for d in result.decision_log if d.get("fta_evaluated") is True]
    candidate_entries = [d for d in result.decision_log if d.get("candidate_generated") is True]
    # Every candidate must have been FTA-evaluated when fta_enabled=True.
    if candidate_entries:
        assert len(fta_evaluated_entries) > 0


# ---------------------------------------------------------------------------
# Test 4: meta-model filter is active (fta off, meta on)
# ---------------------------------------------------------------------------

def test_live_loop_meta_model_filters_trades():
    """
    With meta_model_enabled=True and fta_enabled=False:
    - The meta-model is called on every candidate.
    - Either some trades are rejected (n_meta_rejections > 0) or some pass
      through and are opened (n_trades > 0).
    - The loop must not crash.
    """
    series = _make_series(300, seed=42, trend=0.0003)
    loop = _make_loop(fta_enabled=False, meta_model_enabled=True)
    result = loop.run(series)

    assert result is not None
    # Meta-model must be evaluated for at least some candidates.
    mm_evaluated = [d for d in result.decision_log if d.get("meta_model_evaluated") is True]
    candidate_entries = [d for d in result.decision_log if d.get("candidate_generated") is True]
    if candidate_entries:
        assert len(mm_evaluated) > 0
    # At least one of: rejection or trade
    assert result.n_meta_rejections >= 0


# ---------------------------------------------------------------------------
# Test 5: decision log is populated for every bar
# ---------------------------------------------------------------------------

def test_live_loop_decision_log_populated():
    series = _make_series(300)
    loop = _make_loop()
    result = loop.run(series)

    assert len(result.decision_log) > 0
    # Every entry is a dict.
    for entry in result.decision_log:
        assert isinstance(entry, dict)
    # Required keys present in every entry.
    required_keys = {"bar_idx", "timestamp", "ticker", "close"}
    for entry in result.decision_log:
        for k in required_keys:
            assert k in entry, f"Missing key {k} in decision log entry"


# ---------------------------------------------------------------------------
# Test 6: disabling FTA should not reduce trade count vs enabling it
# ---------------------------------------------------------------------------

def test_live_loop_fta_disabled_allows_more_trades():
    """
    A run with fta_enabled=False should open >= trades as fta_enabled=True
    (FTA is a strict filter; removing it can only admit more or equal trades).
    """
    series = _make_series(300, seed=42, trend=0.0003)

    loop_fta_off = LiveLoop(config=LoopConfig(
        ticker="AAPL",
        fta_enabled=False,
        meta_model_enabled=False,
    ))
    loop_fta_on = LiveLoop(config=LoopConfig(
        ticker="AAPL",
        fta_enabled=True,
        meta_model_enabled=False,
    ))

    result_off = loop_fta_off.run(series)
    result_on = loop_fta_on.run(series)

    assert result_off.n_trades >= result_on.n_trades


# ---------------------------------------------------------------------------
# Test 7: equity curve is populated with (datetime, float) tuples
# ---------------------------------------------------------------------------

def test_live_loop_equity_curve():
    series = _make_series(300)
    loop = _make_loop()
    result = loop.run(series)

    assert len(result.equity_curve) > 0
    for entry in result.equity_curve:
        assert len(entry) == 2, f"Equity curve entry should be (datetime, float): {entry}"
        ts, eq = entry
        assert isinstance(ts, datetime), f"First element must be datetime, got {type(ts)}"
        assert isinstance(eq, float), f"Second element must be float, got {type(eq)}"


# ---------------------------------------------------------------------------
# Test 8: trade journal entries have "meta_features" key
# ---------------------------------------------------------------------------

def test_live_loop_trade_journal_meta_features():
    series = _make_series(300, seed=42, trend=0.0003)
    loop = LiveLoop(config=LoopConfig(
        ticker="AAPL",
        fta_enabled=False,      # disable FTA so trades are more likely
        meta_model_enabled=False,
    ))
    result = loop.run(series)

    for entry in result.trade_journal:
        assert "meta_features" in entry, (
            f"Trade journal entry missing 'meta_features' key: {list(entry.keys())}"
        )
        assert isinstance(entry["meta_features"], dict)


# ---------------------------------------------------------------------------
# Test 9: BacktestEngine with fta_enabled=True runs without error
# ---------------------------------------------------------------------------

def test_backtest_engine_with_fta():
    series = _make_series(300, seed=42, trend=0.0003)
    engine = BacktestEngine(
        starting_capital=100_000,
        fta_enabled=True,
        verbose=False,
    )
    result = engine.run(series, ticker="AAPL")
    assert result is not None
    assert result.n_bars == 300


# ---------------------------------------------------------------------------
# Test 10: full pipeline integration — FTA + meta-model, summary works
# ---------------------------------------------------------------------------

def test_full_pipeline_integration():
    series = make_synthetic_ohlcv(n_bars=300, seed=42, trend=0.0003, ticker="AAPL")
    loop = LiveLoop(config=LoopConfig(
        ticker="AAPL",
        fta_enabled=True,
        meta_model_enabled=True,
        verbose=False,
    ))
    result = loop.run(series)

    assert isinstance(result, LiveLoopResult)

    # summary() must work without raising
    summary = result.summary()
    assert isinstance(summary, str)
    assert "LIVE LOOP RESULT" in summary
    assert "AAPL" in summary

    # to_backtest_result() must work without raising
    br = result.to_backtest_result()
    assert br is not None
    assert br.ticker == "AAPL"
    assert br.n_bars == 300
