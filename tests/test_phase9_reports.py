"""
tests/test_phase9_reports.py — Phase 9 reporting layer tests.

All 15 tests use a shared BacktestResult produced from a 200-bar synthetic
OHLCV series. The fixture is session-scoped so the backtest runs only once.

Tests
-----
1.  test_portfolio_report_keys
2.  test_portfolio_return_consistent
3.  test_per_exit_reason_pnl
4.  test_decision_report_keys
5.  test_acceptance_rate_range
6.  test_trade_diagnostics_count
7.  test_trade_diagnostic_outcome
8.  test_holding_hours_positive
9.  test_model_diagnostics_keys
10. test_threshold_sweep_confidence
11. test_threshold_sweep_rr
12. test_sweep_subset_monotone
13. test_charts_written
14. test_full_report_structure
15. test_report_json_serializable
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.backtest.data_utils import make_synthetic_ohlcv
from src.backtest.engine import BacktestEngine
from src.backtest.result import BacktestResult
from src.reports import (
    generate_charts,
    generate_decision_report,
    generate_full_report,
    generate_model_diagnostics,
    generate_portfolio_report,
    generate_trade_diagnostics,
    sweep_thresholds,
)
from src.reports.runner import _json_default

# ---------------------------------------------------------------------------
# Shared fixture: one backtest result reused across all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def backtest_result() -> BacktestResult:
    series = make_synthetic_ohlcv(n_bars=200, seed=7)
    engine = BacktestEngine(starting_capital=10_000.0, verbose=False)
    return engine.run(series)


# ---------------------------------------------------------------------------
# 1. test_portfolio_report_keys
# ---------------------------------------------------------------------------

def test_portfolio_report_keys(backtest_result):
    report = generate_portfolio_report(backtest_result)
    expected_keys = {
        "ticker", "timeframe", "start_date", "end_date", "n_bars",
        "starting_capital", "final_equity", "total_return_pct",
        "realized_pnl", "unrealized_pnl_at_close",
        "n_trades", "n_winners", "n_losers", "win_rate",
        "avg_winner", "avg_loser", "profit_factor",
        "max_drawdown_pct", "sharpe_ratio",
        "avg_holding_bars", "per_ticker_pnl",
        "per_exit_reason_pnl", "daily_pnl",
    }
    assert expected_keys.issubset(set(report.keys())), (
        f"Missing keys: {expected_keys - set(report.keys())}"
    )


# ---------------------------------------------------------------------------
# 2. test_portfolio_return_consistent
# ---------------------------------------------------------------------------

def test_portfolio_return_consistent(backtest_result):
    report = generate_portfolio_report(backtest_result)
    expected = (report["final_equity"] / report["starting_capital"] - 1) * 100.0
    assert abs(report["total_return_pct"] - expected) < 1e-6, (
        f"total_return_pct mismatch: got {report['total_return_pct']}, "
        f"expected {expected}"
    )


# ---------------------------------------------------------------------------
# 3. test_per_exit_reason_pnl
# ---------------------------------------------------------------------------

def test_per_exit_reason_pnl(backtest_result):
    report = generate_portfolio_report(backtest_result)
    per_reason_total = sum(report["per_exit_reason_pnl"].values())
    realized = report["realized_pnl"]
    assert abs(per_reason_total - realized) < 1e-4, (
        f"per_exit_reason_pnl sum {per_reason_total:.4f} != "
        f"realized_pnl {realized:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. test_decision_report_keys
# ---------------------------------------------------------------------------

def test_decision_report_keys(backtest_result):
    report = generate_decision_report(backtest_result)
    expected_keys = {
        "total_candidates", "accepted_trades",
        "fta_rejected", "meta_model_rejected", "portfolio_rejected",
        "acceptance_rate",
        "forecast_confidence_mean", "forecast_confidence_std",
        "meta_model_prob_mean",
        "rejection_breakdown",
    }
    assert expected_keys.issubset(set(report.keys())), (
        f"Missing keys: {expected_keys - set(report.keys())}"
    )


# ---------------------------------------------------------------------------
# 5. test_acceptance_rate_range
# ---------------------------------------------------------------------------

def test_acceptance_rate_range(backtest_result):
    report = generate_decision_report(backtest_result)
    rate = report["acceptance_rate"]
    assert 0.0 <= rate <= 1.0, f"acceptance_rate out of range: {rate}"


# ---------------------------------------------------------------------------
# 6. test_trade_diagnostics_count
# ---------------------------------------------------------------------------

def test_trade_diagnostics_count(backtest_result):
    diags = generate_trade_diagnostics(backtest_result)
    assert len(diags) == backtest_result.n_trades, (
        f"Expected {backtest_result.n_trades} diagnostics, got {len(diags)}"
    )


# ---------------------------------------------------------------------------
# 7. test_trade_diagnostic_outcome
# ---------------------------------------------------------------------------

def test_trade_diagnostic_outcome(backtest_result):
    diags = generate_trade_diagnostics(backtest_result)
    valid_outcomes = {"win", "loss", "breakeven"}
    for d in diags:
        assert d["outcome"] in valid_outcomes, (
            f"Invalid outcome '{d['outcome']}' for trade {d['trade_id']}"
        )


# ---------------------------------------------------------------------------
# 8. test_holding_hours_positive
# ---------------------------------------------------------------------------

def test_holding_hours_positive(backtest_result):
    diags = generate_trade_diagnostics(backtest_result)
    for d in diags:
        assert d["holding_hours"] >= 0.0, (
            f"Negative holding_hours {d['holding_hours']} for trade {d['trade_id']}"
        )


# ---------------------------------------------------------------------------
# 9. test_model_diagnostics_keys
# ---------------------------------------------------------------------------

def test_model_diagnostics_keys(backtest_result):
    report = generate_model_diagnostics(backtest_result)
    expected_keys = {
        "feature_importance",
        "prediction_distribution",
        "calibration_summary",
        "threshold_sensitivity",
    }
    assert expected_keys.issubset(set(report.keys())), (
        f"Missing keys: {expected_keys - set(report.keys())}"
    )
    # prediction_distribution sub-keys
    pd = report["prediction_distribution"]
    for k in ("mean", "std", "min", "max", "pct_above_0.6"):
        assert k in pd, f"prediction_distribution missing key: {k}"
    # calibration_summary sub-keys
    cs = report["calibration_summary"]
    for k in ("total_trades", "labeled_fraction"):
        assert k in cs, f"calibration_summary missing key: {k}"


# ---------------------------------------------------------------------------
# 10. test_threshold_sweep_confidence
# ---------------------------------------------------------------------------

def test_threshold_sweep_confidence(backtest_result):
    thresholds = [0.0, 0.3, 0.5, 0.7]
    result = sweep_thresholds(backtest_result, confidence_thresholds=thresholds)
    assert "confidence_sweep" in result
    assert len(result["confidence_sweep"]) == len(thresholds), (
        f"Expected {len(thresholds)} entries, got {len(result['confidence_sweep'])}"
    )
    for row in result["confidence_sweep"]:
        for key in ("threshold", "n_trades", "win_rate", "total_pnl", "max_drawdown_pct"):
            assert key in row, f"Missing key '{key}' in confidence_sweep row"


# ---------------------------------------------------------------------------
# 11. test_threshold_sweep_rr
# ---------------------------------------------------------------------------

def test_threshold_sweep_rr(backtest_result):
    rr_mins = [1.0, 2.0, 3.0]
    result = sweep_thresholds(backtest_result, rr_minimums=rr_mins)
    assert "rr_sweep" in result
    assert len(result["rr_sweep"]) == len(rr_mins), (
        f"Expected {len(rr_mins)} entries, got {len(result['rr_sweep'])}"
    )
    for row in result["rr_sweep"]:
        for key in ("rr_minimum", "n_trades", "win_rate", "total_pnl", "max_drawdown_pct"):
            assert key in row, f"Missing key '{key}' in rr_sweep row"


# ---------------------------------------------------------------------------
# 12. test_sweep_subset_monotone
# ---------------------------------------------------------------------------

def test_sweep_subset_monotone(backtest_result):
    """Higher confidence threshold must not increase trade count."""
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    result = sweep_thresholds(backtest_result, confidence_thresholds=thresholds)
    counts = [row["n_trades"] for row in result["confidence_sweep"]]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1], (
            f"Monotonicity violated: threshold {thresholds[i]} gave {counts[i]} trades "
            f"but threshold {thresholds[i+1]} gave {counts[i+1]} trades"
        )


# ---------------------------------------------------------------------------
# 13. test_charts_written
# ---------------------------------------------------------------------------

def _matplotlib_available() -> bool:
    """Return True if matplotlib can be imported successfully."""
    try:
        import matplotlib
        import matplotlib.pyplot  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _matplotlib_available(), reason="matplotlib not available or broken")
def test_charts_written(backtest_result):
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = generate_charts(backtest_result, tmp_dir)
        written_names = {Path(p).name for p in paths}

        # Both equity_curve.png and drawdown_curve.png must be written
        # (they depend only on the equity_curve, which is always populated).
        assert "equity_curve.png" in written_names, (
            f"equity_curve.png not found. Written: {written_names}"
        )
        assert "drawdown_curve.png" in written_names, (
            f"drawdown_curve.png not found. Written: {written_names}"
        )

        # All returned paths must actually exist on disk.
        for p in paths:
            assert Path(p).exists(), f"Chart file not found on disk: {p}"


# ---------------------------------------------------------------------------
# 14. test_full_report_structure
# ---------------------------------------------------------------------------

def test_full_report_structure(backtest_result):
    report = generate_full_report(backtest_result, output_dir=None)
    expected_top_keys = {
        "portfolio", "decisions", "model_diagnostics",
        "threshold_sensitivity", "trade_diagnostics", "charts",
    }
    assert expected_top_keys.issubset(set(report.keys())), (
        f"Missing top-level keys: {expected_top_keys - set(report.keys())}"
    )
    # charts should be [] when output_dir is None
    assert report["charts"] == [], (
        f"Expected empty charts list when output_dir=None, got {report['charts']}"
    )


# ---------------------------------------------------------------------------
# 15. test_report_json_serializable
# ---------------------------------------------------------------------------

def test_report_json_serializable(backtest_result):
    report = generate_full_report(backtest_result, output_dir=None)
    # Should not raise; datetime objects are handled by _json_default.
    serialised = json.dumps(report, default=_json_default)
    assert len(serialised) > 0
    # Verify it round-trips as valid JSON.
    parsed = json.loads(serialised)
    assert isinstance(parsed, dict)
    assert "portfolio" in parsed
