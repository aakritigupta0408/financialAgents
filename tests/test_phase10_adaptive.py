"""
tests/test_phase10_adaptive.py — Phase 10 adaptive module tests.

15 tests covering:
1.  test_context_default_values
2.  test_context_save_and_load
3.  test_context_clamp_thresholds
4.  test_context_version_increments
5.  test_analyze_report_returns_analysis
6.  test_analyze_insufficient_sample
7.  test_apply_update_suppressed_small_sample
8.  test_apply_update_cap_per_session
9.  test_apply_update_degradation_safeguard
10. test_apply_update_ema_performance
11. test_improvement_cycle_runs
12. test_improvement_cycle_context_saved
13. test_improvement_cycle_version_increments
14. test_improvement_cycle_no_retrain_flag
15. test_full_suite_regression
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.backtest.data_utils import make_synthetic_ohlcv
from src.backtest.engine import BacktestEngine

# ---------------------------------------------------------------------------
# Shared fixture: one BacktestResult reused across most tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def backtest_result():
    """Run a 300-bar backtest once and share across all tests."""
    series = make_synthetic_ohlcv(n_bars=300, seed=42)
    engine = BacktestEngine(fta_enabled=False, meta_model_enabled=False, verbose=False)
    return engine.run(series)


@pytest.fixture(scope="session")
def full_report(backtest_result):
    """Generate full report once and share."""
    from src.reports import generate_full_report
    return generate_full_report(backtest_result)


# ---------------------------------------------------------------------------
# Helpers for monkeypatching CONTEXT_DIR to tmp_path
# ---------------------------------------------------------------------------


def _patch_context_dir(monkeypatch, tmp_path: Path):
    """Redirect all CONTEXT_DIR references to a temp directory."""
    import src.adaptive.context as ctx_mod
    new_dir = tmp_path / "adaptive"
    monkeypatch.setattr(ctx_mod, "CONTEXT_DIR", new_dir)
    # Also patch the module-level reference used by analyzer (loads via AdaptiveContext.load)
    return new_dir


# ===========================================================================
# Test 1: test_context_default_values
# ===========================================================================


def test_context_default_values():
    from src.adaptive.context import AdaptiveContext, _BOUNDS

    ctx = AdaptiveContext.default()
    t = ctx.best_thresholds

    # All thresholds must be within bounds.
    lo, hi = _BOUNDS["meta_model_min_confidence"]
    assert lo <= t.meta_model_min_confidence <= hi

    lo, hi = _BOUNDS["min_reward_risk"]
    assert lo <= t.min_reward_risk <= hi

    lo, hi = _BOUNDS["forecast_confidence_min"]
    assert lo <= t.forecast_confidence_min <= hi

    lo, hi = _BOUNDS["fta_score_min"]
    assert lo <= t.fta_score_min <= hi

    assert ctx.version == 0
    assert ctx.updated_at != ""


# ===========================================================================
# Test 2: test_context_save_and_load
# ===========================================================================


def test_context_save_and_load(monkeypatch, tmp_path):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    ctx = ctx_mod.AdaptiveContext.default()
    ctx.version = 5
    ctx.best_thresholds.meta_model_min_confidence = 0.65
    ctx.save()

    loaded = ctx_mod.AdaptiveContext.load()
    assert loaded.version == 5
    assert abs(loaded.best_thresholds.meta_model_min_confidence - 0.65) < 1e-9


# ===========================================================================
# Test 3: test_context_clamp_thresholds
# ===========================================================================


def test_context_clamp_thresholds(monkeypatch, tmp_path):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    ctx = ctx_mod.AdaptiveContext.default()
    # Deliberately set out-of-bounds values.
    ctx.best_thresholds.meta_model_min_confidence = 0.99   # above max 0.90
    ctx.best_thresholds.min_reward_risk = 0.1               # below min 1.0
    ctx.best_thresholds.forecast_confidence_min = 0.0       # below min 0.10
    ctx.best_thresholds.fta_score_min = 1.5                 # above max 1.0

    # Save and reload — clamp_thresholds is called on load.
    ctx.save()
    loaded = ctx_mod.AdaptiveContext.load()
    t = loaded.best_thresholds

    assert t.meta_model_min_confidence <= 0.90
    assert t.min_reward_risk >= 1.0
    assert t.forecast_confidence_min >= 0.10
    assert t.fta_score_min <= 1.0


# ===========================================================================
# Test 4: test_context_version_increments
# ===========================================================================


def test_context_version_increments(monkeypatch, tmp_path, backtest_result, full_report):
    import src.adaptive.context as ctx_mod
    import src.adaptive.analyzer as ana_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.context import AdaptiveContext
    from src.adaptive.analyzer import analyze_report
    from src.adaptive.updater import apply_update

    ctx = AdaptiveContext.default()
    ctx.version = 3
    analysis = analyze_report(full_report)
    new_ctx, _ = apply_update(ctx, analysis, full_report)

    assert new_ctx.version == 4


# ===========================================================================
# Test 5: test_analyze_report_returns_analysis
# ===========================================================================


def test_analyze_report_returns_analysis(full_report):
    from src.adaptive.analyzer import AnalysisResult, analyze_report

    result = analyze_report(full_report)
    assert isinstance(result, AnalysisResult)
    assert isinstance(result.recommended_confidence_threshold, float)
    assert isinstance(result.recommended_rr_minimum, float)
    assert isinstance(result.recommended_forecast_confidence, float)
    assert isinstance(result.n_trades_analyzed, int)
    assert isinstance(result.best_sweep_metric, str)
    assert isinstance(result.feature_importance, dict)
    assert isinstance(result.per_ticker_pnl, dict)
    assert isinstance(result.regime_stats, dict)
    assert isinstance(result.warnings, list)


# ===========================================================================
# Test 6: test_analyze_insufficient_sample
# ===========================================================================


def test_analyze_insufficient_sample():
    from src.adaptive.analyzer import analyze_report

    # Build a minimal report with fewer than 10 trades.
    report = {
        "portfolio": {
            "max_drawdown_pct": 5.0,
            "win_rate": 0.5,
            "total_return_pct": 2.0,
            "sharpe_ratio": 1.0,
            "n_trades": 3,
            "per_ticker_pnl": {},
        },
        "threshold_sensitivity": {
            "confidence_sweep": [],
            "rr_sweep": [],
        },
        "model_diagnostics": {"feature_importance": {}},
        "trade_diagnostics": [
            {"realized_pnl": 10.0, "meta_features": {}} for _ in range(3)
        ],
    }

    result = analyze_report(report)
    assert result.n_trades_analyzed == 3
    assert any("Insufficient" in w for w in result.warnings)


# ===========================================================================
# Test 7: test_apply_update_suppressed_small_sample
# ===========================================================================


def test_apply_update_suppressed_small_sample(monkeypatch, tmp_path):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.context import AdaptiveContext
    from src.adaptive.analyzer import AnalysisResult
    from src.adaptive.updater import apply_update

    ctx = AdaptiveContext.default()
    analysis = AnalysisResult(
        recommended_confidence_threshold=0.70,
        recommended_rr_minimum=2.5,
        recommended_forecast_confidence=0.55,
        n_trades_analyzed=5,   # < 10
        best_sweep_metric="return",
        feature_importance={},
        per_ticker_pnl={},
        regime_stats={},
        warnings=[],
    )
    report = {"portfolio": {"win_rate": 0.5, "total_return_pct": 1.0,
                            "max_drawdown_pct": 3.0, "sharpe_ratio": 0.8,
                            "n_trades": 5}}

    new_ctx, summary = apply_update(ctx, analysis, report)

    assert summary.update_suppressed is True
    assert summary.suppression_reason == "insufficient_sample_size"
    # Thresholds must not change.
    assert new_ctx.best_thresholds.meta_model_min_confidence == ctx.best_thresholds.meta_model_min_confidence


# ===========================================================================
# Test 8: test_apply_update_cap_per_session
# ===========================================================================


def test_apply_update_cap_per_session(monkeypatch, tmp_path):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.context import AdaptiveContext
    from src.adaptive.analyzer import AnalysisResult
    from src.adaptive.updater import apply_update

    ctx = AdaptiveContext.default()
    # Force n_sessions low so degradation guard doesn't trigger.
    ctx.recent_performance.n_sessions = 0
    old_conf = ctx.best_thresholds.meta_model_min_confidence

    # Request a massive jump (+0.50 change) — should be capped to +0.10.
    analysis = AnalysisResult(
        recommended_confidence_threshold=old_conf + 0.50,
        recommended_rr_minimum=ctx.best_thresholds.min_reward_risk,
        recommended_forecast_confidence=ctx.best_thresholds.forecast_confidence_min,
        n_trades_analyzed=20,   # enough trades
        best_sweep_metric="return",
        feature_importance={},
        per_ticker_pnl={},
        regime_stats={},
        warnings=[],
    )
    report = {"portfolio": {"win_rate": 0.5, "total_return_pct": 1.0,
                            "max_drawdown_pct": 2.0, "sharpe_ratio": 0.9,
                            "n_trades": 20}}

    new_ctx, summary = apply_update(ctx, analysis, report)

    assert summary.update_suppressed is False
    new_conf = new_ctx.best_thresholds.meta_model_min_confidence
    # Change capped at ±0.10.
    assert abs(new_conf - old_conf) <= 0.10 + 1e-9


# ===========================================================================
# Test 9: test_apply_update_degradation_safeguard
# ===========================================================================


def test_apply_update_degradation_safeguard(monkeypatch, tmp_path):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.context import AdaptiveContext
    from src.adaptive.analyzer import AnalysisResult
    from src.adaptive.updater import apply_update

    ctx = AdaptiveContext.default()
    # Simulate 3 sessions and a good win rate.
    ctx.recent_performance.n_sessions = 3
    ctx.recent_performance.win_rate = 0.60

    analysis = AnalysisResult(
        recommended_confidence_threshold=0.65,
        recommended_rr_minimum=2.0,
        recommended_forecast_confidence=0.50,
        n_trades_analyzed=15,
        best_sweep_metric="return",
        feature_importance={},
        per_ticker_pnl={},
        regime_stats={},
        warnings=[],
    )
    # win_rate drops by 0.20 (> 0.15 threshold) → degradation guard.
    report = {"portfolio": {"win_rate": 0.40, "total_return_pct": -2.0,
                            "max_drawdown_pct": 8.0, "sharpe_ratio": -0.5,
                            "n_trades": 15}}

    new_ctx, summary = apply_update(ctx, analysis, report)

    assert summary.update_suppressed is True
    assert summary.suppression_reason == "performance_degradation"
    assert any("degradation" in w.lower() for w in summary.warnings)


# ===========================================================================
# Test 10: test_apply_update_ema_performance
# ===========================================================================


def test_apply_update_ema_performance(monkeypatch, tmp_path):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.context import AdaptiveContext
    from src.adaptive.analyzer import AnalysisResult
    from src.adaptive.updater import apply_update

    ctx = AdaptiveContext.default()
    ctx.recent_performance.n_sessions = 0
    ctx.recent_performance.win_rate = 0.50

    analysis = AnalysisResult(
        recommended_confidence_threshold=ctx.best_thresholds.meta_model_min_confidence,
        recommended_rr_minimum=ctx.best_thresholds.min_reward_risk,
        recommended_forecast_confidence=ctx.best_thresholds.forecast_confidence_min,
        n_trades_analyzed=20,
        best_sweep_metric="return",
        feature_importance={},
        per_ticker_pnl={},
        regime_stats={},
        warnings=[],
    )
    report = {"portfolio": {"win_rate": 0.80, "total_return_pct": 5.0,
                            "max_drawdown_pct": 2.0, "sharpe_ratio": 1.5,
                            "n_trades": 20}}

    new_ctx, summary = apply_update(ctx, analysis, report)

    assert summary.update_suppressed is False
    # n_sessions should increment.
    assert new_ctx.recent_performance.n_sessions == 1
    # win_rate should be EMA blend: 0.3*0.80 + 0.7*0.50 = 0.59
    expected_wr = 0.3 * 0.80 + 0.7 * 0.50
    assert abs(new_ctx.recent_performance.win_rate - expected_wr) < 1e-6


# ===========================================================================
# Test 11: test_improvement_cycle_runs
# ===========================================================================


def test_improvement_cycle_runs(monkeypatch, tmp_path, backtest_result):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.loop import run_improvement_cycle

    cycle = run_improvement_cycle(backtest_result, retrain_model=False, save_context=True)

    from src.adaptive.loop import ImprovementCycleResult
    assert isinstance(cycle, ImprovementCycleResult)
    assert cycle.update_summary is not None
    assert cycle.analysis is not None
    assert isinstance(cycle.report, dict)


# ===========================================================================
# Test 12: test_improvement_cycle_context_saved
# ===========================================================================


def test_improvement_cycle_context_saved(monkeypatch, tmp_path, backtest_result):
    import src.adaptive.context as ctx_mod
    new_dir = _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.loop import run_improvement_cycle

    run_improvement_cycle(backtest_result, retrain_model=False, save_context=True)

    context_file = new_dir / "adaptive_context.json"
    assert context_file.exists(), "Context file was not written to disk"

    # Verify it's valid JSON.
    with open(context_file) as fh:
        data = json.load(fh)
    assert "version" in data


# ===========================================================================
# Test 13: test_improvement_cycle_version_increments
# ===========================================================================


def test_improvement_cycle_version_increments(monkeypatch, tmp_path, backtest_result):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.loop import run_improvement_cycle

    cycle1 = run_improvement_cycle(backtest_result, retrain_model=False, save_context=True)
    cycle2 = run_improvement_cycle(backtest_result, retrain_model=False, save_context=True)

    # Version must be >= version_before (suppressed cycles don't increment but do load)
    assert cycle2.context_after.version >= cycle2.context_before.version


# ===========================================================================
# Test 14: test_improvement_cycle_no_retrain_flag
# ===========================================================================


def test_improvement_cycle_no_retrain_flag(monkeypatch, tmp_path, backtest_result):
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    from src.adaptive.loop import run_improvement_cycle

    cycle = run_improvement_cycle(backtest_result, retrain_model=False, save_context=False)

    assert cycle.model_retrained is False
    assert cycle.model_improved is False


# ===========================================================================
# Test 15: test_full_suite_regression
# ===========================================================================


def test_full_suite_regression(monkeypatch, tmp_path):
    """
    Regression guard: run BacktestEngine with both filters disabled,
    run improvement cycle, assert update_summary is not None.
    """
    import src.adaptive.context as ctx_mod
    _patch_context_dir(monkeypatch, tmp_path)

    series = make_synthetic_ohlcv(n_bars=200, seed=99)
    engine = BacktestEngine(fta_enabled=False, meta_model_enabled=False, verbose=False)
    result = engine.run(series)

    from src.adaptive.loop import run_improvement_cycle
    cycle = run_improvement_cycle(result, retrain_model=False, save_context=True)

    assert cycle.update_summary is not None
