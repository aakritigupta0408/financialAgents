"""
tests/test_phase11_validation.py — Phase 11 validation tests.

All 15 tests use synthetic data only (make_synthetic_ohlcv).
n_bars kept small (200–300) so the full suite runs quickly.
"""

from __future__ import annotations

import pytest

from src.backtest.data_utils import make_synthetic_ohlcv
from src.validation import (
    AblationRunner,
    AblationResult,
    BenchmarkRunner,
    BenchmarkResult,
    DataSplitResult,
    DataSplitValidator,
    PassFailCriteria,
    RobustnessResult,
    RobustnessRunner,
    ValidationSummary,
    generate_validation_summary,
    run_full_validation,
)
from src.validation.configs import BENCHMARK_CONFIGS, REGIME_CONFIGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def primary_series():
    return make_synthetic_ohlcv(
        n_bars=200, ticker="AAPL", trend=0.0003, volatility=0.012, seed=1
    )


@pytest.fixture(scope="module")
def benchmark_runner():
    return BenchmarkRunner(n_bars=200, starting_capital=100_000.0, min_bars_required=50)


# ---------------------------------------------------------------------------
# 1. buy_and_hold result has n_trades == 1
# ---------------------------------------------------------------------------

def test_benchmark_runner_buy_and_hold(primary_series, benchmark_runner):
    result = benchmark_runner.run_config("buy_and_hold", primary_series)
    assert isinstance(result, BenchmarkResult)
    assert result.config_name == "buy_and_hold"
    assert result.n_trades == 1


# ---------------------------------------------------------------------------
# 2. forecast_only result is a BenchmarkResult with all fields
# ---------------------------------------------------------------------------

def test_benchmark_runner_forecast_only(primary_series, benchmark_runner):
    result = benchmark_runner.run_config("forecast_only", primary_series)
    assert isinstance(result, BenchmarkResult)
    assert result.config_name == "forecast_only"
    # All required float fields exist
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.realized_pnl, float)
    assert isinstance(result.win_rate, float)
    assert isinstance(result.max_drawdown_pct, float)
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.profit_factor, float)
    assert isinstance(result.n_trades, int)


# ---------------------------------------------------------------------------
# 3. run_all_configs returns one result per config name
# ---------------------------------------------------------------------------

def test_benchmark_all_configs_complete(primary_series, benchmark_runner):
    results = benchmark_runner.run_all_configs(primary_series)
    assert isinstance(results, list)
    config_names = {r.config_name for r in results}
    expected = set(BENCHMARK_CONFIGS.keys())
    assert config_names == expected, (
        f"Missing configs: {expected - config_names}, "
        f"Unexpected: {config_names - expected}"
    )


# ---------------------------------------------------------------------------
# 4. comparison_table returns list sorted by total_return_pct desc
# ---------------------------------------------------------------------------

def test_benchmark_comparison_table_sorted(primary_series, benchmark_runner):
    results = benchmark_runner.run_all_configs(primary_series)
    table = benchmark_runner.comparison_table(results)
    assert isinstance(table, list)
    assert len(table) == len(results)
    returns = [row["total_return_pct"] for row in table]
    assert returns == sorted(returns, reverse=True), "Table is not sorted by total_return_pct desc"


# ---------------------------------------------------------------------------
# 5. run_ablation returns 4 AblationResults
# ---------------------------------------------------------------------------

def test_ablation_returns_all_components(primary_series):
    runner = AblationRunner(n_bars=200, starting_capital=100_000.0, min_bars_required=50)
    results = runner.run_ablation(primary_series)
    assert isinstance(results, list)
    assert len(results) == 4, f"Expected 4 AblationResults, got {len(results)}"
    for r in results:
        assert isinstance(r, AblationResult)


# ---------------------------------------------------------------------------
# 6. each marginal_contribution is one of the valid strings
# ---------------------------------------------------------------------------

def test_ablation_marginal_contribution_values(primary_series):
    runner = AblationRunner(n_bars=200, starting_capital=100_000.0, min_bars_required=50)
    results = runner.run_ablation(primary_series)
    valid = {"positive", "negative", "neutral"}
    for r in results:
        assert r.marginal_contribution in valid, (
            f"component={r.removed_component} "
            f"got marginal_contribution={r.marginal_contribution!r}"
        )


# ---------------------------------------------------------------------------
# 7. increasing slippage → non-increasing stressed_return_pct
# ---------------------------------------------------------------------------

def test_robustness_slippage_monotone(primary_series):
    runner = RobustnessRunner(n_bars=200, starting_capital=100_000.0, min_bars_required=50)
    results = runner.run_slippage_tests(primary_series)
    assert len(results) >= 2
    stressed_returns = [r.stressed_return_pct for r in results]
    for i in range(1, len(stressed_returns)):
        assert stressed_returns[i] <= stressed_returns[i - 1] + 1e-9, (
            f"Slippage returns not monotonically non-increasing: "
            f"{stressed_returns}"
        )


# ---------------------------------------------------------------------------
# 8. run_threshold_sensitivity returns list of RobustnessResults with passed bool
# ---------------------------------------------------------------------------

def test_robustness_threshold_sensitivity(primary_series):
    runner = RobustnessRunner(n_bars=200, starting_capital=100_000.0, min_bars_required=50)
    results = runner.run_threshold_sensitivity(primary_series)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, RobustnessResult)
        assert isinstance(r.passed, bool)


# ---------------------------------------------------------------------------
# 9. run_regime_splits returns one result per REGIME_CONFIG entry
# ---------------------------------------------------------------------------

def test_robustness_regime_splits():
    runner = RobustnessRunner(n_bars=200, starting_capital=100_000.0, min_bars_required=50)
    results = runner.run_regime_splits()
    assert isinstance(results, list)
    assert len(results) == len(REGIME_CONFIGS), (
        f"Expected {len(REGIME_CONFIGS)} regime results, got {len(results)}"
    )
    regime_names = {r.test_name for r in results}
    for cfg in REGIME_CONFIGS:
        assert f"regime_{cfg['name']}" in regime_names


# ---------------------------------------------------------------------------
# 10. small series (80 bars) → adaptive suppression fired → passed=True
# ---------------------------------------------------------------------------

def test_robustness_sample_size_small_suppressed():
    runner = RobustnessRunner(n_bars=200, starting_capital=100_000.0, min_bars_required=50)
    series = make_synthetic_ohlcv(
        n_bars=200, ticker="SYN", trend=0.0003, volatility=0.012, seed=42
    )
    results = runner.run_sample_size_validation(series)
    # Find the small-sample result
    small = next((r for r in results if "sample_size_small" in r.test_name), None)
    assert small is not None, "sample_size_small result not found"
    # passed=True means suppression fired (as expected for small n_trades)
    assert small.passed is True, (
        f"Expected adaptive suppression for small series, but passed={small.passed}. "
        f"n_trades={small.baseline_n_trades}"
    )


# ---------------------------------------------------------------------------
# 11. train/val/test splits use disjoint bar ranges (n_bars sum <= original)
# ---------------------------------------------------------------------------

def test_data_split_no_leakage():
    # Use 400 bars so all 3 splits (70%=280, 15%=60, 15%=60) exceed min_bars_required=50
    series = make_synthetic_ohlcv(
        n_bars=400, ticker="SYN", trend=0.0003, volatility=0.012, seed=10
    )
    validator = DataSplitValidator(
        starting_capital=100_000.0, min_bars_required=50
    )
    results = validator.run(series)
    assert len(results) >= 2, "Expected at least 2 splits (train + test or all three)"
    total_bars = sum(r.n_bars for r in results)
    # Total bars across splits must not exceed the original series length
    assert total_bars <= len(series.bars), (
        f"Split bars ({total_bars}) exceed original ({len(series.bars)}): leakage possible"
    )
    # Split names are unique
    names = [r.split_name for r in results]
    assert len(names) == len(set(names)), "Duplicate split names"


# ---------------------------------------------------------------------------
# 12. walk_forward(n_splits=3) returns 3 DataSplitResults
# ---------------------------------------------------------------------------

def test_walk_forward_n_splits():
    series = make_synthetic_ohlcv(
        n_bars=300, ticker="SYN", trend=0.0003, volatility=0.012, seed=20
    )
    validator = DataSplitValidator(
        starting_capital=100_000.0, min_bars_required=50
    )
    results = validator.walk_forward(series, n_splits=3)
    assert isinstance(results, list)
    assert len(results) == 3, f"Expected 3 walk-forward results, got {len(results)}"
    for i, r in enumerate(results):
        assert isinstance(r, DataSplitResult)
        assert r.split_name == f"wf_test_{i}"


# ---------------------------------------------------------------------------
# 13. ValidationSummary has all required fields
# ---------------------------------------------------------------------------

def test_validation_summary_structure():
    summary = run_full_validation(n_bars=200, n_tickers=2)
    assert isinstance(summary, ValidationSummary)
    assert isinstance(summary.pass_fail, PassFailCriteria)
    assert isinstance(summary.overall_passed, bool)
    assert isinstance(summary.benchmark_comparison, list)
    assert isinstance(summary.ablation_summary, list)
    assert isinstance(summary.robustness_summary, list)
    assert isinstance(summary.data_split_summary, list)
    assert isinstance(summary.ticker_summary, dict)
    assert isinstance(summary.threshold_sensitivity_table, list)
    assert isinstance(summary.report_generated_at, str)
    assert isinstance(summary.notes, list)


# ---------------------------------------------------------------------------
# 14. all PassFailCriteria fields are bool
# ---------------------------------------------------------------------------

def test_pass_fail_criteria_types():
    summary = run_full_validation(n_bars=200, n_tickers=2)
    pf = summary.pass_fail
    assert isinstance(pf.full_system_beats_forecast_only, bool)
    assert isinstance(pf.full_system_reduces_drawdown, bool)
    assert isinstance(pf.consistent_across_tickers, bool)
    assert isinstance(pf.survives_slippage, bool)
    assert isinstance(pf.adaptive_does_not_degrade, bool)


# ---------------------------------------------------------------------------
# 15. run_full_validation(n_bars=200, n_tickers=2) returns ValidationSummary
# ---------------------------------------------------------------------------

def test_run_full_validation_completes():
    summary = run_full_validation(n_bars=200, n_tickers=2)
    assert isinstance(summary, ValidationSummary)
    # Sanity: benchmark comparison table has entries
    assert len(summary.benchmark_comparison) > 0
    # Sanity: ablation has 4 rows
    assert len(summary.ablation_summary) == 4
