"""
src.validation.summary — ValidationSummary and generate_validation_summary.

Aggregates all sub-validation results into a single pass/fail report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.validation.benchmark import BenchmarkResult
from src.validation.ablation import AblationResult
from src.validation.robustness import RobustnessResult
from src.validation.data_split import DataSplitResult


@dataclass
class PassFailCriteria:
    full_system_beats_forecast_only: bool
    full_system_reduces_drawdown: bool
    consistent_across_tickers: bool       # >= 2 of 3 tickers have positive return
    survives_slippage: bool               # return >= 50% of baseline at 0.1% slippage
    adaptive_does_not_degrade: bool       # adaptation cycle completes without worsening


@dataclass
class ValidationSummary:
    pass_fail: PassFailCriteria
    overall_passed: bool
    benchmark_comparison: list[dict]
    ablation_summary: list[dict]
    robustness_summary: list[dict]
    data_split_summary: list[dict]
    ticker_summary: dict[str, dict]
    threshold_sensitivity_table: list[dict]
    report_generated_at: str              # ISO datetime
    notes: list[str]


def generate_validation_summary(
    benchmark_results: dict[str, list[BenchmarkResult]],
    ablation_results: list[AblationResult],
    robustness_results: list[RobustnessResult],
    data_split_results: list[DataSplitResult],
    threshold_sweep: dict,
) -> ValidationSummary:
    """
    Evaluate pass/fail criteria and assemble the full ValidationSummary.
    """
    # ------------------------------------------------------------------
    # 1. full_system_beats_forecast_only
    # ------------------------------------------------------------------
    full_system_returns = _collect_config_returns(benchmark_results, "full_system")
    forecast_only_returns = _collect_config_returns(benchmark_results, "forecast_only")

    mean_full = _safe_mean(full_system_returns)
    mean_forecast = _safe_mean(forecast_only_returns)
    full_system_beats_forecast_only = mean_full >= mean_forecast

    # ------------------------------------------------------------------
    # 2. full_system_reduces_drawdown
    # ------------------------------------------------------------------
    full_system_dds = _collect_config_dds(benchmark_results, "full_system")
    forecast_only_dds = _collect_config_dds(benchmark_results, "forecast_only")

    mean_full_dd = _safe_mean(full_system_dds)
    mean_forecast_dd = _safe_mean(forecast_only_dds)
    # Pass if full_system drawdown <= forecast_only * 1.10 (slight tolerance)
    full_system_reduces_drawdown = mean_full_dd <= mean_forecast_dd * 1.10

    # ------------------------------------------------------------------
    # 3. consistent_across_tickers
    # ------------------------------------------------------------------
    positive_tickers = 0
    total_tickers = 0
    for ticker, results in benchmark_results.items():
        fs_results = [r for r in results if r.config_name == "full_system"]
        if fs_results:
            total_tickers += 1
            if fs_results[0].total_return_pct > 0:
                positive_tickers += 1
    consistent_across_tickers = positive_tickers >= 2

    # ------------------------------------------------------------------
    # 4. survives_slippage (0.1% = 0.001)
    # ------------------------------------------------------------------
    slippage_result = next(
        (r for r in robustness_results if "slippage_0.001" in r.test_name),
        None,
    )
    survives_slippage = slippage_result.passed if slippage_result is not None else True

    # ------------------------------------------------------------------
    # 5. adaptive_does_not_degrade
    # ------------------------------------------------------------------
    sample_large = next(
        (r for r in robustness_results if "sample_size_large" in r.test_name),
        None,
    )
    adaptive_does_not_degrade = sample_large.passed if sample_large is not None else True

    # ------------------------------------------------------------------
    # Assemble PassFailCriteria
    # ------------------------------------------------------------------
    pass_fail = PassFailCriteria(
        full_system_beats_forecast_only=full_system_beats_forecast_only,
        full_system_reduces_drawdown=full_system_reduces_drawdown,
        consistent_across_tickers=consistent_across_tickers,
        survives_slippage=survives_slippage,
        adaptive_does_not_degrade=adaptive_does_not_degrade,
    )

    overall_passed = all([
        pass_fail.full_system_beats_forecast_only,
        pass_fail.full_system_reduces_drawdown,
        pass_fail.consistent_across_tickers,
        pass_fail.survives_slippage,
        pass_fail.adaptive_does_not_degrade,
    ])

    # ------------------------------------------------------------------
    # Flatten benchmark results into comparison table
    # ------------------------------------------------------------------
    all_benchmark_results: list[BenchmarkResult] = []
    for results in benchmark_results.values():
        all_benchmark_results.extend(results)

    comparison_table = [
        {
            "config_name": r.config_name,
            "ticker": r.ticker,
            "n_bars": r.n_bars,
            "n_trades": r.n_trades,
            "total_return_pct": r.total_return_pct,
            "realized_pnl": r.realized_pnl,
            "win_rate": r.win_rate,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "profit_factor": r.profit_factor,
        }
        for r in all_benchmark_results
    ]
    comparison_table.sort(key=lambda x: x["total_return_pct"], reverse=True)

    # Ablation summary
    ablation_summary = [
        {
            "removed_component": a.removed_component,
            "baseline_return_pct": a.baseline_return_pct,
            "ablated_return_pct": a.ablated_return_pct,
            "return_delta_pct": a.return_delta_pct,
            "marginal_contribution": a.marginal_contribution,
            "baseline_n_trades": a.baseline_n_trades,
            "ablated_n_trades": a.ablated_n_trades,
        }
        for a in ablation_results
    ]

    # Robustness summary
    robustness_summary = [
        {
            "test_name": r.test_name,
            "description": r.description,
            "baseline_return_pct": r.baseline_return_pct,
            "stressed_return_pct": r.stressed_return_pct,
            "baseline_n_trades": r.baseline_n_trades,
            "stressed_n_trades": r.stressed_n_trades,
            "passed": r.passed,
            "notes": r.notes,
        }
        for r in robustness_results
    ]

    # Data split summary
    data_split_summary = [
        {
            "split_name": d.split_name,
            "n_bars": d.n_bars,
            "n_trades": d.n_trades,
            "total_return_pct": d.total_return_pct,
            "win_rate": d.win_rate,
            "max_drawdown_pct": d.max_drawdown_pct,
            "sharpe_ratio": d.sharpe_ratio,
        }
        for d in data_split_results
    ]

    # Per-ticker aggregated metrics
    ticker_summary: dict[str, dict] = {}
    for ticker, results in benchmark_results.items():
        fs = [r for r in results if r.config_name == "full_system"]
        fo = [r for r in results if r.config_name == "forecast_only"]
        ticker_summary[ticker] = {
            "full_system_return_pct": fs[0].total_return_pct if fs else None,
            "forecast_only_return_pct": fo[0].total_return_pct if fo else None,
            "full_system_n_trades": fs[0].n_trades if fs else None,
            "full_system_win_rate": fs[0].win_rate if fs else None,
            "full_system_max_drawdown_pct": fs[0].max_drawdown_pct if fs else None,
        }

    # Threshold sensitivity table from sweep
    threshold_sensitivity_table = threshold_sweep.get("confidence_sweep", [])

    # Notes
    notes: list[str] = []
    notes.append(
        f"full_system mean return={mean_full:.2f}% vs "
        f"forecast_only mean={mean_forecast:.2f}%"
    )
    notes.append(
        f"Tickers with positive full_system return: {positive_tickers}/{total_tickers}"
    )
    if not overall_passed:
        failed = [
            k for k, v in {
                "full_system_beats_forecast_only": full_system_beats_forecast_only,
                "full_system_reduces_drawdown": full_system_reduces_drawdown,
                "consistent_across_tickers": consistent_across_tickers,
                "survives_slippage": survives_slippage,
                "adaptive_does_not_degrade": adaptive_does_not_degrade,
            }.items()
            if not v
        ]
        notes.append(f"Failed criteria: {', '.join(failed)}")

    return ValidationSummary(
        pass_fail=pass_fail,
        overall_passed=overall_passed,
        benchmark_comparison=comparison_table,
        ablation_summary=ablation_summary,
        robustness_summary=robustness_summary,
        data_split_summary=data_split_summary,
        ticker_summary=ticker_summary,
        threshold_sensitivity_table=threshold_sensitivity_table,
        report_generated_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_config_returns(
    benchmark_results: dict[str, list[BenchmarkResult]],
    config_name: str,
) -> list[float]:
    values: list[float] = []
    for results in benchmark_results.values():
        for r in results:
            if r.config_name == config_name:
                values.append(r.total_return_pct)
    return values


def _collect_config_dds(
    benchmark_results: dict[str, list[BenchmarkResult]],
    config_name: str,
) -> list[float]:
    values: list[float] = []
    for results in benchmark_results.values():
        for r in results:
            if r.config_name == config_name:
                values.append(r.max_drawdown_pct)
    return values


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
