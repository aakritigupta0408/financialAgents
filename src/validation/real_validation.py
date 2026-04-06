"""
Phase 12: Real market validation runner.
Runs the full benchmark + ablation + robustness suite on real market data fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.validation.benchmark import BenchmarkRunner, BenchmarkResult
from src.validation.ablation import AblationRunner, AblationResult
from src.validation.robustness import RobustnessRunner, RobustnessResult
from src.validation.summary import generate_validation_summary, ValidationSummary, PassFailCriteria
from src.validation.real_data import load_all_tickers, AVAILABLE_TICKERS
from src.validation.configs import BENCHMARK_CONFIGS


@dataclass
class RealValidationResult:
    tickers_validated: list[str]
    per_ticker_benchmark: dict[str, list[BenchmarkResult]]
    per_ticker_ablation: dict[str, list[AblationResult]]
    robustness_results: list[RobustnessResult]          # run on AAPL
    validation_summary: ValidationSummary
    trade_generation_check: dict[str, dict]             # ticker -> {config -> n_trades}
    fta_acceptance_check: dict[str, dict]               # ticker -> {config -> n_trades diff}
    pass_fail_notes: list[str]


def run_real_validation(
    tickers: list[str] | None = None,
    starting_capital: float = 100_000.0,
    min_bars_required: int = 30,       # lower for daily data (100 bars available)
) -> RealValidationResult:
    """
    Run full Phase 12 real-data validation.

    Parameters
    ----------
    tickers : subset of AVAILABLE_TICKERS. Default: all 6.
    starting_capital : portfolio starting capital.
    min_bars_required : lower than default (50) because we only have ~100 daily bars.

    Steps
    -----
    1. Load all requested ticker series.
    2. Run BenchmarkRunner.run_all_configs() on each ticker.
       Use BacktestEngine kwargs: min_bars_required=min_bars_required, starting_capital=starting_capital.
    3. Run AblationRunner.run_ablation() on each ticker.
    4. Run RobustnessRunner.run_all() on AAPL series only (to save time).
    5. Build trade_generation_check: for each ticker+config, record n_trades.
    6. Build fta_acceptance_check: compare n_trades of "forecast_only" vs "forecast_plus_fta"
       per ticker. If forecast_plus_fta has fewer trades, FTA is filtering (expected).
    7. Call generate_validation_summary() from Phase 11.
    8. Build pass_fail_notes (human-readable observations).
    9. Return RealValidationResult.
    """
    if tickers is None:
        tickers = list(AVAILABLE_TICKERS)

    # Step 1: Load ticker series
    all_series = load_all_tickers()
    ticker_series = {t: all_series[t] for t in tickers if t in all_series}

    benchmark_runner = BenchmarkRunner(
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )

    ablation_runner = AblationRunner(
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )

    # Step 2: Run benchmarks on each ticker
    per_ticker_benchmark: dict[str, list[BenchmarkResult]] = {}
    for ticker, series in ticker_series.items():
        per_ticker_benchmark[ticker] = benchmark_runner.run_all_configs(series)

    # Step 3: Run ablation on each ticker
    per_ticker_ablation: dict[str, list[AblationResult]] = {}
    for ticker, series in ticker_series.items():
        try:
            per_ticker_ablation[ticker] = ablation_runner.run_ablation(series)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Ablation failed for %s: %s", ticker, exc)
            per_ticker_ablation[ticker] = []

    # Step 4: Robustness on AAPL only
    robustness_results: list[RobustnessResult] = []
    if "AAPL" in ticker_series:
        robustness_runner = RobustnessRunner(
            starting_capital=starting_capital,
            min_bars_required=min_bars_required,
        )
        try:
            robustness_results = robustness_runner.run_all(ticker_series["AAPL"])
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Robustness runner failed: %s", exc)

    # Step 5: Build trade_generation_check
    trade_generation_check: dict[str, dict] = {}
    for ticker, bench_results in per_ticker_benchmark.items():
        config_trades: dict[str, int] = {}
        for br in bench_results:
            config_trades[br.config_name] = br.n_trades
        trade_generation_check[ticker] = config_trades

    # Step 6: Build fta_acceptance_check
    fta_acceptance_check: dict[str, dict] = {}
    for ticker, config_trades in trade_generation_check.items():
        forecast_only_n = config_trades.get("forecast_only", 0)
        fta_n = config_trades.get("forecast_plus_fta", 0)
        filtered = forecast_only_n - fta_n
        filtered_pct = (filtered / forecast_only_n * 100.0) if forecast_only_n > 0 else 0.0
        fta_acceptance_check[ticker] = {
            "forecast_only_trades": forecast_only_n,
            "fta_trades": fta_n,
            "filtered": filtered,
            "filtered_pct": filtered_pct,
            "fta_is_filtering": fta_n < forecast_only_n,
        }

    # Step 7: generate_validation_summary — needs data_split_results and threshold_sweep
    # We supply empty data_split_results and minimal threshold_sweep for real data context
    try:
        from src.validation.data_split import DataSplitValidator
        # Use AAPL series if available, else first available
        primary_ticker = "AAPL" if "AAPL" in ticker_series else tickers[0]
        primary_series = ticker_series[primary_ticker]
        split_validator = DataSplitValidator(
            starting_capital=starting_capital,
            min_bars_required=min_bars_required,
        )
        split_results = split_validator.run(primary_series)
        wf_results = split_validator.walk_forward(primary_series)
        all_split_results = split_results + wf_results
    except Exception:
        all_split_results = []

    try:
        from src.reports.threshold_tuning import sweep_thresholds
        from src.backtest.engine import BacktestEngine
        primary_ticker = "AAPL" if "AAPL" in ticker_series else tickers[0]
        primary_series = ticker_series[primary_ticker]
        primary_result = BacktestEngine(
            starting_capital=starting_capital,
            min_bars_required=min_bars_required,
            fta_enabled=True,
            meta_model_enabled=True,
        ).run(primary_series)
        threshold_sweep = sweep_thresholds(primary_result)
    except Exception:
        threshold_sweep = {"confidence_sweep": []}

    validation_summary = generate_validation_summary(
        benchmark_results=per_ticker_benchmark,
        ablation_results=per_ticker_ablation.get(tickers[0], []) if tickers else [],
        robustness_results=robustness_results,
        data_split_results=all_split_results,
        threshold_sweep=threshold_sweep,
    )

    # Step 8: Build pass_fail_notes
    pass_fail_notes: list[str] = []

    # FTA trade generation notes
    any_trades = False
    for ticker in tickers:
        config_trades = trade_generation_check.get(ticker, {})
        full_system_n = config_trades.get("full_system", 0)
        if full_system_n > 0:
            any_trades = True
            pass_fail_notes.append(
                f"{ticker}: FTA allows trades on real data (n={full_system_n})"
            )

    if not any_trades:
        pass_fail_notes.append(
            "WARNING: No trades generated on any ticker — FTA may be too restrictive"
        )

    # Compare full_system vs forecast_only total_return_pct
    for ticker, bench_results in per_ticker_benchmark.items():
        fs_results = [r for r in bench_results if r.config_name == "full_system"]
        fo_results = [r for r in bench_results if r.config_name == "forecast_only"]
        if fs_results and fo_results:
            fs_ret = fs_results[0].total_return_pct
            fo_ret = fo_results[0].total_return_pct
            if fs_ret > fo_ret:
                pass_fail_notes.append(
                    f"{ticker}: full_system ({fs_ret:.2f}%) outperforms "
                    f"forecast_only ({fo_ret:.2f}%)"
                )
            else:
                pass_fail_notes.append(
                    f"{ticker}: forecast_only ({fo_ret:.2f}%) >= "
                    f"full_system ({fs_ret:.2f}%)"
                )

    # FTA filtering notes
    for ticker, fta_info in fta_acceptance_check.items():
        if fta_info["fta_is_filtering"]:
            pass_fail_notes.append(
                f"{ticker}: FTA filtered {fta_info['filtered_pct']:.1f}% of "
                f"forecast_only trades ({fta_info['forecast_only_trades']} -> "
                f"{fta_info['fta_trades']})"
            )

    # Overall pass/fail from summary
    if validation_summary.overall_passed:
        pass_fail_notes.append("OVERALL: Phase 12 validation PASSED")
    else:
        pass_fail_notes.append("OVERALL: Phase 12 validation FAILED — see criteria details")

    return RealValidationResult(
        tickers_validated=list(tickers),
        per_ticker_benchmark=per_ticker_benchmark,
        per_ticker_ablation=per_ticker_ablation,
        robustness_results=robustness_results,
        validation_summary=validation_summary,
        trade_generation_check=trade_generation_check,
        fta_acceptance_check=fta_acceptance_check,
        pass_fail_notes=pass_fail_notes,
    )


def print_real_validation_report(result: RealValidationResult) -> None:
    """
    Print a formatted human-readable report to stdout.

    Sections:
    1. Trade Generation Check (table: ticker, config, n_trades)
    2. FTA Acceptance Check (table: ticker, forecast_only_trades, fta_trades, filtered_pct)
    3. Per-ticker Performance (table: ticker, config, return%, drawdown%, win_rate, n_trades)
    4. Ablation Summary (table: component, return_delta_mean_across_tickers, contribution)
    5. Robustness Results (AAPL only)
    6. Pass/Fail Notes
    7. Overall Assessment
    """
    sep = "=" * 72

    # ---- Section 1: Trade Generation Check ----
    print(sep)
    print("  PHASE 12 REAL-DATA VALIDATION REPORT")
    print(sep)
    print()
    print("  1. TRADE GENERATION CHECK")
    print(f"  {'Ticker':<8} {'Config':<35} {'N Trades':>8}")
    print(f"  {'-'*8} {'-'*35} {'-'*8}")
    for ticker in result.tickers_validated:
        config_trades = result.trade_generation_check.get(ticker, {})
        for config_name in BENCHMARK_CONFIGS:
            n = config_trades.get(config_name, "-")
            print(f"  {ticker:<8} {config_name:<35} {str(n):>8}")
    print()

    # ---- Section 2: FTA Acceptance Check ----
    print("  2. FTA ACCEPTANCE CHECK (forecast_only vs forecast_plus_fta)")
    print(f"  {'Ticker':<8} {'Forecast Trades':>16} {'FTA Trades':>12} {'Filtered%':>10}")
    print(f"  {'-'*8} {'-'*16} {'-'*12} {'-'*10}")
    for ticker in result.tickers_validated:
        info = result.fta_acceptance_check.get(ticker, {})
        fo = info.get("forecast_only_trades", 0)
        fta = info.get("fta_trades", 0)
        pct = info.get("filtered_pct", 0.0)
        print(f"  {ticker:<8} {fo:>16} {fta:>12} {pct:>9.1f}%")
    print()

    # ---- Section 3: Per-ticker Performance ----
    print("  3. PER-TICKER PERFORMANCE")
    print(f"  {'Ticker':<8} {'Config':<35} {'Return%':>8} {'Drawdown%':>10} {'WinRate':>8} {'Trades':>7}")
    print(f"  {'-'*8} {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*7}")
    for ticker in result.tickers_validated:
        bench_results = result.per_ticker_benchmark.get(ticker, [])
        for br in bench_results:
            wr = f"{br.win_rate:.1%}" if br.win_rate else "N/A"
            print(
                f"  {ticker:<8} {br.config_name:<35} "
                f"{br.total_return_pct:>7.2f}% {br.max_drawdown_pct:>9.2f}% "
                f"{wr:>8} {br.n_trades:>7}"
            )
    print()

    # ---- Section 4: Ablation Summary ----
    print("  4. ABLATION SUMMARY (mean return_delta across tickers)")
    # Aggregate deltas across tickers
    component_deltas: dict[str, list[float]] = {}
    component_contrib: dict[str, list[str]] = {}
    for ticker, ablation_results in result.per_ticker_ablation.items():
        for ar in ablation_results:
            component_deltas.setdefault(ar.removed_component, []).append(ar.return_delta_pct)
            component_contrib.setdefault(ar.removed_component, []).append(ar.marginal_contribution)
    print(f"  {'Component':<20} {'Mean ReturnDelta%':>18} {'Contribution':>14}")
    print(f"  {'-'*20} {'-'*18} {'-'*14}")
    for comp, deltas in component_deltas.items():
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        # Most common contribution label
        from collections import Counter
        most_common = Counter(component_contrib.get(comp, [])).most_common(1)
        contrib = most_common[0][0] if most_common else "neutral"
        print(f"  {comp:<20} {mean_delta:>17.3f}% {contrib:>14}")
    print()

    # ---- Section 5: Robustness Results (AAPL) ----
    print("  5. ROBUSTNESS RESULTS (AAPL)")
    print(f"  {'Test':<40} {'Base Ret%':>9} {'Stressed%':>10} {'Passed':>7}")
    print(f"  {'-'*40} {'-'*9} {'-'*10} {'-'*7}")
    for rr in result.robustness_results:
        passed_str = "PASS" if rr.passed else "FAIL"
        print(
            f"  {rr.test_name:<40} {rr.baseline_return_pct:>8.2f}% "
            f"{rr.stressed_return_pct:>9.2f}% {passed_str:>7}"
        )
    if not result.robustness_results:
        print("  (no robustness results)")
    print()

    # ---- Section 6: Pass/Fail Notes ----
    print("  6. PASS/FAIL NOTES")
    for note in result.pass_fail_notes:
        print(f"    - {note}")
    print()

    # ---- Section 7: Overall Assessment ----
    print("  7. OVERALL ASSESSMENT")
    pf = result.validation_summary.pass_fail
    criteria = {
        "full_system_beats_forecast_only": pf.full_system_beats_forecast_only,
        "full_system_reduces_drawdown": pf.full_system_reduces_drawdown,
        "consistent_across_tickers": pf.consistent_across_tickers,
        "survives_slippage": pf.survives_slippage,
        "adaptive_does_not_degrade": pf.adaptive_does_not_degrade,
    }
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {name:<45}: {status}")
    overall = "PASS" if result.validation_summary.overall_passed else "FAIL"
    print()
    print(f"  OVERALL: {overall}")
    print(sep)
