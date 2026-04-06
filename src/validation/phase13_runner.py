"""
src.validation.phase13_runner — Phase 13 orchestrator.

Intraday calibration + final acceptance test using structure-rich 1h
synthetic data designed to exercise FTA on realistic intraday price action.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from schemas.market_data import OHLCVSeries
from src.validation.intraday_synthetic import (
    TICKER_PROXIES,
    load_or_generate_1h_series,
)
from src.validation.fta_calibration import (
    CalibrationSummary,
    FTACalibrationResult,
    run_fta_calibration,
    print_calibration_table,
)
from src.validation.benchmark import BenchmarkResult, BenchmarkRunner
from src.validation.configs import BENCHMARK_CONFIGS

_DEFAULT_TICKERS = [p["ticker"] for p in TICKER_PROXIES]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase13Result:
    """Full Phase 13 result: calibration + benchmark + acceptance decision."""

    # Calibration
    calibration_summaries: dict[str, CalibrationSummary]
    recommended_rr_global: float

    # Benchmark at recommended RR
    benchmark_results: dict[str, list[BenchmarkResult]]  # ticker -> list[BenchmarkResult]

    # Acceptance decision
    acceptance_verdict: str   # "READY_FOR_PAPER_TRADING" | "NEEDS_CALIBRATION" | "FAILS_CURRENT_ACCEPTANCE"
    acceptance_reasons: list[str]

    # Summary tables
    per_ticker_summary: dict[str, dict]
    threshold_comparison_table: list[dict]

    # Recommendations
    recommended_settings: dict
    remaining_todos: list[str]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_phase13(
    tickers: Optional[list[str]] = None,
    n_bars: int = 600,
    starting_capital: float = 100_000.0,
    min_bars_required: int = 50,
) -> Phase13Result:
    """
    Full Phase 13 pipeline.

    Steps:
    1. Generate/load 1h series for each ticker.
    2. Run FTA calibration across RR thresholds [1.25, 1.50, 1.75, 2.00].
    3. Choose recommended_rr_global = median of per-ticker recommendations.
    4. Temporarily set FTA_MIN_REWARD_RISK = recommended_rr_global.
    5. Run BenchmarkRunner.run_all_configs() on each ticker.
    6. Restore FTA_MIN_REWARD_RISK.
    7. Build summary tables.
    8. Determine acceptance verdict.
    9. Build recommended_settings and remaining_todos.

    Parameters
    ----------
    tickers          : List of ticker strings to run. Default: all 6 proxies.
    n_bars           : Bars per synthetic series.
    starting_capital : Capital for each backtest.
    min_bars_required: Minimum history before attempting trades.

    Returns
    -------
    Phase13Result
    """
    if tickers is None:
        tickers = _DEFAULT_TICKERS

    # Step 1: Generate/load series
    series_map: dict[str, OHLCVSeries] = {}
    for t in tickers:
        series_map[t] = load_or_generate_1h_series(t, n_bars=n_bars)

    # Step 2: FTA calibration
    calibration_summaries = run_fta_calibration(
        series_map=series_map,
        rr_thresholds=[1.25, 1.50, 1.75, 2.00],
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )

    # Step 3: Global recommended RR = median of per-ticker recommendations
    rr_values = [s.recommended_rr for s in calibration_summaries.values()]
    if rr_values:
        recommended_rr_global = statistics.median(rr_values)
    else:
        recommended_rr_global = 1.25
    # Round to nearest valid threshold
    valid_thresholds = [1.25, 1.50, 1.75, 2.00]
    recommended_rr_global = min(
        valid_thresholds, key=lambda v: abs(v - recommended_rr_global)
    )

    # Steps 4-6: Run benchmarks under recommended RR
    import config.settings as _cfg
    import src.fta.engine as _fta_engine
    orig_rr = _cfg.FTA_MIN_REWARD_RISK
    orig_fta_rr = getattr(_fta_engine, "FTA_MIN_REWARD_RISK", None)

    benchmark_results: dict[str, list[BenchmarkResult]] = {}
    runner = BenchmarkRunner(
        n_bars=n_bars,
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )

    try:
        _cfg.FTA_MIN_REWARD_RISK = recommended_rr_global
        if hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = recommended_rr_global

        for ticker, series in series_map.items():
            try:
                results = runner.run_all_configs(series)
                benchmark_results[ticker] = results
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Phase13: benchmark failed for %s: %s", ticker, exc
                )
                benchmark_results[ticker] = []

    finally:
        _cfg.FTA_MIN_REWARD_RISK = orig_rr
        if orig_fta_rr is not None and hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = orig_fta_rr
        elif hasattr(_fta_engine, "FTA_MIN_REWARD_RISK") and orig_fta_rr is None:
            _fta_engine.FTA_MIN_REWARD_RISK = orig_rr

    # Step 7: Build summary tables
    per_ticker_summary = _build_per_ticker_summary(
        benchmark_results, calibration_summaries, recommended_rr_global
    )
    threshold_comparison_table = _build_threshold_comparison_table(calibration_summaries)

    # Step 8: Acceptance verdict
    verdict, reasons = _determine_acceptance(
        benchmark_results, calibration_summaries, per_ticker_summary, tickers
    )

    # Step 9: Recommended settings and TODOs
    recommended_settings = _build_recommended_settings(
        recommended_rr_global, calibration_summaries
    )
    remaining_todos = _build_remaining_todos(verdict, calibration_summaries, tickers)

    return Phase13Result(
        calibration_summaries=calibration_summaries,
        recommended_rr_global=recommended_rr_global,
        benchmark_results=benchmark_results,
        acceptance_verdict=verdict,
        acceptance_reasons=reasons,
        per_ticker_summary=per_ticker_summary,
        threshold_comparison_table=threshold_comparison_table,
        recommended_settings=recommended_settings,
        remaining_todos=remaining_todos,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_per_ticker_summary(
    benchmark_results: dict[str, list[BenchmarkResult]],
    cal_summaries: dict[str, CalibrationSummary],
    recommended_rr: float,
) -> dict[str, dict]:
    """Build a per-ticker summary dict with key benchmark config metrics."""
    summary: dict[str, dict] = {}
    for ticker, results in benchmark_results.items():
        full_sys = _find_config(results, "full_system")
        fcast_only = _find_config(results, "forecast_only")
        bah = _find_config(results, "buy_and_hold")
        cal = cal_summaries.get(ticker)

        summary[ticker] = {
            "recommended_rr": recommended_rr,
            "cal_recommended_rr": cal.recommended_rr if cal else None,
            "cal_all_zero": cal.all_zero_trades if cal else True,
            "full_system_trades": full_sys.n_trades if full_sys else 0,
            "full_system_win_rate": full_sys.win_rate if full_sys else 0.0,
            "full_system_return_pct": full_sys.total_return_pct if full_sys else 0.0,
            "full_system_max_dd": full_sys.max_drawdown_pct if full_sys else 0.0,
            "full_system_sharpe": full_sys.sharpe_ratio if full_sys else 0.0,
            "forecast_only_trades": fcast_only.n_trades if fcast_only else 0,
            "buy_and_hold_return_pct": bah.total_return_pct if bah else 0.0,
        }
    return summary


def _build_threshold_comparison_table(
    cal_summaries: dict[str, CalibrationSummary],
) -> list[dict]:
    """Build a flat list of dicts for threshold comparison across all tickers."""
    rows: list[dict] = []
    for ticker, summary in cal_summaries.items():
        for r in summary.results:
            rows.append({
                "ticker": ticker,
                "rr_threshold": r.rr_threshold,
                "n_trades": r.n_trades,
                "trades_per_100_bars": r.trades_per_100_bars,
                "win_rate": r.win_rate,
                "total_return_pct": r.total_return_pct,
                "max_drawdown_pct": r.max_drawdown_pct,
                "expectancy": r.expectancy,
                "profit_factor": r.profit_factor,
                "verdict": r.verdict,
                "recommended": (r.rr_threshold == summary.recommended_rr),
            })
    return rows


def _find_config(
    results: list[BenchmarkResult],
    config_name: str,
) -> Optional[BenchmarkResult]:
    """Find a BenchmarkResult by config_name, or None."""
    for r in results:
        if r.config_name == config_name:
            return r
    return None


def _determine_acceptance(
    benchmark_results: dict[str, list[BenchmarkResult]],
    cal_summaries: dict[str, CalibrationSummary],
    per_ticker_summary: dict[str, dict],
    tickers: list[str],
) -> tuple[str, list[str]]:
    """
    Determine the acceptance verdict.

    Returns (verdict_string, list_of_reasons).
    """
    reasons: list[str] = []
    n_tickers = len(tickers)

    # Count tickers with > 0 full_system trades
    tickers_with_trades = sum(
        1 for t in tickers
        if per_ticker_summary.get(t, {}).get("full_system_trades", 0) > 0
    )

    # Count tickers with win_rate > 0.40
    tickers_good_winrate = sum(
        1 for t in tickers
        if per_ticker_summary.get(t, {}).get("full_system_win_rate", 0.0) > 0.40
    )

    # Count tickers with max_drawdown < 20% OR 0 trades
    tickers_low_dd = sum(
        1 for t in tickers
        if (per_ticker_summary.get(t, {}).get("full_system_trades", 0) == 0
            or per_ticker_summary.get(t, {}).get("full_system_max_dd", 100.0) < 20.0)
    )

    # Count tickers where calibration found "good" threshold
    tickers_good_cal = sum(
        1 for t in tickers
        if cal_summaries.get(t) and not cal_summaries[t].all_zero_trades
        and any(r.verdict == "good" for r in cal_summaries[t].results)
    )

    # Count tickers where ALL thresholds gave 0 trades
    tickers_all_zero = sum(
        1 for t in tickers
        if cal_summaries.get(t) and cal_summaries[t].all_zero_trades
    )

    reasons.append(
        f"tickers_with_trades={tickers_with_trades}/{n_tickers}"
    )
    reasons.append(
        f"tickers_good_winrate_gt40pct={tickers_good_winrate}/{n_tickers}"
    )
    reasons.append(
        f"tickers_low_drawdown_lt20pct={tickers_low_dd}/{n_tickers}"
    )
    reasons.append(
        f"tickers_with_good_calibration={tickers_good_cal}/{n_tickers}"
    )
    reasons.append(
        f"tickers_all_zero_cal={tickers_all_zero}/{n_tickers}"
    )

    # FAILS_CURRENT_ACCEPTANCE: all zero trades everywhere
    if tickers_all_zero == n_tickers and tickers_with_trades == 0:
        reasons.append(
            "All tickers produced 0 trades at all RR thresholds including 1.25."
        )
        return "FAILS_CURRENT_ACCEPTANCE", reasons

    # Check if full_system is worse than forecast_only on ALL tickers
    full_sys_worse_count = 0
    for t in tickers:
        ts = per_ticker_summary.get(t, {})
        brs = benchmark_results.get(t, [])
        full_sys = _find_config(brs, "full_system")
        fcast = _find_config(brs, "forecast_only")
        if full_sys and fcast and full_sys.max_drawdown_pct > fcast.max_drawdown_pct:
            full_sys_worse_count += 1

    if full_sys_worse_count == n_tickers and n_tickers > 0 and tickers_with_trades == 0:
        reasons.append(
            "full_system has worse drawdown than forecast_only on all tickers "
            "and no trades were generated."
        )
        return "FAILS_CURRENT_ACCEPTANCE", reasons

    # READY_FOR_PAPER_TRADING: all criteria satisfied
    ready = (
        tickers_with_trades >= 3
        and tickers_good_winrate >= n_tickers // 2
        and tickers_low_dd == n_tickers
        and tickers_good_cal >= 2
    )
    if ready:
        reasons.append("All readiness criteria satisfied.")
        return "READY_FOR_PAPER_TRADING", reasons

    # NEEDS_CALIBRATION: partial trades or partial good calibration
    if tickers_with_trades > 0 or tickers_good_cal > 0:
        if tickers_good_cal < 2:
            reasons.append(
                f"Calibration 'good' threshold found for only {tickers_good_cal} ticker(s) "
                "(need >= 2). Further tuning required."
            )
        if tickers_with_trades < 3:
            reasons.append(
                f"Only {tickers_with_trades} ticker(s) produce trades (need >= 3). "
                "May need looser filters or more data."
            )
        if tickers_good_winrate < n_tickers // 2:
            reasons.append(
                f"Only {tickers_good_winrate} ticker(s) have win_rate > 40% "
                f"(need >= {n_tickers // 2})."
            )
        return "NEEDS_CALIBRATION", reasons

    # Fallback
    reasons.append("Insufficient evidence to reach READY. Defaulting to NEEDS_CALIBRATION.")
    return "NEEDS_CALIBRATION", reasons


def _build_recommended_settings(
    recommended_rr: float,
    cal_summaries: dict[str, CalibrationSummary],
) -> dict:
    """Build the recommended production settings dict."""
    return {
        "FTA_MIN_REWARD_RISK": recommended_rr,
        "META_MODEL_MIN_CONFIDENCE": 0.55,  # slightly looser than default 0.60
        "MIN_BARS_REQUIRED": 50,
        "CONTEXT_BARS": 100,
        "ATR_STOP_MULTIPLE": 1.5,
        "ATR_TARGET_MULTIPLE": 3.0,
        "TIMEFRAME": "1h",
        "NOTE": (
            f"FTA_MIN_REWARD_RISK={recommended_rr:.2f} chosen as median of "
            f"per-ticker calibration recommendations. "
            "Verify on real intraday data before live deployment."
        ),
    }


def _build_remaining_todos(
    verdict: str,
    cal_summaries: dict[str, CalibrationSummary],
    tickers: list[str],
) -> list[str]:
    """Build the remaining TODO list for production deployment."""
    todos: list[str] = [
        "TODO: Acquire real 1h intraday data (Alpha Vantage premium or similar) "
        "and re-run Phase 13 calibration on real data.",
        "TODO: Validate FTA filter on at least 200+ real 1h bars per ticker "
        "before enabling paper trading.",
        "TODO: Train meta-model on historical trade outcomes from real data.",
        "TODO: Implement real-time data feed for live paper trading loop.",
        "TODO: Add slippage and commission model for more realistic simulation.",
    ]

    if verdict == "FAILS_CURRENT_ACCEPTANCE":
        todos.insert(0,
            "URGENT: System produces 0 trades on synthetic data. "
            "Investigate FTA filter calibration — check ATR/RR/FTA_CLEARANCE settings."
        )
    elif verdict == "NEEDS_CALIBRATION":
        todos.insert(0,
            "Calibration incomplete. Run with real 1h data or expand "
            "synthetic series (n_bars >= 1000) for more reliable calibration."
        )

    all_zero_count = sum(1 for s in cal_summaries.values() if s.all_zero_trades)
    if all_zero_count > 0:
        todos.append(
            f"TODO: {all_zero_count} ticker(s) produced 0 trades at all RR thresholds. "
            "Review FTA_MIN_DISTANCE_TO_FTA_PCT and UNSUITABLE_VOLATILITY checks "
            "for these tickers."
        )

    return todos


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_phase13_report(result: Phase13Result) -> None:
    """
    Print the full Phase 13 report to stdout.

    Sections:
    1. Data Summary
    2. FTA Calibration Table
    3. Benchmark Results at Recommended R:R
    4. Acceptance Decision
    5. Recommended Production Settings
    6. Remaining TODOs
    """
    sep = "=" * 80
    thin = "-" * 80

    print()
    print(sep)
    print("  PHASE 13 REPORT — Intraday FTA Calibration + Acceptance Test")
    print(sep)

    # Section 1: Data Summary
    print()
    print("  1. DATA SUMMARY")
    print(thin)
    tickers = list(result.calibration_summaries.keys())
    print(f"  Tickers        : {', '.join(tickers)}")
    if tickers:
        first_ticker = tickers[0]
        cal = result.calibration_summaries[first_ticker]
        if cal.results:
            n_bars = cal.results[0].n_bars
            print(f"  Bars per ticker: {n_bars}")
    print(f"  Timeframe      : 1h (intraday synthetic)")
    print(f"  Date range     : 2025-01-02 → 2025 (7 bars/day US session)")
    print(f"  Data source    : Structure-rich synthetic (wave skeleton)")

    # Section 2: Calibration Table
    print()
    print("  2. FTA CALIBRATION")
    print_calibration_table(result.calibration_summaries)

    # Section 3: Benchmark Results
    print()
    print("  3. BENCHMARK RESULTS AT RECOMMENDED R:R ="
          f" {result.recommended_rr_global:.2f}")
    print(thin)
    configs_to_show = ["buy_and_hold", "forecast_only", "full_system_no_meta_model", "full_system"]
    header = (
        f"  {'Ticker':<10} {'Config':<28} {'Trades':>6} "
        f"{'Ret%':>7} {'WinR':>6} {'DD%':>7} {'Sharpe':>7}"
    )
    print(header)
    print("  " + thin)

    for ticker, bench_list in result.benchmark_results.items():
        for br in bench_list:
            if br.config_name in configs_to_show:
                print(
                    f"  {ticker:<10} {br.config_name:<28} {br.n_trades:>6} "
                    f"{br.total_return_pct:>7.2f} {br.win_rate:>6.1%} "
                    f"{br.max_drawdown_pct:>7.2f} {br.sharpe_ratio:>7.3f}"
                )

    # Section 4: Acceptance Decision
    print()
    print("  4. ACCEPTANCE DECISION")
    print(thin)
    verdict = result.acceptance_verdict
    verdict_display = {
        "READY_FOR_PAPER_TRADING": "READY FOR PAPER TRADING",
        "NEEDS_CALIBRATION": "NEEDS CALIBRATION",
        "FAILS_CURRENT_ACCEPTANCE": "FAILS CURRENT ACCEPTANCE",
    }.get(verdict, verdict)
    print(f"  Verdict: {verdict_display}")
    print()
    print("  Reasons:")
    for reason in result.acceptance_reasons:
        print(f"    - {reason}")

    # Section 5: Recommended Settings
    print()
    print("  5. RECOMMENDED PRODUCTION SETTINGS")
    print(thin)
    for k, v in result.recommended_settings.items():
        if k != "NOTE":
            print(f"  {k:<35}: {v}")
    note = result.recommended_settings.get("NOTE", "")
    if note:
        print(f"\n  Note: {note}")

    # Section 6: Remaining TODOs
    print()
    print("  6. REMAINING TODOs BEFORE REAL DEPLOYMENT")
    print(thin)
    for todo in result.remaining_todos:
        print(f"  - {todo}")

    print()
    print(sep)
    print()
