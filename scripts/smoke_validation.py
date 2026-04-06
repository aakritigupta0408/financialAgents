"""
scripts/smoke_validation.py — Quick Phase 11 validation smoke test.

Runs in < 60 seconds with n_bars=300 and n_tickers=2.
Prints comparison table, ablation summary, robustness results,
data split results, pass/fail criteria, and overall verdict.
"""

from __future__ import annotations

import sys
import time


def _fmt(val, width=10, decimals=2):
    """Format a number or None for tabular display."""
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, bool):
        return str(val).rjust(width)
    if isinstance(val, int):
        return str(val).rjust(width)
    return f"{val:.{decimals}f}".rjust(width)


def _hr(char="-", n=90):
    print(char * n)


def main():
    print("Phase 11 Validation Smoke Test")
    print("=" * 90)
    start = time.time()

    from src.validation import run_full_validation

    summary = run_full_validation(n_bars=300, n_tickers=2)

    elapsed = time.time() - start
    print(f"Validation completed in {elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("BENCHMARK COMPARISON TABLE (sorted by total_return_pct desc)")
    _hr()
    header = (
        f"{'Config':<40} {'Ticker':<6} {'Return%':>8} "
        f"{'Drawdown%':>10} {'N_trades':>9} {'WinRate':>8}"
    )
    print(header)
    _hr()
    for row in summary.benchmark_comparison:
        print(
            f"{row['config_name']:<40} "
            f"{row['ticker']:<6} "
            f"{row['total_return_pct']:>8.2f} "
            f"{row['max_drawdown_pct']:>10.2f} "
            f"{row['n_trades']:>9} "
            f"{row['win_rate']:>8.2f}"
        )
    _hr()
    print()

    # ------------------------------------------------------------------
    # Ablation summary
    # ------------------------------------------------------------------
    print("ABLATION SUMMARY")
    _hr()
    header = (
        f"{'Component removed':<20} {'Baseline%':>10} {'Ablated%':>10} "
        f"{'Delta%':>8} {'Contribution':>14}"
    )
    print(header)
    _hr()
    for row in summary.ablation_summary:
        print(
            f"{row['removed_component']:<20} "
            f"{row['baseline_return_pct']:>10.2f} "
            f"{row['ablated_return_pct']:>10.2f} "
            f"{row['return_delta_pct']:>8.2f} "
            f"{row['marginal_contribution']:>14}"
        )
    _hr()
    print()

    # ------------------------------------------------------------------
    # Robustness results
    # ------------------------------------------------------------------
    print("ROBUSTNESS RESULTS")
    _hr()
    header = (
        f"{'Test name':<38} {'Passed':>7} "
        f"{'Baseline%':>10} {'Stressed%':>10}"
    )
    print(header)
    _hr()
    for row in summary.robustness_summary:
        passed_str = "PASS" if row["passed"] else "FAIL"
        print(
            f"{row['test_name']:<38} "
            f"{passed_str:>7} "
            f"{row['baseline_return_pct']:>10.2f} "
            f"{row['stressed_return_pct']:>10.2f}"
        )
    _hr()
    print()

    # ------------------------------------------------------------------
    # Data split results
    # ------------------------------------------------------------------
    print("DATA SPLIT RESULTS")
    _hr()
    header = (
        f"{'Split':<14} {'N_bars':>7} {'N_trades':>9} "
        f"{'Return%':>8} {'WinRate':>8} {'Drawdown%':>10}"
    )
    print(header)
    _hr()
    for row in summary.data_split_summary:
        print(
            f"{row['split_name']:<14} "
            f"{row['n_bars']:>7} "
            f"{row['n_trades']:>9} "
            f"{row['total_return_pct']:>8.2f} "
            f"{row['win_rate']:>8.2f} "
            f"{row['max_drawdown_pct']:>10.2f}"
        )
    _hr()
    print()

    # ------------------------------------------------------------------
    # Pass/fail criteria
    # ------------------------------------------------------------------
    print("PASS / FAIL CRITERIA")
    _hr()
    pf = summary.pass_fail
    criteria = [
        ("full_system_beats_forecast_only",  pf.full_system_beats_forecast_only),
        ("full_system_reduces_drawdown",      pf.full_system_reduces_drawdown),
        ("consistent_across_tickers",         pf.consistent_across_tickers),
        ("survives_slippage",                 pf.survives_slippage),
        ("adaptive_does_not_degrade",         pf.adaptive_does_not_degrade),
    ]
    for name, value in criteria:
        result_str = "PASS" if value else "FAIL"
        print(f"  {name:<45} {result_str}")
    _hr()

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    if summary.notes:
        print("\nNOTES")
        for note in summary.notes:
            print(f"  - {note}")

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    print()
    if summary.overall_passed:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")

    return 0 if summary.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
