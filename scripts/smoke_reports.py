"""
scripts/smoke_reports.py — Smoke test for the Phase 9 reporting layer.

Usage
-----
    cd /Users/aakritigupta/trading-system
    python scripts/smoke_reports.py

What it does
------------
1. Generates a synthetic OHLCV series with 200 bars.
2. Runs BacktestEngine with verbose=False.
3. Calls generate_full_report(result, output_dir=tmp_path).
4. Prints portfolio summary.
5. Prints first 3 trade diagnostics.
6. Prints confidence sweep table.
7. Asserts charts were written.
"""

import sys
import tempfile
from pathlib import Path

# Ensure the repo root is on the path regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtest.data_utils import make_synthetic_ohlcv
from src.backtest.engine import BacktestEngine
from src.reports import generate_full_report


def main() -> None:
    print("=" * 60)
    print("  SMOKE TEST — Phase 9 Reporting Layer")
    print("=" * 60)

    # Step 1 & 2: Generate data and run backtest.
    print("\n[1/4] Generating 200-bar synthetic series and running backtest...")
    series = make_synthetic_ohlcv(n_bars=200, seed=99)
    engine = BacktestEngine(starting_capital=10_000.0, verbose=False)
    result = engine.run(series)
    print(f"      Backtest complete. n_trades={result.n_trades}, "
          f"total_return_pct={result.total_return_pct:.2f}%")

    # Step 3: Generate full report.
    print("\n[2/4] Generating full report...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = generate_full_report(result, output_dir=tmp_dir)

        # Step 4: Portfolio summary.
        print("\n--- PORTFOLIO SUMMARY ---")
        p = report["portfolio"]
        print(f"  Ticker            : {p['ticker']}")
        print(f"  Timeframe         : {p['timeframe']}")
        print(f"  Start / End       : {p['start_date']} -> {p['end_date']}")
        print(f"  Starting capital  : ${p['starting_capital']:,.2f}")
        print(f"  Final equity      : ${p['final_equity']:,.2f}")
        print(f"  Total return      : {p['total_return_pct']:.2f}%")
        print(f"  Realized PnL      : ${p['realized_pnl']:,.2f}")
        print(f"  N trades          : {p['n_trades']}")
        print(f"  Win rate          : {p['win_rate']}")
        print(f"  Max drawdown      : {p['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe ratio      : {p['sharpe_ratio']}")
        print(f"  Avg holding (hrs) : {p['avg_holding_bars']:.1f}")
        print(f"  Per-exit-reason   : {p['per_exit_reason_pnl']}")

        # Step 5: First 3 trade diagnostics.
        print("\n--- FIRST 3 TRADE DIAGNOSTICS ---")
        diags = report["trade_diagnostics"]
        if diags:
            for i, d in enumerate(diags[:3]):
                print(
                    f"  [{i+1}] {d['trade_id'][:8]}... "
                    f"outcome={d['outcome']} "
                    f"pnl={d['realized_pnl']:.2f} "
                    f"rr={d['reward_risk']:.2f} "
                    f"holding={d['holding_hours']:.1f}h "
                    f"reason={d['exit_reason']}"
                )
        else:
            print("  (no closed trades)")

        # Step 6: Confidence sweep table.
        print("\n--- CONFIDENCE SWEEP ---")
        conf_sweep = report["threshold_sensitivity"]["confidence_sweep"]
        print(f"  {'Threshold':>10}  {'N Trades':>8}  {'Win Rate':>8}  "
              f"{'Total PnL':>10}  {'Max DD%':>8}")
        for row in conf_sweep:
            print(
                f"  {row['threshold']:>10.2f}  {row['n_trades']:>8}  "
                f"{row['win_rate']:>8.2%}  {row['total_pnl']:>10.2f}  "
                f"{row['max_drawdown_pct']:>8.2f}"
            )

        # Step 7: Assert charts were written (when matplotlib is available).
        print("\n[3/4] Checking chart files...")
        charts = report["charts"]

        def _matplotlib_ok() -> bool:
            try:
                import matplotlib
                import matplotlib.pyplot  # noqa: F401
                return True
            except Exception:
                return False

        if _matplotlib_ok():
            required_charts = ["equity_curve.png", "drawdown_curve.png"]
            for chart_name in required_charts:
                found = any(Path(c).name == chart_name for c in charts)
                assert found, f"Expected chart not found: {chart_name}"
                print(f"  OK: {chart_name}")

            print(f"\n  Total charts written: {len(charts)}")
            for c in charts:
                print(f"    {c}")
        else:
            print("  (matplotlib not available or broken — charts skipped gracefully)")
            assert charts == [], f"Expected empty charts list, got {charts}"
            print("  OK: generate_charts returned empty list as expected")

        print("\n[4/4] Verifying report.json was written...")
        report_json = Path(tmp_dir) / "report.json"
        assert report_json.exists(), "report.json not found"
        print(f"  OK: report.json ({report_json.stat().st_size} bytes)")

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
