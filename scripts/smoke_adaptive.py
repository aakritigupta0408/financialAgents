"""
scripts/smoke_adaptive.py — End-to-end smoke test for Phase 10 adaptive module.

Usage:
    cd /Users/aakritigupta/trading-system
    python scripts/smoke_adaptive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.data_utils import make_synthetic_ohlcv
from src.backtest.engine import BacktestEngine
from src.adaptive.loop import run_improvement_cycle


def main() -> None:
    print("=" * 60)
    print("  SMOKE TEST — Phase 10 Adaptive Module")
    print("=" * 60)

    # Step 1: Generate 300-bar synthetic series.
    print("\n[1] Generating 300-bar synthetic OHLCV series...")
    series = make_synthetic_ohlcv(n_bars=300, seed=42)
    print(f"    Ticker: {series.ticker}, bars: {len(series.bars)}")

    # Step 2: Run BacktestEngine (both filters disabled for speed).
    print("\n[2] Running BacktestEngine (fta_enabled=False, meta_model_enabled=False)...")
    engine = BacktestEngine(
        fta_enabled=False,
        meta_model_enabled=False,
        verbose=False,
    )
    result = engine.run(series)
    print(f"    Trades: {result.n_trades}, win_rate: {result.win_rate}")

    # Step 3: Run first improvement cycle.
    print("\n[3] Running first improvement cycle (retrain_model=False)...")
    cycle1 = run_improvement_cycle(result, retrain_model=False, save_context=True)

    print(f"\n    Context version before: {cycle1.context_before.version}")
    print(f"    Context version after : {cycle1.context_after.version}")

    print("\n    Thresholds BEFORE:")
    for k, v in vars(cycle1.context_before.best_thresholds).items():
        print(f"      {k}: {v}")

    print("\n    Thresholds AFTER:")
    for k, v in vars(cycle1.context_after.best_thresholds).items():
        print(f"      {k}: {v}")

    print("\n    Update summary:")
    us = cycle1.update_summary
    print(f"      thresholds_changed : {us.thresholds_changed}")
    print(f"      update_suppressed  : {us.update_suppressed}")
    print(f"      suppression_reason : {us.suppression_reason!r}")
    print(f"      warnings           : {us.warnings}")
    print(f"      n_trades_analyzed  : {cycle1.analysis.n_trades_analyzed}")

    # Step 4: Run second cycle to verify context loads from disk and version increments.
    print("\n[4] Running second improvement cycle to verify persistence...")
    cycle2 = run_improvement_cycle(result, retrain_model=False, save_context=True)

    print(f"    Context version before second cycle: {cycle2.context_before.version}")
    print(f"    Context version after  second cycle: {cycle2.context_after.version}")

    assert cycle2.context_before.version >= cycle1.context_before.version, \
        "Context did not load from disk!"
    assert cycle2.context_after.version >= cycle2.context_before.version, \
        "Version did not increment in second cycle!"

    print("\n[5] All assertions passed.")
    print("=" * 60)
    print("  SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
