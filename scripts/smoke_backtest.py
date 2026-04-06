"""
scripts/smoke_backtest.py — Phase 6 smoke test for the historical backtest engine.

Run:
    cd /Users/aakritigupta/trading-system && python scripts/smoke_backtest.py

Steps
-----
1. Generate a 500-bar synthetic 1h series (uptrend, seed=42).
2. Run BacktestEngine with verbose=True.
3. Print BacktestResult.summary().
4. Print first 5 and last 5 equity curve entries.
5. Print the full trade journal.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import BacktestEngine, make_synthetic_ohlcv

# ── 1. Generate synthetic series ──────────────────────────────────────────────

print("=" * 60)
print("  Generating 500-bar synthetic 1h series (uptrend, seed=42)")
print("=" * 60)

series = make_synthetic_ohlcv(
    n_bars=500,
    start_price=100.0,
    ticker="SYN",
    timeframe="1h",
    seed=42,
    trend=0.0003,
    volatility=0.008,
)

print(f"  Series: {series.ticker} / {series.timeframe} — {len(series.bars)} bars")
print(f"  First bar: {series.bars[0].timestamp}  close={series.bars[0].close:.4f}")
print(f"  Last  bar: {series.bars[-1].timestamp}  close={series.bars[-1].close:.4f}")
print()

# ── 2. Run backtest ───────────────────────────────────────────────────────────

print("=" * 60)
print("  Running BacktestEngine (verbose=True)...")
print("=" * 60)

engine = BacktestEngine(
    starting_capital=100_000,
    verbose=True,
)

result = engine.run(series)
print()

# ── 3. Print summary ──────────────────────────────────────────────────────────

print(result.summary())
print()

# ── 4. Equity curve (first 5 and last 5) ─────────────────────────────────────

print("=" * 60)
print("  EQUITY CURVE  (first 5 entries)")
print("=" * 60)
for ts, eq in result.equity_curve[:5]:
    print(f"    {ts.isoformat()}  equity={eq:>12,.2f}")

print()
print("  EQUITY CURVE  (last 5 entries)")
print("  ...")
for ts, eq in result.equity_curve[-5:]:
    print(f"    {ts.isoformat()}  equity={eq:>12,.2f}")
print()

# ── 5. Trade journal ──────────────────────────────────────────────────────────

print("=" * 60)
print(f"  TRADE JOURNAL  ({len(result.trade_journal)} trades)")
print("=" * 60)

if not result.trade_journal:
    print("  No trades executed.")
else:
    fmt = (
        "  {trade_id}  {ticker}  {side:4s}  qty={quantity:6.0f}"
        "  entry={entry_price:8.4f}  exit={exit_price}  pnl={realized_pnl:>9.2f}"
        "  reason={exit_reason}"
    )
    for t in result.trade_journal:
        exit_fmt = f"{t['exit_price']:8.4f}" if t["exit_price"] is not None else "      None"
        print(
            f"  {t['trade_id']}  {t['ticker']}  {t['side']:4s}"
            f"  qty={t['quantity']:6.0f}"
            f"  entry={t['entry_price']:8.4f}"
            f"  exit={exit_fmt}"
            f"  pnl={t['realized_pnl']:>9.2f}"
            f"  reason={t['exit_reason']}"
        )

print()
print("Smoke test complete.")
