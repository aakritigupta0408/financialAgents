"""
scripts/smoke_live_loop.py — Smoke test for the Phase 8 live intraday loop.

Usage
-----
    cd /Users/aakritigupta/trading-system
    python scripts/smoke_live_loop.py

What it does
------------
1. Generates a 500-bar synthetic OHLCV series (AAPL, 1h, uptrend).
2. Runs the full live loop with FTA and meta-model enabled.
3. Prints result summary.
4. Prints decision log table for the first 15 bars after min_bars_required.
5. Prints first 3 and last 3 equity curve entries.
6. Prints full trade journal.
"""

from __future__ import annotations

import sys
import os

# Ensure repo root is on sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.backtest.data_utils import make_synthetic_ohlcv
from src.loop import LiveLoop, LoopConfig

# ---------------------------------------------------------------------------
# 1. Generate series
# ---------------------------------------------------------------------------
print("Generating 500-bar synthetic OHLCV series (AAPL, 1h, uptrend=0.0003)...")
series = make_synthetic_ohlcv(n_bars=500, seed=42, trend=0.0003, ticker="AAPL")
print(f"  Series: {len(series.bars)} bars, ticker={series.ticker}, timeframe={series.timeframe}")
print()

# ---------------------------------------------------------------------------
# 2. Build and run loop
# ---------------------------------------------------------------------------
cfg = LoopConfig(
    ticker="AAPL",
    fta_enabled=True,
    meta_model_enabled=True,
    verbose=True,
)
loop = LiveLoop(config=cfg)

print("Running live loop...")
print("-" * 60)
result = loop.run(series)
print("-" * 60)
print()

# ---------------------------------------------------------------------------
# 3. Print result summary
# ---------------------------------------------------------------------------
print(result.summary())
print()

# ---------------------------------------------------------------------------
# 4. Print decision log table for first 15 bars AFTER min_bars_required
# ---------------------------------------------------------------------------
min_bars = cfg.min_bars_required
post_warmup = [d for d in result.decision_log if d["bar_idx"] >= min_bars]
sample = post_warmup[:15]

print(f"Decision log — first 15 bars after min_bars_required={min_bars}:")
header = (
    f"{'bar_idx':>8}  {'fta_eval':>8}  {'fta_ok':>6}  "
    f"{'mm_prob':>8}  {'mm_ok':>6}  {'trade':>6}"
)
print(header)
print("-" * len(header))
for d in sample:
    fta_eval = str(d.get("fta_evaluated", ""))
    fta_ok = str(d.get("fta_accepted", ""))
    mm_prob_raw = d.get("meta_model_prob")
    mm_prob = f"{mm_prob_raw:.3f}" if mm_prob_raw is not None else "N/A"
    mm_ok = str(d.get("meta_model_accepted", ""))
    trade = str(d.get("trade_opened", ""))
    print(
        f"{d['bar_idx']:>8}  {fta_eval:>8}  {fta_ok:>6}  "
        f"{mm_prob:>8}  {mm_ok:>6}  {trade:>6}"
    )
print()

# ---------------------------------------------------------------------------
# 5. Equity curve: first 3 and last 3 entries
# ---------------------------------------------------------------------------
ec = result.equity_curve
print("Equity curve:")
if len(ec) >= 6:
    sample_ec = ec[:3] + [("...", "...")] + ec[-3:]
elif ec:
    sample_ec = ec
else:
    sample_ec = []

for entry in sample_ec:
    if entry[0] == "...":
        print("  ...")
    else:
        ts, eq = entry
        print(f"  {ts}  ${eq:>14,.2f}")
print()

# ---------------------------------------------------------------------------
# 6. Trade journal
# ---------------------------------------------------------------------------
print(f"Trade journal ({len(result.trade_journal)} trades):")
if not result.trade_journal:
    print("  (no trades opened — filters rejected all candidates)")
else:
    for i, trade in enumerate(result.trade_journal, 1):
        pnl = trade.get("realized_pnl", "N/A")
        pnl_str = f"${pnl:,.2f}" if isinstance(pnl, (int, float)) else str(pnl)
        meta_keys = list(trade.get("meta_features", {}).keys())
        print(
            f"  [{i:2d}] id={trade.get('trade_id','?'):<8}  "
            f"side={trade.get('side','?'):<5}  "
            f"entry={trade.get('entry_price', 0):.2f}  "
            f"exit={trade.get('exit_price', 0) or 0:.2f}  "
            f"pnl={pnl_str}  "
            f"reason={trade.get('exit_reason','?')}  "
            f"meta_feature_keys({len(meta_keys)})"
        )

print()
print("Smoke test complete.")
