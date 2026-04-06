"""
scripts/smoke_portfolio.py — Phase 5 smoke test for the portfolio engine.

Run:
    cd /Users/aakritigupta/trading-system && python scripts/smoke_portfolio.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

# Ensure project root is on sys.path when run directly
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.portfolio import create_portfolio

_UTC = timezone.utc


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main() -> None:
    # ── 1. Create portfolio ────────────────────────────────────────────────
    banner("1. Create portfolio (starting_capital=100_000, max_trades_per_day=2)")
    p = create_portfolio(starting_capital=100_000, max_trades_per_day=2)
    print(f"  cash:             {p.cash:,.2f}")
    print(f"  equity:           {p.equity:,.2f}")
    print(f"  starting_capital: {p.starting_capital:,.2f}")

    ts0 = datetime(2026, 1, 5, 9, 30, 0, tzinfo=_UTC)

    # ── 2. Open AAPL long ──────────────────────────────────────────────────
    banner("2. Open AAPL long @ 150.00  stop=147.00  target=156.00")
    aapl = p.open_trade(
        "AAPL", side="long",
        entry_price=150.00, stop_price=147.00, target_price=156.00,
        timestamp=ts0,
    )
    if aapl:
        print(f"  trade_id:  {aapl.trade_id}")
        print(f"  quantity:  {aapl.quantity}")
        print(f"  cost_basis:{aapl.cost_basis():,.2f}")
        print(f"  cash after open: {p.cash:,.2f}")
    else:
        print("  ERROR: AAPL trade was rejected — aborting smoke test")
        sys.exit(1)

    # ── 3. Open MSFT long ──────────────────────────────────────────────────
    banner("3. Open MSFT long @ 380.00  stop=375.00  target=390.00")
    msft = p.open_trade(
        "MSFT", side="long",
        entry_price=380.00, stop_price=375.00, target_price=390.00,
        timestamp=ts0,
    )
    if msft:
        print(f"  trade_id:  {msft.trade_id}")
        print(f"  quantity:  {msft.quantity}")
        print(f"  cost_basis:{msft.cost_basis():,.2f}")
        print(f"  cash after open: {p.cash:,.2f}")
    else:
        print("  ERROR: MSFT trade was rejected — aborting smoke test")
        sys.exit(1)

    # ── 4. Update: AAPL winning, MSFT near stop ────────────────────────────
    banner("4. update_positions  AAPL=153.00  MSFT=373.00")
    ts1 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
    closed4 = p.update_positions({"AAPL": 153.00, "MSFT": 373.00}, timestamp=ts1)
    print(f"  Trades auto-closed this update: {len(closed4)}")
    print(f"  AAPL unrealized_pnl: {aapl.unrealized_pnl(153.00):+,.2f}")
    print(f"  MSFT unrealized_pnl: {msft.unrealized_pnl(373.00):+,.2f}")
    print(f"  equity: {p.equity:,.2f}")

    # ── 5. Update: MSFT stop hit ───────────────────────────────────────────
    banner("5. update_positions  AAPL=153.00  MSFT=374.50  (MSFT stop hit)")
    ts2 = datetime(2026, 1, 5, 10, 30, 0, tzinfo=_UTC)
    closed5 = p.update_positions({"AAPL": 153.00, "MSFT": 374.50}, timestamp=ts2)
    print(f"  Trades auto-closed this update: {len(closed5)}")
    if closed5:
        for t in closed5:
            print(f"    closed: {t.ticker} status={t.status} exit={t.exit_price} "
                  f"pnl={t.realized_pnl():+,.2f}")
    print(f"  equity: {p.equity:,.2f}")
    print(f"  cash:   {p.cash:,.2f}")

    # ── 6. Close AAPL manually at 154.00 ──────────────────────────────────
    banner("6. Close AAPL manually @ 154.00")
    ts3 = datetime(2026, 1, 5, 11, 0, 0, tzinfo=_UTC)
    closed_aapl = p.close_trade(aapl.trade_id, exit_price=154.00, reason="manual", timestamp=ts3)
    if closed_aapl:
        print(f"  AAPL closed. status={closed_aapl.status}  "
              f"realized_pnl={closed_aapl.realized_pnl():+,.2f}")
    print(f"  equity: {p.equity:,.2f}")
    print(f"  cash:   {p.cash:,.2f}")

    # ── 7. Attempt a 3rd trade (should be rejected) ────────────────────────
    banner("7. Attempt 3rd trade (NVDA long @ 500)  — expect rejection")
    # Re-open portfolio's max_concurrent_positions=3 so let's exhaust daily count
    # All positions are now closed so the real rejection should be daily drawdown
    # or max_positions. Let's just try and show the reason.
    ok, reason = p.can_open_trade("NVDA", 500.0, 490.0)
    if not ok:
        print(f"  Rejected. Reason: {reason}")
    else:
        nvda = p.open_trade("NVDA", "long", 500.0, 490.0, timestamp=ts3)
        if nvda is None:
            print("  open_trade returned None (rejected internally)")
        else:
            print(f"  Opened (unexpected): {nvda.trade_id}")

    # ── 8. Print snapshot ─────────────────────────────────────────────────
    banner("8. portfolio_snapshot()")
    snap = p.portfolio_snapshot(timestamp=ts3)
    print(f"  snapshot_at:      {snap.snapshot_at}")
    print(f"  starting_capital: {snap.starting_capital:,.2f}")
    print(f"  cash:             {snap.cash:,.2f}")
    print(f"  equity:           {snap.equity:,.2f}")
    print(f"  realized_pnl:     {snap.realized_pnl:+,.2f}")
    print(f"  unrealized_pnl:   {snap.unrealized_pnl:+,.2f}")
    print(f"  open_positions:   {len(snap.open_positions)}")
    print(f"  closed_positions: {len(snap.closed_positions)}")
    print(f"  total_return_pct: {snap.total_return_pct:+.4f}%")

    # ── 9. Print metrics ──────────────────────────────────────────────────
    banner("9. get_metrics()")
    metrics = p.get_metrics()
    for k, v in metrics.items():
        if v is None:
            print(f"  {k:30s}: None")
        elif isinstance(v, float):
            print(f"  {k:30s}: {v:,.4f}")
        else:
            print(f"  {k:30s}: {v}")

    # ── 10. Trade journal ─────────────────────────────────────────────────
    banner("10. export_trade_journal()")
    journal = p.export_trade_journal()
    print(f"  Total closed trades in journal: {len(journal)}")
    for entry in journal:
        print(
            f"    [{entry['trade_id']}] {entry['ticker']:4s} {entry['side']:5s} "
            f"entry={entry['entry_price']:.2f}  exit={entry.get('exit_price')}  "
            f"pnl={entry['realized_pnl']:+.2f}  reason={entry['exit_reason']}"
        )

    print("\nSmoke test completed successfully.\n")


if __name__ == "__main__":
    main()
