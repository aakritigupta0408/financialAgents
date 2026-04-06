#!/usr/bin/env python
"""
scripts/run_eod.py — End-of-day learning cycle CLI.

Usage examples
──────────────
  # Run EOD for today
  python scripts/run_eod.py

  # Run EOD for a specific date
  python scripts/run_eod.py --date 2026-04-06

  # Run for specific tickers only
  python scripts/run_eod.py --ticker AAPL,MSFT

  # Save retrained model
  python scripts/run_eod.py --save-model --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("run_eod")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-of-day learning cycle")
    parser.add_argument("--date", default=None,
                        help="Session date YYYY-MM-DD (default: today)")
    parser.add_argument("--ticker", default=None,
                        help="Comma-separated ticker(s) (default: all SCAN_TICKERS)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Persist retrained meta-model")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    session_date = date.fromisoformat(args.date) if args.date else None
    tickers = (
        [t.strip().upper() for t in args.ticker.split(",")]
        if args.ticker else None
    )

    from src.loop.eod import run_eod_cycle
    summary = run_eod_cycle(
        session_date=session_date,
        tickers=tickers,
        save_model=args.save_model,
        verbose=args.verbose,
    )

    print(f"\nEOD Cycle Summary — {summary['session_date']}")
    print(f"  Tickers processed: {summary['tickers_processed']}")
    for r in summary["results"]:
        if r["status"] == "ok":
            print(f"  {r['ticker']}: {r['n_trades']} trades, "
                  f"pnl=${r['daily_pnl']:+,.2f}, "
                  f"retrained={r['model_retrained']}, improved={r['model_improved']}")
        else:
            print(f"  {r['ticker']}: ERROR — {r.get('error', '?')}")


if __name__ == "__main__":
    main()
