#!/usr/bin/env python
"""
scripts/run_intraday.py — Intraday recommendation loop CLI.

Usage examples
──────────────
  # Replay mode (processes stored daily data)
  python scripts/run_intraday.py --ticker AAPL --replay

  # Live mode (polls at 1h intervals during market hours)
  python scripts/run_intraday.py --ticker AAPL,MSFT,NVDA --live --interval 3600

  # With a specific risk profile
  python scripts/run_intraday.py --ticker AAPL --replay --risk conservative

  # Verbose output
  python scripts/run_intraday.py --ticker AAPL --replay --verbose
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import SCAN_TICKERS
from src.loop.config import LoopConfig
from src.loop.intraday import IntradayRecommendationLoop
from src.risk_appetite.loader import load_risk_appetite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("run_intraday")


def _load_series(ticker: str):
    """Load OHLCVSeries from local data store or fixture fallback."""
    try:
        from src.data_store.replay import ReplayLoader
        loader = ReplayLoader()
        series = loader.load(ticker=ticker, timeframe="1d")
        if series and len(series.bars) >= 20:
            return series
    except Exception as e:
        log.warning("data_store.load_failed for %s: %s — trying fixture", ticker, e)

    # Fixture fallback
    from src.data_store.sync import sync_from_fixtures
    sync_from_fixtures()
    try:
        from src.data_store.replay import ReplayLoader
        loader = ReplayLoader()
        return loader.load(ticker=ticker, timeframe="1d")
    except Exception as e:
        log.error("fixture.load_failed for %s: %s", ticker, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday recommendation loop")
    parser.add_argument("--ticker", default="AAPL",
                        help="Comma-separated ticker(s) or 'all' for SCAN_TICKERS")
    parser.add_argument("--replay", action="store_true", default=True,
                        help="Run in replay mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Run in live polling mode")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Polling interval in seconds (live mode only)")
    parser.add_argument("--risk", default=None,
                        choices=["conservative", "moderate", "aggressive", "custom"],
                        help="Risk appetite profile (default: from settings)")
    parser.add_argument("--fta", action="store_true", default=True,
                        help="Enable FTA filter")
    parser.add_argument("--no-fta", dest="fta", action="store_false")
    parser.add_argument("--meta", action="store_true", default=True,
                        help="Enable meta-model filter")
    parser.add_argument("--no-meta", dest="meta", action="store_false")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tickers = (
        SCAN_TICKERS if args.ticker == "all"
        else [t.strip().upper() for t in args.ticker.split(",")]
    )

    ra = load_risk_appetite(args.risk)
    loop_config = LoopConfig(
        fta_enabled=args.fta,
        meta_model_enabled=args.meta,
        verbose=args.verbose,
        eod_retrain=False,
    )

    loop = IntradayRecommendationLoop(
        loop_config=loop_config,
        risk_appetite=ra,
        verbose=args.verbose,
    )

    if args.live:
        log.info("Starting LIVE mode: tickers=%s interval=%ds", tickers, args.interval)
        loop.run_live(
            fetch_fn=_load_series,
            tickers=tickers,
            interval_seconds=args.interval,
        )
    else:
        log.info("Starting REPLAY mode: tickers=%s", tickers)
        for ticker in tickers:
            series = _load_series(ticker)
            if series is None:
                log.warning("Skipping %s — no data", ticker)
                continue
            loop.cfg.ticker = ticker
            result, recs = loop.run_replay(series)
            print(result.summary())
            if recs:
                latest = recs[-1]
                print(f"\n  Latest recommendation for {ticker}:")
                from src.recommendation.reporter import format_recommendation
                print(format_recommendation(latest))


if __name__ == "__main__":
    main()
