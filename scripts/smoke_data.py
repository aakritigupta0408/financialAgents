"""
Phase 2 smoke test — market-data layer.

Fetches 5m OHLCV for AAPL and prints a brief summary.
Run from the project root:

    python scripts/smoke_data.py

Requires ALPHA_VANTAGE_API_KEY to be set; if it is not, the script
prints a warning and exits gracefully.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the path so `config` and `src` import cleanly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)

log = structlog.get_logger("smoke_data")


def main() -> None:
    from src.data import get_provider, DataFetchError

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        print(
            "\nWARNING: ALPHA_VANTAGE_API_KEY is not set.\n"
            "  Live fetch will fail.  Set the env var and re-run.\n"
            "  Example:\n"
            "    export ALPHA_VANTAGE_API_KEY=your_key_here\n"
            "    python scripts/smoke_data.py\n"
        )
        sys.exit(0)

    provider = get_provider(api_key=api_key)
    ticker = "AAPL"
    timeframe = "5m"
    limit = 50

    print(f"\n{'='*60}")
    print(f"Phase 2 Smoke Test — market-data layer")
    print(f"Provider : {provider.name}")
    print(f"Ticker   : {ticker}")
    print(f"Timeframe: {timeframe}")
    print(f"Limit    : {limit} bars")
    print(f"{'='*60}\n")

    try:
        series = provider.fetch_ohlcv(ticker=ticker, timeframe=timeframe, limit=limit)
    except DataFetchError as exc:
        print(f"ERROR: DataFetchError — {exc}")
        sys.exit(1)

    print(f"OHLCVSeries summary")
    print(f"  ticker     : {series.ticker}")
    print(f"  timeframe  : {series.timeframe}")
    print(f"  bar count  : {len(series.bars)}")
    print(f"  fetched_at : {series.fetched_at}")

    if series.bars:
        first = series.bars[0]
        last = series.bars[-1]
        print(f"\n  First bar  : {first.timestamp}  O={first.open}  H={first.high}  L={first.low}  C={first.close}  V={first.volume}")
        print(f"  Last bar   : {last.timestamp}  O={last.open}  H={last.high}  L={last.low}  C={last.close}  V={last.volume}")
        print(f"\n  Latest close: {series.latest_close}")

        df = series.to_dataframe()
        print(f"\n  DataFrame shape : {df.shape}")
        print(f"  DataFrame dtypes:\n{df.dtypes.to_string()}")
        print(f"\n  Head:\n{df.head(3).to_string()}")
    else:
        print("\n  No bars returned (data may be outside market hours).")

    print(f"\n{'='*60}")
    print("Smoke test complete.\n")


if __name__ == "__main__":
    main()
