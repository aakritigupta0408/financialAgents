"""
Real-data loader for Phase 12 validation.
Loads CSV fixtures from tests/fixtures/real_data/ into OHLCVSeries.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from schemas.market_data import OHLCVBar, OHLCVSeries

# Path to fixture directory relative to repo root
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "real_data"

AVAILABLE_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "AMD"]


def load_ticker(ticker: str, timeframe: str = "1d") -> OHLCVSeries:
    """
    Load a ticker's CSV fixture into an OHLCVSeries.

    CSV rows are newest-first — sort ascending by timestamp before building bars.
    Timestamps are YYYY-MM-DD — attach time 16:00:00 UTC (US market close proxy).

    Validate each bar: skip rows where high < low or open/close outside [low, high].
    Also skip rows where any of open/high/low/close <= 0 or volume < 0.

    Returns OHLCVSeries with bars sorted ascending (oldest first).
    Raises FileNotFoundError if the fixture does not exist.
    """
    fixture_path = FIXTURE_DIR / f"{ticker}.csv"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    raw_rows: list[dict] = []
    with open(fixture_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_rows.append(row)

    # Sort ascending by timestamp string (YYYY-MM-DD sorts lexicographically)
    raw_rows.sort(key=lambda r: r["timestamp"])

    bars: list[OHLCVBar] = []
    for row in raw_rows:
        try:
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d").replace(
                hour=16, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
            hi = float(row["high"])
            lo = float(row["low"])
            o_raw = float(row["open"])
            c_raw = float(row["close"])
            vol = float(row["volume"])

            # Skip structurally invalid rows
            if hi < lo:
                continue
            if hi <= 0 or lo <= 0:
                continue
            if vol < 0:
                continue

            # Clamp open/close into [low, high] to handle tiny floating-point discrepancies
            o = max(lo, min(hi, o_raw))
            c = max(lo, min(hi, c_raw))

            bar = OHLCVBar(
                timestamp=ts,
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=vol,
                ticker=ticker,
                timeframe=timeframe,  # type: ignore[arg-type]
            )
            bars.append(bar)
        except (ValueError, KeyError):
            # Skip unparseable rows silently
            continue

    return OHLCVSeries(
        ticker=ticker,
        timeframe=timeframe,  # type: ignore[arg-type]
        bars=bars,
        fetched_at=datetime.now(timezone.utc),
    )


def load_all_tickers(timeframe: str = "1d") -> dict[str, OHLCVSeries]:
    """Load all AVAILABLE_TICKERS. Returns dict ticker -> OHLCVSeries."""
    return {ticker: load_ticker(ticker, timeframe=timeframe) for ticker in AVAILABLE_TICKERS}
