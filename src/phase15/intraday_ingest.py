"""
src.phase15.intraday_ingest — Populate the local data store with 1h synthetic data.

Primary:  structured synthetic 1h series clearly labelled "simulation" in inventory.
Secondary: real CSV import hook for future use.

Convention: simulation data uses last_fetch_at = datetime(2000, 1, 1, tzinfo=UTC)
as a sentinel so get_1h_inventory_summary() can distinguish simulated vs real data.
"""
from __future__ import annotations

import csv as _csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from schemas.market_data import OHLCVBar, OHLCVSeries
from src.data_store.ingest_logger import IngestLogger
from src.data_store.inventory import DataInventory
from src.data_store.store import DataStore
from src.validation.intraday_synthetic import TICKER_PROXIES, make_structured_1h_series

_UTC = timezone.utc

# Sentinel datetime that marks data as simulation (not real fetched data).
_SIMULATION_SENTINEL = datetime(2000, 1, 1, 0, 0, 0, tzinfo=_UTC)

_PROXY_MAP = {p["ticker"]: p for p in TICKER_PROXIES}


def populate_1h_store(
    store: DataStore | None = None,
    inventory: DataInventory | None = None,
    logger: IngestLogger | None = None,
    n_bars: int = 800,
    tickers: list[str] | None = None,
) -> dict[str, int]:
    """
    Generate structured synthetic 1h series for each ticker in TICKER_PROXIES
    and upsert into the local store.

    Returns dict: ticker_1h_name -> rows_written.
    Logs each ingest with provider="synthetic_1h", response_status="ok".
    Updates inventory with last_fetch_at=_SIMULATION_SENTINEL to mark as simulation.
    """
    store = store or DataStore()
    inventory = inventory or DataInventory()
    logger = logger or IngestLogger()

    if tickers is None:
        tickers = [p["ticker"] for p in TICKER_PROXIES]

    results: dict[str, int] = {}

    for ticker in tickers:
        proxy = _PROXY_MAP.get(ticker, {"seed": 42, "base_price": 200.0})
        fetched_at = datetime.now(_UTC)

        try:
            series = make_structured_1h_series(
                ticker=ticker,
                n_bars=n_bars,
                seed=proxy["seed"],
                base_price=proxy["base_price"],
            )

            rows_written = store.upsert(series)

            # Update inventory with simulation sentinel
            ranges = store.available_ranges(ticker, "1h")
            if ranges:
                first_ts = min(r.first_ts for r in ranges)
                last_ts = max(r.last_ts for r in ranges)
                row_count = sum(r.row_count for r in ranges)
                inventory.update(
                    ticker=ticker,
                    timeframe="1h",
                    first_ts=first_ts,
                    last_ts=last_ts,
                    row_count=row_count,
                    last_fetch_at=_SIMULATION_SENTINEL,  # simulation marker
                )

            logger.log(
                provider="synthetic_1h",
                endpoint="make_structured_1h_series",
                ticker=ticker,
                timeframe="1h",
                request_params={"n_bars": n_bars, "seed": proxy["seed"]},
                fetched_at=fetched_at,
                response_status="ok",
                row_count=rows_written,
            )

            results[ticker] = rows_written

        except Exception as exc:
            logger.log(
                provider="synthetic_1h",
                endpoint="make_structured_1h_series",
                ticker=ticker,
                timeframe="1h",
                request_params={"n_bars": n_bars},
                fetched_at=fetched_at,
                response_status="error",
                row_count=0,
                error_message=str(exc),
            )
            results[ticker] = 0

    return results


def import_real_1h_csv(
    csv_path: Path,
    ticker: str,
    store: DataStore | None = None,
    inventory: DataInventory | None = None,
    logger: IngestLogger | None = None,
) -> int:
    """
    Import a real 1h intraday CSV into the store.

    CSV must have columns: timestamp,open,high,low,close,volume
    timestamp must be parseable datetime (ISO or YYYY-MM-DD HH:MM:SS).
    Returns rows written.

    This is the hook for future real data ingestion.
    """
    store = store or DataStore()
    inventory = inventory or DataInventory()
    logger = logger or IngestLogger()

    fetched_at = datetime.now(_UTC)

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        # Coerce numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        if df.empty:
            raise ValueError("CSV parsed to empty DataFrame after cleaning")

        rows_written = store.upsert_from_dataframe(df, ticker, "1h")

        # Update inventory with real fetch timestamp (not sentinel)
        ranges = store.available_ranges(ticker, "1h")
        if ranges:
            first_ts = min(r.first_ts for r in ranges)
            last_ts = max(r.last_ts for r in ranges)
            row_count = sum(r.row_count for r in ranges)
            inventory.update(
                ticker=ticker,
                timeframe="1h",
                first_ts=first_ts,
                last_ts=last_ts,
                row_count=row_count,
                last_fetch_at=fetched_at,  # real fetch time, NOT sentinel
            )

        logger.log(
            provider="real_1h_csv",
            endpoint="csv_import",
            ticker=ticker,
            timeframe="1h",
            request_params={"csv_path": str(csv_path)},
            fetched_at=fetched_at,
            response_status="ok",
            row_count=rows_written,
        )

        return rows_written

    except Exception as exc:
        logger.log(
            provider="real_1h_csv",
            endpoint="csv_import",
            ticker=ticker,
            timeframe="1h",
            request_params={"csv_path": str(csv_path)},
            fetched_at=fetched_at,
            response_status="error",
            row_count=0,
            error_message=str(exc),
        )
        return 0


def get_1h_inventory_summary(
    inventory: DataInventory | None = None,
    tickers: list[str] | None = None,
) -> list[dict]:
    """
    Return list of dicts with coverage info for 1h tickers.

    Keys: ticker, timeframe, first_date, last_date, row_count, is_fresh, source.
    source="simulation" if last_fetch_at matches the sentinel year 2000,
    else source="real".
    """
    inventory = inventory or DataInventory()

    if tickers is None:
        tickers = [p["ticker"] for p in TICKER_PROXIES]

    summary: list[dict] = []

    for ticker in tickers:
        cov = inventory.get(ticker, "1h")
        if cov is None:
            summary.append({
                "ticker": ticker,
                "timeframe": "1h",
                "first_date": None,
                "last_date": None,
                "row_count": 0,
                "is_fresh": False,
                "source": "missing",
            })
            continue

        # Determine source by checking if last_fetch_at is the sentinel
        source = "missing"
        if cov.last_fetch_at is not None:
            # Sentinel: year 2000 marks simulation data
            if cov.last_fetch_at.year == 2000:
                source = "simulation"
            else:
                source = "real"

        summary.append({
            "ticker": ticker,
            "timeframe": "1h",
            "first_date": cov.first_ts.date() if cov.first_ts else None,
            "last_date": cov.last_ts.date() if cov.last_ts else None,
            "row_count": cov.row_count,
            "is_fresh": cov.is_fresh,
            "source": source,
        })

    return summary
