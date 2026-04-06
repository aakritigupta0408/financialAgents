"""
sync_from_fixtures(): one-time migration of Phase 12 daily fixture CSVs
into the local data store. Used to seed the store with the real data we have.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_store.store import DataStore
from src.data_store.inventory import DataInventory
from src.data_store.ingest_logger import IngestLogger
from src.validation.real_data import AVAILABLE_TICKERS, load_ticker, FIXTURE_DIR


def sync_from_fixtures(
    store: DataStore | None = None,
    inventory: DataInventory | None = None,
    logger: IngestLogger | None = None,
    tickers: list[str] | None = None,
) -> dict[str, int]:
    """
    Load all Phase 12 daily fixture CSVs (tests/fixtures/real_data/*.csv)
    into the local data store. Returns dict: ticker -> rows_written.

    This seeds the store so it has the 6 tickers x ~100 daily bars we already have.
    """
    store = store or DataStore()
    inventory = inventory or DataInventory()
    logger = logger or IngestLogger()
    tickers_to_sync = tickers or AVAILABLE_TICKERS

    results: dict[str, int] = {}
    for ticker in tickers_to_sync:
        fetched_at = datetime.now(timezone.utc)
        try:
            series = load_ticker(ticker, timeframe="1d")
            rows_written = 0
            if series.bars:
                rows_written = store.upsert(series)
                # Update inventory
                ranges = store.available_ranges(ticker, "1d")
                if ranges:
                    first_ts = min(r.first_ts for r in ranges)
                    last_ts = max(r.last_ts for r in ranges)
                    row_count = sum(r.row_count for r in ranges)
                    inventory.update(
                        ticker=ticker,
                        timeframe="1d",
                        first_ts=first_ts,
                        last_ts=last_ts,
                        row_count=row_count,
                        last_fetch_at=fetched_at,
                    )
            logger.log(
                provider="sync_from_fixtures",
                endpoint="fixture_csv",
                ticker=ticker,
                timeframe="1d",
                request_params={"fixture_dir": str(FIXTURE_DIR)},
                fetched_at=fetched_at,
                response_status="ok",
                row_count=rows_written,
            )
            results[ticker] = rows_written
        except Exception as exc:
            logger.log(
                provider="sync_from_fixtures",
                endpoint="fixture_csv",
                ticker=ticker,
                timeframe="1d",
                request_params={"fixture_dir": str(FIXTURE_DIR)},
                fetched_at=fetched_at,
                response_status="error",
                row_count=0,
                error_message=str(exc),
            )
            results[ticker] = 0

    return results


def sync_from_csv(
    csv_path: Path,
    ticker: str,
    timeframe: str,
    store: DataStore | None = None,
    inventory: DataInventory | None = None,
    logger: IngestLogger | None = None,
) -> int:
    """
    Load an arbitrary OHLCV CSV file (timestamp,open,high,low,close,volume)
    into the local store. Returns rows written.
    """
    store = store or DataStore()
    inventory = inventory or DataInventory()
    logger = logger or IngestLogger()

    fetched_at = datetime.now(timezone.utc)
    try:
        df = pd.read_csv(csv_path)
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        rows_written = store.upsert_from_dataframe(df, ticker, timeframe)

        # Update inventory
        ranges = store.available_ranges(ticker, timeframe)
        if ranges:
            first_ts = min(r.first_ts for r in ranges)
            last_ts = max(r.last_ts for r in ranges)
            row_count = sum(r.row_count for r in ranges)
            inventory.update(
                ticker=ticker,
                timeframe=timeframe,
                first_ts=first_ts,
                last_ts=last_ts,
                row_count=row_count,
                last_fetch_at=fetched_at,
            )

        logger.log(
            provider="sync_from_csv",
            endpoint="csv_file",
            ticker=ticker,
            timeframe=timeframe,
            request_params={"csv_path": str(csv_path)},
            fetched_at=fetched_at,
            response_status="ok",
            row_count=rows_written,
        )
        return rows_written

    except Exception as exc:
        logger.log(
            provider="sync_from_csv",
            endpoint="csv_file",
            ticker=ticker,
            timeframe=timeframe,
            request_params={"csv_path": str(csv_path)},
            fetched_at=fetched_at,
            response_status="error",
            row_count=0,
            error_message=str(exc),
        )
        return 0


def get_alpha_vantage_fetch_fn(api_key: str | None = None):
    """
    Return a FetchFn that calls AlphaVantageProvider.fetch_ohlcv().
    Returns None if no API key is available (ALPHA_VANTAGE_API_KEY env var).
    Used by LocalFirstRetrieval to fill gaps from real API when available.
    """
    import os
    from schemas.market_data import OHLCVSeries as _OHLCVSeries

    key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        return None

    def _fetch(
        ticker: str,
        timeframe: str,
        start: datetime | None,
        end: datetime | None,
    ) -> _OHLCVSeries:
        from src.data.alpha_vantage import AlphaVantageProvider
        provider = AlphaVantageProvider(api_key=key)
        return provider.fetch_ohlcv(ticker, timeframe)

    return _fetch
