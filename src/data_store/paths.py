"""
Canonical path definitions for the local data store.
All other modules import from here so paths are changed in one place.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_STORE_DIR   = _REPO_ROOT / "data" / "store"
INGEST_LOG_DIR   = _REPO_ROOT / "data" / "ingest_log"
METADATA_DB_PATH = _REPO_ROOT / "data" / "metadata.sqlite"


def ohlcv_dir(ticker: str, timeframe: str) -> Path:
    """Return the directory for a ticker/timeframe partition."""
    return DATA_STORE_DIR / ticker.upper() / timeframe


def ohlcv_file(ticker: str, timeframe: str, year_month: str, use_parquet: bool = True) -> Path:
    """Return path for a YYYY-MM partition file."""
    ext = "parquet" if use_parquet else "csv"
    return ohlcv_dir(ticker, timeframe) / f"{year_month}.{ext}"


def ingest_log_file(date_str: str) -> Path:
    """Return path for today's ingest log JSONL file (YYYY-MM-DD)."""
    return INGEST_LOG_DIR / f"{date_str}.jsonl"
