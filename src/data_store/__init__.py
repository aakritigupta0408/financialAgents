"""
src.data_store — local-first OHLCV data store for the paper-trading system.

Provides:
- DataStore: partitioned on-disk OHLCV storage (CSV or parquet)
- IngestLogger: JSONL audit log of every data ingestion event
- DataInventory: SQLite-backed metadata tracking coverage and freshness
- LocalFirstRetrieval: gap-aware retrieval layer
- ReplayLoader: unified loader for backtest/live-paper replay
- get_data_store: singleton DataStore factory

Usage
-----
    from src.data_store import DataStore, ReplayLoader, get_data_store

    store = get_data_store()
    loader = ReplayLoader(store=store)
    series = loader.load("AAPL", timeframe="1d")
"""
from __future__ import annotations

from src.data_store.store import DataStore
from src.data_store.ingest_logger import IngestLogger
from src.data_store.inventory import DataInventory
from src.data_store.retrieval import LocalFirstRetrieval
from src.data_store.replay import ReplayLoader

__all__ = [
    "DataStore",
    "IngestLogger",
    "DataInventory",
    "LocalFirstRetrieval",
    "ReplayLoader",
    "get_data_store",
]

_store_singleton: DataStore | None = None


def get_data_store(force_new: bool = False) -> DataStore:
    """
    Return a module-level singleton DataStore.

    Parameters
    ----------
    force_new : bool
        If True, discard the singleton and create a fresh store.
    """
    global _store_singleton
    if _store_singleton is None or force_new:
        _store_singleton = DataStore()
    return _store_singleton
