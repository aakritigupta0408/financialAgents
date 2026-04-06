"""
Phase 14: Local Data Store — 15 tests.

All tests use tmp_path to isolate filesystem state from the real data directory.
monkeypatch redirects DATA_STORE_DIR, INGEST_LOG_DIR, and METADATA_DB_PATH so
that no test writes to the production data/ directory.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from schemas.market_data import OHLCVBar, OHLCVSeries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    """Redirect all data_store paths to tmp_path so tests are isolated."""
    import src.data_store.paths as paths_mod
    import src.data_store.store as store_mod

    store_dir = tmp_path / "store"
    ingest_dir = tmp_path / "ingest_log"
    db_path = tmp_path / "metadata.sqlite"

    monkeypatch.setattr(paths_mod, "DATA_STORE_DIR", store_dir)
    monkeypatch.setattr(paths_mod, "INGEST_LOG_DIR", ingest_dir)
    monkeypatch.setattr(paths_mod, "METADATA_DB_PATH", db_path)
    monkeypatch.setattr(store_mod, "DATA_STORE_DIR", store_dir)


def _make_bar(
    ticker: str,
    timeframe: str,
    ts: datetime,
    price: float = 100.0,
    volume: float = 1_000_000.0,
) -> OHLCVBar:
    """Helper: build a valid OHLCVBar."""
    lo = price * 0.99
    hi = price * 1.01
    return OHLCVBar(
        timestamp=ts,
        open=price,
        high=hi,
        low=lo,
        close=price,
        volume=volume,
        ticker=ticker,
        timeframe=timeframe,  # type: ignore[arg-type]
    )


def _make_series(
    ticker: str,
    timeframe: str,
    n: int,
    base_ts: datetime | None = None,
    price: float = 100.0,
) -> OHLCVSeries:
    """Helper: build an OHLCVSeries with n bars, one per day."""
    if base_ts is None:
        base_ts = datetime(2025, 1, 2, 16, 0, 0, tzinfo=timezone.utc)
    bars = [
        _make_bar(ticker, timeframe, base_ts + timedelta(days=i), price=price + i)
        for i in range(n)
    ]
    return OHLCVSeries(
        ticker=ticker,
        timeframe=timeframe,  # type: ignore[arg-type]
        bars=bars,
        fetched_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataStore:

    def test_data_store_upsert_and_load(self, tmp_path):
        """T1: upsert OHLCVSeries, load it back, bars match."""
        from src.data_store.store import DataStore

        store = DataStore(store_dir=tmp_path / "store")
        series = _make_series("AAPL", "1d", 5)

        rows = store.upsert(series)
        assert rows > 0

        loaded = store.load("AAPL", "1d")
        assert len(loaded.bars) == 5
        assert loaded.ticker == "AAPL"
        assert loaded.timeframe == "1d"

        # Timestamps should match (modulo UTC normalization)
        orig_ts = sorted(b.timestamp for b in series.bars)
        loaded_ts = sorted(b.timestamp for b in loaded.bars)
        for a, b in zip(orig_ts, loaded_ts):
            assert a.replace(tzinfo=timezone.utc) == b.replace(tzinfo=timezone.utc) or a == b

    def test_data_store_deduplication(self, tmp_path):
        """T2: upsert same data twice, load returns deduplicated rows."""
        from src.data_store.store import DataStore

        store = DataStore(store_dir=tmp_path / "store")
        series = _make_series("MSFT", "1d", 10)

        store.upsert(series)
        store.upsert(series)  # second upsert of identical data

        loaded = store.load("MSFT", "1d")
        assert len(loaded.bars) == 10  # no duplicates

    def test_data_store_multi_partition(self, tmp_path):
        """T3: upsert bars spanning 3 months, all 3 partition files created."""
        from src.data_store.store import DataStore

        store = DataStore(store_dir=tmp_path / "store")

        # Create bars across Jan, Feb, Mar 2025
        bars = []
        for month in [1, 2, 3]:
            ts = datetime(2025, month, 15, 16, 0, 0, tzinfo=timezone.utc)
            bars.append(_make_bar("NVDA", "1d", ts))

        series = OHLCVSeries(
            ticker="NVDA",
            timeframe="1d",  # type: ignore[arg-type]
            bars=bars,
            fetched_at=datetime.now(timezone.utc),
        )
        store.upsert(series)

        # Check partition files exist
        nvda_dir = tmp_path / "store" / "NVDA" / "1d"
        assert nvda_dir.exists()
        files = list(nvda_dir.iterdir())
        assert len(files) == 3

        # Stems should be 2025-01, 2025-02, 2025-03
        stems = sorted(f.stem for f in files)
        assert stems == ["2025-01", "2025-02", "2025-03"]

    def test_data_store_range_filter(self, tmp_path):
        """T4: load with start/end returns only bars in range."""
        from src.data_store.store import DataStore

        store = DataStore(store_dir=tmp_path / "store")
        # 30 bars starting 2025-01-02
        series = _make_series("TSLA", "1d", 30)
        store.upsert(series)

        # Filter to first 10 days
        start = datetime(2025, 1, 2, tzinfo=timezone.utc)
        end = datetime(2025, 1, 11, tzinfo=timezone.utc)
        loaded = store.load("TSLA", "1d", start=start, end=end)
        assert 1 <= len(loaded.bars) <= 10
        for bar in loaded.bars:
            assert bar.timestamp >= start
            assert bar.timestamp <= end

    def test_data_store_empty_load(self, tmp_path):
        """T5: load for nonexistent ticker returns empty OHLCVSeries (0 bars)."""
        from src.data_store.store import DataStore

        store = DataStore(store_dir=tmp_path / "store")
        loaded = store.load("FAKE", "1d")
        assert len(loaded.bars) == 0
        assert loaded.ticker == "FAKE"


class TestIngestLogger:

    def test_ingest_logger_write_and_read(self, tmp_path):
        """T6: log a record, read_today() returns it."""
        from src.data_store.ingest_logger import IngestLogger

        logger = IngestLogger()
        ts = datetime.now(timezone.utc)
        logger.log(
            provider="test_provider",
            endpoint="test_endpoint",
            ticker="AAPL",
            timeframe="1d",
            request_params={"limit": 100},
            fetched_at=ts,
            response_status="ok",
            row_count=50,
        )

        records = logger.read_today()
        assert len(records) >= 1
        last = records[-1]
        assert last["provider"] == "test_provider"
        assert last["ticker"] == "AAPL"
        assert last["response_status"] == "ok"
        assert last["row_count"] == 50


class TestDataInventory:

    def test_inventory_update_and_get(self, tmp_path):
        """T7: update coverage, get() returns matching TickerCoverage."""
        from src.data_store.inventory import DataInventory

        inv = DataInventory(db_path=tmp_path / "meta.sqlite")
        first = datetime(2025, 1, 2, tzinfo=timezone.utc)
        last = datetime(2025, 3, 31, tzinfo=timezone.utc)

        inv.update("AAPL", "1d", first, last, row_count=88)
        coverage = inv.get("AAPL", "1d")

        assert coverage is not None
        assert coverage.ticker == "AAPL"
        assert coverage.timeframe == "1d"
        assert coverage.row_count == 88
        assert coverage.first_ts is not None
        assert coverage.last_ts is not None

    def test_inventory_detect_gaps_full(self, tmp_path):
        """T8: no data stored -> one gap covering full desired range."""
        from src.data_store.inventory import DataInventory

        inv = DataInventory(db_path=tmp_path / "meta.sqlite")
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 31, tzinfo=timezone.utc)

        gaps = inv.detect_gaps("AAPL", "1d", start, end)
        assert len(gaps) == 1
        assert gaps[0].gap_start == start
        assert gaps[0].gap_end == end
        assert gaps[0].estimated_missing_bars > 0

    def test_inventory_detect_gaps_partial(self, tmp_path):
        """T9: partial data -> gap detected for missing period."""
        from src.data_store.inventory import DataInventory

        inv = DataInventory(db_path=tmp_path / "meta.sqlite")
        # We have data from Feb 1 to Mar 31
        inv.update(
            "MSFT", "1d",
            first_ts=datetime(2025, 2, 1, tzinfo=timezone.utc),
            last_ts=datetime(2025, 3, 31, tzinfo=timezone.utc),
            row_count=40,
        )

        # Desired range is Jan 1 to Mar 31 — gap should be detected before Feb 1
        gaps = inv.detect_gaps(
            "MSFT", "1d",
            desired_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            desired_end=datetime(2025, 3, 31, tzinfo=timezone.utc),
        )
        assert len(gaps) >= 1
        # The gap before stored data should be detected
        pre_gaps = [g for g in gaps if g.gap_start == datetime(2025, 1, 1, tzinfo=timezone.utc)]
        assert len(pre_gaps) == 1
        assert pre_gaps[0].estimated_missing_bars > 0

    def test_inventory_list_all(self, tmp_path):
        """T10: list_all() returns records after updates."""
        from src.data_store.inventory import DataInventory

        inv = DataInventory(db_path=tmp_path / "meta.sqlite")
        inv.update("AAPL", "1d", datetime(2025, 1, 1, tzinfo=timezone.utc),
                   datetime(2025, 3, 31, tzinfo=timezone.utc), 88)
        inv.update("MSFT", "1h", datetime(2025, 1, 1, tzinfo=timezone.utc),
                   datetime(2025, 1, 31, tzinfo=timezone.utc), 154)

        records = inv.list_all()
        assert len(records) == 2
        tickers = {r.ticker for r in records}
        assert "AAPL" in tickers
        assert "MSFT" in tickers


class TestLocalFirstRetrieval:

    def test_local_first_retrieval_cache_hit(self, tmp_path):
        """T11: load data into store; get() returns from store without calling fetch_fn."""
        from src.data_store.store import DataStore
        from src.data_store.inventory import DataInventory
        from src.data_store.ingest_logger import IngestLogger
        from src.data_store.retrieval import LocalFirstRetrieval

        store = DataStore(store_dir=tmp_path / "store")
        inv = DataInventory(db_path=tmp_path / "meta.sqlite")
        logger = IngestLogger()

        series = _make_series("AAPL", "1d", 30)
        store.upsert(series)
        # Update inventory so gaps are not detected
        ranges = store.available_ranges("AAPL", "1d")
        inv.update(
            "AAPL", "1d",
            first_ts=min(r.first_ts for r in ranges),
            last_ts=max(r.last_ts for r in ranges),
            row_count=30,
        )

        fetch_called = []

        def _fake_fetch(ticker, timeframe, start, end):
            fetch_called.append(True)
            return OHLCVSeries(ticker=ticker, timeframe=timeframe, bars=[],  # type: ignore
                               fetched_at=datetime.now(timezone.utc))

        retrieval = LocalFirstRetrieval(
            store=store, inventory=inv, logger=logger, fetch_fn=_fake_fetch
        )

        # Request a range fully covered by stored data
        start = datetime(2025, 1, 2, tzinfo=timezone.utc)
        end = datetime(2025, 1, 15, tzinfo=timezone.utc)
        loaded = retrieval.get("AAPL", "1d", start=start, end=end)

        assert len(loaded.bars) > 0
        # fetch_fn should NOT have been called — data was already in store
        # (gaps detection: stored range covers desired range)
        assert len(fetch_called) == 0

    def test_local_first_retrieval_gap_fill(self, tmp_path):
        """T12: empty store + fetch_fn stub -> fetch_fn called, data stored."""
        from src.data_store.store import DataStore
        from src.data_store.inventory import DataInventory
        from src.data_store.ingest_logger import IngestLogger
        from src.data_store.retrieval import LocalFirstRetrieval

        store = DataStore(store_dir=tmp_path / "store")
        inv = DataInventory(db_path=tmp_path / "meta.sqlite")
        logger = IngestLogger()

        fetch_called = []
        fill_series = _make_series("SPY", "1d", 20)

        def _fake_fetch(ticker, timeframe, start, end):
            fetch_called.append((ticker, timeframe))
            return fill_series

        retrieval = LocalFirstRetrieval(
            store=store, inventory=inv, logger=logger, fetch_fn=_fake_fetch
        )

        start = datetime(2025, 1, 2, tzinfo=timezone.utc)
        end = datetime(2025, 1, 31, tzinfo=timezone.utc)
        loaded = retrieval.get("SPY", "1d", start=start, end=end)

        # fetch_fn should have been called (gap fill for empty store)
        assert len(fetch_called) >= 1
        assert fetch_called[0][0] == "SPY"

        # Store should now have data
        assert store.has_data("SPY", "1d")


class TestSyncFromFixtures:

    def test_sync_from_fixtures(self, tmp_path):
        """T13: sync_from_fixtures() writes >0 rows for AAPL."""
        from src.data_store.store import DataStore
        from src.data_store.inventory import DataInventory
        from src.data_store.ingest_logger import IngestLogger
        from src.data_store.sync import sync_from_fixtures

        store = DataStore(store_dir=tmp_path / "store")
        inv = DataInventory(db_path=tmp_path / "meta.sqlite")
        logger = IngestLogger()

        results = sync_from_fixtures(
            store=store, inventory=inv, logger=logger, tickers=["AAPL"]
        )
        assert "AAPL" in results
        assert results["AAPL"] > 0

        # Data should be loadable
        loaded = store.load("AAPL", "1d")
        assert len(loaded.bars) > 0


class TestReplayLoader:

    def test_replay_loader_local_store(self, tmp_path):
        """T14: upsert data, ReplayLoader.load() returns it from store."""
        from src.data_store.store import DataStore
        from src.data_store.replay import ReplayLoader

        store = DataStore(store_dir=tmp_path / "store")
        series = _make_series("NVDA", "1d", 120)
        store.upsert(series)

        loader = ReplayLoader(store=store)
        loaded = loader.load("NVDA", timeframe="1d", min_bars=100)

        assert loaded.ticker == "NVDA"
        assert len(loaded.bars) >= 100

    def test_replay_loader_fallback_synthetic(self, tmp_path):
        """T15: empty store + unknown ticker -> falls back to synthetic 1h series."""
        from src.data_store.store import DataStore
        from src.data_store.replay import ReplayLoader

        store = DataStore(store_dir=tmp_path / "store")
        loader = ReplayLoader(store=store)

        # "FAKE_TICKER" is not in fixtures or TICKER_PROXIES — must generate synthetic
        series = loader.load("FAKE_TICKER", timeframe="1h", min_bars=100)

        assert series.ticker == "FAKE_TICKER"
        assert series.timeframe == "1h"
        assert len(series.bars) >= 100
