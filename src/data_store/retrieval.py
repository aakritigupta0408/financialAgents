"""
Local-first retrieval layer.
Checks local store first, detects gaps, fetches only missing data,
upserts, updates inventory, returns merged clean OHLCVSeries.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from schemas.market_data import OHLCVSeries
from src.data_store.store import DataStore
from src.data_store.inventory import DataInventory
from src.data_store.ingest_logger import IngestLogger

# Type for a fetch function: (ticker, timeframe, start, end) -> OHLCVSeries
FetchFn = Callable[[str, str, datetime | None, datetime | None], OHLCVSeries]


class LocalFirstRetrieval:
    """
    Retrieval layer that prioritizes local store.

    get() flow:
    1. Check local store.
    2. Detect gaps vs requested range.
    3. For each gap: call fetch_fn(ticker, timeframe, gap_start, gap_end).
    4. Upsert fetched data into store.
    5. Update inventory.
    6. Return merged data from store for full requested range.

    If fetch_fn is None or raises, log the error and return whatever is in local store.
    """

    def __init__(
        self,
        store: DataStore | None = None,
        inventory: DataInventory | None = None,
        logger: IngestLogger | None = None,
        fetch_fn: FetchFn | None = None,
        freshness_hours: float = 6.0,
    ):
        self._store = store or DataStore()
        self._inventory = inventory or DataInventory()
        self._logger = logger or IngestLogger()
        self._fetch_fn = fetch_fn
        self._freshness_hours = freshness_hours

    def get(
        self,
        ticker: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVSeries:
        """
        Local-first get. Fills gaps from fetch_fn if available.
        """
        # Always attempt to fill gaps if we have a fetch_fn and a desired range
        if self._fetch_fn is not None and start is not None and end is not None:
            self._fill_gaps(ticker, timeframe, start, end)

        return self._store.load(ticker, timeframe, start, end)

    def refresh(
        self,
        ticker: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVSeries:
        """
        Force a fresh fetch regardless of local store state.
        """
        if self._fetch_fn is None:
            return self._store.load(ticker, timeframe, start, end)

        fetched_at = datetime.now(timezone.utc)
        try:
            series = self._fetch_fn(ticker, timeframe, start, end)
            row_count = 0
            if series.bars:
                row_count = self._store.upsert(series)
                self._update_inventory(ticker, timeframe, fetched_at)
            self._logger.log(
                provider="refresh",
                endpoint="refresh",
                ticker=ticker,
                timeframe=timeframe,
                request_params={"start": str(start), "end": str(end)},
                fetched_at=fetched_at,
                response_status="ok",
                row_count=row_count,
            )
        except Exception as exc:
            self._logger.log(
                provider="refresh",
                endpoint="refresh",
                ticker=ticker,
                timeframe=timeframe,
                request_params={"start": str(start), "end": str(end)},
                fetched_at=fetched_at,
                response_status="error",
                row_count=0,
                error_message=str(exc),
            )

        return self._store.load(ticker, timeframe, start, end)

    def bulk_sync(
        self,
        tickers: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, OHLCVSeries]:
        """
        Sync multiple tickers. Returns dict ticker -> OHLCVSeries.
        Skips tickers where fetch fails, logs errors.
        """
        result: dict[str, OHLCVSeries] = {}
        for ticker in tickers:
            try:
                result[ticker] = self.get(ticker, timeframe, start, end)
            except Exception as exc:
                self._logger.log(
                    provider="bulk_sync",
                    endpoint="bulk_sync",
                    ticker=ticker,
                    timeframe=timeframe,
                    request_params={"start": str(start), "end": str(end)},
                    fetched_at=datetime.now(timezone.utc),
                    response_status="error",
                    row_count=0,
                    error_message=str(exc),
                )
                # Return empty series for failed tickers
                result[ticker] = OHLCVSeries(
                    ticker=ticker,
                    timeframe=timeframe,  # type: ignore[arg-type]
                    bars=[],
                    fetched_at=datetime.now(timezone.utc),
                )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fill_gaps(
        self,
        ticker: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """Detect and fill gaps for the requested range."""
        gaps = self._inventory.detect_gaps(ticker, timeframe, start, end)
        for gap in gaps:
            if gap.estimated_missing_bars <= 0:
                continue
            fetched_at = datetime.now(timezone.utc)
            try:
                series = self._fetch_fn(  # type: ignore[misc]
                    ticker, timeframe, gap.gap_start, gap.gap_end
                )
                row_count = 0
                if series.bars:
                    row_count = self._store.upsert(series)
                    self._update_inventory(ticker, timeframe, fetched_at)
                self._logger.log(
                    provider="local_first_retrieval",
                    endpoint="gap_fill",
                    ticker=ticker,
                    timeframe=timeframe,
                    request_params={
                        "gap_start": gap.gap_start.isoformat(),
                        "gap_end": gap.gap_end.isoformat(),
                    },
                    fetched_at=fetched_at,
                    response_status="ok",
                    row_count=row_count,
                )
            except Exception as exc:
                self._logger.log(
                    provider="local_first_retrieval",
                    endpoint="gap_fill",
                    ticker=ticker,
                    timeframe=timeframe,
                    request_params={
                        "gap_start": gap.gap_start.isoformat(),
                        "gap_end": gap.gap_end.isoformat(),
                    },
                    fetched_at=fetched_at,
                    response_status="error",
                    row_count=0,
                    error_message=str(exc),
                )

    def _update_inventory(
        self,
        ticker: str,
        timeframe: str,
        fetched_at: datetime,
    ) -> None:
        """Sync inventory from store after an upsert."""
        ranges = self._store.available_ranges(ticker, timeframe)
        if not ranges:
            return
        first_ts = min(r.first_ts for r in ranges)
        last_ts = max(r.last_ts for r in ranges)
        row_count = sum(r.row_count for r in ranges)
        self._inventory.update(
            ticker=ticker,
            timeframe=timeframe,
            first_ts=first_ts,
            last_ts=last_ts,
            row_count=row_count,
            last_fetch_at=fetched_at,
        )
