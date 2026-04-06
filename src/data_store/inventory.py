"""
DataInventory: SQLite-backed metadata store tracking what data is available,
when it was last fetched, and what gaps remain.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import src.data_store.paths as _paths_module

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS coverage (
    ticker         TEXT NOT NULL,
    timeframe      TEXT NOT NULL,
    first_ts       TEXT,
    last_ts        TEXT,
    row_count      INTEGER DEFAULT 0,
    last_fetch_at  TEXT,
    updated_at     TEXT,
    PRIMARY KEY (ticker, timeframe)
);
"""


@dataclass
class TickerCoverage:
    ticker: str
    timeframe: str
    first_ts: datetime | None
    last_ts: datetime | None
    row_count: int
    last_fetch_at: datetime | None
    is_fresh: bool
    freshness_hours: float


@dataclass
class GapSpec:
    ticker: str
    timeframe: str
    gap_start: datetime
    gap_end: datetime
    estimated_missing_bars: int


class DataInventory:
    """
    SQLite-backed inventory of stored data coverage.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        freshness_hours: float = 6.0,
    ):
        self._db_path = db_path  # None means use paths module at runtime
        self._freshness_hours = freshness_hours
        self._init_db()

    @property
    def _resolved_db_path(self) -> Path:
        if self._db_path is not None:
            return self._db_path
        return _paths_module.METADATA_DB_PATH

    def _connect(self) -> sqlite3.Connection:
        path = self._resolved_db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(path))

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def update(
        self,
        ticker: str,
        timeframe: str,
        first_ts: datetime | None,
        last_ts: datetime | None,
        row_count: int,
        last_fetch_at: datetime | None = None,
    ) -> None:
        """Upsert coverage record for ticker/timeframe."""
        now_iso = datetime.now(timezone.utc).isoformat()
        first_iso = first_ts.isoformat() if first_ts is not None else None
        last_iso = last_ts.isoformat() if last_ts is not None else None
        fetch_iso = last_fetch_at.isoformat() if last_fetch_at is not None else None

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coverage
                    (ticker, timeframe, first_ts, last_ts, row_count, last_fetch_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, timeframe) DO UPDATE SET
                    first_ts      = excluded.first_ts,
                    last_ts       = excluded.last_ts,
                    row_count     = excluded.row_count,
                    last_fetch_at = excluded.last_fetch_at,
                    updated_at    = excluded.updated_at
                """,
                (ticker, timeframe, first_iso, last_iso, row_count, fetch_iso, now_iso),
            )
            conn.commit()

    def get(self, ticker: str, timeframe: str) -> TickerCoverage | None:
        """Return coverage record or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ticker, timeframe, first_ts, last_ts, row_count, last_fetch_at "
                "FROM coverage WHERE ticker=? AND timeframe=?",
                (ticker, timeframe),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_coverage(row)

    def list_all(self) -> list[TickerCoverage]:
        """Return all coverage records."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, timeframe, first_ts, last_ts, row_count, last_fetch_at "
                "FROM coverage ORDER BY ticker, timeframe"
            ).fetchall()
        return [self._row_to_coverage(r) for r in rows]

    def detect_gaps(
        self,
        ticker: str,
        timeframe: str,
        desired_start: datetime,
        desired_end: datetime,
        bars_per_day: int = 7,
    ) -> list[GapSpec]:
        """
        Compare desired range against stored coverage.
        Returns list of GapSpec for missing periods.
        """
        coverage = self.get(ticker, timeframe)

        if coverage is None or coverage.first_ts is None or coverage.last_ts is None:
            # No data at all: one gap covering the full desired range
            return [
                GapSpec(
                    ticker=ticker,
                    timeframe=timeframe,
                    gap_start=desired_start,
                    gap_end=desired_end,
                    estimated_missing_bars=_estimate_bars(
                        desired_start, desired_end, bars_per_day
                    ),
                )
            ]

        first_ts = _ensure_utc(coverage.first_ts)
        last_ts = _ensure_utc(coverage.last_ts)
        desired_start = _ensure_utc(desired_start)
        desired_end = _ensure_utc(desired_end)

        # Normalize coverage boundaries to day granularity so that time-of-day
        # differences in bar timestamps (e.g. 16:00 close price) don't create
        # spurious sub-day gaps against a midnight desired_start/end.
        first_ts_day = first_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        last_ts_day = last_ts.replace(hour=23, minute=59, second=59, microsecond=0)

        gaps: list[GapSpec] = []

        # Gap before stored data
        if desired_start < first_ts_day:
            gap_end = min(first_ts_day, desired_end)
            estimated = _estimate_bars(desired_start, gap_end, bars_per_day)
            if estimated > 0:
                gaps.append(
                    GapSpec(
                        ticker=ticker,
                        timeframe=timeframe,
                        gap_start=desired_start,
                        gap_end=gap_end,
                        estimated_missing_bars=estimated,
                    )
                )

        # Gap after stored data
        if desired_end > last_ts_day:
            gap_start = max(last_ts_day, desired_start)
            estimated = _estimate_bars(gap_start, desired_end, bars_per_day)
            if estimated > 0:
                gaps.append(
                    GapSpec(
                        ticker=ticker,
                        timeframe=timeframe,
                        gap_start=gap_start,
                        gap_end=desired_end,
                        estimated_missing_bars=estimated,
                    )
                )

        return gaps

    def print_inventory(self) -> None:
        """Print a human-readable inventory table to stdout."""
        records = self.list_all()
        if not records:
            print("DataInventory: (empty)")
            return

        print(
            f"{'Ticker':<10} {'TF':<6} {'First':<22} {'Last':<22} "
            f"{'Rows':>7} {'Fresh':>7} {'LastFetch':<26}"
        )
        print("-" * 100)
        for r in records:
            first = r.first_ts.strftime("%Y-%m-%d %H:%M") if r.first_ts else "N/A"
            last = r.last_ts.strftime("%Y-%m-%d %H:%M") if r.last_ts else "N/A"
            fetch = r.last_fetch_at.strftime("%Y-%m-%d %H:%M") if r.last_fetch_at else "N/A"
            fresh = "yes" if r.is_fresh else "no"
            print(
                f"{r.ticker:<10} {r.timeframe:<6} {first:<22} {last:<22} "
                f"{r.row_count:>7} {fresh:>7} {fetch:<26}"
            )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _row_to_coverage(self, row: tuple) -> TickerCoverage:
        ticker, timeframe, first_iso, last_iso, row_count, fetch_iso = row
        first_ts = _parse_iso(first_iso)
        last_ts = _parse_iso(last_iso)
        last_fetch_at = _parse_iso(fetch_iso)

        # Compute is_fresh
        is_fresh = False
        if last_fetch_at is not None:
            age_hours = (
                datetime.now(timezone.utc) - _ensure_utc(last_fetch_at)
            ).total_seconds() / 3600.0
            is_fresh = age_hours <= self._freshness_hours

        return TickerCoverage(
            ticker=ticker,
            timeframe=timeframe,
            first_ts=first_ts,
            last_ts=last_ts,
            row_count=row_count or 0,
            last_fetch_at=last_fetch_at,
            is_fresh=is_fresh,
            freshness_hours=self._freshness_hours,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_iso(iso_str: str | None) -> datetime | None:
    if iso_str is None:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        return _ensure_utc(dt)
    except (ValueError, TypeError):
        return None


def _estimate_bars(start: datetime, end: datetime, bars_per_day: int) -> int:
    """Estimate number of bars between start and end, excluding weekends."""
    if end <= start:
        return 0
    delta = end - start
    total_days = delta.days + delta.seconds / 86400.0
    # Count weekdays
    trading_days = 0
    current = start
    while current < end:
        if current.weekday() < 5:  # 0=Mon ... 4=Fri
            trading_days += 1
        current += timedelta(days=1)
    return max(0, int(trading_days * bars_per_day))
