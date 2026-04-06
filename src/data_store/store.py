"""
DataStore: read/write OHLCV data to partitioned local files.
Handles parquet (preferred) with CSV fallback.

Partition scheme: one file per calendar month (YYYY-MM).
Within each file, rows are sorted ascending by timestamp.
Deduplication: on upsert, rows with identical timestamp are deduplicated —
newer write wins (last occurrence kept after concat + drop_duplicates).
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from schemas.market_data import OHLCVBar, OHLCVSeries
import src.data_store.paths as _paths_module

# Module-level alias for monkeypatching in tests.
# DataStore._root reads from _paths_module.DATA_STORE_DIR at runtime, so
# patching _paths_module.DATA_STORE_DIR is sufficient for isolation.
# This attribute exists so the test fixture's monkeypatch.setattr succeeds.
DATA_STORE_DIR = _paths_module.DATA_STORE_DIR

# Detect parquet availability at import time.
try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except (ImportError, AttributeError, ValueError, Exception):
    _PARQUET_AVAILABLE = False

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


class StoredRange(NamedTuple):
    ticker: str
    timeframe: str
    first_ts: datetime
    last_ts: datetime
    row_count: int


class DataStore:
    """
    Local partitioned OHLCV store.

    One file per calendar month (YYYY-MM) per ticker/timeframe combination.
    Files are stored under DATA_STORE_DIR/{TICKER}/{timeframe}/YYYY-MM.{ext}.
    """

    def __init__(self, store_dir: Path | None = None):
        # Read from the module attribute at call time so monkeypatching works.
        self._store_dir = store_dir  # None means "use paths module at runtime"
        self._use_parquet = _PARQUET_AVAILABLE

    @property
    def _root(self) -> Path:
        if self._store_dir is not None:
            return self._store_dir
        return _paths_module.DATA_STORE_DIR

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert(self, series: OHLCVSeries) -> int:
        """
        Write bars from series to local store.
        Groups bars by calendar month (YYYY-MM), reads existing partition,
        merges, deduplicates on timestamp, sorts ascending, writes back.
        Returns total rows written (across all partitions).
        """
        if not series.bars:
            return 0
        df = self._series_to_df(series)
        return self.upsert_from_dataframe(df, series.ticker, series.timeframe)

    def upsert_from_dataframe(
        self, df: pd.DataFrame, ticker: str, timeframe: str
    ) -> int:
        """
        Same as upsert but accepts a DataFrame with columns matching _OHLCV_COLS.
        timestamp column must be timezone-aware datetime or parseable string.
        """
        if df.empty:
            return 0

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Group by YYYY-MM partition
        df["_ym"] = df["timestamp"].dt.strftime("%Y-%m")
        total_written = 0

        for ym, group in df.groupby("_ym"):
            group = group.drop(columns=["_ym"])

            # Read existing partition
            existing = self._read_partition(ticker, timeframe, str(ym))

            if existing.empty:
                merged = group
            else:
                merged = pd.concat([existing, group], ignore_index=True)

            # Deduplicate on timestamp — keep last (newer write wins)
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
            merged = merged.sort_values("timestamp").reset_index(drop=True)

            self._write_partition(merged, ticker, timeframe, str(ym))
            total_written += len(merged)

        return total_written

    # ── Read ─────────────────────────────────────────────────────────────────

    def load(
        self,
        ticker: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVSeries:
        """
        Load bars for ticker/timeframe from local store, optionally filtered by [start, end].
        Returns an OHLCVSeries. Returns empty OHLCVSeries (no bars) if no data found.
        """
        df = self.load_as_dataframe(ticker, timeframe, start, end)
        if df.empty:
            return OHLCVSeries(
                ticker=ticker,
                timeframe=timeframe,  # type: ignore[arg-type]
                bars=[],
                fetched_at=datetime.now(timezone.utc),
            )
        return self._df_to_series(df, ticker, timeframe)

    def load_as_dataframe(
        self,
        ticker: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load as DataFrame with _OHLCV_COLS columns, timestamp as UTC datetime."""
        ticker_dir = self._root / ticker.upper() / timeframe
        if not ticker_dir.exists():
            return pd.DataFrame(columns=_OHLCV_COLS)

        # Determine which partition files to read
        partition_files = self._list_partition_files(ticker, timeframe)
        if not partition_files:
            return pd.DataFrame(columns=_OHLCV_COLS)

        # Filter partitions by date range if requested
        if start is not None or end is not None:
            partition_files = self._filter_partitions_by_range(
                partition_files, start, end
            )

        frames: list[pd.DataFrame] = []
        for ym in partition_files:
            part = self._read_partition(ticker, timeframe, ym)
            if not part.empty:
                frames.append(part)

        if not frames:
            return pd.DataFrame(columns=_OHLCV_COLS)

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Apply time filters
        if start is not None:
            start_utc = _ensure_utc(start)
            df = df[df["timestamp"] >= start_utc]
        if end is not None:
            end_utc = _ensure_utc(end)
            df = df[df["timestamp"] <= end_utc]

        return df.reset_index(drop=True)

    # ── Inventory ─────────────────────────────────────────────────────────────

    def available_ranges(self, ticker: str, timeframe: str) -> list[StoredRange]:
        """
        Return list of StoredRange per partition file found for this ticker/timeframe.
        """
        partition_files = self._list_partition_files(ticker, timeframe)
        ranges: list[StoredRange] = []
        for ym in partition_files:
            part = self._read_partition(ticker, timeframe, ym)
            if part.empty:
                continue
            part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
            first_ts = part["timestamp"].min().to_pydatetime()
            last_ts = part["timestamp"].max().to_pydatetime()
            ranges.append(
                StoredRange(
                    ticker=ticker,
                    timeframe=timeframe,
                    first_ts=first_ts,
                    last_ts=last_ts,
                    row_count=len(part),
                )
            )
        return ranges

    def has_data(self, ticker: str, timeframe: str) -> bool:
        """True if any partition files exist for this ticker/timeframe."""
        return bool(self._list_partition_files(ticker, timeframe))

    def list_tickers(self, timeframe: str | None = None) -> list[str]:
        """Return list of tickers that have stored data, optionally filtered by timeframe."""
        root = self._root
        if not root.exists():
            return []
        tickers: list[str] = []
        for ticker_dir in sorted(root.iterdir()):
            if not ticker_dir.is_dir():
                continue
            if timeframe is not None:
                tf_dir = ticker_dir / timeframe
                if tf_dir.exists() and any(tf_dir.iterdir()):
                    tickers.append(ticker_dir.name)
            else:
                # Any timeframe directory with files
                has_any = False
                for tf_dir in ticker_dir.iterdir():
                    if tf_dir.is_dir() and any(tf_dir.iterdir()):
                        has_any = True
                        break
                if has_any:
                    tickers.append(ticker_dir.name)
        return tickers

    # ── Internals ─────────────────────────────────────────────────────────────

    def _partition_key(self, ts: datetime) -> str:
        """Return YYYY-MM string for a timestamp."""
        return ts.strftime("%Y-%m")

    def _list_partition_files(self, ticker: str, timeframe: str) -> list[str]:
        """Return sorted list of YYYY-MM partition keys that exist on disk."""
        ticker_dir = self._root / ticker.upper() / timeframe
        if not ticker_dir.exists():
            return []
        keys: list[str] = []
        for f in ticker_dir.iterdir():
            if f.suffix in (".parquet", ".csv") and len(f.stem) == 7:
                keys.append(f.stem)  # YYYY-MM
        return sorted(keys)

    def _filter_partitions_by_range(
        self,
        partition_keys: list[str],
        start: datetime | None,
        end: datetime | None,
    ) -> list[str]:
        """Keep only partition keys that could overlap [start, end]."""
        result: list[str] = []
        for ym in partition_keys:
            year, month = int(ym[:4]), int(ym[5:7])
            # Partition covers the entire month
            part_start_str = f"{year:04d}-{month:02d}-01"
            # End of month: next month minus one day
            if month == 12:
                part_end_str = f"{year + 1:04d}-01-31"
            else:
                part_end_str = f"{year:04d}-{month + 1:02d}-01"

            # Compare only year-month against requested range
            if start is not None:
                start_ym = start.strftime("%Y-%m")
                if ym < start_ym[:7]:
                    # This partition ends before start month
                    # Actually check: partition month < start month
                    if ym < start.strftime("%Y-%m"):
                        continue
            if end is not None:
                end_ym = end.strftime("%Y-%m")
                if ym > end.strftime("%Y-%m"):
                    continue
            result.append(ym)
        return result

    def _read_partition(
        self, ticker: str, timeframe: str, ym: str
    ) -> pd.DataFrame:
        """Read a single partition file. Returns empty DataFrame if not found."""
        # Try parquet first, then CSV
        if self._use_parquet:
            p_path = self._root / ticker.upper() / timeframe / f"{ym}.parquet"
            if p_path.exists():
                try:
                    df = pd.read_parquet(p_path)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    return df[_OHLCV_COLS]
                except Exception:
                    pass

        c_path = self._root / ticker.upper() / timeframe / f"{ym}.csv"
        if c_path.exists():
            try:
                df = pd.read_csv(c_path, parse_dates=False)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df[_OHLCV_COLS]
            except Exception:
                pass

        return pd.DataFrame(columns=_OHLCV_COLS)

    def _write_partition(
        self, df: pd.DataFrame, ticker: str, timeframe: str, ym: str
    ) -> None:
        """Write a partition file. Creates directories as needed."""
        part_dir = self._root / ticker.upper() / timeframe
        part_dir.mkdir(parents=True, exist_ok=True)

        # Ensure only the canonical columns are written, in order
        df = df[_OHLCV_COLS].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if self._use_parquet:
            path = part_dir / f"{ym}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = part_dir / f"{ym}.csv"
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            df.to_csv(path, index=False)

    def _df_to_series(
        self, df: pd.DataFrame, ticker: str, timeframe: str
    ) -> OHLCVSeries:
        """Convert DataFrame with _OHLCV_COLS to OHLCVSeries."""
        bars: list[OHLCVBar] = []
        for row in df.itertuples(index=False):
            ts = row.timestamp
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            lo = float(row.low)
            hi = float(row.high)
            # Clamp open/close to [low, high] to prevent Pydantic validation errors
            o = max(lo, min(hi, float(row.open)))
            c = max(lo, min(hi, float(row.close)))

            try:
                bar = OHLCVBar(
                    timestamp=ts,
                    open=o,
                    high=hi,
                    low=lo,
                    close=c,
                    volume=float(row.volume),
                    ticker=ticker,
                    timeframe=timeframe,  # type: ignore[arg-type]
                )
                bars.append(bar)
            except Exception:
                # Skip bars that fail validation
                continue

        return OHLCVSeries(
            ticker=ticker,
            timeframe=timeframe,  # type: ignore[arg-type]
            bars=bars,
            fetched_at=datetime.now(timezone.utc),
        )

    def _series_to_df(self, series: OHLCVSeries) -> pd.DataFrame:
        """Convert OHLCVSeries to DataFrame with _OHLCV_COLS."""
        rows = []
        for bar in series.bars:
            rows.append(
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )
        if not rows:
            return pd.DataFrame(columns=_OHLCV_COLS)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
