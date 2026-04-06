"""
Logs every raw API/MCP pull to disk as JSONL.

Each record contains: provider, endpoint, ticker, timeframe,
request_params, fetched_at, response_status, row_count, error_message.
Does NOT log the raw payload (too large) — logs metadata only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import src.data_store.paths as _paths_module


class IngestLogger:
    """Append-only JSONL logger for data ingestion events."""

    def log(
        self,
        provider: str,
        endpoint: str,
        ticker: str,
        timeframe: str,
        request_params: dict,
        fetched_at: datetime,
        response_status: str,    # "ok", "error", "rate_limited", "cached"
        row_count: int = 0,
        error_message: str = "",
    ) -> None:
        """Append one JSON record to today's JSONL log file."""
        ingest_log_dir: Path = _paths_module.INGEST_LOG_DIR
        ingest_log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "provider": provider,
            "endpoint": endpoint,
            "ticker": ticker,
            "timeframe": timeframe,
            "request_params": request_params,
            "fetched_at": fetched_at.isoformat(),
            "response_status": response_status,
            "row_count": row_count,
            "error_message": error_message,
        }
        log_path = _paths_module.ingest_log_file(fetched_at.strftime("%Y-%m-%d"))
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def read_today(self) -> list[dict]:
        """Read today's log records."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = _paths_module.ingest_log_file(today)
        if not path.exists():
            return []
        records = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def read_date(self, date_str: str) -> list[dict]:
        """Read log records for a specific date (YYYY-MM-DD)."""
        path = _paths_module.ingest_log_file(date_str)
        if not path.exists():
            return []
        records = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
