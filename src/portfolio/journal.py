"""
src.portfolio.journal — Persistent trade journal.

Trades are stored as newline-delimited JSON (JSONL) — one JSON object per
line. This format is append-friendly and readable with standard tools
(grep, jq, pandas).

The default location is JOURNAL_DIR / "trades.jsonl" from config/settings.py.
Pass an explicit path to __init__ to override (useful in tests).
"""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import JOURNAL_DIR
from src.portfolio.trade import Trade


class TradeJournal:
    """
    Append-only, persistent journal for closed paper trades.

    Each entry is a single JSON line written by Trade.to_dict().
    Reading uses stdlib json exclusively — no external dependencies.

    Thread safety: not thread-safe. For single-threaded use only.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path: Path = path if path is not None else JOURNAL_DIR / "trades.jsonl"
        # Ensure the parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trade: Trade) -> None:
        """
        Serialize trade to JSON and append one line to the journal file.

        The file is opened in append mode; the line ends with a newline.
        Datetime fields are already serialised to ISO-8601 strings by
        Trade.to_dict().
        """
        record = trade.to_dict()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def load_all(self) -> list[dict]:
        """
        Read and parse all journal entries.

        Returns an empty list if the file does not exist or is empty.
        Malformed lines are skipped with a warning rather than raising.
        """
        if not self.path.exists():
            return []
        records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    import warnings
                    warnings.warn(
                        f"TradeJournal: skipping malformed line {lineno} in {self.path}: {exc}"
                    )
        return records

    def to_dataframe(self):  # -> pd.DataFrame
        """
        Load all journal entries and return a pandas DataFrame.

        Returns an empty DataFrame if there are no entries.
        pandas is imported lazily so the rest of the engine does not
        require it at import time.
        """
        import pandas as pd  # noqa: PLC0415

        records = self.load_all()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def clear(self) -> None:
        """Delete the journal file if it exists."""
        if self.path.exists():
            self.path.unlink()
