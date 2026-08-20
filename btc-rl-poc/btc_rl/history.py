"""Append-only metric history: results/metrics_history.jsonl.

Every retrain (hourly, kind="retrain") and every batch training run
(kind="batch") appends one row here, so before/after comparisons survive —
previously gate outcomes lived only in online_status.json and were
overwritten within one 30-second poll.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HISTORY = RESULTS_DIR / "metrics_history.jsonl"
MAX_ROWS = 5000


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RESULTS_DIR.parent, capture_output=True, text=True,
            timeout=5).stdout.strip() or None
    except Exception:
        return None


def append_history(kind: str, payload: dict) -> None:
    row = {"kind": kind, "ts": int(time.time()), "git": _git_sha(), **payload}
    RESULTS_DIR.mkdir(exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps(row) + "\n")
    # occasional trim so the file never grows unbounded
    try:
        lines = HISTORY.read_text().splitlines()
        if len(lines) > MAX_ROWS:
            tmp = HISTORY.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[-MAX_ROWS:]) + "\n")
            tmp.replace(HISTORY)
    except OSError:
        pass


def load_history(kind: str | None = None) -> list[dict]:
    if not HISTORY.exists():
        return []
    rows = []
    for line in HISTORY.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if kind is None or r.get("kind") == kind:
            rows.append(r)
    return rows
