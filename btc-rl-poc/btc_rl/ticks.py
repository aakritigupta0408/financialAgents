"""Tick-level trade collector — the data foundation for sub-minute models.

Polls Coinbase's public trades endpoint every few seconds (captures every
trade unless volume exceeds ~1000 trades per poll, which is rare) and
appends deduplicated ticks to results/ticks.jsonl with rotation. This is
the honest groundwork for the "tick-level modeling" future-work rung; no
model consumes it yet.

Usage:  python -m btc_rl.ticks            # run forever
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .sources import fetch_recent_trades

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TICKS = RESULTS_DIR / "ticks.jsonl"
POLL_S = 4
MAX_LINES = 500_000     # ~ a few days of BTC-USD ticks
TRIM_CHECK_EVERY = 500  # polls between rotation checks


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    seen: set[int] = set()
    if TICKS.exists():  # resume without duplicating
        for line in TICKS.read_text().splitlines()[-50_000:]:
            try:
                seen.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    polls = 0
    print("tick collector up — results/ticks.jsonl")
    while True:
        try:
            fresh = [t for t in fetch_recent_trades(1000) if t["id"] not in seen]
            if fresh:
                fresh.sort(key=lambda t: t["id"])
                with TICKS.open("a") as f:
                    for t in fresh:
                        f.write(json.dumps(t) + "\n")
                        seen.add(t["id"])
                if len(seen) > 200_000:  # bound memory
                    seen = set(sorted(seen)[-100_000:])
            polls += 1
            if polls % TRIM_CHECK_EVERY == 0 and TICKS.exists():
                lines = TICKS.read_text().splitlines()
                if len(lines) > MAX_LINES:
                    TICKS.write_text("\n".join(lines[-MAX_LINES:]) + "\n")
        except Exception as exc:
            print(f"tick poll error (retrying): {exc}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    run()
