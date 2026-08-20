"""Verify kb now commits every minute: wait until 3+ fresh rows exist."""
import json
import time
from pathlib import Path

LOG = (Path(__file__).resolve().parent.parent / "results"
       / "kalshi_binary_log.jsonl")
start = time.time()

for _ in range(30):
    rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    fresh = [r for r in rows if r["made_ts"] >= start - 60]
    if len(fresh) >= 3:
        break
    time.sleep(15)

print(f"total rows: {len(rows)}  fresh (this restart): {len(fresh)}")
for r in fresh:
    print(f"  made={time.strftime('%H:%M', time.localtime(r['made_ts']))} "
          f"close={time.strftime('%H:%M', time.localtime(r['close_ts']))} "
          f"left={r['mins_left']}m p_up={r['p_up']} call={r['call']}")
