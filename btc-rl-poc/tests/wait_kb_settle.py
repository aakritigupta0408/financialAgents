"""Wait for the first kb row to settle, then print the outcome."""
import json
import time
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
LOG = RES / "kalshi_binary_log.jsonl"

for _ in range(25):
    rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    done = [r for r in rows if r["actual"] is not None]
    if done:
        break
    time.sleep(20)

print(f"rows: {len(rows)}  settled: {len(done)}")
for r in done[-3:]:
    print(f"  {r['ticker']} p_up={r['p_up']} call={r['call']} "
          f"outcome={r['actual']} hit={r['hit']} brier={r['brier']} "
          f"mkt_brier={r.get('mkt_brier')}")
