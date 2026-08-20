"""Wait for the first t10 rows to land in the ledger, then print one."""
import json
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "results" / "prediction_log.jsonl"

for _ in range(20):
    rows = [json.loads(l) for l in LOG.read_text().splitlines()
            if '"t10-' in l]
    if rows:
        break
    time.sleep(20)

print(f"t10 rows: {len(rows)}")
for r in rows[-4:]:
    print(f"  {r['variant']}: dims={len(r['x'])} kalshi={r['x'][-4:]} "
          f"pred={r['pred']} now={r['price_now']} arm={r['arm']}")
