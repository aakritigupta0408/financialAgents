"""Wait for the first kb (kalshi-binary) rows, then print them."""
import json
import time
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
LOG = RES / "kalshi_binary_log.jsonl"

rows = []
for _ in range(20):
    if LOG.exists():
        rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    if rows:
        break
    time.sleep(15)

print(f"kb rows: {len(rows)}")
for r in rows[-3:]:
    print(f"  {r['ticker']} slot={r['made_ts']} strike={r['strike']} "
          f"base={r['base']} left={r['mins_left']}m "
          f"p_up={r['p_up']} call={r['call']} mkt={r['mkt_p_up']}")
s = json.load(open(RES / "online_status.json"))
print("status kalshi_binary:", json.dumps(s.get("kalshi_binary"))[:200])
