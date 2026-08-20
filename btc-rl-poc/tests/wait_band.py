"""Wait for freshly scored rows carrying the vol-scaled hit band."""
import json
import time
from pathlib import Path

LOG = (Path(__file__).resolve().parent.parent / "results"
       / "prediction_log.jsonl")
start = time.time()

for _ in range(30):
    rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    fresh = [r for r in rows if r.get("hit_band") is not None]
    if fresh:
        break
    time.sleep(20)

print(f"rows with hit_band: {len(fresh)}")
for h in (1, 5, 15, 30):
    g = [r for r in fresh if r["horizon"] == h]
    if g:
        bands = sorted(r["hit_band"] for r in g)
        print(f"  +{h:2}m: n={len(g):3}  band range "
              f"${bands[0]:.0f}-${bands[-1]:.0f}  "
              f"hits {sum(r['hit'] for r in g)}")
