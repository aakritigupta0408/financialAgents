"""Are h1 predictions genuinely delayed? Cross-correlate: for each arm's
h1 rows, compare |pred - actual(target)| vs |pred - actual(target±60s)|.
If pred matches the PAST actual better than the target actual, predictions
are anchored to stale data (real lag). Also check the anchor: how far is
pred from the base price at make time (martingale illusion check)."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").open()]

# actual price by target minute from settled rows (any arm, h1)
actual_at = {}
for r in rows:
    if r.get("actual") is not None and r.get("target_ts"):
        actual_at[r["target_ts"]] = r["actual"]

stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
for r in rows:
    if (r.get("horizon") != 1 or r.get("actual") is None
            or not r.get("target_ts")):
        continue
    t = r["target_ts"]
    past, fut = actual_at.get(t - 60), actual_at.get(t + 60)
    if past is None or fut is None:
        continue
    s = stats[r["variant"]]
    s[0] += abs(r["pred"] - r["actual"])      # vs target actual
    s[1] += abs(r["pred"] - past)             # vs 1 min earlier
    s[2] += abs(r["pred"] - fut)              # vs 1 min later
    s[3] += abs(r["pred"] - r.get("price_now", r["pred"]))  # move size
    s[4] += 1

print(f"{'arm':10s} {'n':>5s} {'MAE@T':>8s} {'MAE@T-1':>8s} "
      f"{'MAE@T+1':>8s} {'|pred-now|':>10s}  verdict")
for v, s in sorted(stats.items()):
    if s[4] < 30:
        continue
    n = s[4]
    m0, mp, mf, mv = s[0]/n, s[1]/n, s[2]/n, s[3]/n
    verdict = ("STALE-ANCHORED (matches past)" if mp < m0 * 0.9
               else "ok: target is best fit")
    print(f"{v:10s} {n:5d} {m0:8.1f} {mp:8.1f} {mf:8.1f} {mv:10.2f}  "
          f"{verdict}")
