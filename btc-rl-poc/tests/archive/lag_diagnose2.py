"""Lag diagnosis v2 against the per-minute snapshot price series.
Tests, for each arm's h1 rows:
  A) alignment: |pred - px(target)| vs |pred - px(target-60)| vs +60
  B) anchor freshness: |price_now(row) - px(made_ts)|
  C) movement: |pred - price_now| vs |px(target) - px(made)| (does the
     model predict any move at all, or is it persistence?)"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
px = {int(r["ts"]): float(r["c"])
      for r in json.loads((ROOT / "results"
                           / "recent_prices.json").read_text())}
print(f"{len(px)} price minutes")

rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").open()]
st = defaultdict(lambda: defaultdict(float))
for r in rows:
    if r.get("horizon") != 1 or r.get("actual") is None:
        continue
    t = (r.get("target_ts") or 0) // 60 * 60
    m = (r.get("made_ts") or 0) // 60 * 60
    if not all(k in px for k in (t, t - 60, t + 60, m)):
        continue
    s = st[r["variant"]]
    s["n"] += 1
    s["at_T"] += abs(r["pred"] - px[t])
    s["at_Tm1"] += abs(r["pred"] - px[t - 60])
    s["at_Tp1"] += abs(r["pred"] - px[t + 60])
    s["anchor_gap"] += abs(r.get("price_now", 0) - px[m])
    s["pred_move"] += abs(r["pred"] - r.get("price_now", r["pred"]))
    s["true_move"] += abs(px[t] - px[m])

print(f"{'arm':10s} {'n':>4s} {'@T':>7s} {'@T-1':>7s} {'@T+1':>7s} "
      f"{'anchor':>7s} {'predmv':>7s} {'truemv':>7s}")
for v, s in sorted(st.items()):
    n = int(s["n"])
    if n < 20:
        continue
    print(f"{v:10s} {n:4d} {s['at_T']/n:7.1f} {s['at_Tm1']/n:7.1f} "
          f"{s['at_Tp1']/n:7.1f} {s['anchor_gap']/n:7.1f} "
          f"{s['pred_move']/n:7.2f} {s['true_move']/n:7.1f}")
