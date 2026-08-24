"""Do kb2/kb3 gated misses cluster in fixed time windows?
(a) miss rate by hour-of-day PT (flag hours where a binomial z vs the
    model's own average exceeds 2)
(b) miss rate by window-close slot (:00/:15/:30/:45)
(c) window-level concentration: share of misses contributed by the worst
    k windows, vs what shared outcomes alone would predict
"""
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
TAU = {"kb2": .62, "kb3": .82}

for v in ("kb2", "kb3"):
    rows = [r for r in kb if r.get("variant") == v
            and r.get("actual") is not None
            and max(r["p_up"], 1 - r["p_up"]) >= TAU[v]]
    n = len(rows)
    base = sum(1 - r["hit"] for r in rows) / n
    print(f"\n== {v} gated: {n} calls, overall miss rate {base:.1%} ==")

    byh = defaultdict(list)
    for r in rows:
        byh[datetime.fromtimestamp(r["made_ts"], PT).hour].append(r)
    print("hour-of-day PT (n>=25 shown; * = binomial z >= 2 vs own avg):")
    for h in sorted(byh):
        rs = byh[h]
        if len(rs) < 25:
            continue
        m = sum(1 - r["hit"] for r in rs) / len(rs)
        se = math.sqrt(base * (1 - base) / len(rs))
        z = (m - base) / se if se else 0
        flag = " *HOT*" if z >= 2 else (" *cold*" if z <= -2 else "")
        print(f"  {h:02d}:00  n={len(rs):3d}  miss {m:5.1%}  z={z:+.1f}{flag}")

    byslot = defaultdict(list)
    for r in rows:
        byslot[datetime.fromtimestamp(r["close_ts"], PT).minute].append(r)
    print("window-close slot:", {f":{m:02d}":
          f"{sum(1-r['hit'] for r in rs)/len(rs):.1%}({len(rs)})"
          for m, rs in sorted(byslot.items())})

    bywin = defaultdict(list)
    for r in rows:
        bywin[r["ticker"]].append(r)
    wins = sorted(bywin.values(),
                  key=lambda rs: -sum(1 - r["hit"] for r in rs))
    total_miss = sum(1 - r["hit"] for r in rows)
    n_win = len(bywin)
    miss_any = sum(1 for rs in bywin.values()
                   if any(1 - r["hit"] for r in rs))
    for k in (5, 10, 20):
        share = sum(sum(1 - r["hit"] for r in rs) for rs in wins[:k]) \
            / max(1, total_miss)
        print(f"  worst {k:2d} of {n_win} windows hold {share:.0%} of misses")
    print(f"  windows with >=1 gated miss: {miss_any}/{n_win} "
          f"({miss_any/n_win:.0%}) — the rest are fully clean")
