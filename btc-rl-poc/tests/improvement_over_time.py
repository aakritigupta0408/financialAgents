"""Are the models getting better with time? Per-PT-day: ungated and
gated accuracy per variant, market accuracy on the same rows, and the
model-minus-market edge (the regime-controlled trajectory)."""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
TAU = {"kb2": .64, "kb3": .83, "kb4": .78}

for v in ("kb2", "kb3", "kb4"):
    rows = [r for r in kb if r.get("variant") == v
            and r.get("actual") is not None]
    byday = defaultdict(list)
    for r in rows:
        byday[datetime.fromtimestamp(r["made_ts"], PT)
              .strftime("%m-%d")].append(r)
    print(f"\n== {v} (gate tau {TAU[v]}) ==")
    print(f"{'day':6s} {'n':>5s} {'ungated':>8s} {'gated':>7s} "
          f"{'gcov':>5s} {'market':>7s} {'edge':>6s}")
    edges = []
    for day in sorted(byday):
        rs = byday[day]
        if len(rs) < 30:
            continue
        g = [r for r in rs if max(r["p_up"], 1 - r["p_up"]) >= TAU[v]]
        ung = sum(r["hit"] for r in rs) / len(rs)
        ga = sum(r["hit"] for r in g) / len(g) if g else float("nan")
        mk = [r for r in g if r.get("mkt_p_up") is not None]
        ma = (sum(((r["mkt_p_up"] >= .5) == bool(r["actual"]))
                  for r in mk) / len(mk)) if mk else float("nan")
        edge = (ga - ma) * 100 if g and mk else float("nan")
        edges.append(edge)
        print(f"{day:6s} {len(rs):5d} {ung:8.1%} {ga:7.1%} "
              f"{len(g)/len(rs):5.0%} {ma:7.1%} {edge:+6.1f}")
    if len(edges) >= 4:
        h = len(edges) // 2
        a, b = edges[:h], edges[h:]
        print(f"   edge first-half avg {sum(a)/len(a):+.2f} -> "
              f"second-half avg {sum(b)/len(b):+.2f} "
              f"({'improving' if sum(b)/len(b) > sum(a)/len(a) else 'not improving'})")
