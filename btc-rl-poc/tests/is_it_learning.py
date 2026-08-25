"""Is each arm actually learning? Evidence per learning arm:
- SGD update counter: value 3h ago vs now (ticking = learning happening)
- rolling accuracy: prior window vs recent window
- prediction variance (a stuck model goes flat)"""
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
now = time.time()
print(f"{'arm':5s} {'upd 3h ago':>10s} {'upd now':>8s} {'settled':>8s} "
      f"{'acc prev':>8s} {'acc last3h':>10s} {'p stdev':>8s}")
for v in ("kb3", "kb4", "kb5", "kb6"):
    rows = [r for r in kb if r.get("variant") == v
            and r.get("trained") is not None]
    if not rows:
        print(f"{v:5s}  no rows")
        continue
    old = [r for r in rows if r["made_ts"] <= now - 3 * 3600]
    u0 = old[-1]["trained"] if old else 0
    u1 = rows[-1]["trained"]
    st = [r for r in rows if r.get("actual") is not None]
    recent = [r for r in st if r["made_ts"] > now - 3 * 3600]
    prior = [r for r in st if r["made_ts"] <= now - 3 * 3600][-200:]
    a0 = (sum(r["hit"] for r in prior) / len(prior)) if prior else None
    a1 = (sum(r["hit"] for r in recent) / len(recent)) if recent else None
    ps = [r["p_up"] for r in rows[-120:]]
    sd = statistics.pstdev(ps) if len(ps) > 5 else 0
    print(f"{v:5s} {u0:10d} {u1:8d} {len(st):8d} "
          f"{('%.1f%%' % (100*a0)) if a0 is not None else '–':>8s} "
          f"{('%.1f%%' % (100*a1)) if a1 is not None else '–':>10s} "
          f"{sd:8.3f}")
print("\nkb7: zero-shot by design — it never updates (that's the point:")
print("a frozen pretrained forecaster as the no-learning baseline).")
