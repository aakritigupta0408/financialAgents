"""Precision-first critique of kb2:
(a) per-class precision vs coverage across tau (where does 90% live?)
(b) precision at one tau, split by phase (should tau depend on phase?)
(c) gated FALSE calls: did kb2 agree with the market (both wrong — only
    faster info fixes) or defy it (suppressible)?
(d) gated precision inside vs outside the hot session-open hours
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb2"
        and r.get("actual") is not None]
n = len(rows)


def prec(sel, val):
    tp = sum(1 for r in sel if r["call"] == val and r["actual"] == val)
    fp = sum(1 for r in sel if r["call"] == val and r["actual"] != val)
    return (tp / (tp + fp) if tp + fp else None), tp + fp


print(f"(a) precision-coverage frontier ({n} settled calls):")
print(f"{'tau':>5s} {'cov':>5s} {'UP prec(n)':>13s} {'DOWN prec(n)':>13s} "
      f"{'min':>6s}")
for t100 in range(55, 96, 5):
    t = t100 / 100
    sel = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= t]
    if len(sel) < 30:
        break
    pu, nu = prec(sel, 1)
    pd, nd = prec(sel, 0)
    print(f"{t:5.2f} {len(sel)/n:5.0%} {pu:8.3f}({nu:3d}) "
          f"{pd:8.3f}({nd:3d}) {min(pu, pd):6.3f}")

print("\n(b) precision at tau .62 by phase:")
sel62 = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= .62]
for name, f in (("early >10m", lambda m: m > 10),
                ("mid 5-10m", lambda m: 5 <= m <= 10),
                ("late <5m", lambda m: m < 5)):
    s = [r for r in sel62 if f(r["mins_left"])]
    pu, nu = prec(s, 1)
    pd, nd = prec(s, 0)
    print(f"  {name:11s} n={len(s):4d}  UP {pu:.3f}({nu})  DOWN {pd:.3f}({nd})")

print("\n(c) gated FALSE calls — market agreement:")
false_g = [r for r in sel62 if not r["hit"] and r.get("mkt_p_up") is not None]
agree = [r for r in false_g
         if (r["mkt_p_up"] >= .5) == (r["call"] == 1)]
print(f"  {len(false_g)} gated misses with quotes: "
      f"{len(agree)} agreed with market (both wrong), "
      f"{len(false_g)-len(agree)} defied it")
defy_all = [r for r in sel62 if r.get("mkt_p_up") is not None
            and (r["mkt_p_up"] >= .5) != (r["call"] == 1)]
if defy_all:
    print(f"  ALL gated market-defying calls: {len(defy_all)}, "
          f"kb2 right {sum(r['hit'] for r in defy_all)/len(defy_all):.1%}")

print("\n(d) gated precision by session-open hot hours (18-20, 01 PT):")
hot = [r for r in sel62
       if datetime.fromtimestamp(r["made_ts"], PT).hour in (18, 19, 20, 1)]
cold = [r for r in sel62
        if datetime.fromtimestamp(r["made_ts"], PT).hour
        not in (18, 19, 20, 1)]
for name, s in (("hot hours", hot), ("other hours", cold)):
    pu, nu = prec(s, 1)
    pd, nd = prec(s, 0)
    print(f"  {name:12s} n={len(s):4d}  UP {pu:.3f}({nu})  DOWN {pd:.3f}({nd})")
