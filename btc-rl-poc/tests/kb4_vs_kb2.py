"""kb4 vs kb2 head-to-head on identical settled minutes: accuracy,
Brier, gated precision, phase splits, and the disagreement subset."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
by = defaultdict(dict)
for r in kb:
    if r.get("variant") in ("kb2", "kb4") and r.get("actual") is not None:
        by[(r["ticker"], r["made_ts"])][r["variant"]] = r
pairs = [v for v in by.values() if "kb2" in v and "kb4" in v]
print(f"paired settled minutes: {len(pairs)}")

acc = lambda rs, v: sum(x[v]["hit"] for x in rs) / len(rs)
brier = lambda rs, v: sum((x[v]["p_up"] - x[v]["actual"]) ** 2
                          for x in rs) / len(rs)
print(f"{'':14s} {'kb2':>7s} {'kb4':>7s}")
print(f"{'accuracy':14s} {acc(pairs,'kb2'):7.1%} {acc(pairs,'kb4'):7.1%}")
print(f"{'brier':14s} {brier(pairs,'kb2'):7.3f} {brier(pairs,'kb4'):7.3f}")

for name, f in (("early >10m", lambda m: m > 10),
                ("mid 5-10m", lambda m: 5 <= m <= 10),
                ("late <5m", lambda m: m < 5)):
    rs = [x for x in pairs if f(x["kb2"]["mins_left"])]
    if len(rs) < 20:
        continue
    print(f"{name:14s} {acc(rs,'kb2'):7.1%} {acc(rs,'kb4'):7.1%}  (n={len(rs)})")

for tau in (0.62, 0.75, 0.85):
    g2 = [x for x in pairs
          if max(x["kb2"]["p_up"], 1 - x["kb2"]["p_up"]) >= tau]
    g4 = [x for x in pairs
          if max(x["kb4"]["p_up"], 1 - x["kb4"]["p_up"]) >= tau]
    a2 = acc(g2, "kb2") if g2 else float("nan")
    a4 = acc(g4, "kb4") if g4 else float("nan")
    print(f"gated tau {tau:.2f}: kb2 {a2:.1%} (cov {len(g2)/len(pairs):.0%}) "
          f"| kb4 {a4:.1%} (cov {len(g4)/len(pairs):.0%})")

dis = [x for x in pairs if x["kb2"]["call"] != x["kb4"]["call"]]
if dis:
    w2 = sum(x["kb2"]["hit"] for x in dis)
    print(f"disagreements: {len(dis)} ({len(dis)/len(pairs):.0%}) — "
          f"kb2 right {w2}/{len(dis)} ({w2/len(dis):.0%}), "
          f"kb4 right {len(dis)-w2} ({(len(dis)-w2)/len(dis):.0%})")
