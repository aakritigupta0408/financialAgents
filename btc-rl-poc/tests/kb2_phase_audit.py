"""Entry-time bias audit for kb2's gated calls: are the calls that clear
the confidence gate concentrated late in the window (market-converged),
and does per-class 80/80 survive within each phase?"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb2"
        and r.get("actual") is not None]

# reproduce the page's gate: smallest tau with both class precisions >= .8
def prf(sel):
    out = {}
    for cls, val in (("up", 1), ("down", 0)):
        tp = sum(1 for r in sel if r["call"] == val and r["actual"] == val)
        fp = sum(1 for r in sel if r["call"] == val and r["actual"] != val)
        fn = sum(1 for r in sel if r["call"] != val and r["actual"] == val)
        out[cls] = (tp / (tp + fp) if tp + fp else None,
                    tp / (tp + fn) if tp + fn else None)
    return out

tau = None
for t100 in range(50, 96):
    t = t100 / 100
    sel = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= t]
    if len(sel) < 40:
        break
    s = prf(sel)
    if (s["up"][0] or 0) >= 0.8 and (s["down"][0] or 0) >= 0.8:
        tau = t
        break
sel = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= tau]
print(f"gate tau {tau:.2f} | gated {len(sel)}/{len(rows)} calls")

PHASES = (("early >10m", lambda m: m > 10),
          ("mid 5-10m", lambda m: 5 <= m <= 10),
          ("late <5m", lambda m: m < 5))
print(f"{'phase':11s} {'all':>5s} {'gated':>6s} {'cov':>5s} "
      f"{'acc':>6s} {'mkt':>6s} {'UP P/R':>11s} {'DOWN P/R':>11s}")
for name, f in PHASES:
    allp = [r for r in rows if f(r["mins_left"])]
    g = [r for r in sel if f(r["mins_left"])]
    if not g:
        print(f"{name:11s} {len(allp):5d} {0:6d}")
        continue
    acc = sum(r["hit"] for r in g) / len(g)
    mk = [r for r in g if r.get("mkt_p_up") is not None]
    macc = (sum(((r["mkt_p_up"] >= 0.5) == bool(r["actual"])) for r in mk)
            / len(mk)) if mk else float("nan")
    s = prf(g)
    fmt = lambda t2: "/".join("–" if x is None else f"{x:.2f}" for x in t2)
    print(f"{name:11s} {len(allp):5d} {len(g):6d} "
          f"{len(g)/len(allp):5.0%} {acc:6.1%} {macc:6.1%} "
          f"{fmt(s['up']):>11s} {fmt(s['down']):>11s}")
