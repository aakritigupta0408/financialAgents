"""kb8 feature-design lab — offline, on the cached kb7 replay.

The first warm start showed a REAL design flaw (not cold start): Q4
prequential acc 64.5% vs 71.8/71.9 for kb7/market. Hypothesis: centered
probabilities saturate at +-1, so a linear logit can't express confident
parent calls; fusing in LOG-ODDS space (log-opinion pool) makes
"copy the market" learnable as weight ~= 1.

Variants (all 12-dim, all decision-time only, settle-grouped prequential):
  V0 current  centered probs (what shipped to _kb8_features)
  V1 logit    log-odds pool + time interactions
  V2 logit-   V1 without time interactions (ablate: is time the driver?)
Report Q4 acc and window-clustered Brier t vs kb7-replay and market.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.agents import BinaryLogit                            # noqa: E402

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb3"
        and r.get("actual") is not None and r.get("bx")
        and len(r["bx"]) >= 4 and r.get("pf")]
cache = {}
for l in (ROOT / "results" / "kb8_replay.jsonl").open():
    c = json.loads(l)
    cache[(c["made_ts"], c["ticker"])] = (c["p7"], c["w80"])


def lg(p):
    return max(-3.0, min(3.0, math.log(p / (1 - p)))) if p is not None else 0.0


def v0(p7, w80, mkt, bx, pf, ml):
    p7c = (p7 - 0.5) * 2.0
    mktc = (mkt - 0.5) * 2.0 if mkt is not None else 0.0
    return [1.0, p7c, mktc, 1.0 if mkt is not None else 0.0,
            p7c - mktc, p7c * mktc * 2.0, min(w80 / 200.0, 2.0),
            bx[3], ml / 15.0] + pf[:3]


def v1(p7, w80, mkt, bx, pf, ml):
    l7, lm = lg(p7), lg(mkt)
    t = ml / 15.0
    return [1.0, l7, lm, 1.0 if mkt is not None else 0.0,
            l7 - lm, l7 * t, lm * (1.0 - t),
            min(w80 / 200.0, 2.0), bx[3], t] + pf[:2]


def v2(p7, w80, mkt, bx, pf, ml):
    l7, lm = lg(p7), lg(mkt)
    return [1.0, l7, lm, 1.0 if mkt is not None else 0.0,
            l7 - lm, l7 * lm * 0.5, min(w80 / 200.0, 2.0),
            bx[3], ml / 15.0] + pf[:3]


def run(fx):
    samples = []
    for r in rows:
        ck = (r["made_ts"], r["ticker"])
        if ck not in cache:
            continue
        p7, w80 = cache[ck]
        pf = ((r.get("pf") or []) + [0.0] * 4)[:4]
        x = fx(p7, w80, r.get("mkt_p_up"), r["bx"], pf, r["mins_left"])
        samples.append((r["close_ts"], r["ticker"], x, r["actual"],
                        p7, r.get("mkt_p_up")))
    samples.sort(key=lambda s: s[0])
    m = BinaryLogit(12)
    groups = defaultdict(list)
    for s in samples:
        groups[s[0]].append(s)
    preds = []
    for cts in sorted(groups):
        g = groups[cts]
        for _, tk, x, y, p7, mkt in g:
            preds.append((tk, m.predict(x), y, p7, mkt))
        for _, _, x, y, _, _ in g:
            m.update(x, y)
    n = len(preds)
    q4 = preds[-n // 4:]
    a = sum((p >= .5) == bool(y) for _, p, y, _, _ in preds) / n
    a4 = sum((p >= .5) == bool(y) for _, p, y, _, _ in q4) / len(q4)

    def ct(diffs):
        byw = defaultdict(list)
        for tk, d in diffs:
            byw[tk].append(d)
        ms = [sum(v) / len(v) for v in byw.values()]
        k = len(ms)
        mu = sum(ms) / k
        sd = math.sqrt(sum((x - mu) ** 2 for x in ms) / (k - 1))
        return mu / (sd / math.sqrt(k)) if sd else float("nan")

    t7 = ct([(tk, (p - y) ** 2 - (p7 - y) ** 2) for tk, p, y, p7, _ in preds])
    tm = ct([(tk, (p - y) ** 2 - (mk - y) ** 2)
             for tk, p, y, _, mk in preds if mk is not None])
    return n, a, a4, t7, tm, m


print(f"{'variant':>8s} {'n':>5s} {'acc':>6s} {'Q4 acc':>7s} "
      f"{'t vs kb7':>9s} {'t vs mkt':>9s}")
for name, fx in (("V0-cur", v0), ("V1-logit", v1), ("V2-nolag", v2)):
    n, a, a4, t7, tm, _ = run(fx)
    print(f"{name:>8s} {n:5d} {a:6.1%} {a4:7.1%} {t7:9.2f} {tm:9.2f}")
print("(negative t = kb8 variant better; Q4 = post-training regime)")
