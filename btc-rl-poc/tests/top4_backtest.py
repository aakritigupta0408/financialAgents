"""Top-4 backtest: kb2, kb4, kb7, kb8 on three axes.

  1. performance      minute acc / Brier vs market (clustered), and
                      gated acc + coverage at the ledger tau (0.62)
  2. improvement      per-window accuracy by chronological quartile of
                      WINDOWS + least-squares slope (acc per 100 windows)
  3. saturation left  late-quartile gap to the market ceiling, whether
                      the slope survives its own noise, and for the
                      online learners the fraction of initial learning
                      rate remaining (lr = lr0 / (1 + updates/400))

All effective-n is windows, never minutes. kb8 has ~1 live settled
window, so its long view is the REPLAY prequential from
results/kb8_replay.jsonl (127 windows, decision-time inputs only) —
labeled kb8-replay to keep provenance honest.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.agents import BinaryLogit                # noqa: E402
from btc_rl.online import _kb8_features              # noqa: E402

TAU = 0.62
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]


def rows_for(variant):
    return [r for r in kb if r.get("variant") == variant
            and r.get("actual") is not None]


def kb8_replay_preds():
    """Rebuild kb8's prequential replay predictions (cold model, settle-
    grouped) so its 'backtest' spans 127 windows, not 1."""
    cache = {}
    for l in (ROOT / "results" / "kb8_replay.jsonl").open():
        c = json.loads(l)
        cache[(c["made_ts"], c["ticker"])] = (c["p7"], c["w80"])
    src = [r for r in kb if r.get("variant") == "kb3"
           and r.get("actual") is not None and r.get("bx")]
    samples = []
    for r in src:
        ck = (r["made_ts"], r["ticker"])
        if ck not in cache:
            continue
        p7, w80 = cache[ck]
        x = _kb8_features(p7, w80, r.get("mkt_p_up"), r["bx"],
                          ((r.get("pf") or []) + [0.0] * 4)[:4],
                          r["mins_left"])
        samples.append((r["close_ts"], r["ticker"], x, r["actual"],
                        r.get("mkt_p_up"), r["mins_left"]))
    samples.sort(key=lambda s: s[0])
    m = BinaryLogit(len(samples[0][2]))
    groups = defaultdict(list)
    for s in samples:
        groups[s[0]].append(s)
    out = []
    for cts in sorted(groups):
        g = groups[cts]
        for cts2, tk, x, y, mkt, ml in g:
            out.append({"ticker": tk, "close_ts": cts2,
                        "p_up": m.predict(x), "actual": y,
                        "hit": None, "mkt_p_up": mkt, "mins_left": ml})
        for _, _, x, y, _, _ in g:
            m.update(x, y)
    for r in out:
        r["hit"] = int((r["p_up"] >= 0.5) == bool(r["actual"]))
    return out


def clustered_t(pairs):
    byw = defaultdict(list)
    for tk, d in pairs:
        byw[tk].append(d)
    ms = [sum(v) / len(v) for v in byw.values()]
    k = len(ms)
    if k < 5:
        return float("nan")
    mu = sum(ms) / k
    sd = math.sqrt(sum((x - mu) ** 2 for x in ms) / (k - 1))
    return mu / (sd / math.sqrt(k)) if sd else float("nan")


def analyze(name, rows, updates=None, lr0=0.05):
    n = len(rows)
    if not n:
        print(f"\n=== {name}: no settled rows ===")
        return
    acc = sum(r["hit"] for r in rows) / n
    br = sum((r["p_up"] - r["actual"]) ** 2 for r in rows) / n
    wm = [r for r in rows if r.get("mkt_p_up") is not None]
    macc = sum(int((r["mkt_p_up"] >= .5) == bool(r["actual"]))
               for r in wm) / max(1, len(wm))
    tb = clustered_t([(r["ticker"],
                       (r["p_up"] - r["actual"]) ** 2
                       - (r["mkt_p_up"] - r["actual"]) ** 2) for r in wm])
    conf = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= TAU]
    cacc = sum(r["hit"] for r in conf) / max(1, len(conf))

    # window series, chronological
    byw = defaultdict(list)
    for r in rows:
        byw[r["ticker"]].append(r)
    wins = sorted(byw.values(), key=lambda g: g[0]["close_ts"])
    wacc = [sum(r["hit"] for r in g) / len(g) for g in wins]
    W = len(wacc)
    wmkt = []
    for g in wins:
        gm = [r for r in g if r.get("mkt_p_up") is not None]
        wmkt.append(sum(int((r["mkt_p_up"] >= .5) == bool(r["actual"]))
                        for r in gm) / len(gm) if gm else None)

    # quartiles of WINDOWS
    qs = max(1, W // 4)
    quart = []
    for qi in range(4):
        seg = wacc[qi * qs: (qi + 1) * qs if qi < 3 else W]
        segm = [m for m in wmkt[qi * qs: (qi + 1) * qs if qi < 3 else W]
                if m is not None]
        if seg:
            quart.append((sum(seg) / len(seg),
                          sum(segm) / len(segm) if segm else float("nan")))

    # slope: acc per 100 windows, with its own t (windows independent)
    xb = (W - 1) / 2
    yb = sum(wacc) / W
    sxx = sum((i - xb) ** 2 for i in range(W))
    slope = sum((i - xb) * (y - yb) for i, y in enumerate(wacc)) / sxx \
        if sxx else 0.0
    resid = [y - (yb + slope * (i - xb)) for i, y in enumerate(wacc)]
    se = math.sqrt(sum(e * e for e in resid) / max(1, W - 2) / sxx) \
        if sxx and W > 2 else float("inf")
    tslope = slope / se if se else float("nan")

    print(f"\n=== {name}  ({W} windows, {n} minute-rows) ===")
    print(f"  performance : acc {acc:.1%}  brier {br:.3f}  "
          f"market {macc:.1%}  clustered t(brier vs mkt) {tb:+.2f}")
    print(f"  gated @0.62 : acc {cacc:.1%}  coverage {len(conf)/n:.0%}  "
          f"(n={len(conf)})")
    qtxt = "  ".join(f"Q{i+1} {a:.0%}(mkt {m:.0%})"
                     for i, (a, m) in enumerate(quart))
    print(f"  improvement : {qtxt}")
    # slope is fraction/window; ×100 windows ×100 to percentage points
    print(f"  slope       : {slope*10000:+.1f} acc-pts per 100 windows "
          f"(t = {tslope:+.1f})")
    lastq, lastm = quart[-1]
    head = lastq - lastm
    sat = []
    sat.append(f"gap to market ceiling in Q4: {head:+.1%}")
    if updates is not None:
        lr_frac = 1.0 / (1.0 + updates / 400.0)
        sat.append(f"lr remaining {lr_frac:.0%} of initial "
                   f"({updates} updates)")
    else:
        sat.append("frozen — no learning channel (headroom 0 by design)"
                   if name.startswith("kb7") else
                   "daily re-blend only — improves with data, not SGD")
    print(f"  saturation  : " + "; ".join(sat))


upd = {}
for nm, f in (("kb4", "kb4_logit.json"), ("kb8", "kb8_logit.json")):
    try:
        upd[nm] = json.loads((ROOT / "results" / f).read_text())["updates"]
    except Exception:
        pass

analyze("kb2 (market-anchored blend)", rows_for("kb2"))
analyze("kb4 (stack kb2+kb3)", rows_for("kb4"), updates=upd.get("kb4"))
analyze("kb7 (frozen foundation model)", rows_for("kb7"))
analyze("kb8-live (log pool, young)", rows_for("kb8"),
        updates=upd.get("kb8"))
analyze("kb8-replay (127-window prequential)", kb8_replay_preds(),
        updates=upd.get("kb8"))
print("\nnotes: effective n = windows; kb8-replay is decision-time replay,"
      "\nnot live rows; slope t uses window independence (valid), Brier t"
      "\nclusters by window; tau fixed at the ledger's 0.62 for all arms.")
