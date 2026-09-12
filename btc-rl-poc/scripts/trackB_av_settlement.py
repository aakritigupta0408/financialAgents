"""TRACK B — does Alpha Vantage crypto-linked equity state improve the FINAL
15-min ABOVE/BELOW settlement probability, conditional on Kalshi's current
probability? (handoff Track B). Target = settlement, NOT Kalshi's next tick.

Features: COIN/MSTR/IBIT/QQQ 5m & 15m returns, cross-asset consensus &
dispersion, QQQ (macro risk), all PIT (latest AV bar <= decision). Equities
are not 24/7 -> explicit session freshness; the primary test is on the
SESSION-FRESH subset, reported with coverage. Residual model
sigmoid(logit(p_mkt)+delta_AV); BSS vs Kalshi, window-split, window-weighted.
"""
import calendar
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "av_cache"
SHARDS = ROOT / "results" / "events"
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
OUT = ROOT / "research" / "t05_repricing" / "trackB_result.json"
EDT = 4 * 3600
SYMS = ["COIN", "MSTR", "IBIT", "QQQ"]
LAMBDAS = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
np.random.seed(20260912)


def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def load_sym(sym):
    bars = []
    for m in ("2026-08", "2026-09"):
        fp = CACHE / f"intraday_{sym}_{m}.json"
        if not fp.exists():
            continue
        doc = json.loads(fp.read_text())
        key = next((k for k in doc if "Time Series" in k), None)
        if not key:
            continue
        for ts, row in doc[key].items():
            st = time.strptime(ts, "%Y-%m-%d %H:%M:%S")
            bars.append((calendar.timegm(st) + EDT, float(row["4. close"])))
    bars.sort()
    return bars


def ret(bars, t, back):
    """return over [t-back, t] using bars with epoch <= t (PIT). None if stale."""
    j = None
    for i, (e, _) in enumerate(bars):
        if e <= t:
            j = i
        else:
            break
    if j is None:
        return None, None
    last_e, last_v = bars[j]
    stale = t - last_e
    # find bar ~back seconds earlier
    v0 = None
    for (e, v) in bars[:j + 1][::-1]:
        if e <= t - back:
            v0 = v; break
    r = (last_v / v0 - 1.0) if v0 else 0.0
    return r, stale


def settlement_map(n_shards=96):
    files = sorted(glob.glob(str(SHARDS / "*.jsonl")), key=os.path.getmtime)[-n_shards:]
    last = {}
    for fp in files:
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("src") == "kalshi" and e.get("kind") == "quote":
                yb, ya, ts = e.get("yes_bid"), e.get("yes_ask"), e.get("receive_ts")
                if yb is None or ya is None or ts is None:
                    continue
                p = (yb + ya) / 200.0; tk = e.get("ticker")
                if tk not in last or ts > last[tk][0]:
                    last[tk] = (ts, p)
    return {tk: (1 if p > 0.5 else 0, ts) for tk, (ts, p) in last.items()
            if abs(p - 0.5) > 0.35}


def brier(p, y, w): return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def window_brier(tk, p, y):
    agg = {}
    for t, pp, yy in zip(tk, p, y):
        a = agg.setdefault(t, [0.0, 0, yy]); a[0] += pp; a[1] += 1
    return float(np.mean([((v[0] / v[1]) - v[2]) ** 2 for v in agg.values()])), len(agg)


def fit(X, y, w, zmkt, lam, it=600, lr=0.3):
    b = np.zeros(X.shape[1]); W = w / w.sum()
    for _ in range(it):
        p = sigmoid(zmkt + X @ b); b -= lr * (X.T @ (W * (p - y)) + lam * b)
    return b


def main():
    series = {s: load_sym(s) for s in SYMS}
    settle = settlement_map()
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    data = []
    perwin = {}
    for r in rows:
        s = settle.get(r["ticker"])
        if not s:
            continue
        t = r["ts"]
        feat = {}; stales = []
        for sym in SYMS:
            r5, st5 = ret(series[sym], t, 300)
            r15, _ = ret(series[sym], t, 900)
            feat[f"{sym}_ret5m"] = r5 if r5 is not None else 0.0
            feat[f"{sym}_ret15m"] = r15 if r15 is not None else 0.0
            stales.append(st5 if st5 is not None else 1e9)
        crypto = [feat["COIN_ret5m"], feat["MSTR_ret5m"], feat["IBIT_ret5m"]]
        feat["xasset_consensus"] = float(np.mean([np.sign(x) for x in crypto]))
        feat["xasset_dispersion"] = float(np.std(crypto))
        min_stale = min(stales)
        feat["av_stale_s"] = min_stale
        fresh = min_stale < 900        # within 15 min => RTH-fresh
        data.append({"ticker": r["ticker"], "ts": t, "y": s[0],
                     "k_prob": r["k_prob"], "fresh": fresh,
                     "tte_min": max(0.0, (s[1] - t) / 60.0), **feat})
        perwin[r["ticker"]] = perwin.get(r["ticker"], 0) + 1

    fresh = [d for d in data if d["fresh"]]
    cov = len(fresh) / len(data) if data else 0
    fresh_wins = len({d["ticker"] for d in fresh})
    print(f"Track B — {len(data)} settled points, session-FRESH {len(fresh)}"
          f" ({cov:.1%}) across {fresh_wins} windows")
    if len(fresh) < 400 or fresh_wins < 30:
        OUT.write_text(json.dumps({"verdict": "AV_NO_CANDIDATE",
            "reason": "insufficient session-fresh coverage",
            "coverage": cov, "fresh_points": len(fresh),
            "fresh_windows": fresh_wins}, indent=1))
        print("VERDICT: AV_NO_CANDIDATE (insufficient fresh coverage)")
        return

    FEATS = [f"{s}_ret5m" for s in SYMS] + [f"{s}_ret15m" for s in SYMS] + \
            ["xasset_consensus", "xasset_dispersion"]
    for d in fresh:
        d["w"] = 1.0 / perwin[d["ticker"]]
    first = {}
    for d in fresh:
        first[d["ticker"]] = min(first.get(d["ticker"], 1e18), d["ts"])
    order = sorted(first, key=lambda t: first[t]); n = len(order)
    trw = set(order[:int(.6 * n)]); vaw = set(order[int(.6 * n):int(.8 * n)])
    tew = set(order[int(.8 * n):])

    def sp(S):
        rs = [d for d in fresh if d["ticker"] in S]
        X = np.array([[d[f] for f in FEATS] for d in rs], float)
        y = np.array([d["y"] for d in rs], float)
        w = np.array([d["w"] for d in rs], float)
        pm = np.array([d["k_prob"] for d in rs], float)
        tk = [d["ticker"] for d in rs]
        return X, y, w, pm, tk
    Xtr, ytr, wtr, pmtr, _ = sp(trw)
    Xva, yva, wva, pmva, _ = sp(vaw)
    Xte, yte, wte, pmte, tkte = sp(tew)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd
    bmkt = brier(pmte, yte, wte); bmkt_w, _ = window_brier(tkte, pmte, yte)

    path = []; best = None
    for lam in LAMBDAS:
        b = fit(Xtr, ytr, wtr, logit(pmtr), lam)
        pv = sigmoid(logit(pmva) + Xva @ b)
        bss_v = 1 - brier(pv, yva, wva) / brier(pmva, yva, wva)
        pt = sigmoid(logit(pmte) + Xte @ b)
        bss_t = 1 - brier(pt, yte, wte) / bmkt
        bw, _ = window_brier(tkte, pt, yte); bss_tw = 1 - bw / bmkt_w
        ptr = sigmoid(logit(pmtr) + Xtr @ b)
        bss_tr = 1 - brier(ptr, ytr, wtr) / brier(pmtr, ytr, wtr)
        rec = {"lambda": lam, "bss_val": round(bss_v, 4), "bss_test_row": round(bss_t, 4),
               "bss_test_window": round(bss_tw, 4), "bss_train": round(bss_tr, 4)}
        path.append(rec)
        if best is None or bss_v > best["bss_val"]:
            best = rec
    print("λ path (BSS vs market):")
    for r in path:
        print(f"  λ={r['lambda']:<7} val {r['bss_val']:+.4f} test_row {r['bss_test_row']:+.4f}"
              f" test_win {r['bss_test_window']:+.4f} train {r['bss_train']:+.4f}")

    tr0 = path[0]["bss_train"]
    if tr0 < 0.01 and best["bss_test_window"] < 0.005:
        diag = "AV_INFORMATION_LIMITED"
    elif tr0 > 0.02 and best["bss_test_window"] < 0.005:
        diag = "AV_CLASSIC_OVERFIT"
    elif best["bss_test_window"] >= 0.005:
        diag = "AV_SETTLEMENT_SIGNAL_PROMISING"
    else:
        diag = "AV_NO_CANDIDATE"
    out = {"total_settled_points": len(data), "fresh_points": len(fresh),
           "session_coverage": round(cov, 3), "fresh_windows": fresh_wins,
           "windows_split": [len(trw), len(vaw), len(tew)],
           "market_brier_test": round(bmkt, 5), "market_brier_window": round(bmkt_w, 5),
           "base_rate": round(float(yte.mean()), 3),
           "lambda_path": path, "best": best, "verdict": diag,
           "note": "session-fresh subset only; AV equities not 24/7; target=settlement."}
    OUT.write_text(json.dumps(out, indent=1))
    print("VERDICT:", diag)


if __name__ == "__main__":
    main()
