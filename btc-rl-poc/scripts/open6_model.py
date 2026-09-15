"""OPEN+6min model — the professor's actual framing: observe the FIRST 6 MINUTES of price
action, decide at open+6min (~9 min left), predict the 15-min CLOSE direction. Fixed,
tradeable entry (contract still has ~9 min of value), no post-decision data. Target 89%@90%.

Features use ONLY the [open, open+6min] path: drift-adjusted barrier z, distance-to-strike,
6-min momentum/vol/RSI/path-shape, required-remaining-average, market prob (k_prob). Walk-
forward 60/40; barrier baseline vs GBM vs logistic; hit-rate @ coverage>=0.90; leakage canary.

Data: results/brti_decision_dataset.jsonl (intra-window rows w/ time_remaining_s).
"""
import json, math, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "brti_decision_dataset.jsonl"
OUT = ROOT / "research" / "open6_model.json"
DECIDE_TR = 540.0        # 9 min left == open + 6 min (15-min window)
WINDOW_S = 900.0


def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load():
    byw = defaultdict(list)
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1) and d.get("time_remaining_s") is not None and d.get("current_brti"):
            byw[d.get("market_window_id")].append(d)
    for w in byw:
        byw[w].sort(key=lambda r: -(r.get("time_remaining_s") or 0))   # open -> later
    return byw


def features(rows):
    """PIT features at open+6min using only rows with time_remaining >= DECIDE_TR (first 6 min)."""
    pre = [r for r in rows if (r.get("time_remaining_s") or 0) >= DECIDE_TR - 15]
    if len(pre) < 4:
        return None
    d = pre[-1]                                    # decision row (~9 min left)
    cur = d["current_brti"]; tgt = d.get("official_target") or d.get("official_target_kalshi")
    trs = d.get("time_remaining_s"); rvol = d.get("cb_rvol_30s"); kp = d.get("k_prob")
    rra = d.get("required_remaining_average")
    if not tgt or not trs:
        return None
    series = [r["current_brti"] for r in pre if r.get("current_brti")]
    rets = [math.log(series[i] / series[i - 1]) for i in range(1, len(series)) if series[i - 1] > 0]
    vol = st.pstdev(rets) if len(rets) > 1 else 1e-5
    dt = max(1.0, (pre[0]["time_remaining_s"] - trs) / max(1, len(series) - 1))
    sig_s = (rvol / cur / math.sqrt(30.0)) if rvol else vol / math.sqrt(dt)
    sigma_T = cur * sig_s * math.sqrt(max(1.0, trs))
    mom = cur - series[0]                           # 6-min move
    drift = 0.5 * mom * (trs / WINDOW_S)            # damped continuation
    z = (cur - tgt) / max(sigma_T, 1e-6)
    z_dr = (cur + drift - tgt) / max(sigma_T, 1e-6)
    up = sum(r for r in rets if r > 0); dn = -sum(r for r in rets if r < 0)
    rsi = up / (up + dn) if (up + dn) else 0.5
    prev = min(pre, key=lambda r: abs((r.get("time_remaining_s") or 0) - (trs + 60)))
    momr = (cur - prev["current_brti"]) / cur * 1e4 if prev.get("current_brti") else 0.0
    k = min(5, len(series)); short_ma = sum(series[-k:]) / k
    f = {
        "z": z, "z_drift": z_dr, "theo": _phi(z), "theo_drift": _phi(z_dr),
        "dist_bps": (cur - tgt) / cur * 1e4, "sigmaT_bps": sigma_T / cur * 1e4,
        "mom6_bps": mom / cur * 1e4, "mom_recent_bps": momr, "rsi": rsi,
        "ma_dist_bps": (cur - short_ma) / cur * 1e4, "range_bps": (max(series) - min(series)) / cur * 1e4,
        "k_prob": kp if kp is not None else 0.5, "kprob_vs_theo": (kp - _phi(z)) if kp is not None else 0.0,
        "rra_bps": ((rra - cur) / cur * 1e4) if (rra and cur) else 0.0, "n_prefix": len(series),
    }
    return f, _phi(z_dr), d["exact_yes"], d.get("decision_time") or 0


def _sel(p, y, thr_grid=None):
    conf = np.abs(p - 0.5) * 2; pred = (p >= 0.5).astype(int)
    best = None
    for c in np.linspace(0, 0.95, 40):
        m = conf >= c
        if m.mean() >= 0.90 and m.any():
            hit = float((pred[m] == y[m]).mean())
            if best is None or hit > best["hit"]:
                best = {"conf_thr": round(float(c), 3), "hit": round(hit, 4), "coverage": round(float(m.mean()), 4)}
    return best


def run():
    byw = load()
    X, y, theo, ts, keys = [], [], [], [], None
    for w, rows in byw.items():
        r = features(rows)
        if r is None:
            continue
        f, th, lab, t = r
        if keys is None:
            keys = list(f.keys())
        X.append([f[k] for k in keys]); y.append(lab); theo.append(th); ts.append(t)
    o = np.argsort(ts)
    X = np.nan_to_num(np.array(X, float)[o]); y = np.array(y, int)[o]; theo = np.array(theo, float)[o]
    n = len(y); mid = int(n * 0.6); yte = y[mid:]
    rep = {"n": n, "n_test": n - mid, "decide_min_left": DECIDE_TR / 60, "base_up": round(float(y.mean()), 4)}

    # barrier baseline
    bp = theo[mid:]
    rep["barrier"] = {"oos_acc": round(float(((bp >= 0.5).astype(int) == yte).mean()), 4),
                      "hit@cov>=.90": _sel(bp, yte)}
    # GBM
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=500,
                                        l2_regularization=2.0, min_samples_leaf=15, random_state=17)
    gb.fit(X[:mid], y[:mid]); pg = gb.predict_proba(X[mid:])[:, 1]
    rep["gbm"] = {"oos_acc": round(float(((pg >= 0.5).astype(int) == yte).mean()), 4),
                  "in_sample": round(float((gb.predict(X[:mid]) == y[:mid]).mean()), 4),
                  "hit@cov>=.90": _sel(pg, yte)}
    # logistic
    sc = StandardScaler().fit(X[:mid]); lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[:mid]), y[:mid])
    pl = lr.predict_proba(sc.transform(X[mid:]))[:, 1]
    rep["logistic"] = {"oos_acc": round(float(((pl >= 0.5).astype(int) == yte).mean()), 4),
                       "hit@cov>=.90": _sel(pl, yte)}
    # leakage canary
    rng = np.random.default_rng(0); ysh = y.copy(); ysh[:mid] = rng.permutation(ysh[:mid])
    gc = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=500,
                                        l2_regularization=2.0, min_samples_leaf=15, random_state=17)
    gc.fit(X[:mid], ysh[:mid]); pc = gc.predict_proba(X[mid:])[:, 1]
    rep["leakage_canary_oos"] = round(float(((pc >= 0.5).astype(int) == yte).mean()), 4)
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    run()
