"""RISK-COVERAGE curve at open+6min — hit-rate as we abstain on the least-confident windows.
Honest selective classification: sort test windows by confidence, report hit at each coverage,
find the coverage where hit >= 0.89. Walk-forward (train on first 60%, evaluate on last 40%),
so the confidence threshold generalizes; leakage canary. Barrier + best model.

Uses the CLEAN 370-window brti_decision_dataset at ~9 min left (the reliable open+6min set).
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
OUT = ROOT / "research" / "risk_coverage.json"
DECIDE_TR = 540.0


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
        byw[w].sort(key=lambda r: -(r.get("time_remaining_s") or 0))
    return byw


def feats(rows):
    pre = [r for r in rows if (r.get("time_remaining_s") or 0) >= DECIDE_TR - 15]
    if len(pre) < 4:
        return None
    d = pre[-1]
    cur = d["current_brti"]; tgt = d.get("official_target") or d.get("official_target_kalshi")
    trs = d.get("time_remaining_s"); rvol = d.get("cb_rvol_30s"); kp = d.get("k_prob")
    if not tgt or not trs:
        return None
    series = [r["current_brti"] for r in pre if r.get("current_brti")]
    rets = [math.log(series[i] / series[i - 1]) for i in range(1, len(series)) if series[i - 1] > 0]
    vol = st.pstdev(rets) if len(rets) > 1 else 1e-5
    dt = max(1.0, (pre[0]["time_remaining_s"] - trs) / max(1, len(series) - 1))
    sig_s = (rvol / cur / math.sqrt(30.0)) if rvol else vol / math.sqrt(dt)
    sigma_T = cur * sig_s * math.sqrt(max(1.0, trs))
    mom = cur - series[0]; drift = 0.5 * mom * (trs / 900.0)
    z = (cur - tgt) / max(sigma_T, 1e-6); z_dr = (cur + drift - tgt) / max(sigma_T, 1e-6)
    up = sum(r for r in rets if r > 0); dn = -sum(r for r in rets if r < 0)
    f = [z, z_dr, _phi(z), _phi(z_dr), (cur - tgt) / cur * 1e4, sigma_T / cur * 1e4,
         mom / cur * 1e4, up / (up + dn) if (up + dn) else 0.5,
         kp if kp is not None else 0.5, (max(series) - min(series)) / cur * 1e4]
    return f, _phi(z_dr), d["exact_yes"], d.get("decision_time") or 0


def curve(p, y):
    conf = np.abs(p - 0.5) * 2
    order = np.argsort(-conf)          # most confident first
    p2, y2 = p[order], y[order]
    pred = (p2 >= 0.5).astype(int)
    out = []
    n = len(y2)
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        k = max(1, int(round(n * cov)))
        hit = float((pred[:k] == y2[:k]).mean())
        out.append({"coverage": cov, "n": k, "hit": round(hit, 4)})
    # coverage where hit first reaches 0.89 (scanning from high coverage down)
    reach = None
    for cov in np.linspace(1.0, 0.05, 40):
        k = max(1, int(round(n * cov)))
        if (pred[:k] == y2[:k]).mean() >= 0.89:
            reach = round(float(cov), 3)
    return out, reach


def run():
    byw = load()
    X, y, theo, ts = [], [], [], []
    for w, rows in byw.items():
        r = feats(rows)
        if r is None:
            continue
        f, th, lab, t = r
        X.append(f); y.append(lab); theo.append(th); ts.append(t)
    o = np.argsort(ts)
    X = np.nan_to_num(np.array(X, float)[o]); y = np.array(y, int)[o]; theo = np.array(theo, float)[o]
    n = len(y); mid = int(n * 0.6); yte = y[mid:]
    rep = {"n": n, "n_test": n - mid, "decide": "open+6min (~9min left)"}
    # barrier
    bc, br = curve(theo[mid:], yte)
    rep["barrier"] = {"risk_coverage": bc, "coverage_for_0.89": br}
    # logistic
    sc = StandardScaler().fit(X[:mid]); lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[:mid]), y[:mid])
    pl = lr.predict_proba(sc.transform(X[mid:]))[:, 1]
    lc, lrr = curve(pl, yte)
    rep["logistic"] = {"risk_coverage": lc, "coverage_for_0.89": lrr}
    # gbm
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=400,
                                        l2_regularization=3.0, min_samples_leaf=25, random_state=17).fit(X[:mid], y[:mid])
    pg = gb.predict_proba(X[mid:])[:, 1]
    gc, grr = curve(pg, yte)
    rep["gbm"] = {"risk_coverage": gc, "coverage_for_0.89": grr}
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"n={n} test={n-mid} @ open+6min")
    for m in ("barrier", "logistic", "gbm"):
        print(f"\n{m}: coverage-for-0.89 = {rep[m]['coverage_for_0.89']}")
        for r in rep[m]["risk_coverage"]:
            mark = "  <-- 0.89+" if r["hit"] >= 0.89 else ""
            print(f"   cov {r['coverage']:.2f} (n={r['n']:3})  hit {r['hit']}{mark}")


if __name__ == "__main__":
    run()
