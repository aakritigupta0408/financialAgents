"""FEATURE POWER SCAN at ENTRY (window open) — hunt for the signal that unlocks 89%.

For every numeric feature logged at the entry decision (earliest PIT observation per window),
measure its STANDALONE predictive accuracy for the close direction (exact_yes): best single-
threshold accuracy + AUC. If any feature reaches ~0.89 it is the oracle (then verify it is
truly point-in-time, not a leak). If nothing exceeds ~0.60, the unlock is not in our current
features and must come from new data. Also reports the market (k_prob) accuracy at entry and a
full-model (all features, walk-forward) number as the current combined ceiling.
"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "brti_decision_dataset.jsonl"
KB = ROOT / "results" / "kalshi_binary_log.jsonl"
OUT = ROOT / "research" / "feature_power_scan.json"


def earliest_rows():
    byw = defaultdict(list)
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1) and d.get("time_remaining_s") is not None:
            byw[d.get("market_window_id")].append(d)
    out = {}
    for wid, rows in byw.items():
        rows.sort(key=lambda r: -(r.get("time_remaining_s") or 0))
        out[wid] = rows[0]
    return out


def univ_acc(x, y):
    """Best single-threshold accuracy (either direction) + AUC, ignoring NaN."""
    x = np.asarray(x, float); y = np.asarray(y, int)
    m = np.isfinite(x)
    if m.sum() < 30 or len(set(y[m])) < 2:
        return None
    xs, ys = x[m], y[m]
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    # threshold sweep: predict up if x>=t (and the flipped rule); best accuracy
    uniq = np.unique(xs)
    if len(uniq) > 200:
        uniq = np.quantile(xs, np.linspace(0, 1, 200))
    best = 0.5
    for t in uniq:
        pred = (xs >= t).astype(int)
        best = max(best, (pred == ys).mean(), (1 - pred == ys).mean())
    # AUC (rank)
    pos = ys == 1; neg = ys == 0
    if pos.sum() and neg.sum():
        ranks = np.argsort(np.argsort(xs))
        auc = (ranks[pos].mean() - ranks[neg].mean()) / len(xs) + 0.5
        auc = max(auc, 1 - auc)
    else:
        auc = None
    return {"n": int(m.sum()), "best_acc": round(float(best), 4),
            "auc": round(float(auc), 4) if auc is not None else None}


def run():
    E = earliest_rows()
    wids = list(E)
    y = np.array([E[w]["exact_yes"] for w in wids], int)
    # collect every numeric field present
    fields = set()
    for w in wids:
        for k, v in E[w].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "exact_yes":
                fields.add(k)
    scan = {}
    for f in sorted(fields):
        x = [E[w].get(f, np.nan) for w in wids]
        r = univ_acc(x, y)
        if r:
            scan[f] = r
    # derived: barrier distance normalized, dist sign
    ranked = sorted(scan.items(), key=lambda kv: -(kv[1]["best_acc"]))
    # full model (all numeric features, walk-forward)
    keys = [f for f in sorted(fields)]
    X = np.array([[E[w].get(k, np.nan) for k in keys] for w in wids], float)
    X = np.nan_to_num(X)
    ts = np.array([E[w].get("decision_time") or 0 for w in wids])
    o = np.argsort(ts); X, yy = X[o], y[o]
    mid = int(len(yy) * 0.6)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                        l2_regularization=2.0, min_samples_leaf=15, random_state=17)
    gb.fit(X[:mid], yy[:mid]); full_acc = round(float((gb.predict(X[mid:]) == yy[mid:]).mean()), 4)
    rep = {"n_windows": len(wids), "base_up": round(float(y.mean()), 4),
           "top_features": ranked[:15], "full_model_oos_acc": full_acc}
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"n={len(wids)} base-up={rep['base_up']}  full-model OOS acc={full_acc}")
    print("TOP standalone features (best single-threshold acc | AUC):")
    for f, r in ranked[:15]:
        print(f"  {f:32} acc {r['best_acc']}  auc {r['auc']}  (n {r['n']})")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    run()
