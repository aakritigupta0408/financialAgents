"""THE ANALYST (deep v2) — validated quant/TA methods on the intra-window BRTI path.

Improvements over v1, each grounded in a validated method:
  * EWMA realized vol (lambda .94, RiskMetrics) from the intra-window BRTI returns, blended
    with the logged 30s realized vol -> a sharper sigma over the remaining time.
  * DRIFT-ADJUSTED barrier: intraday time-series momentum (early-window move predictably
    continues; Bitcoin Intraday TS-Momentum, Univ. Reading 2021) added as drift in the
    first-passage probability.
  * AVERAGED (Asian) settlement: Kalshi settles on the trailing-60s BRTI mean, whose
    variance is ~1/3 of the point value -> variance-reduced barrier.
  * TA features on the prefix path: multi-scale momentum, RSI, distance-from-short-MA
    (mean reversion), realized vol, coinbase-BRTI basis.
Model: regularized logistic + GBM + an ENSEMBLE of (drift barrier, market k_prob, logistic).
PIT at entry (~11 min left); label = official exact_yes. Report OOS + feature impact.
"""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "brti_decision_dataset.jsonl"
OUT = ROOT / "research" / "analyst_deep_v2_report.json"
ENTRY_TR = 660.0
LAM = 0.94                # RiskMetrics EWMA decay
AVG_VAR_FACTOR = 1.0 / 3  # variance of a uniform trailing average vs point (~1/3)


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ewma_vol(rets):
    if len(rets) < 2:
        return None
    v = rets[0] ** 2
    for r in rets[1:]:
        v = LAM * v + (1 - LAM) * r * r
    return math.sqrt(max(v, 0.0))


def _rsi(rets):
    if not rets:
        return 0.5
    up = sum(r for r in rets if r > 0); dn = -sum(r for r in rets if r < 0)
    if up + dn == 0:
        return 0.5
    return up / (up + dn)


def build():
    byw = defaultdict(list)
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1) and d.get("time_remaining_s") is not None \
                and d.get("current_brti"):
            byw[d.get("market_window_id")].append(d)
    feats, labels, ts = [], [], []
    for wid, rows in byw.items():
        rows.sort(key=lambda r: -(r.get("time_remaining_s") or 0))       # early -> late
        pre = [r for r in rows if (r.get("time_remaining_s") or 0) >= ENTRY_TR]
        if len(pre) < 1:
            pre = rows[:max(1, len(rows) // 6)]
        d = pre[-1]                                                      # entry row
        cur = d["current_brti"]
        tgt = d.get("official_target") or d.get("official_target_kalshi")
        tr = d.get("time_remaining_s")
        kp = d.get("k_prob")
        rvol30 = d.get("cb_rvol_30s")
        if not tgt or not tr:
            continue
        series = [r["current_brti"] for r in pre if r.get("current_brti")]
        rets = [math.log(series[i] / series[i - 1]) for i in range(1, len(series))
                if series[i - 1] > 0]
        ew = _ewma_vol(rets)
        # per-second vol: prefer EWMA (per-sample) mapped to seconds via avg dt; blend rvol30
        dt = max(1.0, (pre[0]["time_remaining_s"] - tr) / max(1, len(series) - 1)) if len(series) > 1 else 30.0
        sig_s_ew = (ew / math.sqrt(dt)) if ew else None                  # per-sqrt-second
        sig_s_r = (rvol30 / cur) / math.sqrt(30.0) if rvol30 else None   # rvol30 is $/30s -> frac
        sig_s = np.nanmean([x for x in (sig_s_ew, sig_s_r) if x is not None]) if (sig_s_ew or sig_s_r) else 1e-5
        # variance-reduced sigma over remaining time to the averaged settlement
        sigma_T = cur * sig_s * math.sqrt(tr) * math.sqrt(AVG_VAR_FACTOR + (1 - AVG_VAR_FACTOR) * 0)
        sigma_T = max(sigma_T, 1e-6)
        # intraday TS-momentum: early-window drift, assumed to partly continue
        mom = (cur - series[0]) if series else 0.0
        drift = 0.5 * mom * (tr / max(1.0, pre[0]["time_remaining_s"]))   # damped continuation
        z_plain = (cur - tgt) / sigma_T
        z_drift = (cur + drift - tgt) / sigma_T
        # short-MA mean reversion + RSI on the prefix
        k = min(5, len(series))
        short_ma = sum(series[-k:]) / k if k else cur
        f = {
            "z_drift": z_drift,
            "theo_up_drift": _phi(z_drift),
            "z_plain": z_plain,
            "dist_bps": (cur - tgt) / cur * 1e4,
            "sigma_T_bps": sigma_T / cur * 1e4,
            "mom_bps": (mom / cur * 1e4) if cur else 0.0,
            "ma_dist_bps": (cur - short_ma) / cur * 1e4,
            "rsi": _rsi(rets),
            "k_prob": kp if kp is not None else 0.5,
            "kprob_vs_theo": (kp - _phi(z_drift)) if kp is not None else 0.0,
            "time_remaining_s": tr,
            "n_prefix": len(series),
        }
        feats.append(f); labels.append(d["exact_yes"]); ts.append(d.get("decision_time") or 0)
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return [feats[i] for i in order], [labels[i] for i in order]


def evaluate(feats, labels):
    keys = list(feats[0].keys())
    X = np.nan_to_num(np.array([[f[k] for k in keys] for f in feats], float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels, int)
    n = len(y); mid = int(n * 0.6); yte = y[mid:]
    maj = 1 if y[:mid].mean() >= 0.5 else 0
    out = {"n": n, "n_test": n - mid, "base_rate_up": round(float(y.mean()), 4),
           "majority_hit": round(float((yte == maj).mean()), 4), "features": keys}

    def hit(pred):
        return round(float((pred == yte).mean()), 4)
    out["theo_drift_only_hit"] = hit((X[mid:, keys.index("theo_up_drift")] >= 0.5).astype(int))
    out["market_kprob_hit"] = hit((X[mid:, keys.index("k_prob")] >= 0.5).astype(int))

    probs = {}
    for name, mk, scale in (("logistic", lambda: LogisticRegression(C=0.5, max_iter=4000), True),
                            ("gbm", lambda: HistGradientBoostingClassifier(
                                max_depth=3, learning_rate=0.04, max_iter=400,
                                l2_regularization=2.0, min_samples_leaf=15, random_state=17), False)):
        sc = StandardScaler().fit(X[:mid]) if scale else None
        Xtr = sc.transform(X[:mid]) if scale else X[:mid]
        Xte = sc.transform(X[mid:]) if scale else X[mid:]
        clf = mk().fit(Xtr, y[:mid]); p = clf.predict_proba(Xte)[:, 1]; probs[name] = p
        pred = (p >= 0.5).astype(int)
        conf = np.abs(p - 0.5) * 2
        sel = {c: {"cov": round(float((conf >= c).mean()), 3),
                   "hit": round(float((pred[conf >= c] == yte[conf >= c]).mean()), 4) if (conf >= c).any() else None}
               for c in (0.2, 0.4, 0.6)}
        pi = permutation_importance(clf, Xte, yte, n_repeats=20, random_state=17)
        out[name] = {"oos_hit": hit(pred), "oos_brier": round(float(((p - yte) ** 2).mean()), 4),
                     "selective": sel,
                     "impact": sorted([(k, round(float(v), 4)) for k, v in zip(keys, pi.importances_mean)],
                                      key=lambda kv: -kv[1])[:6]}
    # ENSEMBLE: average drift-barrier, market, logistic
    ens = (X[mid:, keys.index("theo_up_drift")] + X[mid:, keys.index("k_prob")] + probs["logistic"]) / 3
    epred = (ens >= 0.5).astype(int); econf = np.abs(ens - 0.5) * 2
    out["ensemble"] = {"oos_hit": hit(epred), "oos_brier": round(float(((ens - yte) ** 2).mean()), 4),
                       "selective": {c: {"cov": round(float((econf >= c).mean()), 3),
                                         "hit": round(float((epred[econf >= c] == yte[econf >= c]).mean()), 4) if (econf >= c).any() else None}
                                     for c in (0.2, 0.4, 0.6)}}
    return out


def run():
    feats, labels = build()
    res = evaluate(feats, labels)
    OUT.write_text(json.dumps({"schema_version": "analyst-deep-v2", "entry_tr": ENTRY_TR, "result": res}, indent=1))
    print(f"ANALYST DEEP v2 (validated quant/TA, PIT @ {int(ENTRY_TR)}s) — n={res['n']} test={res['n_test']} base-up {res['base_rate_up']}")
    print(f"  baselines: majority {res['majority_hit']} | drift-barrier {res['theo_drift_only_hit']} | market {res['market_kprob_hit']}")
    for m in ("logistic", "gbm", "ensemble"):
        r = res[m]
        print(f"  {m:9} OOS hit {r['oos_hit']} Brier {r['oos_brier']} selective {r['selective']}")
        if "impact" in r:
            print(f"     impact: {r['impact']}")


if __name__ == "__main__":
    run()
