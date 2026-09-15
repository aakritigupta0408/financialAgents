"""Evaluate exogenous (order-flow) + time-of-inference features for window direction.

Answers: (1) does time-of-day/day-of-week structure predict direction on ALL ~6.3k labeled
windows? (2) do the off-path order-flow features (results/exo_features.jsonl) carry OOS
signal? (3) do they ADD to a simple on-path barrier baseline? Walk-forward 60/40 by time,
leakage canary on the fused model. Time features are backfilled from each window's timestamp.
"""
import json, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RES = Path("results")
OUT = Path("research/exo_eval_report.json")
ET = timezone(timedelta(hours=-4))   # EDT for Aug-Sep 2026 (user trades ET)


def _ep(t):
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def time_feats(ep):
    """Cyclical + raw calendar features of the inference time. Backfillable from the
    timestamp alone, so valid for every historic window."""
    u = datetime.fromtimestamp(ep, tz=timezone.utc)
    e = datetime.fromtimestamp(ep, tz=ET)
    hu, he, dow, dom, mon = u.hour + u.minute/60, e.hour + e.minute/60, e.weekday(), e.day, e.month
    def sc(v, p): return [math.sin(2*math.pi*v/p), math.cos(2*math.pi*v/p)]
    return {
        "hour_utc": hu, "hour_et": he, "dow": dow, "dom": dom, "month": mon,
        "sin_hu": sc(hu, 24)[0], "cos_hu": sc(hu, 24)[1],
        "sin_he": sc(he, 24)[0], "cos_he": sc(he, 24)[1],
        "sin_dow": sc(dow, 7)[0], "cos_dow": sc(dow, 7)[1],
        "sin_dom": sc(dom, 31)[0], "cos_dom": sc(dom, 31)[1],
        "sin_mon": sc(mon, 12)[0], "cos_mon": sc(mon, 12)[1],
        "us_session": 1.0 if (9.5 <= he < 16 and dow < 5) else 0.0,
        "asia_session": 1.0 if (20 <= he or he < 3) else 0.0,
        "weekend": 1.0 if dow >= 5 else 0.0,
    }


def labels():
    m = {}
    for l in (RES / "contract_outcomes.jsonl").open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1):
            ct = _ep(d.get("close_time"))
            if ct:
                m[d["ticker"]] = (int(d["exact_yes"]), ct, d.get("floor_strike"))
    return m


def _ev(name, Xtr, ytr, Xte, yte, kind="both"):
    out = {}
    if kind in ("lin", "both"):
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr), ytr)
        p = lr.predict(sc.transform(Xte))
        out["logistic_oos_acc"] = round(float((p == yte).mean()), 4)
    if kind in ("gbm", "both"):
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                            l2_regularization=2.0, min_samples_leaf=20, random_state=17)
        gb.fit(Xtr, ytr); p = gb.predict(Xte)
        out["gbm_oos_acc"] = round(float((p == yte).mean()), 4)
        out["gbm_in_sample_acc"] = round(float((gb.predict(Xtr) == ytr).mean()), 4)
    return out


def run():
    lab = labels()
    rep = {"n_labeled": len(lab), "tests": {}}

    # ---- (1) TIME-ONLY on ALL labeled windows ----
    items = sorted(lab.items(), key=lambda kv: kv[1][1])   # by close time
    tkeys = list(time_feats(items[0][1][1]).keys())
    Xt = np.array([[time_feats(v[1])[k] for k in tkeys] for _, v in items], float)
    yt = np.array([v[0] for _, v in items], int)
    m = int(len(yt)*0.6)
    rep["tests"]["time_only_ALL"] = {"n": len(yt), "n_test": len(yt)-m,
                                     "base_up": round(float(yt.mean()), 4),
                                     **_ev("time", Xt[:m], yt[:m], Xt[m:], yt[m:])}

    # ---- (2,3) EXO (+time, +barrier) on windows with captured flow ----
    exo_p = RES / "exo_features.jsonl"
    if exo_p.exists():
        rows = [json.loads(l) for l in exo_p.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("ticker") in lab]
        rows.sort(key=lambda r: lab[r["ticker"]][1])
        y = np.array([lab[r["ticker"]][0] for r in rows], int)
        exok = ["cb_ofi", "cb_ofi_notional", "cb_trades", "book_imb", "spread_bps",
                "bn_ofi", "bn_trades", "basis_bps"]
        Xe = np.nan_to_num(np.array([[r.get(k) or 0.0 for k in exok] for r in rows], float))
        # on-path barrier baseline from the captured coinbase mids vs floor
        def barrier_row(r):
            me, mo, fl = r.get("cb_mid_entry"), r.get("cb_mid_open"), r.get("floor")
            dist = ((me - fl)/fl*1e4) if (me and fl) else 0.0
            drift = ((me - mo)/mo*1e4) if (me and mo) else 0.0
            return [dist, drift]
        Xb = np.nan_to_num(np.array([barrier_row(r) for r in rows], float))
        Xtm = np.array([[time_feats(lab[r["ticker"]][1])[k] for k in tkeys] for r in rows], float)
        m2 = int(len(y)*0.6)
        def sl(X): return (X[:m2], X[m2:])
        seg = {"exo_only": Xe, "barrier_only": Xb, "time_only": Xtm,
               "barrier+exo": np.column_stack([Xb, Xe]),
               "barrier+exo+time": np.column_stack([Xb, Xe, Xtm])}
        rep["tests"]["_flow_subset"] = {"n": len(y), "n_test": len(y)-m2, "base_up": round(float(y.mean()), 4)}
        for nm, X in seg.items():
            a, b = sl(X)
            rep["tests"][nm] = _ev(nm, a, y[:m2], b, y[m2:])
        # leakage canary on the richest model
        rng = np.random.default_rng(0); ysh = y.copy(); ysh[:m2] = rng.permutation(ysh[:m2])
        Xall = np.column_stack([Xb, Xe, Xtm]); a, b = sl(Xall)
        rep["leakage_canary_oos"] = _ev("canary", a, ysh[:m2], b, y[m2:], kind="gbm")
    else:
        rep["tests"]["_flow_subset"] = "exo_features.jsonl not ready yet"

    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    run()
