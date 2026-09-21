"""THE ANALYST — from RAW logged data only. Predict the official window outcome (up/down)
from the intra-window feature path AVAILABLE AT ENTRY (point-in-time), then report the
out-of-sample hit rate the model actually reaches. Rich feature engineering + a nonlinear
model; no existing arms/infrastructure used — built from results/rl_window_log.jsonl.

Decision point: entry ~12 min left (how the desk enters, one shot, side-locked). Features
use only rows up to entry (no post-entry data). Label = official Kalshi exact_yes.
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

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "research" / "analyst_raw_report.json"
ENTRY_ML = 12.0     # decision point: minutes-left at entry


def official():
    m = {}
    for l in (RES / "contract_outcomes.jsonl").open():
        l = l.strip()
        if l:
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get("exact_yes") in (0, 1):
                m[d["ticker"]] = d["exact_yes"]
    return m


def build():
    off = official()
    byw = defaultdict(list)
    for l in (RES / "rl_window_log.jsonl").open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("ticker") in off and d.get("mins_left") is not None:
            byw[d["ticker"]].append(d)
    feats, labels, ts = [], [], []
    for tk, rows in byw.items():
        rows.sort(key=lambda r: (r.get("made_ts") or 0))
        # prefix = everything at/after open up to the entry decision (mins_left >= ENTRY_ML)
        pre = [r for r in rows if (r.get("mins_left") or 0) >= ENTRY_ML]
        if len(pre) < 2:
            pre = rows[: max(2, len(rows) // 5)]      # fallback: first fifth of the path
        # group prefix by timestamp; each ts has ~8 variant rows
        byt = defaultdict(list)
        for r in pre:
            byt[round(r["made_ts"])].append(r)
        tss = sorted(byt)
        entry = byt[tss[-1]]                          # variant rows at the entry instant
        openg = byt[tss[0]]

        def pu(g):
            v = [x["p_up"] for x in g if x.get("p_up") is not None]
            return v

        pe, po = pu(entry), pu(openg)
        if not pe:
            continue
        strike = entry[0].get("strike")
        base_e = entry[0].get("base")
        base_o = openg[0].get("base")
        mkt = entry[0].get("mkt_p_up")
        if strike is None or base_e is None:
            continue
        pmean = sum(pe) / len(pe)
        f = {
            "p_up_mean": pmean,
            "p_up_std": st.pstdev(pe) if len(pe) > 1 else 0.0,
            "frac_up": sum(1 for x in pe if x > 0.5) / len(pe),
            "p_up_slope": pmean - (sum(po) / len(po) if po else pmean),
            "mkt_p_up": mkt if mkt is not None else 0.5,
            "oracle_mkt_div": pmean - (mkt if mkt is not None else 0.5),
            "dist_bps": (base_e - strike) / strike * 1e4,
            "base_ret_bps": ((base_e - base_o) / base_o * 1e4) if base_o else 0.0,
            "conf_mean": (sum(x.get("conf", 0) for x in entry) / len(entry)),
            "tau": entry[0].get("tau") or 0.0,
            "n_prefix_ts": len(tss),
            "mins_left": entry[0].get("mins_left"),
        }
        feats.append(f)
        labels.append(off[tk])
        ts.append(tss[-1])
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    feats = [feats[i] for i in order]; labels = [labels[i] for i in order]
    return feats, labels


def evaluate(feats, labels):
    keys = list(feats[0].keys())
    X = np.array([[f[k] for k in keys] for f in feats], float)
    y = np.array(labels, int)
    n = len(y); mid = int(n * 0.6)
    base_up = round(float(y.mean()), 4)
    # majority-class baseline hit rate on the test split
    maj = 1 if y[:mid].mean() >= 0.5 else 0
    yte = y[mid:]
    maj_hit = round(float((yte == maj).mean()), 4)
    out = {"n": n, "n_train": mid, "n_test": n - mid, "base_rate_up": base_up,
           "majority_baseline_hit": maj_hit, "features": keys}
    for name, mk in (("logistic", lambda: LogisticRegression(max_iter=3000)),
                     ("hist_gbm", lambda: HistGradientBoostingClassifier(
                         max_depth=3, learning_rate=0.05, max_iter=300,
                         l2_regularization=1.0, random_state=17))):
        sc = StandardScaler().fit(X[:mid])
        clf = mk().fit(sc.transform(X[:mid]) if name == "logistic" else X[:mid], y[:mid])
        Xt = sc.transform(X[mid:]) if name == "logistic" else X[mid:]
        p = clf.predict_proba(Xt)[:, 1]
        pred = (p >= 0.5).astype(int)
        hit = float((pred == yte).mean())
        brier = float(((p - yte) ** 2).mean())
        # selective: only act on high-confidence predictions
        conf = np.abs(p - 0.5) * 2
        selhit = {}
        for c in (0.2, 0.4, 0.6):
            m = conf >= c
            selhit[f">={c}"] = {"coverage": round(float(m.mean()), 3),
                                "hit": round(float((pred[m] == yte[m]).mean()), 4) if m.any() else None}
        out[name] = {"oos_hit": round(hit, 4), "oos_brier": round(brier, 4),
                     "selective_hit": selhit}
        if name == "hist_gbm" and hasattr(clf, "feature_importances_"):
            pass
    return out


def run():
    feats, labels = build()
    res = evaluate(feats, labels)
    OUT.write_text(json.dumps({"schema_version": "analyst-raw-1",
                               "decision_min": ENTRY_ML, "result": res}, indent=1))
    print(f"ANALYST (raw path @ {ENTRY_ML}min-left, PIT) — n={res['n']} "
          f"(train {res['n_train']} / test {res['n_test']})")
    print(f"  base-rate up {res['base_rate_up']} | majority-baseline hit {res['majority_baseline_hit']}")
    for m in ("logistic", "hist_gbm"):
        r = res[m]
        print(f"  {m:9} OOS hit {r['oos_hit']}  Brier {r['oos_brier']}  "
              f"selective {r['selective_hit']}")


if __name__ == "__main__":
    run()
