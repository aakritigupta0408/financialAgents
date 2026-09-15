"""IN-SAMPLE RECONSTRUCTION across ALL models, on the FULL logged history (~1.3k windows).

Answers the user's question: can each model reproduce the labels on windows it has SEEN
(in-sample), and how far does that fall out-of-sample (OOS)? The in-sample>>OOS gap is the
overfitting signature; a model that can't even fit in-sample is capacity- or signal-limited.

Data: results/feature_snapshots.jsonl gives, per window, the PIT price path the models saw
(`features.closes`, up to the decision instant) + `strike`; labels from
results/contract_outcomes.jsonl (`exact_yes`). ~1364 joined windows (vs 368 in
brti_decision_dataset). NOTE: these snapshots were logged at the model's DECISION time, so
the entry regime is later than the desk's 12-min entry — median mins_left is reported.

FROZEN models (barrier, market, zero-shot Chronos small/base/local-ft, TimesFM) can't be
overfit -> in-sample == fixed skill == OOS. TRAINABLE-on-features (high-capacity GBM, linear
logistic) can memorise -> in-sample vs walk-forward OOS exposes the gap.
Reuses fm_benchmark's Chronos backend for parity with the deployed T2. Read-only research.
"""
import json, math, statistics as st, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "research" / "insample_reconstruction.json"
ENTRY_FRAC = None    # use full logged PIT context as-is (already decision-time)
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from fm_benchmark import chronos_backend, LOCAL_BOLT, _p_ge, QL  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier      # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402


def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _epoch(t):
    """ISO string or epoch number -> epoch seconds (float), or None."""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


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
            m[d["ticker"]] = (d["exact_yes"], _epoch(d.get("close_time")), _epoch(d.get("open_time")))
    return m


def load_windows():
    lab = labels()
    W = {}
    for l in (RES / "feature_snapshots.jsonl").open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        wid = d.get("window_id")
        f = d.get("features") or {}
        if not (wid and isinstance(f, dict) and f.get("closes") and f.get("strike")):
            continue
        if wid in W or wid not in lab:
            continue
        y, close_t, open_t = lab[wid]
        series = [c for c in f["closes"] if c]
        if len(series) < 8:
            continue
        dec = _epoch(d.get("decision_ts") or d.get("event_ts") or d.get("persist_ts"))
        tr = (close_t - dec) if (close_t and dec) else 180.0          # time remaining (s)
        span = (dec - open_t) if (open_t and dec) else (len(series) * 2.0)
        dt = max(1.0, span / max(1, len(series) - 1))
        cur = series[-1]; tgt = f["strike"]
        rets = [math.log(series[i] / series[i - 1]) for i in range(1, len(series)) if series[i - 1] > 0]
        vol = st.pstdev(rets) if len(rets) > 1 else 1e-5
        sigma_T = cur * vol * math.sqrt(max(1.0, tr / dt))
        theo = _phi((cur - tgt) / max(sigma_T, 1e-6))
        W[wid] = {"id": wid, "series": series, "cur": cur, "tgt": tgt, "tr": max(1.0, tr), "dt": dt,
                  "kprob": 0.5, "theo": theo, "y": int(y), "mins_left": max(0.0, tr / 60.0),
                  "ts": dec or 0}
    out = sorted(W.values(), key=lambda w: w["ts"])
    return out


def _metrics(pred, y):
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    return round(acc, 4), round(f1, 4)


def feats(W):
    X = []
    for w in W:
        s = w["series"]; cur = w["cur"]; tgt = w["tgt"]
        rets = [math.log(s[i] / s[i - 1]) for i in range(1, len(s)) if s[i - 1] > 0]
        vol = st.pstdev(rets) if len(rets) > 1 else 1e-5
        sigma_T = cur * vol * math.sqrt(max(1.0, w["tr"] / w["dt"]))
        z = (cur - tgt) / max(sigma_T, 1e-6)
        up = sum(r for r in rets if r > 0); dn = -sum(r for r in rets if r < 0)
        rsi = up / (up + dn) if (up + dn) else 0.5
        X.append([z, _phi(z), (cur - tgt) / cur * 1e4, sigma_T / cur * 1e4,
                  (cur - s[0]) / cur * 1e4, rsi, w["tr"], len(s)])
    return np.array(X, float)


def run():
    W = load_windows()
    y = np.array([w["y"] for w in W], int)
    n = len(W); mid = int(n * 0.6); yte = y[mid:]
    ml = np.median([w["mins_left"] for w in W])
    print(f"n={n} windows (median mins_left {ml:.1f}), base-up {y.mean():.3f}, test {n-mid}")
    rep = {"n": n, "n_test": n - mid, "base_up": round(float(y.mean()), 4),
           "median_mins_left": round(float(ml), 2), "models": {}}

    frozen = {"barrier_closed_form": lambda: (lambda w: w["theo"]),
              "chronos_bolt_base": lambda: chronos_backend("amazon/chronos-bolt-base"),
              "chronos_bolt_small": lambda: chronos_backend("amazon/chronos-bolt-small"),
              "chronos_bolt_local_ft": lambda: chronos_backend(str(LOCAL_BOLT))}
    for name, mk in frozen.items():
        t0 = time.time()
        try:
            fn = mk()
            p = np.nan_to_num(np.array([fn(w) for w in W], float), nan=0.5, posinf=1.0, neginf=0.0)
            ia, if1 = _metrics((p >= 0.5).astype(int), y)
            oa, of1 = _metrics((p[mid:] >= 0.5).astype(int), yte)
            rep["models"][name] = {"trainable": False, "in_sample": {"acc": ia, "f1": if1},
                                   "oos": {"acc": oa, "f1": of1}, "overfit_gap_acc": round(ia - oa, 4),
                                   "ms_per_win": round((time.time() - t0) * 1000 / n, 1)}
            print(f"  {name:24} FROZEN  in-sample acc {ia} F1 {if1} | OOS acc {oa} F1 {of1}")
        except Exception as e:
            rep["models"][name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"  {name:24} UNAVAILABLE: {type(e).__name__}: {str(e)[:100]}")

    X = feats(W)
    hi = HistGradientBoostingClassifier(max_depth=None, min_samples_leaf=1, l2_regularization=0.0,
                                        learning_rate=0.3, max_iter=2000, early_stopping=False, random_state=0)
    hi.fit(X, y); ia, if1 = _metrics(hi.predict(X), y)
    hi2 = HistGradientBoostingClassifier(max_depth=None, min_samples_leaf=1, l2_regularization=0.0,
                                         learning_rate=0.3, max_iter=2000, early_stopping=False, random_state=0)
    hi2.fit(X[:mid], y[:mid]); oa, of1 = _metrics(hi2.predict(X[mid:]), yte)
    rep["models"]["gbm_high_capacity"] = {"trainable": True, "in_sample": {"acc": ia, "f1": if1},
                                          "oos": {"acc": oa, "f1": of1}, "overfit_gap_acc": round(ia - oa, 4)}
    print(f"  {'gbm_high_capacity':24} TRAIN   in-sample acc {ia} F1 {if1} | OOS acc {oa} F1 {of1}  gap {ia-oa:+.3f}")

    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1e6, max_iter=6000).fit(sc.transform(X), y)
    ia, if1 = _metrics(lr.predict(sc.transform(X)), y)
    sc2 = StandardScaler().fit(X[:mid])
    lr2 = LogisticRegression(C=1e6, max_iter=6000).fit(sc2.transform(X[:mid]), y[:mid])
    oa, of1 = _metrics(lr2.predict(sc2.transform(X[mid:])), yte)
    rep["models"]["logistic_min_reg"] = {"trainable": True, "in_sample": {"acc": ia, "f1": if1},
                                         "oos": {"acc": oa, "f1": of1}, "overfit_gap_acc": round(ia - oa, 4)}
    print(f"  {'logistic_min_reg':24} TRAIN   in-sample acc {ia} F1 {if1} | OOS acc {oa} F1 {of1}  gap {ia-oa:+.3f}")

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\n  -> {OUT.name} | base-up {rep['base_up']} n {n}")


if __name__ == "__main__":
    run()
