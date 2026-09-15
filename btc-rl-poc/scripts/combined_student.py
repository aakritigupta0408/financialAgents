"""COMBINED STUDENT — fuse the large-model predictions (Chronos base/small, TimesFM, the
closed-form barrier) with a rich engineered feature set, a retrieval memory (kNN over past
windows), and a GBM, into one student that learns WHEN to trust WHICH source.

Architecture (stacking + retrieval, RAG-style for time-series):
  teachers   : per-window P(up) from chronos-bolt-base, chronos-bolt-small, TimesFM 2.5,
               barrier closed-form   (cached to research/teacher_cache.json — expensive)
  features   : ~30 PIT features from the intra-window path (momentum multi-scale, EWMA vol,
               RSI, MACD, Bollinger pos, range, skew/kurt, barrier z & drift, autocorr, ...)
  retrieval  : k nearest PAST windows in feature space -> their mean label (non-parametric
               memory). PAST-ONLY so it cannot borrow the future.
  gbm        : gradient boosting on the features, fit on the PAST only.
  students   : (A) logistic meta-stacker on [teachers + retrieval + gbm]
               (B) logistic on rich features + degree-2 basis  -> the "near-100% in-sample"
                   capacity demo (report in-sample AND OOS)
               (C) GBM on EVERYTHING (teachers + features + retrieval + gbm prob)  -> the fused model

Everything is walk-forward (60/40 by time). A label-shuffle LEAKAGE CANARY re-runs student C:
if OOS stays >~0.55 on shuffled labels, a feature is peeking and the result is void.
Data = feature_snapshots (PIT closes) x contract_outcomes (labels), ~1364 windows.
PAPER / SIMULATION research. Read-only w.r.t. the live desk.
"""
import json, math, statistics as st, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from insample_reconstruction import load_windows            # noqa: E402
from fm_benchmark import chronos_backend, timesfm_backend, LOCAL_BOLT  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # noqa: E402
from sklearn.neighbors import NearestNeighbors               # noqa: E402

CACHE = ROOT / "research" / "teacher_cache.json"
OUT = ROOT / "research" / "combined_student_report.json"
K_RETRIEVAL = 25


def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def teacher_probs(W):
    """Per-window P(up) from each large model, disk-cached by window_id (recompute only new)."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    need = [w for w in W if w["id"] not in cache]
    print(f"teachers: {len(W)} windows, {len(need)} to compute, {len(W)-len(need)} cached")
    if need:
        backends = {}
        for nm, mk in (("chronos_base", lambda: chronos_backend("amazon/chronos-bolt-base")),
                       ("chronos_small", lambda: chronos_backend("amazon/chronos-bolt-small")),
                       ("timesfm", timesfm_backend)):
            try:
                t0 = time.time(); backends[nm] = mk(); print(f"  loaded {nm} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {nm} UNAVAILABLE: {type(e).__name__}: {str(e)[:90]}"); backends[nm] = None
        for i, w in enumerate(need):
            row = {"barrier": float(w["theo"])}
            for nm, fn in backends.items():
                try:
                    row[nm] = float(fn(w)) if fn else None
                except Exception:
                    row[nm] = None
            cache[w["id"]] = row
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(need)}"); CACHE.write_text(json.dumps(cache))
        CACHE.write_text(json.dumps(cache))
    return cache


def rich_features(w):
    s = np.array(w["series"], float); cur = s[-1]; tgt = w["tgt"]
    r = np.diff(np.log(np.clip(s, 1e-9, None)))
    n = len(s)
    def sl(a, b): return s[int(n*a):int(n*b)] if int(n*b) > int(n*a) else s[-1:]
    vol = r.std() if len(r) > 1 else 1e-5
    # RiskMetrics EWMA vol
    ew = r[0]**2 if len(r) else 1e-10
    for x in r[1:]:
        ew = 0.94*ew + 0.06*x*x
    ew = math.sqrt(max(ew, 1e-12))
    sig_T = cur*ew*math.sqrt(max(1.0, w["tr"]/w["dt"]))
    z = (cur-tgt)/max(sig_T, 1e-9)
    up = r[r > 0].sum(); dn = -r[r < 0].sum()
    rsi = up/(up+dn) if (up+dn) else 0.5
    # multi-scale momentum
    def mom(a): seg = sl(a, 1.0); return (seg[-1]-seg[0])/seg[0]*1e4 if len(seg) > 1 and seg[0] else 0.0
    ma_s = s[-max(2, n//10):].mean(); ma_l = s[-max(2, n//3):].mean()
    boll = (cur-s.mean())/(s.std()+1e-9)
    rng = (s.max()-s.min())/cur*1e4
    sk = float(((r-r.mean())**3).mean()/(r.std()**3+1e-12)) if len(r) > 2 else 0.0
    ku = float(((r-r.mean())**4).mean()/(r.std()**4+1e-12)) if len(r) > 2 else 0.0
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 3 else 0.0
    drift = 0.5*(cur-s[0])*(w["tr"]/max(1.0, w["tr"]+w["dt"]*n))
    z_dr = (cur+drift-tgt)/max(sig_T, 1e-9)
    return {
        "z": z, "theo": _phi(z), "z_drift": z_dr, "theo_drift": _phi(z_dr),
        "dist_bps": (cur-tgt)/cur*1e4, "sigT_bps": sig_T/cur*1e4, "vol_bps": vol*1e4,
        "ewvol_bps": ew*1e4, "rsi": rsi, "mom_all": mom(0.0), "mom_half": mom(0.5),
        "mom_q": mom(0.75), "mom_last": mom(0.9), "ma_cross": (ma_s-ma_l)/cur*1e4,
        "boll": boll, "range_bps": rng, "skew": sk, "kurt": ku, "acf1": ac1,
        "curve": (sl(0.5, 1.0).mean()-sl(0.0, 0.5).mean())/cur*1e4, "npref": n,
        "tr_min": w["tr"]/60.0, "last_vs_ma": (cur-ma_s)/cur*1e4,
    }


def _metrics(pred, y):
    pred = np.asarray(pred); y = np.asarray(y)
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr = tp/(tp+fp) if (tp+fp) else 0.0; rc = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*pr*rc/(pr+rc) if (pr+rc) else 0.0
    return {"acc": round(acc, 4), "f1": round(f1, 4), "prec": round(pr, 4), "rec": round(rc, 4)}


def run():
    if "--benchmark-windows" in sys.argv:
        from fm_benchmark import windows as bw
        W = bw()
        for i, w in enumerate(W):
            w.setdefault("id", f"bw{i}_{int(w.get('ts') or 0)}")
        global CACHE, OUT
        CACHE = ROOT / "research" / "teacher_cache_midwin.json"
        OUT = ROOT / "research" / "combined_student_midwindow.json"
        print(f"REGIME: mid-window (fm_benchmark, ~11min left), {len(W)} windows")
    else:
        W = load_windows()
    tp = teacher_probs(W)
    y = np.array([w["y"] for w in W], int)
    n = len(W); mid = int(n*0.6); yte = y[mid:]
    tnames = ["barrier", "chronos_base", "chronos_small", "timesfm"]
    T = np.array([[tp.get(w["id"], {}).get(t, 0.5) if tp.get(w["id"], {}).get(t) is not None else 0.5
                   for t in tnames] for w in W], float)
    feats = [rich_features(w) for w in W]
    fkeys = list(feats[0].keys())
    F = np.nan_to_num(np.array([[f[k] for k in fkeys] for f in feats], float), nan=0.0, posinf=0.0, neginf=0.0)

    rep = {"n": n, "n_test": n-mid, "base_up": round(float(y.mean()), 4),
           "median_mins_left": round(float(np.median([w["tr"]/60 for w in W])), 2),
           "teachers": {}, "students": {}}

    # teacher OOS (each large model alone, argmax at 0.5)
    for j, t in enumerate(tnames):
        rep["teachers"][t] = _metrics((T[mid:, j] >= 0.5).astype(int), yte)

    # retrieval memory (PAST-only): fit kNN on train features, neighbor mean label
    scF = StandardScaler().fit(F[:mid])
    nn = NearestNeighbors(n_neighbors=min(K_RETRIEVAL, mid)).fit(scF.transform(F[:mid]))
    def retr(idx_rows):
        _, idx = nn.kneighbors(scF.transform(idx_rows))
        return y[:mid][idx].mean(axis=1)
    retr_tr = retr(F[:mid]); retr_te = retr(F[mid:])
    rep["teachers"]["retrieval_knn"] = _metrics((retr_te >= 0.5).astype(int), yte)

    # GBM on features, PAST-only
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                        l2_regularization=2.0, min_samples_leaf=20, random_state=17)
    gb.fit(F[:mid], y[:mid]); gp_tr = gb.predict_proba(F[:mid])[:, 1]; gp_te = gb.predict_proba(F[mid:])[:, 1]
    rep["teachers"]["gbm"] = _metrics((gp_te >= 0.5).astype(int), yte)

    # ---- Student A: logistic meta-stacker on [teachers + retrieval + gbm] ----
    stack_tr = np.column_stack([T[:mid], retr_tr, gp_tr])
    stack_te = np.column_stack([T[mid:], retr_te, gp_te])
    sc = StandardScaler().fit(stack_tr)
    la = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(stack_tr), y[:mid])
    rep["students"]["A_stacker_logistic"] = {
        "in_sample": _metrics((la.predict(sc.transform(stack_tr))), y[:mid]),
        "oos": _metrics(la.predict(sc.transform(stack_te)), yte),
        "inputs": tnames + ["retrieval", "gbm"]}

    # ---- Student B: logistic on rich features + degree-2 basis (capacity / near-100% in-sample) ----
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Fp_tr = poly.fit_transform(F[:mid]); Fp_te = poly.transform(F[mid:])
    scb = StandardScaler().fit(Fp_tr)
    lb = LogisticRegression(C=1e4, max_iter=8000).fit(scb.transform(Fp_tr), y[:mid])
    rep["students"]["B_logistic_poly2"] = {
        "in_sample": _metrics(lb.predict(scb.transform(Fp_tr)), y[:mid]),
        "oos": _metrics(lb.predict(scb.transform(Fp_te)), yte),
        "n_features": Fp_tr.shape[1]}

    # ---- Student C: GBM on EVERYTHING (teachers + features + retrieval + gbm prob) ----
    X_tr = np.column_stack([T[:mid], F[:mid], retr_tr, gp_tr])
    X_te = np.column_stack([T[mid:], F[mid:], retr_te, gp_te])
    sc_all = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=500,
                                            l2_regularization=2.0, min_samples_leaf=20, random_state=13)
    sc_all.fit(X_tr, y[:mid])
    rep["students"]["C_gbm_fused"] = {
        "in_sample": _metrics(sc_all.predict(X_tr), y[:mid]),
        "oos": _metrics(sc_all.predict(X_te), yte),
        "n_inputs": X_tr.shape[1]}

    # ---- LEAKAGE CANARY: shuffle labels, refit student C, OOS should be ~0.5 ----
    rng = np.random.default_rng(0)
    ysh = y.copy(); ysh[:mid] = rng.permutation(ysh[:mid])
    can = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=500,
                                         l2_regularization=2.0, min_samples_leaf=20, random_state=13)
    can.fit(X_tr, ysh[:mid])
    rep["leakage_canary_oos"] = _metrics(can.predict(X_te), yte)

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nn={n} test={n-mid} base-up {rep['base_up']} median_mins_left {rep['median_mins_left']}")
    print("TEACHERS (OOS):")
    for k, v in rep["teachers"].items():
        print(f"  {k:16} acc {v['acc']} F1 {v['f1']} prec {v['prec']}")
    print("STUDENTS:")
    for k, v in rep["students"].items():
        print(f"  {k:20} in-sample acc {v['in_sample']['acc']} -> OOS acc {v['oos']['acc']} F1 {v['oos']['f1']} prec {v['oos']['prec']}")
    print(f"LEAKAGE CANARY (shuffled labels) OOS acc {rep['leakage_canary_oos']['acc']} (must be ~0.5)")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    run()
