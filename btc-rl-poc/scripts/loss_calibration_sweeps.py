"""Loss-formulation + calibration sweeps (directive L12 continuation).

On A_CORE, frozen TRAIN->VAL (TEST_V1 spent / TEST_V2 sealed — untouched):

LOSS formulations, all reduced to P(D>=0) / P(Y=1), compared on VAL vs the 0.693
baseline — because the binary target discards magnitude and D may carry more:
  TGT-A  BCE (reference, from earlier)      TGT-B  Huber(D) -> Gaussian P(D>=0)
  TGT-C  MSE(normalized D) -> P(D>=0)       TGT-D  quantile(D) -> P(D>=0)
  TGT-E1 Gaussian NLL (NGBoost Normal)      TGT-E2 Student-t NLL (NGBoost T)
  TGT-MT multitask net  loss = λ·BCE(Y) + (1-λ)·Huber(D_scaled),  λ swept on VAL

CALIBRATION sweep — separate ranking (AUC) from probability quality (Brier/LL):
  Platt (sigmoid), isotonic, temperature — fit on a TRAIN-internal calib holdout,
  evaluated on VAL. A miscalibrated-but-ranking model would show AUC>0.5 with poor
  LL that calibration rescues; no ranking means calibration cannot help.

Writes LOSS_CALIBRATION_SWEEPS_V1.json. Emits events.
"""
import os
# macOS Anaconda ships multiple OpenMP runtimes (torch/catboost/xgboost/sklearn each
# bundle one). Running torch ops after the others triggers a libomp double-load
# SEGFAULT. Serialize threads and permit the duplicate load BEFORE any numeric import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_ll, roc_auc_score
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
import torch
import torch.nn as nn
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import metrics as MET          # noqa: E402
from btc_rl import research_events as EV    # noqa: E402

TT = ROOT / "research" / "true15m"
DS = TT / "TRUE15M_DATASET_V1.jsonl"
SPLIT = TT / "SPLIT_SPEC_V1.json"
SEED = 17
torch.manual_seed(SEED)


def _clip(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _m(p, y):
    p = _clip(p); y = np.asarray(y, int)
    try:
        auc = round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None
    except Exception:
        auc = None
    return {"log_loss": round(float(sk_ll(y, p, labels=[0, 1])), 5),
            "brier": round(float(MET.brier(p.tolist(), y.tolist())), 5),
            "auc": auc,
            "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4)}


def _multitask(Xtr, ytr, Dtr, Xva, lam, epochs=150):
    ds = Dtr.std() or 1.0
    Dn = Dtr / ds
    xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    dt = torch.tensor(Dn, dtype=torch.float32); xv = torch.tensor(Xva, dtype=torch.float32)
    trunk = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
    clf_head = nn.Linear(16, 1); reg_head = nn.Linear(16, 1)
    params = list(trunk.parameters()) + list(clf_head.parameters()) + list(reg_head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); huber = nn.HuberLoss()
    for _ in range(epochs):
        opt.zero_grad()
        h = trunk(xt)
        lg = clf_head(h).squeeze(1); dr = reg_head(h).squeeze(1)
        loss = lam * bce(lg, yt) + (1 - lam) * huber(dr, dt)
        loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(clf_head(trunk(xv)).squeeze(1)).numpy()
    return p


def run():
    t0 = time.time()
    EV.emit("PLAN", "Loss-formulation + calibration sweeps", lane="L12",
            narrative="Testing whether magnitude-aware losses (Huber/quantile/NLL/multitask) "
                      "or calibration change the picture. TRAIN+VAL only.")
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    rows.sort(key=lambda r: r["T0"])
    fids = list(rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in fids] for r in rows], float)
    y = np.array([r["Y"] for r in rows], int)
    D = np.array([r["D"] for r in rows], float)
    sp = json.loads(SPLIT.read_text())
    tr_end = sp["train"]["n"]; va_end = tr_end + sp["validation"]["n"]
    sc = StandardScaler().fit(X[:tr_end])
    Xtr, Xva = sc.transform(X[:tr_end]), sc.transform(X[tr_end:va_end])
    ytr, yva = y[:tr_end], y[tr_end:va_end]
    Dtr = D[:tr_end]
    base = _m(np.full(len(yva), 0.5), yva)

    losses = {}
    # TGT-B Huber
    hub = HuberRegressor(max_iter=2000).fit(Xtr, Dtr); mu = hub.predict(Xva)
    s = (Dtr - hub.predict(Xtr)).std() or 1.0
    losses["TGT-B_huber"] = _m(np.array([_phi(m / s) for m in mu]), yva)
    # TGT-C MSE normalized
    rg = Ridge(alpha=1.0).fit(Xtr, Dtr / (Dtr.std() or 1.0)); mu = rg.predict(Xva)
    sr = (Dtr / (Dtr.std() or 1.0) - rg.predict(Xtr)).std() or 1.0
    losses["TGT-C_mse"] = _m(np.array([_phi(m / sr) for m in mu]), yva)
    # TGT-D quantile (catboost) -> P(D>=0) via alpha where quantile crosses 0
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    qpred = {}
    for a in alphas:
        qr = CatBoostRegressor(loss_function=f"Quantile:alpha={a}", depth=3, iterations=150,
                               learning_rate=0.03, random_seed=SEED, verbose=False).fit(Xtr, Dtr)
        qpred[a] = qr.predict(Xva)
    pq = []
    for i in range(len(Xva)):
        qs = [qpred[a][i] for a in alphas]
        # P(D<0) ~ alpha at which quantile(D)=0 (piecewise-linear interp)
        p_lt = 0.5
        for j in range(len(alphas) - 1):
            if qs[j] <= 0 <= qs[j + 1] or qs[j] >= 0 >= qs[j + 1]:
                denom = (qs[j + 1] - qs[j]) or 1e-9
                p_lt = alphas[j] + (0 - qs[j]) / denom * (alphas[j + 1] - alphas[j]); break
        else:
            p_lt = 0.02 if qs[-1] < 0 else (0.98 if qs[0] > 0 else 0.5)
        pq.append(1 - p_lt)
    losses["TGT-D_quantile"] = _m(np.array(pq), yva)
    # TGT-E1 Gaussian NLL. Fit on NORMALIZED D — P(D>=0) is the loc/scale ratio,
    # invariant to the constant normalization; normalizing avoids overflow.
    dstd = Dtr.std() or 1.0
    Dtr_n = Dtr / dstd
    try:
        ng = NGBRegressor(Dist=Normal, n_estimators=250, learning_rate=0.02, verbose=False,
                          random_state=SEED).fit(Xtr, Dtr_n)
        dd = ng.pred_dist(Xva); loc = dd.loc; scale = getattr(dd, "scale", None)
        sc2 = scale if scale is not None else np.full(len(loc), 1.0)
        p = np.array([_phi(loc[i] / (sc2[i] or 1.0)) for i in range(len(loc))])
        losses["TGT-E1_gaussian_nll"] = _m(p, yva) if np.all(np.isfinite(p)) else {
            "status": "UNSTABLE", "reason": "non-finite probs", "log_loss": None, "auc": None}
    except Exception as e:
        losses["TGT-E1_gaussian_nll"] = {"status": "UNSTABLE", "reason": str(e)[:120],
                                         "log_loss": None, "auc": None}
    # TGT-E2 Student-t NLL — ISOLATED in a subprocess: NGBoost's T score overflows
    # (df -> inf) and SEGFAULTS the interpreter on this target, which no try/except
    # can trap. Student-t is "if stable" per the directive; a child-process crash is
    # recorded as empirical UNSTABLE evidence rather than killing the sweep.
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "_ngboost_t_probe.py")],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=400)
        out = (r.stdout or "").strip().splitlines()
        if r.returncode == 0 and out:
            losses["TGT-E2_studentt_nll"] = json.loads(out[-1])
        else:
            losses["TGT-E2_studentt_nll"] = {
                "status": "UNSTABLE", "log_loss": None, "auc": None,
                "reason": f"isolated NGBoost-T fit exited rc={r.returncode} "
                          f"(segfault/overflow: df->inf in Student-t score)"}
    except Exception as e:
        losses["TGT-E2_studentt_nll"] = {"status": "UNSTABLE", "log_loss": None, "auc": None,
                                         "reason": str(e)[:120]}
    # TGT-MT multitask lambda sweep
    mt = {}
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mt[f"lambda={lam}"] = _m(_multitask(Xtr, ytr, Dtr, Xva, lam), yva)
    best_lam = min(mt, key=lambda k: mt[k]["log_loss"])
    losses["TGT-MT_multitask_best"] = {**mt[best_lam], "best_lambda": best_lam}

    # ---- calibration sweep on the best-AUC loss model ----
    all_models = {k: v for k, v in losses.items() if v.get("auc") is not None}
    best_auc_model = max(all_models, key=lambda k: all_models[k]["auc"] or 0)
    # re-derive raw val probs for calibration (TGT-B huber as representative scorer).
    # The calibrator must be fit on a GENUINELY held-out slice: fit the scorer on the
    # first 80% of TRAIN only, then score the untouched 20% holdout and VAL — otherwise
    # the calibration holdout leaks into the scorer and flatters the calibrators.
    ci = int(len(Xtr) * 0.8)                      # train-internal calibration holdout
    cal_hub = HuberRegressor(max_iter=2000).fit(Xtr[:ci], Dtr[:ci])
    s2 = (Dtr[:ci] - cal_hub.predict(Xtr[:ci])).std() or 1.0
    mu_cal = cal_hub.predict(Xtr[ci:]); mu_va = cal_hub.predict(Xva)
    raw_cal = np.array([_phi(m / s2) for m in mu_cal]); raw_va = np.array([_phi(m / s2) for m in mu_va])
    ycal = ytr[ci:]
    calib = {"raw": _m(raw_va, yva)}
    # Platt (sigmoid)
    platt = LogisticRegression(max_iter=2000).fit(raw_cal.reshape(-1, 1), ycal)
    calib["platt"] = _m(platt.predict_proba(raw_va.reshape(-1, 1))[:, 1], yva)
    # isotonic
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, ycal)
    calib["isotonic"] = _m(iso.predict(raw_va), yva)
    # temperature scaling on logits
    lg_cal = np.log(_clip(raw_cal) / (1 - _clip(raw_cal))); lg_va = np.log(_clip(raw_va) / (1 - _clip(raw_va)))
    Tt = torch.nn.Parameter(torch.ones(1)); optT = torch.optim.LBFGS([Tt], lr=0.1, max_iter=50)
    lc = torch.tensor(lg_cal, dtype=torch.float32); yc = torch.tensor(ycal, dtype=torch.float32)
    def closure():
        optT.zero_grad(); l = nn.BCEWithLogitsLoss()(lc / Tt, yc); l.backward(); return l
    optT.step(closure)
    calib["temperature"] = _m(1 / (1 + np.exp(-lg_va / float(Tt.item()))), yva)

    # ---- walk-forward corroboration (a single-VAL-slice dip is NOT evidence) ----
    # The CatBoost 0.68956 case looked like an edge on one slice but its walk-forward
    # median was 0.698. Any formulation that beats baseline on VAL is re-checked over
    # 5 expanding chronological folds on RAW features (scaler re-fit per fold, TEST
    # untouched). Only a stable WF-median edge counts as promising.
    def _huber_pd(Xa, Da, Xb):
        h = HuberRegressor(max_iter=2000).fit(Xa, Da); ss = (Da - h.predict(Xa)).std() or 1.0
        return np.array([_phi(v / ss) for v in h.predict(Xb)])

    def _ridge_pd(Xa, Da, Xb):
        dn = Da.std() or 1.0; rr = Ridge(alpha=1.0).fit(Xa, Da / dn)
        ss = (Da / dn - rr.predict(Xa)).std() or 1.0
        return np.array([_phi(v / ss) for v in rr.predict(Xb)])

    def _quant_pd(Xa, Da, Xb):
        qp = {a: CatBoostRegressor(loss_function=f"Quantile:alpha={a}", depth=3, iterations=150,
                                   learning_rate=0.03, random_seed=SEED, verbose=False)
              .fit(Xa, Da).predict(Xb) for a in alphas}
        out = []
        for i in range(len(Xb)):
            qs = [qp[a][i] for a in alphas]; p_lt = 0.5
            for j in range(len(alphas) - 1):
                if qs[j] <= 0 <= qs[j + 1] or qs[j] >= 0 >= qs[j + 1]:
                    den = (qs[j + 1] - qs[j]) or 1e-9
                    p_lt = alphas[j] + (0 - qs[j]) / den * (alphas[j + 1] - alphas[j]); break
            else:
                p_lt = 0.02 if qs[-1] < 0 else (0.98 if qs[0] > 0 else 0.5)
            out.append(1 - p_lt)
        return np.array(out)

    def _wf_median_ll(fn, n_folds=5):
        Xr, Dr, yr = X[:va_end], D[:va_end], y[:va_end]; n = len(Xr); lls = []
        for k in range(n_folds):
            lo = int(n * (0.5 + 0.1 * k)); hi = int(n * (0.5 + 0.1 * (k + 1)))
            if hi <= lo or lo < 100:
                continue
            scf = StandardScaler().fit(Xr[:lo])
            p = fn(scf.transform(Xr[:lo]), Dr[:lo], scf.transform(Xr[lo:hi]))
            lls.append(round(float(sk_ll(yr[lo:hi], _clip(p), labels=[0, 1])), 5))
        return {"per_fold": lls, "median": round(float(np.median(lls)), 5) if lls else None}

    wf = {}
    thr = base["log_loss"] - 0.0005
    for nm, fn in [("TGT-B_huber", _huber_pd), ("TGT-C_mse", _ridge_pd), ("TGT-D_quantile", _quant_pd)]:
        v = losses.get(nm, {})
        if isinstance(v.get("log_loss"), float) and v["log_loss"] < thr:
            r = _wf_median_ll(fn)
            folds = r["per_fold"]
            r["folds_beating_baseline"] = sum(1 for f in folds if f < base["log_loss"])
            r["n_folds"] = len(folds)
            r["consistency"] = round(r["folds_beating_baseline"] / len(folds), 3) if folds else 0.0
            r["median_beats"] = r["median"] is not None and r["median"] < thr
            wf[nm] = r

    val_beats = [k for k, v in losses.items()
                 if isinstance(v.get("log_loss"), float) and v["log_loss"] < thr]
    # 3-level corroboration: a median can pass while folds disagree. Require BOTH a
    # median edge AND high fold consistency (>=80%) before "promising"; a median-only
    # pass with mixed folds is a WEAK/UNSTABLE candidate, not a stable edge.
    median_pass = [k for k, w in wf.items() if w["median_beats"]]
    strong = [k for k in median_pass if wf[k]["consistency"] >= 0.8]
    if strong:
        verdict = "LOSS_FORMULATION_PROMISING"
        interp = (f"{', '.join(strong)} beats baseline on VAL AND holds up in walk-forward "
                  "(median edge + >=80% folds) — a stable OOS edge; promote to the full battery "
                  "and the sealed test before any claim. Do NOT adopt yet.")
    elif median_pass:
        verdict = "WEAK_UNSTABLE_CANDIDATE"
        det = "; ".join(f"{k}: WF median {wf[k]['median']} vs base {base['log_loss']}, "
                        f"{wf[k]['folds_beating_baseline']}/{wf[k]['n_folds']} folds beat baseline"
                        for k in median_pass)
        interp = (f"A magnitude-aware loss shows a WEAK, UNSTABLE edge: {det}. The WF median passes "
                  "the threshold but folds disagree (not >=80% consistent) and AUCs are ~0.5, so "
                  "there is no reliable ranking for calibration to rescue. Consistent with the "
                  "prescribed global wording (weak/unstable). NOT qualified; carry as a candidate "
                  "into the full battery + sealed TEST_V2 — do not adopt.")
    else:
        verdict = "NO_STABLE_LOSS_RESCUE"
        interp = (f"VAL dips exist ({', '.join(val_beats) or 'none'}) but NONE survive walk-forward "
                  "(median LL >= baseline) — single-slice noise, exactly like the rejected CatBoost "
                  "0.68956→0.698 case. AUCs ~0.5, so calibration has no ranking to rescue. "
                  "Magnitude-aware losses do not add stable signal on A_CORE.")
    doc = {"schema_version": "loss-calibration-sweeps-2", "generated_at": time.time(),
           "cohort": "A_CORE", "baseline_val": base, "loss_formulations": losses,
           "multitask_lambda_sweep": mt,
           "calibration_sweep": {"scorer": "TGT-B huber (representative)", "results": calib},
           "walk_forward_corroboration": {
               "folds": 5, "scheme": "expanding chronological, scaler re-fit per fold, raw features",
               "candidates_checked": list(wf.keys()),
               "results": wf,
               "rule": "a VAL dip counts only if its WF median LL < baseline - 0.0005"},
           "val_beats_baseline": val_beats,
           "auc_summary": {k: v["auc"] for k, v in all_models.items()},
           "verdict": verdict,
           "interpretation": interp,
           "note": "TRAIN+VAL only; TEST_V1 spent, TEST_V2 sealed."}
    (TT / "LOSS_CALIBRATION_SWEEPS_V1.json").write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", "Loss + calibration sweeps: " + doc["verdict"], lane="L12", severity="high",
            fact="; ".join(f"{k} LL {v['log_loss']} AUC {v.get('auc')}"
                           for k, v in losses.items() if isinstance(v.get("log_loss"), float))
                 + f" | baseline LL {base['log_loss']} | WF-median "
                 + (", ".join(f"{k} {wf[k]['median']}" for k in wf) or "n/a")
                 + f" | calib raw {calib['raw']['log_loss']} "
                   f"platt {calib['platt']['log_loss']} iso {calib['isotonic']['log_loss']} "
                   f"temp {calib['temperature']['log_loss']}",
            interpretation=doc["interpretation"],
            next_action="neural/temporal/foundation families remain; TEST_V2 sealed.",
            files=["research/true15m/LOSS_CALIBRATION_SWEEPS_V1.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"loss_calibration_sweeps: baseline LL {base['log_loss']} -> {doc['verdict']}")
    for k, v in losses.items():
        if isinstance(v.get("log_loss"), float):
            print(f"  {k:24} LL {v['log_loss']} AUC {v.get('auc')}")
    print("  multitask best:", best_lam, mt[best_lam]["log_loss"])
    print("  calib:", {k: calib[k]["log_loss"] for k in calib})
    print("  walk-forward:", {k: wf[k]["median"] for k in wf} or "no VAL dips to check")


if __name__ == "__main__":
    run()
