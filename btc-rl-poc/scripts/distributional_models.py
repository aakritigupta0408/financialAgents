"""Level 5 — distributional / probabilistic model family (CatBoost, NGBoost).

Runs on TRUE15M_DATASET_V1 (A_CORE) using the FROZEN split: train + validation +
chronological walk-forward ONLY. TEST_V1 is SPENT and TEST_V2 is SEALED — neither
is touched here (these are additional development candidates).

Models:
  catboost_clf      TGT-A: P(Y=1) directly
  catboost_gauss    TGT-C: CatBoost regresses D, Gaussian residual -> P(D>=0)
  ngboost_normal    TGT-C: NGBoost Normal(D) -> P(D>=0) (proper distributional)

Each gets validation + walk-forward metrics + a label-shuffle placebo. Appends to
OFFLINE_MODEL_FEATURE_MATRIX_V1 and writes distributional_models_result.json.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss as sk_ll, roc_auc_score
from catboost import CatBoostClassifier, CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import metrics as MET          # noqa: E402
from btc_rl import research_events as EV    # noqa: E402
from btc_rl.test_v2_firewall import guard_rows  # noqa: E402

T = ROOT / "research" / "true15m"
DS = T / "TRUE15M_DATASET_V1.jsonl"
SPLIT = T / "SPLIT_SPEC_V1.json"
SEED = 17


def _clip(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def _m(p, y):
    p = _clip(p); y = np.asarray(y, int)
    try:
        auc = round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None
    except Exception:
        auc = None
    cb = MET.calibration_bins(p.tolist(), y.tolist(), n_bins=10)
    ece = sum((b["n"] / len(y)) * abs(b["p_mean"] - b["y_freq"]) for b in cb if b["n"])
    return {"log_loss": round(float(sk_ll(y, p, labels=[0, 1])), 5),
            "brier": round(float(MET.brier(p.tolist(), y.tolist())), 5),
            "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4),
            "auc": auc, "ece": round(float(ece), 4)}


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _fit_predict(kind, Xtr, ytr, Dtr, Xte):
    if kind == "catboost_clf":
        clf = CatBoostClassifier(depth=3, iterations=200, learning_rate=0.03,
                                 l2_leaf_reg=3.0, random_seed=SEED, verbose=False)
        clf.fit(Xtr, ytr); return clf.predict_proba(Xte)[:, 1]
    if kind == "catboost_gauss":
        reg = CatBoostRegressor(depth=3, iterations=200, learning_rate=0.03,
                                l2_leaf_reg=3.0, random_seed=SEED, verbose=False)
        reg.fit(Xtr, Dtr)
        mu = reg.predict(Xte); s = (Dtr - reg.predict(Xtr)).std() or 1.0
        return np.array([_phi(m / s) for m in mu])
    if kind == "ngboost_normal":
        ng = NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.02,
                          verbose=False, random_state=SEED)
        ng.fit(np.asarray(Xtr), np.asarray(Dtr))
        d = ng.pred_dist(np.asarray(Xte))
        loc, scale = d.loc, d.scale
        return np.array([_phi(loc[i] / (scale[i] or 1.0)) for i in range(len(loc))])
    raise ValueError(kind)


def _walk(kind, X, y, D, folds=5):
    n = len(y); step = n // (folds + 1); out = []
    for k in range(1, folds + 1):
        tr = slice(0, step * k); va = slice(step * k, step * (k + 1) if k < folds else n)
        if (va.stop - va.start) < 20 or (tr.stop - tr.start) < 100:
            continue
        p = _fit_predict(kind, X[tr], y[tr], D[tr], X[va])
        out.append(_m(p, y[va]))
    return out


def run():
    t0 = time.time()
    EV.emit("PLAN", "Level-5 distributional models (CatBoost, NGBoost)", lane="L8",
            narrative="TGT-A P(Y=1) + TGT-C distributional D->P(D>=0). TRAIN+VAL+walk-forward "
                      "only; TEST_V1 spent, TEST_V2 sealed — untouched.")
    rows = guard_rows([json.loads(l) for l in DS.open() if l.strip()], "distributional_models")
    rows.sort(key=lambda r: r["T0"])
    fids = list(rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in fids] for r in rows], float)
    y = np.array([r["Y"] for r in rows], int)
    D = np.array([r["D"] for r in rows], float)
    sp = json.loads(SPLIT.read_text())
    tr_end = sp["train"]["n"]; va_end = tr_end + sp["validation"]["n"]     # dev = train+val
    Xtr, ytr, Dtr = X[:tr_end], y[:tr_end], D[:tr_end]
    Xva, yva = X[tr_end:va_end], y[tr_end:va_end]
    Xdev, ydev, Ddev = X[:va_end], y[:va_end], D[:va_end]
    base_ll = _m(np.full(len(yva), 0.5), yva)["log_loss"]

    models, res = ["catboost_clf", "catboost_gauss", "ngboost_normal"], {}
    for kind in models:
        pva = _fit_predict(kind, Xtr, ytr, Dtr, Xva)          # train->val
        vm = _m(pva, yva)
        wf = _walk(kind, Xdev, ydev, Ddev)
        wf_ll = [f["log_loss"] for f in wf]
        # label-shuffle placebo (dev-internal)
        rng = np.random.default_rng(SEED); ysh = ytr.copy(); rng.shuffle(ysh)
        pf = _fit_predict(kind, Xtr, ysh, Dtr, Xva); fll = _m(pf, yva)["log_loss"]
        beats = vm["log_loss"] < base_ll - 0.0005 and fll > vm["log_loss"] + 0.0005
        res[kind] = {"target": "TGT-A" if kind == "catboost_clf" else "TGT-C",
                     "val": vm, "baseline_val_logloss": base_ll,
                     "walk_forward_logloss": {"median": round(float(np.median(wf_ll)), 5) if wf_ll else None,
                                              "worst": round(float(max(wf_ll)), 5) if wf_ll else None,
                                              "n_folds": len(wf)},
                     "label_shuffle_val_logloss": fll,
                     "verdict": "PROMISING" if beats else "REJECTED_NO_OOS_VALUE"}
        EV.emit("MODEL_TRAINING_COMPLETE", f"{kind}: val logloss {vm['log_loss']} "
                f"(baseline {base_ll})", lane="L8", job_id=kind,
                metrics={"val_logloss": vm["log_loss"], "wf_median": res[kind]["walk_forward_logloss"]["median"]},
                interpretation=res[kind]["verdict"])
    any_promising = [k for k, v in res.items() if v["verdict"] == "PROMISING"]
    doc = {"schema_version": "distributional-models-1", "generated_at": time.time(),
           "cohort": "A_CORE", "feature_set": "FS0_CORE(20)", "dev_n": int(va_end),
           "baseline_val_logloss": base_ll, "models": res,
           "family_verdict": "PROMISING:" + any_promising[0] if any_promising else "NO_OOS_VALUE",
           "note": "development metrics only (val + walk-forward); TEST_V1 spent, TEST_V2 sealed."}
    (T / "distributional_models_result.json").write_text(json.dumps(doc, indent=1))

    # append to the model×feature matrix
    mfm_p = T / "OFFLINE_MODEL_FEATURE_MATRIX_V1.json"
    mfm = json.loads(mfm_p.read_text()) if mfm_p.exists() else {"models": {}}
    for k, v in res.items():
        mfm["models"][k] = {"target": v["target"], "feature_set": "FS0_CORE(20)",
                            "cohort": "A_CORE", "val": v["val"],
                            "walk_forward_logloss": v["walk_forward_logloss"],
                            "label_shuffle_val_logloss": v["label_shuffle_val_logloss"],
                            "verdict": v["verdict"]}
    mfm["distributional_appended_at"] = time.time()
    mfm_p.write_text(json.dumps(mfm, indent=1))

    EV.emit("DISCOVERY", "Level-5 distributional family: " + doc["family_verdict"], lane="L8",
            severity="high",
            fact="; ".join(f"{k} val {v['val']['log_loss']} (base {base_ll}) -> {v['verdict']}"
                           for k, v in res.items()),
            interpretation="Distributional modelling of D does not beat the baseline OOS on "
                           "A_CORE either." if not any_promising else
                           "A distributional candidate looks promising on validation — needs "
                           "the full battery before any TEST_V2 consideration.",
            next_action="neural/temporal/foundation families remain (compute); TEST_V2 sealed.",
            files=["research/true15m/distributional_models_result.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"distributional_models: baseline_val {base_ll}")
    for k, v in res.items():
        print(f"  {k:16} val {v['val']['log_loss']} wf_median {v['walk_forward_logloss']['median']} "
              f"shuffle {v['label_shuffle_val_logloss']} -> {v['verdict']}")
    print("  family:", doc["family_verdict"])


if __name__ == "__main__":
    run()
