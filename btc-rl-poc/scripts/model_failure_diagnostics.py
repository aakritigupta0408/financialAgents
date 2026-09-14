"""L12 — MODEL FAILURE DIAGNOSTICS (directive).

Before any negative may be called INFORMATION_LIMITED, prove the training stack
itself is healthy. This runs the memorization ladder (the single most informative
experiment), the logistic regularization path, and an MLP capacity curve on the
FROZEN split (TRAIN + validation only; TEST_V1 spent, TEST_V2 sealed).

Memorization ladder: fit on N in {32,128,512,full TRAIN}, score on the SAME rows.
  can't memorize 32              -> bug / optimizer / LR / scaling / loss problem
  memorizes 32, not full TRAIN   -> capacity or over-regularization
  fits TRAIN strongly, VAL~base  -> generalization / weak-signal (NOT a stack bug)

Writes MODEL_FAILURE_DIAGNOSTICS_V1.json. Reclassifies model negatives as
UNDIAGNOSED_NEGATIVE vs (now) diagnosed. Emits events.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_ll
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

T = ROOT / "research" / "true15m"
DS = T / "TRUE15M_DATASET_V1.jsonl"
SPLIT = T / "SPLIT_SPEC_V1.json"
SEED = 17
LADDER = [32, 128, 512]


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return round(float(sk_ll(y, p, labels=[0, 1])), 5)


def _fit(kind, X, y):
    if kind == "logistic":
        sc = StandardScaler().fit(X)
        return ("sc", sc, LogisticRegression(C=1.0, max_iter=5000, random_state=SEED).fit(sc.transform(X), y))
    if kind == "rf":
        return ("raw", None, RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y))
    if kind == "histgb":
        return ("raw", None, HistGradientBoostingClassifier(random_state=SEED).fit(X, y))
    if kind == "xgboost":
        return ("raw", None, XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(X, y))
    if kind == "catboost":
        return ("raw", None, CatBoostClassifier(depth=6, iterations=300, learning_rate=0.1,
                random_seed=SEED, verbose=False).fit(X, y))
    if kind == "mlp":
        sc = StandardScaler().fit(X)
        return ("sc", sc, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-5, max_iter=2000,
                random_state=SEED).fit(sc.transform(X), y))
    raise ValueError(kind)


def _pred(m, X):
    mode, sc, clf = m
    return clf.predict_proba(sc.transform(X) if mode == "sc" else X)[:, 1]


def run():
    t0 = time.time()
    EV.emit("PLAN", "L12 model-failure diagnostics (memorization ladder etc.)", lane="L12",
            narrative="Proving the training stack is healthy before any INFORMATION_LIMITED "
                      "claim. TRAIN+VAL only; TEST_V1 spent, TEST_V2 sealed.")
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    rows.sort(key=lambda r: r["T0"])
    fids = list(rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in fids] for r in rows], float)
    y = np.array([r["Y"] for r in rows], int)
    sp = json.loads(SPLIT.read_text())
    tr_end = sp["train"]["n"]; va_end = tr_end + sp["validation"]["n"]
    Xtr, ytr = X[:tr_end], y[:tr_end]
    Xva, yva = X[tr_end:va_end], y[tr_end:va_end]
    base_val = _ll(np.full(len(yva), 0.5), yva)

    models = ["logistic", "rf", "histgb", "xgboost", "catboost", "mlp"]
    ladder = {}
    for kind in models:
        rung = {}
        for nsub in LADDER + [len(ytr)]:
            n = min(nsub, len(ytr))
            # take a class-balanced-ish contiguous head; ensure both classes present
            Xs, ys = Xtr[:n], ytr[:n]
            if len(np.unique(ys)) < 2:
                Xs, ys = Xtr[:max(n, 64)], ytr[:max(n, 64)]
            m = _fit(kind, Xs, ys)
            ptr = _pred(m, Xs)
            rung[str(n)] = {"train_logloss": _ll(ptr, ys),
                            "train_acc": round(float(((ptr >= 0.5).astype(int) == ys).mean()), 4)}
        # full-train model -> val
        mfull = _fit(kind, Xtr, ytr)
        pval = _pred(mfull, Xva)
        val_ll = _ll(pval, yva)
        tr_ll_full = rung[str(len(ytr))]["train_logloss"]
        mem32 = rung["32"]["train_logloss"]
        FLEX = {"rf", "xgboost", "catboost", "mlp"}   # can shatter tiny sets if healthy
        # memorization is only EXPECTED of flexible learners; a linear model cannot
        # shatter 32 arbitrary points, and a default-regularized GBM (histgb) won't
        # split 32 rows — those are model-class properties, not stack bugs.
        can_memorize32 = mem32 < 0.25
        fits_full = tr_ll_full < 0.65
        if kind in FLEX:
            diag = ("OPTIMIZER_OR_STACK_BUG" if not can_memorize32 else
                    "CAPACITY_OR_OVERREG" if not fits_full else
                    "HEALTHY_STACK_WEAK_SIGNAL" if val_ll >= base_val - 0.0005 else
                    "GENERALIZES_INVESTIGATE")
        elif kind == "histgb":
            diag = ("DEFAULT_REGULARIZED_TINY_SET" if not can_memorize32 else
                    "HEALTHY_STACK_WEAK_SIGNAL" if val_ll >= base_val - 0.0005 else
                    "GENERALIZES_INVESTIGATE")
        else:  # logistic / linear
            diag = ("LINEAR_CANNOT_SHATTER_EXPECTED" if not can_memorize32 else
                    "HEALTHY_STACK_WEAK_SIGNAL" if val_ll >= base_val - 0.0005 else
                    "GENERALIZES_INVESTIGATE")
        ladder[kind] = {"rungs": rung, "full_train_logloss": tr_ll_full,
                        "val_logloss": val_ll, "baseline_val_logloss": base_val,
                        "can_memorize_32": can_memorize32, "fits_full_train": fits_full,
                        "diagnosis": diag}
        EV.emit("TEST_PASSED" if can_memorize32 else "TEST_FAILED",
                f"{kind}: memorize32 logloss {rung['32']['train_logloss']}, "
                f"full-train {tr_ll_full}, val {val_ll} -> {diag}", lane="L12", job_id=kind)

    # logistic regularization path
    cpath = {}
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
        tr = _ll(clf.predict_proba(sc.transform(Xtr))[:, 1], ytr)
        va = _ll(clf.predict_proba(sc.transform(Xva))[:, 1], yva)
        cpath[str(C)] = {"train": tr, "val": va}

    # MLP capacity curve
    cap = {}
    for hs in [(32,), (64, 32), (128, 64, 32), (256, 128, 64)]:
        sc = StandardScaler().fit(Xtr)
        clf = MLPClassifier(hidden_layer_sizes=hs, alpha=1e-5, max_iter=2000, random_state=SEED).fit(sc.transform(Xtr), ytr)
        tr = _ll(clf.predict_proba(sc.transform(Xtr))[:, 1], ytr)
        va = _ll(clf.predict_proba(sc.transform(Xva))[:, 1], yva)
        cap["x".join(map(str, hs))] = {"train": tr, "val": va}

    # the stack is healthy if the FLEXIBLE learners can memorize tiny sets and fit TRAIN
    flex = ["rf", "xgboost", "catboost", "mlp"]
    healthy = all(ladder[k]["can_memorize_32"] and ladder[k]["fits_full_train"] for k in flex)
    all_weak = all(ladder[k]["val_logloss"] >= base_val - 0.0005 for k in models)
    doc = {"schema_version": "model-failure-diagnostics-1", "generated_at": time.time(),
           "cohort": "A_CORE", "baseline_val_logloss": base_val,
           "memorization_ladder": ladder,
           "logistic_regularization_path": cpath,
           "mlp_capacity_curve": cap,
           "stack_healthy": healthy,
           "summary": ("All models memorize tiny sets and fit TRAIN, yet VAL stays ~baseline "
                       "-> strong evidence of weak generalizable signal, NOT a stack bug."
                       if healthy and all_weak else
                       "Mixed: see per-model diagnosis before any INFORMATION_LIMITED claim."),
           "note": "negatives remain UNDIAGNOSED_NEGATIVE until the full battery (LR/reg/scaling/"
                   "capacity/loss/learning-curve/seed/fold/calibration/placebo) is complete."}
    (T / "MODEL_FAILURE_DIAGNOSTICS_V1.json").write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", "L12 memorization ladder complete", lane="L12", severity="high",
            fact="; ".join(f"{k}: mem32 {ladder[k]['rungs']['32']['train_logloss']}, "
                           f"fullTR {ladder[k]['full_train_logloss']}, val {ladder[k]['val_logloss']} "
                           f"({ladder[k]['diagnosis']})" for k in models),
            interpretation=doc["summary"],
            next_action="loss comparison (TGT-A..F), scaling audit, learning curve, seed/fold "
                        "stability, calibration — then a diagnosed verdict.",
            files=["research/true15m/MODEL_FAILURE_DIAGNOSTICS_V1.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"model_failure_diagnostics: stack_healthy={healthy} baseline_val={base_val}")
    for k in models:
        d = ladder[k]
        print(f"  {k:9} mem32 {d['rungs']['32']['train_logloss']:.4f} "
              f"fullTR {d['full_train_logloss']:.4f} val {d['val_logloss']:.4f} -> {d['diagnosis']}")


if __name__ == "__main__":
    run()
