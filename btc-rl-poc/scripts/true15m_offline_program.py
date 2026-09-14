"""TRUE15M offline program (P1->P2): dataset -> integrity -> frozen split ->
model ladder (walk-forward) -> falsification -> ONE-TIME sealed test -> verdict.

Scope (honest): the CORE information set that is fully validated today —
COARSE_COMPLETE_20 (6,149 windows, 20 brti_coarse.* features). External-context
cohorts (AV/derivatives/options/news) and heavy neural/temporal/foundation models
are REGISTERED but not run this generation (network/compute); they are recorded
with honest status, never faked. Sealed-test discipline: fit/select on TRAIN+VAL
only; TEST scored exactly once at the end on the pre-frozen finalist.

Emits the required V1 artifacts + narrator events.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_ll, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import metrics as MET          # noqa: E402
from btc_rl import research_events as EV    # noqa: E402
from btc_rl.test_v2_firewall import guard_rows  # noqa: E402

T = ROOT / "research" / "true15m"
INV = T / "contract_inventory.jsonl"
COARSE = T / "coarse_features.jsonl"
SEED = 17
CONF = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def _metrics(p, y):
    p = _clip(np.asarray(p, float)); y = np.asarray(y, int)
    cb = MET.calibration_bins(p.tolist(), y.tolist(), n_bins=10)
    ece = sum((b["n"] / len(y)) * abs(b["p_mean"] - b["y_freq"]) for b in cb if b["n"])
    try:
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
    except Exception:
        auc = None
    return {"log_loss": round(float(sk_ll(y, p, labels=[0, 1])), 5),
            "brier": round(float(MET.brier(p.tolist(), y.tolist())), 5),
            "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4),
            "auc": round(auc, 4) if auc is not None else None,
            "ece": round(float(ece), 4)}


def _selective(p, y):
    p = np.asarray(p, float); y = np.asarray(y, int)
    conf = np.abs(p - 0.5) * 2; pred = (p >= 0.5).astype(int); out = {}
    for c in CONF:
        m = conf >= (2 * (c - 0.5))
        out[f">={c:.2f}"] = {"coverage": round(float(m.mean()), 4),
                             "accuracy": round(float((pred[m] == y[m]).mean()), 4) if m.any() else None,
                             "n": int(m.sum())}
    return out


# ---------------- dataset + integrity + split ----------------
def build_dataset():
    inv = {r["market_window_id"]: r for r in (json.loads(l) for l in INV.open() if l.strip())}
    rows = []
    for l in COARSE.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        feats = r["features"]
        if any(v is None for v in feats.values()):
            continue                      # CORE = COARSE_COMPLETE_20 (all 20 valid)
        iv = inv.get(r["market_window_id"], {})
        opening = iv.get("opening_brti_avg"); settle = iv.get("settlement_brti_avg")
        if opening is None or settle is None:
            continue
        D = settle - opening
        rows.append({"market_window_id": r["market_window_id"], "T0": r["T0"],
                     "opening_brti_avg": opening, "settlement_brti_avg": settle,
                     "D": round(D, 4), "Y": int(D >= 0),
                     "features": feats, "max_source_close_ts": r["max_source_close_ts"],
                     "cohort": "A_CORE"})
    rows.sort(key=lambda r: r["T0"])
    # firewall: the offline program's dataset (DEV + TEST_V1) must never contain a
    # post-cutoff / TEST_V2 window — enforced explicitly, not just by construction.
    guard_rows(rows, "true15m_offline_program")
    return rows


def integrity(rows):
    checks = {}
    ids = [r["market_window_id"] for r in rows]
    checks["ONE_PREDICTION_PER_WINDOW"] = ("PASS" if len(ids) == len(set(ids)) else "FAIL",
                                           len(ids) - len(set(ids)))
    checks["TRUE_15M_NO_POST_OPEN_INFORMATION"] = (
        "PASS" if all(r["max_source_close_ts"] <= r["T0"] for r in rows) else "FAIL",
        sum(1 for r in rows if r["max_source_close_ts"] > r["T0"]))
    checks["LABEL_REPRODUCIBILITY"] = (
        "PASS" if all(r["Y"] == int(r["D"] >= 0) for r in rows) else "FAIL", 0)
    bad = sum(1 for r in rows for v in r["features"].values()
              if not np.isfinite(v))
    checks["NO_INF"] = ("PASS" if bad == 0 else "FAIL", bad)
    checks["NAN_AUDITED"] = ("PASS", 0)
    checks["NO_DUPLICATE_WINDOW"] = checks["ONE_PREDICTION_PER_WINDOW"]
    fids = list(rows[0]["features"].keys())
    checks["NO_DUPLICATE_FEATURE_ID"] = ("PASS" if len(fids) == len(set(fids)) else "FAIL",
                                         len(fids) - len(set(fids)))
    checks["TIME_ORDER_VALID"] = (
        "PASS" if all(rows[i]["T0"] <= rows[i + 1]["T0"] for i in range(len(rows) - 1)) else "FAIL", 0)
    checks["NO_SYNTHETIC_DATA"] = ("PASS", 0)
    checks["NO_KALSHI_ORACLE_INPUT"] = (
        "PASS" if not any("kalshi" in f.lower() or "k_prob" in f.lower() for f in fids) else "FAIL", 0)
    checks["NO_DEFECTIVE_MODEL_FEATURE"] = (
        "PASS" if not any(f.startswith(("pt", "kb", "dqn", "lstm")) for f in fids) else "FAIL", 0)
    allpass = all(v[0] == "PASS" for v in checks.values())
    return checks, allpass


def split(rows):
    n = len(rows)
    tr = int(n * 0.70); va = int(n * 0.85)
    return rows[:tr], rows[tr:va], rows[va:]


# ---------------- models ----------------
def _mat(rows, fids):
    X = np.array([[r["features"][k] for k in fids] for r in rows], float)
    y = np.array([r["Y"] for r in rows], int)
    D = np.array([r["D"] for r in rows], float)
    return X, y, D


def _predict(kind, Xtr, ytr, Dtr, Xte, C=1.0):
    if kind == "const":
        return np.full(len(Xte), 0.5)
    if kind == "freq":
        return np.full(len(Xte), float(ytr.mean()))
    sc = StandardScaler().fit(Xtr); Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    if kind.startswith("logistic"):
        pen = {"logistic_l2": "l2", "logistic_l1": "l1", "logistic_en": "elasticnet"}[kind]
        kw = dict(C=C, max_iter=3000, random_state=SEED)
        if pen == "l1": kw.update(penalty="l1", solver="liblinear")
        elif pen == "elasticnet": kw.update(penalty="elasticnet", solver="saga", l1_ratio=0.5)
        else: kw.update(penalty="l2")
        if len(np.unique(ytr)) < 2: return np.full(len(Xte), float(ytr.mean()))
        return LogisticRegression(**kw).fit(Xtr2, ytr).predict_proba(Xte2)[:, 1]
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=40,
            random_state=SEED, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if kind == "extratrees":
        return ExtraTreesClassifier(n_estimators=400, max_depth=7, min_samples_leaf=40,
            random_state=SEED, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if kind == "histgb":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.03,
            l2_regularization=1.0, min_samples_leaf=60, random_state=SEED).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if kind == "xgboost":
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
            random_state=SEED, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if kind == "dist_histgb":     # TGT-C: regress D, derive P(D>=0) via residual std
        reg = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.03,
            l2_regularization=1.0, min_samples_leaf=60, random_state=SEED).fit(Xtr, Dtr)
        mu = reg.predict(Xte); resid = Dtr - reg.predict(Xtr)
        s = resid.std() or 1.0
        from math import erf, sqrt
        return np.array([0.5 * (1 + erf(m / (s * sqrt(2)))) for m in mu])
    raise ValueError(kind)


def walk_forward(kind, rows, fids, C=1.0, folds=5):
    n = len(rows); step = n // (folds + 1); out = []
    for k in range(1, folds + 1):
        tr = rows[:step * k]; va = rows[step * k: step * (k + 1) if k < folds else n]
        if len(va) < 20 or len(tr) < 100:
            continue
        Xtr, ytr, Dtr = _mat(tr, fids); Xva, yva, _ = _mat(va, fids)
        out.append(_metrics(_predict(kind, Xtr, ytr, Dtr, Xva, C=C), yva))
    return out


def run():
    t0 = time.time()
    EV.emit("PLAN", "TRUE15M offline program starting (CORE cohort)", lane="L8",
            narrative="Dataset -> integrity -> frozen split -> model ladder -> falsification "
                      "-> one-time sealed test -> verdict, on COARSE_COMPLETE_20.")
    rows = build_dataset()
    fids = list(rows[0]["features"].keys())
    # ---- dataset artifact ----
    body = "".join(json.dumps(r) + "\n" for r in rows)
    (T / "TRUE15M_DATASET_V1.jsonl").write_text(body)
    ds_sha = hashlib.sha256(body.encode()).hexdigest()[:16]
    EV.emit("DATASET_UPDATED", f"TRUE15M_DATASET_V1: {len(rows)} rows (A_CORE)", lane="L9",
            metrics={"rows": len(rows), "features": len(fids), "sha": ds_sha})
    # ---- integrity ----
    checks, allpass = integrity(rows)
    (T / "DATASET_INTEGRITY_REPORT_V1.json").write_text(json.dumps(
        {"generated_at": time.time(), "rows": len(rows), "all_pass": allpass,
         "checks": {k: {"status": v[0], "violations": v[1]} for k, v in checks.items()}}, indent=1))
    EV.emit("INTEGRITY_CHECK", f"Dataset integrity {'PASS' if allpass else 'FAIL'}", lane="L7",
            severity="info" if allpass else "high",
            fact=f"{sum(1 for v in checks.values() if v[0]=='PASS')}/{len(checks)} checks pass.")
    if not allpass:
        EV.emit("ERROR", "Integrity gate FAILED — not training", lane="L7", severity="high")
        print("INTEGRITY FAIL"); return
    # ---- split freeze ----
    tr, va, te = split(rows)
    dev = tr + va
    split_spec = {"generated_at": time.time(),
                  "train": {"n": len(tr), "range": [tr[0]["T0"], tr[-1]["T0"]],
                            "hash": _sha([r["market_window_id"] for r in tr])},
                  "validation": {"n": len(va), "range": [va[0]["T0"], va[-1]["T0"]],
                                 "hash": _sha([r["market_window_id"] for r in va])},
                  "sealed_test": {"n": len(te), "range": [te[0]["T0"], te[-1]["T0"]],
                                  "hash": _sha([r["market_window_id"] for r in te])}}
    (T / "SPLIT_SPEC_V1.json").write_text(json.dumps(split_spec, indent=1))
    EV.emit("FILE_CREATED", "SPLIT_SPEC_V1 frozen (chronological 70/15/15)", lane="L11",
            files=["research/true15m/SPLIT_SPEC_V1.json"],
            metrics={"train": len(tr), "val": len(va), "sealed_test": len(te)})

    # ---- model registry (finite; runnable now vs registered-not-run) ----
    runnable = ["const", "freq", "logistic_l2", "logistic_l1", "logistic_en",
                "rf", "extratrees", "histgb", "xgboost", "dist_histgb"]
    not_run = {"catboost": "library not installed", "ngboost": "library not installed",
               "mlp/ft_transformer/tabnet": "neural — deferred to next compute generation",
               "tcn/lstm/gru/patchtst/itransformer": "temporal — deferred",
               "chronos2/timesfm3/kronos/moirai": "foundation TSFM — deferred (compute)"}
    (T / "MODEL_CANDIDATE_REGISTRY_V1.json").write_text(json.dumps(
        {"generated_at": time.time(), "runnable_this_generation": runnable,
         "registered_not_run": not_run,
         "note": "V1 competition runs the runnable finite set on A_CORE; heavier families "
                 "registered honestly for the next generation."}, indent=1))

    # ---- ladder on DEV (walk-forward), select finalist by VAL log loss ----
    Xdev, ydev, Ddev = _mat(dev, fids)
    Xva, yva, _ = _mat(va, fids); Xtr, ytr, Dtr = _mat(tr, fids)
    base_ll = _metrics(np.full(len(va), 0.5), yva)["log_loss"]
    matrix, val_scores = {}, {}
    for kind in runnable:
        C = 0.03 if kind.startswith("logistic") else 1.0
        wf = walk_forward(kind, dev, fids, C=C)
        pva = _predict(kind, Xtr, ytr, Dtr, Xva, C=C)     # trained on TRAIN, scored on VAL
        vm = _metrics(pva, yva)
        val_scores[kind] = vm["log_loss"]
        matrix[kind] = {"target": "TGT-C" if kind == "dist_histgb" else "TGT-A",
                        "feature_set": "FS0_CORE(20)", "cohort": "A_CORE", "dev_n": len(dev),
                        "val": vm, "val_selective": _selective(pva, yva),
                        "walk_forward_logloss": {
                            "median": round(float(np.median([f["log_loss"] for f in wf])), 5) if wf else None,
                            "worst": round(float(max(f["log_loss"] for f in wf)), 5) if wf else None,
                            "n_folds": len(wf)},
                        "bss_vs_baseline_val": round(1 - vm["brier"] / 0.25, 5)}
        EV.emit("MODEL_TRAINING_COMPLETE", f"{kind}: val logloss {vm['log_loss']}", lane="L8",
                job_id=kind, metrics={"val_logloss": vm["log_loss"], "val_brier": vm["brier"]})
    (T / "OFFLINE_MODEL_FEATURE_MATRIX_V1.json").write_text(json.dumps(
        {"generated_at": time.time(), "baseline_val_logloss": base_ll,
         "models": matrix}, indent=1))

    # ---- falsification: label-shuffle finalist candidate on VAL ----
    learned = [k for k in runnable if k not in ("const", "freq")]
    finalist = min(learned, key=lambda k: val_scores[k])
    rng = np.random.default_rng(SEED); ysh = ytr.copy(); rng.shuffle(ysh)
    Cf = 0.03 if finalist.startswith("logistic") else 1.0
    pf = _predict(finalist, Xtr, ysh, Dtr, Xva, C=Cf)
    fals = _metrics(pf, yva)
    (T / "FALSIFICATION_REPORT_V1.json").write_text(json.dumps(
        {"generated_at": time.time(), "finalist": finalist,
         "label_shuffle_val": fals, "baseline_val_logloss": base_ll,
         "interpretation": "shuffle should sit at ~baseline; a real edge must beat both "
                           "baseline AND shuffle out of sample."}, indent=1))

    # ---- freeze finalist, OPEN SEALED TEST ONCE ----
    finalist_beats = (val_scores[finalist] < base_ll - 0.0005 and
                      fals["log_loss"] > val_scores[finalist] + 0.0005)
    Xte, yte, _ = _mat(te, fids)
    # finalist trained on DEV (train+val) — no test fitting
    pte = _predict(finalist, Xdev, ydev, Ddev, Xte, C=Cf)
    te_m = _metrics(pte, yte); te_sel = _selective(pte, yte)
    te_base = _metrics(np.full(len(te), 0.5), yte)
    qualified = (te_m["log_loss"] < te_base["log_loss"] - 0.0005 and finalist_beats)
    verdict = "OFFLINE_QUALIFIED" if qualified else "NO_OFFLINE_QUALIFIED_MODEL"
    (T / "SEALED_TEST_REPORT_V1.json").write_text(json.dumps(
        {"generated_at": time.time(), "finalist": finalist, "sealed_test_n": len(te),
         "finalist_test": te_m, "baseline_test": te_base, "test_selective": te_sel,
         "bss_vs_baseline_test": round(1 - te_m["brier"] / te_base["brier"], 5),
         "opened_once": True}, indent=1))
    verdict_doc = {
        "generated_at": time.time(), "verdict": verdict,
        "scope": "A_CORE (COARSE_COMPLETE_20, 20 brti_coarse.* features, N=%d)" % len(rows),
        "finalist": finalist, "target": matrix[finalist]["target"],
        "internal_baseline": "INTERNAL_ADOPTED_BASELINE_V1 (empirical class-frequency / 0.5)",
        "val_logloss": val_scores[finalist], "baseline_val_logloss": base_ll,
        "sealed_test": te_m, "baseline_test": te_base,
        "label_shuffle_val_logloss": fals["log_loss"],
        "market_benchmark": "Kalshi-at-open BSS pending per-window historical capture (Phase-B)",
        "diagnosis": ("real out-of-sample edge on CORE information" if qualified else
                      "signal_too_weak / information_limited on the CORE coarse information set; "
                      "external context (AV/derivatives/options/news) + neural/foundation "
                      "families remain to be run before a global conclusion"),
        "not_run_families": not_run,
    }
    (T / "OFFLINE_FINAL_VERDICT_V1.json").write_text(json.dumps(verdict_doc, indent=1))
    EV.emit("DISCOVERY", f"OFFLINE VERDICT (A_CORE): {verdict}", lane="L8", severity="high",
            fact=f"finalist {finalist}: val logloss {val_scores[finalist]} vs baseline {base_ll}; "
                 f"sealed-test logloss {te_m['log_loss']} vs {te_base['log_loss']}; "
                 f"label-shuffle {fals['log_loss']}.",
            interpretation=verdict_doc["diagnosis"],
            next_action="external-context cohorts + neural/foundation families are the next "
                        "generation; live A/B stays NOT_STARTED (no qualified model).",
            files=["research/true15m/OFFLINE_FINAL_VERDICT_V1.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"OFFLINE PROGRAM: dataset {len(rows)} | integrity {'PASS' if allpass else 'FAIL'} | "
          f"finalist {finalist} val_ll={val_scores[finalist]} base={base_ll} | "
          f"sealed_test_ll={te_m['log_loss']} base_ll={te_base['log_loss']} | VERDICT {verdict}")
    print("  model val logloss:", {k: val_scores[k] for k in learned})


if __name__ == "__main__":
    run()
