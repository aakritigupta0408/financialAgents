"""OPEN_ORACLE_15M model ladder — chronological walk-forward + final holdout.

Directive §11,14,15,16,18,31. Tests whether ANY pre-open (<=T0) information beats
the driftless mechanics control (0.5) at forecasting the exact 15-minute BRTI
outcome. One window = one example.

Ladder:
  M0  50/50                      (sanity floor)
  M1  MECH_FAIR_15M == 0.5       (driftless mechanics at the open)
  M2  ridge logistic residual    (offset logit(p_mech)=0 -> L2 logistic on features)
  M4  HistGradientBoosting        (conservative nonlinear residual; sklearn LightGBM-equiv)

Protocol (no holdout leakage — §15 NO_HOLDOUT_FIT_15M):
  * chronological sort; final 20% = UNTOUCHED holdout.
  * walk-forward on the first 80% (expanding train -> next validation block).
  * scalers/hyperparameters fit on train/validation ONLY; holdout scored once.
Falsification (§16): a label-shuffled run must collapse to ~chance on holdout.

Metrics via the canonical owners where they exist (btc_rl.metrics.brier /
calibration_bins) + sklearn log_loss. Writes results/open_oracle_15m_ladder.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_log_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import metrics as M   # noqa: E402

DS = ROOT / "results" / "open_oracle_15m_dataset.jsonl"
OUT = ROOT / "results" / "open_oracle_15m_ladder.json"

HOLDOUT_FRAC = 0.20
N_WF_FOLDS = 6
CONF_LEVELS = [0.60, 0.70, 0.80, 0.90]   # registered pre-holdout selective thresholds
SEED = 17


def _load():
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    rows.sort(key=lambda r: r["T0"])
    feat_keys = sorted(rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in feat_keys] for r in rows], float)
    y = np.array([r["official_outcome"] for r in rows], int)
    return rows, feat_keys, X, y


def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def _prob_metrics(p, y):
    p = _clip(np.asarray(p, float))
    y = np.asarray(y, int)
    ll = float(sk_log_loss(y, p, labels=[0, 1]))
    br = float(M.brier(p.tolist(), y.tolist()))
    acc = float((( p >= 0.5).astype(int) == y).mean())
    cb = M.calibration_bins(p.tolist(), y.tolist(), n_bins=10)
    # ECE from the canonical calibration bins
    ece = 0.0
    for b in cb:
        nb = b.get("n", 0)
        if nb:
            ece += (nb / len(y)) * abs(b.get("p_mean", 0) - b.get("y_freq", 0))
    return {"log_loss": round(ll, 5), "brier": round(br, 5),
            "accuracy": round(acc, 4), "ece": round(ece, 4)}


def _selective(p, y):
    p = np.asarray(p, float); y = np.asarray(y, int)
    conf = np.abs(p - 0.5) * 2                       # 0..1 confidence
    pred = (p >= 0.5).astype(int)
    out = {}
    for c in CONF_LEVELS:
        m = conf >= (2 * (c - 0.5))                  # p>=c or p<=1-c
        cov = float(m.mean())
        acc = float((pred[m] == y[m]).mean()) if m.any() else None
        out[f"conf>={c:.2f}"] = {"coverage": round(cov, 4),
                                 "accuracy": round(acc, 4) if acc is not None else None,
                                 "n": int(m.sum())}
    return out


def _fit_predict(kind, Xtr, ytr, Xte, C=1.0):
    if kind == "const":
        return np.full(len(Xte), 0.5)
    sc = StandardScaler().fit(Xtr)
    Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    if kind == "logistic":
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xte), float(ytr.mean()))
        clf = LogisticRegression(C=C, penalty="l2", max_iter=2000, random_state=SEED)
        clf.fit(Xtr2, ytr)
        return clf.predict_proba(Xte2)[:, 1]
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.03,
            l2_regularization=1.0, min_samples_leaf=60, random_state=SEED)
        clf.fit(Xtr, ytr)                            # trees: raw features ok
        return clf.predict_proba(Xte)[:, 1]
    raise ValueError(kind)


def _walk_forward(kind, X, y, C=1.0):
    """Expanding-window walk-forward on the dev slice; returns per-fold val metrics."""
    n = len(y)
    fold = n // (N_WF_FOLDS + 1)
    folds = []
    for k in range(1, N_WF_FOLDS + 1):
        tr_end = fold * k
        va_end = fold * (k + 1) if k < N_WF_FOLDS else n
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xva, yva = X[tr_end:va_end], y[tr_end:va_end]
        if len(yva) < 20 or len(ytr) < 50:
            continue
        p = _fit_predict(kind, Xtr, ytr, Xva, C=C)
        folds.append(_prob_metrics(p, yva))
    return folds


def _summ(folds, key):
    vals = [f[key] for f in folds]
    return {"median": round(float(np.median(vals)), 5),
            "worst": round(float(max(vals)), 5),        # higher loss = worse
            "variance": round(float(np.var(vals)), 6),
            "n_folds": len(vals)}


def run():
    rows, feat_keys, X, y = _load()
    n = len(y)
    h = int(n * (1 - HOLDOUT_FRAC))
    Xdev, ydev, Xhold, yhold = X[:h], y[:h], X[h:], y[h:]

    result = {
        "schema_version": "open-oracle-15m-ladder-1", "generated_at": time.time(),
        "dataset_sha": None, "market_window_n": n,
        "dev_n": int(h), "holdout_n": int(n - h),
        "holdout_base_rate_up": round(float(yhold.mean()), 4),
        "features": feat_keys, "conf_levels": CONF_LEVELS,
        "protocol": "chronological expanding walk-forward on dev; final 20% untouched holdout; "
                    "scalers/C fit on dev only (NO_HOLDOUT_FIT_15M).",
        "models": {}, "falsification": {},
    }
    try:
        m = json.loads((ROOT / "results" / "open_oracle_15m_dataset.meta.json").read_text())
        result["dataset_sha"] = m.get("content_sha256_16")
    except Exception:
        pass

    # ----- baselines -----
    for mid, kind in [("M0_5050", "const"), ("M1_MECH_15M", "const")]:
        p_hold = _fit_predict(kind, Xdev, ydev, Xhold)
        result["models"][mid] = {"holdout": _prob_metrics(p_hold, yhold),
                                 "holdout_selective": _selective(p_hold, yhold),
                                 "walk_forward": {}}

    # ----- M2 ridge logistic: pick C on walk-forward, then score holdout once -----
    best_C, best_ll = 1.0, 1e9
    C_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    C_search = {}
    for C in C_grid:
        folds = _walk_forward("logistic", Xdev, ydev, C=C)
        med = float(np.median([f["log_loss"] for f in folds])) if folds else 1e9
        C_search[str(C)] = round(med, 5)
        if med < best_ll:
            best_ll, best_C = med, C
    wf2 = _walk_forward("logistic", Xdev, ydev, C=best_C)
    p2 = _fit_predict("logistic", Xdev, ydev, Xhold, C=best_C)
    result["models"]["M2_ridge_logistic"] = {
        "chosen_C": best_C, "C_search_wf_median_logloss": C_search,
        "walk_forward": {"log_loss": _summ(wf2, "log_loss"), "brier": _summ(wf2, "brier"),
                         "accuracy": _summ(wf2, "accuracy")},
        "holdout": _prob_metrics(p2, yhold), "holdout_selective": _selective(p2, yhold)}

    # ----- M4 HistGradientBoosting -----
    wf4 = _walk_forward("hgb", Xdev, ydev)
    p4 = _fit_predict("hgb", Xdev, ydev, Xhold)
    result["models"]["M4_hgb"] = {
        "walk_forward": {"log_loss": _summ(wf4, "log_loss"), "brier": _summ(wf4, "brier"),
                         "accuracy": _summ(wf4, "accuracy")},
        "holdout": _prob_metrics(p4, yhold), "holdout_selective": _selective(p4, yhold)}

    # ----- falsification: shuffle labels in dev, expect ~chance on holdout (§16) -----
    rng = np.random.default_rng(SEED)
    yshuf = ydev.copy(); rng.shuffle(yshuf)
    pf = _fit_predict("logistic", Xdev, yshuf, Xhold, C=best_C)
    result["falsification"]["label_shuffle_M2_holdout"] = _prob_metrics(pf, yhold)
    result["falsification"]["interpretation"] = (
        "label-shuffle holdout log_loss should sit at ~0.693 (chance); a real edge "
        "must beat that AND beat M0/M1 (0.25 brier / 0.693 log_loss).")

    # ----- offline verdict (§31,32,61) — computed, not asserted -----
    base_ll = result["models"]["M1_MECH_15M"]["holdout"]["log_loss"]
    shuf_ll = result["falsification"]["label_shuffle_M2_holdout"]["log_loss"]
    challengers = {}
    for mid in ("M2_ridge_logistic", "M4_hgb"):
        hl = result["models"][mid]["holdout"]["log_loss"]
        wf = result["models"][mid]["walk_forward"]["log_loss"]
        beats_base = hl < base_ll                          # better (lower) than mechanics
        beats_noise = hl < shuf_ll - 0.0005                # meaningfully better than shuffle
        wf_beats = wf["median"] < base_ll and wf["worst"] < base_ll
        challengers[mid] = {"holdout_beats_mech": beats_base,
                            "holdout_beats_shuffle_noise": beats_noise,
                            "walkforward_median_and_worst_beat_mech": wf_beats,
                            "qualifies": bool(beats_base and beats_noise and wf_beats)}
    qualified = [m for m, v in challengers.items() if v["qualifies"]]
    result["qualification"] = challengers
    result["verdict"] = ("ONE_QUALIFIED_CHALLENGER:" + qualified[0]) if qualified else "NO_CANDIDATE"
    result["classification"] = ("PROMISING" if qualified else "INFORMATION_LIMITED")
    result["scope"] = ("Tested F-CONTRACT price-path family on the full "
                       f"{n}-window universe. Independent families (F-SPOT/F-XVENUE/"
                       "F-DERIVATIVES/F-OPTIONS/F-NEWS) have ~370-window coverage only "
                       "and are a separate Phase-B test, not evaluated here.")
    OUT.write_text(json.dumps(result, indent=1))
    # console summary
    def hl(mid): return result["models"][mid]["holdout"]
    print(f"OPEN_ORACLE_15M ladder  (holdout n={n-h}, base_up={result['holdout_base_rate_up']})")
    for mid in ("M0_5050", "M1_MECH_15M", "M2_ridge_logistic", "M4_hgb"):
        m = hl(mid)
        print(f"  {mid:20} holdout  logloss={m['log_loss']}  brier={m['brier']}  acc={m['accuracy']}  ece={m['ece']}")
    fs = result["falsification"]["label_shuffle_M2_holdout"]
    print(f"  label-shuffle M2      holdout  logloss={fs['log_loss']}  brier={fs['brier']} (expect ~0.693/0.25)")
    print(f"  M2 chosen C={best_C}")
    print(f"  VERDICT: {result['verdict']}  ({result['classification']})")


if __name__ == "__main__":
    run()
