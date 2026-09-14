"""TRACK C — T2 META-ML trader (SHADOW). Meta-labeling, NOT direction prediction.

The Oracle owns UP/DOWN. T2 answers: given an already-eligible T0 candidate trade,
should we TAKE or SKIP? Meta-label Y = 1 iff taking the trade at the registered
executable entry produces positive realized paper economics after fees.

Strict chronological split (window unit): TRAIN / VAL / HOLDOUT. The Oracle is fit
on TRAIN+VAL only (holdout OOS); the meta-model + scaler + accept-threshold are fit
on TRAIN and selected on VAL only. Holdout is opened once. NO settlement info in
features. Small-n: window count is the independent unit, not decision rows.

Models: logistic, L2 logistic, HistGradientBoosting (LightGBM proxy; lightgbm N/A).
Compares T2 vs T0 (take-all) and T1 (fixed 0.15 disagreement) on identical windows.
Writes research/traders/t2_meta_ml_result.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import replay_backtest as RB          # noqa: E402 (frozen replay + policies)
from btc_rl import traders_v2 as T    # noqa: E402
from btc_rl import economics as E     # noqa: E402

OUT = ROOT / "research" / "traders" / "t2_meta_ml_result.json"
FEATS = ["p_mech", "p_oracle", "oracle_delta", "abs_oracle_delta", "tte_min",
         "brti_dist", "norm_dist", "rvol", "cost", "spread", "quoted_ev"]


def candidate(rows_of_window, sigma):
    """T0's candidate trade for a window: first eligible row. Returns (features, label,
    econ) or None. Label = realized pnl>0. NO settlement info in features."""
    elig = RB.eligible_rows(rows_of_window)
    if not elig:
        return None
    r, side, cost, ev = elig[0]
    pm = RB._p_mech(r, sigma)
    tte = max(1.0, r["time_remaining_s"])
    norm = abs(r["brti_distance_to_target"]) / (sigma * r["current_brti"] * math.sqrt(tte))
    feat = {"p_mech": pm, "p_oracle": r["p_oracle"], "oracle_delta": r["p_oracle"] - r["k_prob"],
            "abs_oracle_delta": abs(r["p_oracle"] - r["k_prob"]), "tte_min": tte / 60.0,
            "brti_dist": r["brti_distance_to_target"], "norm_dist": norm,
            "rvol": r.get("cb_rvol_30s") or 0.0, "cost": cost, "spread": r["half_spread"] * 2,
            "quoted_ev": ev}
    pnl, win = T.settle_pnl(side, cost, T.FIXED_STAKE, r["exact_yes"])
    return feat, int(pnl > 0), {"pnl_c": pnl, "side": side, "wid": r["market_window_id"]}


def build(wins_rows, sigma):
    X, y, meta = [], [], []
    for wid, wr in wins_rows.items():
        c = candidate(wr, sigma)
        if c:
            f, lab, m = c
            X.append([f[k] for k in FEATS]); y.append(lab); meta.append(m)
    return np.array(X, float), np.array(y, float), meta


def econ_from_mask(meta, take_mask, n_eligible):
    pnls = [m["pnl_c"] for m, t in zip(meta, take_mask) if t]
    return {"n_eligible": n_eligible, "trades": int(sum(take_mask)),
            "coverage": E.coverage(int(sum(take_mask)), n_eligible),
            "ev_per_eligible_c": E.ev_per_eligible_window(pnls, n_eligible),
            "ev_per_trade_c": E.ev_per_trade(pnls),
            "total_pnl_c": E.realized_pnl(pnls),
            "max_drawdown_c": E.drawdown(E.equity_curve(0.0, pnls)),
            "bad_entry_rate": E.bad_entry_rate(pnls)}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows, leak = RB.load()
    dev, val, hold = RB.split(rows)
    devval = dev | val
    sigma, iso = RB.fit_oracle([r for r in rows if r["market_window_id"] in devval])
    RB.attach_oracle(rows, sigma, iso)
    # TRAIN=dev, VAL=val, HOLDOUT=hold (windows)
    Xtr, ytr, mtr = build(RB.by_window(rows, dev), sigma)
    Xva, yva, mva = build(RB.by_window(rows, val), sigma)
    Xte, yte, mte = build(RB.by_window(rows, hold), sigma)
    if len(Xtr) < 40 or len(Xte) < 20:
        print("insufficient candidates:", len(Xtr), len(Xte)); return

    scaler = StandardScaler().fit(Xtr)            # fit on TRAIN only
    Ztr, Zva, Zte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    models = {}
    models["logistic"] = LogisticRegression(penalty=None, max_iter=500).fit(Ztr, ytr)
    models["logistic_l2"] = LogisticRegression(C=0.5, max_iter=500).fit(Ztr, ytr)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        models["histgbm"] = HGB(max_depth=3, max_iter=200, learning_rate=0.05,
                                l2_regularization=1.0, min_samples_leaf=15).fit(Xtr, ytr)
    except Exception:
        pass

    n_elig_hold = len(mte)
    # baselines on holdout: T0 take-all; T1 disagreement rule (>=0.15)
    t0_take = [True] * len(mte)
    t1_take = [abs(f[2]) >= T.T1_EDGE_TAU for f in Xte]   # index 2 = oracle_delta
    results = {"T0_take_all": econ_from_mask(mte, t0_take, n_elig_hold),
               "T1_disagree_0.15": econ_from_mask(mte, t1_take, n_elig_hold)}
    model_report = {}
    for name, mdl in models.items():
        use_scaled = name.startswith("logistic")
        pva = mdl.predict_proba(Zva if use_scaled else Xva)[:, 1]
        # select accept-threshold on VAL by EV/eligible (economic, not accuracy)
        best_thr, best_ev = 0.5, -1e9
        for thr in np.linspace(0.3, 0.8, 26):
            ev = econ_from_mask(mva, pva >= thr, len(mva))["ev_per_eligible_c"] or -1e9
            if ev > best_ev:
                best_ev, best_thr = ev, thr
        pte = mdl.predict_proba(Zte if use_scaled else Xte)[:, 1]
        take = pte >= best_thr
        econ = econ_from_mask(mte, take, n_elig_hold)
        # diagnostics
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
        try:
            auc = round(float(roc_auc_score(yte, pte)), 4)
        except Exception:
            auc = None
        model_report[name] = {"accept_threshold_from_val": round(best_thr, 3),
                              "economics_holdout": econ,
                              "diagnostics": {"auc": auc,
                                              "brier": round(float(brier_score_loss(yte, pte)), 4),
                                              "log_loss": round(float(log_loss(yte, np.clip(pte, 1e-6, 1 - 1e-6))), 4)}}
        results[f"T2_{name}"] = econ

    # verdict: a learned meta-trader must (a) actually PREDICT its label OOS (holdout
    # AUC meaningfully > 0.5) before any EV improvement can be trusted, then (b) beat
    # T0 AND T1 on EV/eligible. An EV gain from a non-predictive model (AUC<=0.5) is
    # threshold-selection noise on small n — NOT qualification.
    best_auc = max((mr["diagnostics"]["auc"] or 0.0) for mr in model_report.values())
    best_t2 = max((v for k, v in results.items() if k.startswith("T2_")),
                  key=lambda e: e["ev_per_eligible_c"] or -1e9)
    beats_t0 = (best_t2["ev_per_eligible_c"] or -1e9) > (results["T0_take_all"]["ev_per_eligible_c"] or -1e9)
    beats_t1 = (best_t2["ev_per_eligible_c"] or -1e9) > (results["T1_disagree_0.15"]["ev_per_eligible_c"] or -1e9)
    if best_t2["trades"] < 20:
        verdict = "INSUFFICIENT_EVIDENCE"
    elif best_auc <= 0.52:
        verdict = "INFORMATION_LIMITED"      # meta-label not learnable OOS; EV gain is noise
    elif beats_t0 and beats_t1:
        verdict = "QUALIFIED_FOR_PROSPECTIVE_SHADOW"
    else:
        verdict = "SHADOW_ONLY"

    doc = {
        "schema_version": "t2-metaml-1", "role": "SHADOW",
        "target": "Y_trade = 1 if T0 candidate trade realized pnl>0 after fees (meta-label)",
        "features": FEATS, "no_settlement_in_features": True,
        "small_n": {"rows": len(rows), "windows": len(dev | val | hold),
                    "train_windows": len(Xtr), "val_windows": len(Xva),
                    "holdout_windows": len(Xte),
                    "effective_independent_n": len(Xte),
                    "note": "unit = market_window_id; decision rows are NOT independent"},
        "oracle_fit": "DEV+VAL only (holdout OOS)",
        "models_tested": list(models.keys()),
        "model_report": model_report,
        "comparison_holdout": results,
        "best_holdout_auc": best_auc,
        "verdict": verdict,
        "caveats": ["OFFLINE holdout n=%d windows — small; SHADOW only, may not replace "
                    "T1 from retrospective evidence." % len(Xte),
                    "holdout AUC <= 0.5 -> meta-label NOT learnable OOS from these "
                    "features; any EV 'improvement' is threshold-selection noise, not "
                    "signal. Verdict reflects that, not the raw EV number.",
                    "meta-label uses proxy T0 entry economics on official BRTI settlement."],
        "note": "T2 is downstream of the Oracle; market valuation features allowed. "
                "Oracle probability is NOT altered by T2.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"T2 META-ML — holdout {len(Xte)} windows (eff n {len(Xte)}), models {list(models)}")
    print(f"  T0 take-all  EV/elig {results['T0_take_all']['ev_per_eligible_c']}c  "
          f"cov {results['T0_take_all']['coverage']}")
    print(f"  T1 0.15      EV/elig {results['T1_disagree_0.15']['ev_per_eligible_c']}c  "
          f"cov {results['T1_disagree_0.15']['coverage']}")
    for name in models:
        e = results[f"T2_{name}"]
        print(f"  T2 {name:10s} EV/elig {e['ev_per_eligible_c']}c  cov {e['coverage']}  "
              f"auc {model_report[name]['diagnostics']['auc']}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
