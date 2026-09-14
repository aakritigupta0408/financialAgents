"""ModellingSnapshot for the OPEN_ORACLE_15M program (directive §54-59).

Pure presentation adapter: joins the canonical dataset meta + ladder evaluation
(computed by scripts/open_oracle_ladder.py, which owns all the science) into one
typed UI resource. Computes NO metrics itself. Writes results/modelling_snapshot.json.

Selected model logic: the ladder's verdict decides. On NO_CANDIDATE the serving
model for the 15-minute open forecast is MECH_FAIR_15M (the driftless 0.5
control) — shown honestly, not dressed up as a fancy learned model.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DMETA = ROOT / "results" / "open_oracle_15m_dataset.meta.json"
LADDER = ROOT / "results" / "open_oracle_15m_ladder.json"
OUT = ROOT / "results" / "modelling_snapshot.json"


def _j(p):
    return json.loads(p.read_text()) if p.exists() else {}


def build():
    dm = _j(DMETA)
    L = _j(LADDER)
    models = L.get("models", {})
    verdict = L.get("verdict", "NO_CANDIDATE")
    qualified = verdict.startswith("ONE_QUALIFIED")
    sel_id = verdict.split(":", 1)[1] if qualified else "M1_MECH_15M"

    # architecture actually serving (honest): mechanics control unless a challenger won
    if qualified:
        arch = ["T0 input features (≤ open)", "standardize", "MECH_FAIR_15M offset",
                "learned residual (" + sel_id + ")", "calibration", "p_oracle_15m", "FREEZE"]
    else:
        arch = ["T0 input features (≤ open)", "MECH_FAIR_15M (driftless, distance 0)",
                "p_mech_15m = 0.50", "FREEZE until settlement"]

    doc = {
        "schema_version": "modelling-snapshot-1",
        "generated_at": time.time(),
        "program": "OPEN_ORACLE_15M",
        "prediction_target": "P(settlement 60s-BRTI ≥ opening 60s-BRTI target) 15 min later",
        "prediction_time": "CONTRACT OPEN (T0)",
        "forecast_horizon": "15 minutes",
        "update_policy": "FROZEN UNTIL SETTLEMENT",
        "selected_model": {
            "id": sel_id,
            "family": "residual" if qualified else "mechanics-control",
            "role": "TREATMENT" if qualified else "CONTROL",
            "architecture": arch,
            "primary_loss": "log loss (binary)",
            "target_loss": "log loss / Brier vs exact BRTI outcome",
        },
        "dataset": {
            "market_window_n": dm.get("market_window_n"),
            "raw_observation_n": dm.get("raw_observation_n"),
            "class_balance_up": dm.get("class_balance_up"),
            "feature_families": dm.get("feature_families"),
            "content_sha256_16": dm.get("content_sha256_16"),
            "note": "one KXBTC15M window = one example (no tick inflation)",
        },
        "training_summary": {
            "protocol": L.get("protocol"),
            "dev_n": L.get("dev_n"), "holdout_n": L.get("holdout_n"),
            "holdout_base_rate_up": L.get("holdout_base_rate_up"),
            "note": "no epoch curves — logistic/tree models; walk-forward + single holdout",
        },
        "model_ladder": {mid: {"holdout": m.get("holdout"),
                               "walk_forward": m.get("walk_forward", {})}
                         for mid, m in models.items()},
        "offline_evaluation": {
            "verdict": verdict,
            "classification": L.get("classification"),
            "qualification": L.get("qualification"),
            "scope": L.get("scope"),
        },
        "live_experiment": {
            "state": "RUNNING" if qualified else "NOT_STARTED",
            "reason": ("qualified challenger registered" if qualified
                       else "no challenger beat the mechanics control out of sample"),
        },
        "diagnostics": {
            "TRUE_15M_NO_POST_OPEN_INFORMATION":
                "PASS" if dm.get("post_open_information_violations") == 0 else "FAIL",
            "ONE_PREDICTION_PER_WINDOW":
                "PASS" if dm.get("raw_observation_n") == dm.get("market_window_n") else "FAIL",
            "LABEL_SHUFFLE_PLACEBO": L.get("falsification", {}).get("label_shuffle_M2_holdout"),
            "label_shuffle_interpretation": L.get("falsification", {}).get("interpretation"),
        },
        "benchmarks": {"note": "MECH_FAIR_15M and 50/50 are controls; Kalshi-at-open is a "
                               "benchmark only (never a model input). Kalshi-at-open live "
                               "capture is a Phase-B item."},
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"modelling_snapshot: selected={sel_id} verdict={verdict} "
          f"windows={dm.get('market_window_n')} classification={L.get('classification')}")


if __name__ == "__main__":
    build()
