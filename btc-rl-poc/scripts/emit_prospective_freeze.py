"""P11/P12 — freeze the prospective version set and register the first formal
exact-BRTI economic experiment (T0 CONTROL vs T1 TREATMENT). Once prospective
evidence begins these MUST NOT change mid-experiment (P11); new research runs in
shadow. Writes research/replay/prospective_freeze.json.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import traders_v2 as T  # noqa: E402

RES = ROOT / "results"


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def main():
    replay = _jload(ROOT / "research" / "replay" / "replay_result.json") or {}
    frozen = _jload(ROOT / "research" / "oracle" / "oracle_frozen.json") or {}
    spec = {
        "contract_spec": "research/contract_specs/KXBTC15M_2026-09.json",
        "mech_fair_version": "MECH_FAIR_BRTI",
        "oracle_model_version": frozen.get("spec_hash"),
        "oracle_note": "recalibrated MECH_FAIR_BRTI; NO Kalshi input",
        "feature_version": "brti-decision-v1",
        "lock_policy": {"lock_max_s": T.LOCK_MAX_S, "min_edge": T.MIN_EDGE},
        "trader_policies": T.FROZEN,
        "t1_edge_tau_frozen_on": "DEV+VAL",
        "t1_edge_tau": replay.get("frozen_t1_tau_from_devval", T.T1_EDGE_TAU),
        "fee_model": "0.07*p*(1-p) per contract",
        "execution_assumption": "enter at Kalshi mid + half-spread",
        "metric_version": "econ-1/prob-1",
        "settlement": "OFFICIAL_EXACT_BRTI",
    }
    experiment = {
        "experiment_id": "PROSPECTIVE-T0-vs-T1-exactBRTI",
        "spec_hash": _hash(spec),
        "hypothesis": "Abstaining unless the independent Oracle materially disagrees "
                      "with the executable market (|p_oracle - p_market| >= tau) "
                      "improves prospective realized paper EV per eligible window.",
        "control": "T0", "treatment": "T1", "unit": "market_window_id",
        "primary_metric": "paired_delta(REALIZED_PAPER_EV_PER_ELIGIBLE_WINDOW)",
        "secondary_metrics": ["ev_per_trade", "coverage", "accuracy", "earliness",
                              "total_pnl", "max_drawdown", "bad_entry_rate"],
        "backtest_holdout": {
            "T0": (replay.get("final_holdout", {}) or {}).get("T0"),
            "T1": (replay.get("final_holdout", {}) or {}).get("T1"),
            "verdicts": replay.get("offline_verdicts")},
        "prospective_state": "REGISTERED_PENDING_LIVE_CAPTURE",
        "blocked_on": "DT-01 live activation (EXACT_BRTI_RUNTIME_ENABLED=1) — prospective "
                      "windows accrue only once the live daemon settles on exact BRTI",
        "promotion_rule": "No promotion from retrospective/backtest evidence; requires "
                          "prospective paired-Δ CI excluding 0 on unseen live windows.",
    }
    doc = {"schema_version": "prospective-freeze-1", "frozen_versions": spec,
           "first_experiment": experiment,
           "lineage_map": {                     # P15
               "Follower/pt": "T0", "MLE/pt6 selective abstention": "T1",
               "execution-timing/exec_timing": "T2",
               "Gambler/Saver sizing": "T3",
               "other legacy traders": "EVIDENCE_ONLY / RETIRED"},
           "note": "Frozen prospective version set (P11). Do not change mid-experiment; "
                   "new research runs in shadow."}
    out = ROOT / "research" / "replay" / "prospective_freeze.json"
    out.write_text(json.dumps(doc, indent=1))
    print(f"prospective freeze — spec_hash {experiment['spec_hash']} · "
          f"first experiment {experiment['experiment_id']} · "
          f"state {experiment['prospective_state']}")


if __name__ == "__main__":
    main()
