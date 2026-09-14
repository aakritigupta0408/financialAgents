"""Sealed-test governance (critical scientific correction).

TEST_V1 has been OBSERVED (the RF finalist was scored on it) while the research
program is still changing (derivatives/news/neural/foundation lanes remain). It is
therefore no longer a blind final audit — status EXPOSED_SPENT. The data are fine;
it is simply no longer eligible as the champion's final test.

TEST_V2 is sealed FORWARD from a later chronological cutoff that has never
influenced any feature/model/architecture/hyperparameter/calibration/threshold/
ensemble decision. Because L0 live capture never stopped, it accumulates
automatically (~96 windows/day). It must not be opened until the model registry
and finalists are frozen.

Invalidated results are preserved explicitly (never erased) — the AV -0.042 edge
that was pure future leakage is kept as INVALIDATED_RESULT.

Writes research/true15m/SEALED_TEST_STATUS.json + emits events.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

T = ROOT / "research" / "true15m"
SPLIT = T / "SPLIT_SPEC_V1.json"
INV = T / "contract_inventory.jsonl"
V2_MEMBERSHIP = T / "TEST_V2_MEMBERSHIP.jsonl"   # forward-rolling sealed accrual
OUT = T / "SEALED_TEST_STATUS.json"
TARGET_V2 = 672          # ~7 days @ 96/day — minimum for a meaningful blind test


def build():
    split = json.loads(SPLIT.read_text())
    v1 = split["sealed_test"]
    cutoff = v1["range"][1]                 # TEST_V2 starts strictly after TEST_V1 end
    # TEST_V2 accrues FORWARD — it must be counted from the live settled feed's
    # membership (test_v2_capture_audit.py), NOT from the FROZEN research inventory
    # (contract_inventory.jsonl ends at the cutoff by construction, so counting it
    # always yielded 0 — the original accumulator bug). If the membership file is
    # missing, run scripts/test_v2_capture_audit.py first.
    if V2_MEMBERSHIP.exists():
        v2_n = sum(1 for _ in V2_MEMBERSHIP.open() if _.strip())
    else:
        v2_n = 0

    doc = {
        "schema_version": "sealed-test-status-1", "generated_at": time.time(),
        "TEST_V1": {
            "status": "EXPOSED_SPENT",
            "range": v1["range"], "n": v1["n"], "hash": v1["hash"],
            "reason": "Observed during ongoing research (RF finalist scored on it) while "
                      "derivatives/news/neural/foundation lanes remain untested. No longer blind.",
            "data_quality": "FINE — not contaminated data; simply no longer eligible as the "
                            "final blind audit for the eventual champion.",
            "valid_historical_results": {
                "A_CORE": "NO_OFFLINE_QUALIFIED_MODEL on TEST_V1 (remains a valid historical result)"},
            "rule": "Future model selection must NOT optimize against TEST_V1 and then claim "
                    "success on TEST_V1.",
        },
        "TEST_V2": {
            "status": "SEALED_ACCUMULATING",
            "cutoff_T0": cutoff, "start_after": v1["range"][1],
            "accumulated_windows": v2_n, "target_windows": TARGET_V2,
            "eta_note": "L0 live capture accrues ~96 windows/day; ~7 days -> 672, ~14 -> 1,344.",
            "source": "forward settled feed (results/contract_outcomes.jsonl) -> "
                      "research/true15m/TEST_V2_MEMBERSHIP.jsonl (built by rule, not outcome)",
            "membership_file": "research/true15m/TEST_V2_MEMBERSHIP.jsonl",
            "capture_audit": "research/true15m/TEST_V2_CAPTURE_AUDIT.json",
            "accumulator_fix": "counts the forward membership, NOT the frozen inventory "
                               "(contract_inventory.jsonl ends at the cutoff -> old 0-count bug).",
            "seal_rule": "NEVER used for feature selection, model selection, architecture, "
                         "hyperparameters, calibration, thresholds, or ensembles.",
            "open_when": "model registry + finalists frozen — opened exactly once.",
        },
        "INVALIDATED_RESULTS": [{
            "id": "AV_incremental_v0",
            "claim": "BASE+AV cross-asset context improves log loss by ~0.042",
            "delta_logloss_raw": -0.042,
            "status": "INVALIDATED_RESULT",
            "cause": "future 5-minute bar leakage — AV bars timestamped at bar START; as-of "
                     "join on ts<=T0 admitted a bar whose interval extends ~5 min past T0.",
            "corrected_delta_logloss": -0.001,
            "corrected_verdict": "AV_NO_OOS_VALUE",
            "lesson": "the system caught itself cheating; a strong result collapsed under "
                      "causal alignment. Kept in history, never erased.",
        }],
        "test_state_taxonomy": ["DEVELOPMENT/WALK-FORWARD", "VALIDATION",
                                "TEST_V1 — SPENT", "TEST_V2 — SEALED", "PROSPECTIVE — COLLECTING"],
    }
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", "TEST_V1 marked SPENT; TEST_V2 sealed forward", lane="L9", severity="high",
            fact=f"TEST_V1 (N={v1['n']}, ends {v1['range'][1]}) observed -> EXPOSED_SPENT. "
                 f"TEST_V2 sealed after cutoff; accumulated {v2_n}/{TARGET_V2} so far.",
            interpretation="A_CORE NO_OFFLINE_QUALIFIED_MODEL stays a valid historical result, "
                           "but the champion needs a fresh blind test. Live capture builds it.",
            next_action="run remaining lanes/models on TRAIN+VAL+walk-forward only; freeze "
                        "finalists; open TEST_V2 once when it reaches target N.",
            files=["research/true15m/SEALED_TEST_STATUS.json"])
    EV.emit("WARNING", "INVALIDATED_RESULT preserved: AV -0.042 (future-bar leakage)", lane="L9",
            fact="raw -0.042 -> corrected -0.001 (AV_NO_OOS_VALUE). Retained in research history.")
    print(f"sealed_test_governance: TEST_V1 SPENT (N={v1['n']}); "
          f"TEST_V2 sealed, {v2_n}/{TARGET_V2} accumulated")


if __name__ == "__main__":
    build()
