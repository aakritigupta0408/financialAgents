"""L11 — Research Intelligence queue (directive P1.3).

Accumulates ready-to-run experiment specs while the dataset lanes finish. It ONLY
produces specs; it never trains against the (unfrozen, and later sealed) task.
Every item is QUEUED_AFTER_DATASET_GATE. Writes
research/true15m/RESEARCH_INTELLIGENCE_QUEUE.json and emits an event.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

OUT = ROOT / "research" / "true15m" / "RESEARCH_INTELLIGENCE_QUEUE.json"
GATE = "QUEUED_AFTER_DATASET_GATE"

QUEUE = [
    {"id": "RI-001", "paper": "Chronos-2 (covariate-aware TSFM)",
     "transfer": "multivariate/covariate temporal representation over BRTI+context",
     "our_test": "coarse BRTI state + AV context -> frozen Chronos-2 repr -> small prob head",
     "risk": "generic TSFM may not preserve financial direction signal at 15m",
     "data_requirement": "A_AV cohort", "status": GATE},
    {"id": "RI-002", "paper": "Kronos (financial temporal encoder)",
     "transfer": "finance-pretrained encoder with our own PIT normalization",
     "our_test": "PIT-safe sequence -> frozen Kronos embedding -> logistic/CatBoost head",
     "risk": "pretraining window / leakage vs our T0 cutoff must be verified",
     "data_requirement": "B_FINE or A_CORE sequences", "status": GATE},
    {"id": "RI-003", "paper": "TimesFM 3 representation",
     "transfer": "zero-shot forecast features as covariates",
     "our_test": "frozen forecast features -> ridge residual on p_mech",
     "risk": "horizon/resolution mismatch (15m marks vs model pretraining)",
     "data_requirement": "A_CORE", "status": GATE},
    {"id": "RI-004", "paper": "Moirai / Moirai-MoE (probabilistic)",
     "transfer": "distributional forecast of BRTI displacement D",
     "our_test": "predict distribution of D -> derive P(D>=0)",
     "risk": "small N (esp. FINE) vs model capacity", "data_requirement": "A_CORE / A_AV", "status": GATE},
    {"id": "RI-005", "paper": "PatchTST", "transfer": "patch-based small-sequence challenger",
     "our_test": "coarse+fine sequences -> PatchTST -> prob head, capacity-capped",
     "risk": "overfit at N~6k; needs strong regularization", "data_requirement": "A_CORE", "status": GATE},
    {"id": "RI-006", "paper": "iTransformer", "transfer": "inverted-attention over feature channels",
     "our_test": "multi-family channels -> iTransformer -> prob head",
     "risk": "channel count vs N; interpretability", "data_requirement": "A_AV_DERIV", "status": GATE},
    {"id": "RI-007", "paper": "TCN baseline", "transfer": "causal dilated conv sequence model",
     "our_test": "fine sequence -> small TCN -> prob head (Family-B)",
     "risk": "363 windows is very small for a sequence net", "data_requirement": "B_FINE", "status": GATE},
    {"id": "RI-008", "paper": "Distributional CatBoost (quantile) predicting D",
     "transfer": "quantile regression of displacement -> P(D>=0)",
     "our_test": "tabular coarse+context -> quantile CatBoost -> integrate over 0",
     "risk": "calibration of derived probability", "data_requirement": "A_AV", "status": GATE},
    {"id": "RI-009", "paper": "NGBoost / conditional distribution of D",
     "transfer": "natural-gradient boosting for predictive distribution",
     "our_test": "tabular -> NGBoost(Normal) -> P(D>=0) with calibration",
     "risk": "distributional assumption (heavy tails of BRTI moves)",
     "data_requirement": "A_CORE / A_AV", "status": GATE},
]


def build():
    doc = {"schema_version": "research-intelligence-queue-1", "generated_at": time.time(),
           "policy": "specs only; NO training against the task until the dataset+split are "
                     "frozen. Every item stays QUEUED_AFTER_DATASET_GATE.",
           "n_specs": len(QUEUE), "queue": QUEUE}
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("PLAN", f"Research Intelligence queue: {len(QUEUE)} experiment specs", lane="L11",
            fact="RI-001..RI-009 queued (Chronos-2, Kronos, TimesFM-3, Moirai, PatchTST, "
                 "iTransformer, TCN, distributional CatBoost, NGBoost).",
            interpretation="Model experiments are pre-specified so the race can start the "
                           "instant the dataset gate passes — no scramble, no post-hoc design.",
            next_action="hold at QUEUED_AFTER_DATASET_GATE until TRUE15M_DATASET_V1 + sealed split.",
            files=[str(OUT.relative_to(ROOT))], metrics={"specs": len(QUEUE)})
    print(f"research_intelligence: {len(QUEUE)} specs queued (all {GATE})")


if __name__ == "__main__":
    build()
