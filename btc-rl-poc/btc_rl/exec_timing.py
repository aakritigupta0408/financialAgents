"""EXEC-TIMING treatment (EXEC_TIMING_SPEC.yaml, hash 9c0c89ae).

The ONE formal treatment. Given the frozen upstream decision (side, size,
enter-intent), the confirmed T0.5 reprice model predicts the near-term
Kalshi move and may only defer/skip the entry — never the side.

SAFETY: feature-flagged. When EXEC_TIMING_TREATMENT_ENABLED is false the
decision is byte-for-byte the control action (see tests/test_exec_timing.py).
An orthogonal-source outage or a missing model file also falls back to
control. This module is pure and importable without side effects.
"""
import json
import os
from pathlib import Path

_MODEL_PATH = (Path(__file__).resolve().parent.parent
               / "research" / "t05_repricing" / "exec_timing_model.json")
_MODEL = None


def enabled():
    """Runtime flag. Default OFF; activated only in the live checkout."""
    return os.environ.get("EXEC_TIMING_TREATMENT_ENABLED", "false").lower() \
        in ("1", "true", "yes", "on")


def _model():
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = json.loads(_MODEL_PATH.read_text())
        except Exception:
            _MODEL = {}
    return _MODEL


def predict_yes_move(features):
    """Predicted 15s change in Kalshi YES-probability. 0.0 if model/features
    unavailable (safe: yields no deferral)."""
    m = _model()
    if not m or "beta" not in m:
        return 0.0
    z = 0.0
    for i, f in enumerate(m["feats"]):
        v = features.get(f)
        if v is None:
            return 0.0                       # incomplete features -> no action
        z += m["beta"][i] * (v - m["mu"][i]) / m["sd"][i]
    return z


def decide(control_action, side, features, *, force_enabled=None):
    """Return the timing action for an intended entry.

    control_action : the frozen control decision, one of ENTER_NOW/SKIP/... —
                     returned unchanged unless the treatment is active AND
                     chooses to DEFER.
    side           : "yes" or "no" — the FROZEN settlement side (never changed).
    features       : dict of the T0.5 microstructure features at decision time.
    Returns (action, meta). action in {control_action, "DEFER"}.
    """
    on = enabled() if force_enabled is None else force_enabled
    if not on or control_action != "ENTER_NOW":
        return control_action, {"treatment": "off" if not on else "n/a",
                                "predicted_yes_move": None}
    m = _model()
    thr = m.get("defer_threshold", 0.002)
    yes_move = predict_yes_move(features)
    # cost move of the INTENDED contract: a YES buyer is hurt by YES rising;
    # a NO buyer is hurt by YES falling. "Cheaper soon" => defer.
    intended_move = yes_move if side == "yes" else -yes_move
    if intended_move <= -thr:                # contract predicted to get cheaper
        return "DEFER", {"treatment": "on", "action": "DEFER",
                         "predicted_yes_move": round(yes_move, 5),
                         "defer_horizon_s": m.get("defer_horizon_s", 15)}
    return control_action, {"treatment": "on", "action": "ENTER_NOW",
                            "predicted_yes_move": round(yes_move, 5)}
