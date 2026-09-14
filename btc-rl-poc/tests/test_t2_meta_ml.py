"""Track C integrity — T2 meta-ML small-n discipline + NO_HOLDOUT_FIT + honest verdict."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "research" / "traders" / "t2_meta_ml_result.json"


def _r():
    return json.loads(RESULT.read_text())


def test_no_settlement_in_features():
    d = _r()
    assert d["no_settlement_in_features"] is True
    for f in d["features"]:
        assert "settle" not in f and "exact_yes" not in f and "outcome" not in f


def test_no_holdout_fit():
    # the Oracle + meta-model + scaler + threshold are fit on TRAIN/VAL only;
    # holdout windows are disjoint and only evaluated. Verify via replay split.
    import replay_backtest as RB
    rows, _ = RB.load()
    dev, val, hold = RB.split(rows)
    assert dev.isdisjoint(hold) and val.isdisjoint(hold)   # holdout untouched by fit
    d = _r()
    assert d["oracle_fit"].startswith("DEV+VAL")


def test_small_n_reports_windows_not_rows():
    d = _r()["small_n"]
    assert d["rows"] > d["windows"]                        # rows are NOT the unit
    assert d["effective_independent_n"] == d["holdout_windows"]
    assert "market_window_id" in d["note"]


def test_verdict_from_allowed_set_and_auc_gated():
    d = _r()
    assert d["verdict"] in {"QUALIFIED_FOR_PROSPECTIVE_SHADOW", "SHADOW_ONLY",
                            "REJECTED", "INSUFFICIENT_EVIDENCE", "INFORMATION_LIMITED"}
    # if no model is meaningfully predictive OOS, verdict cannot be QUALIFIED
    if d["best_holdout_auc"] <= 0.52:
        assert d["verdict"] != "QUALIFIED_FOR_PROSPECTIVE_SHADOW"


def test_mandatory_model_sequence_present():
    d = _r()
    assert "logistic" in d["models_tested"]                # simplest first (mandatory)
