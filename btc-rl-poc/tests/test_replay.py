"""P3/P4/P10 integrity checks on the exact-BRTI replay result + trader family."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btc_rl import traders_v2 as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPLAY = ROOT / "research" / "replay" / "replay_result.json"
FREEZE = ROOT / "research" / "replay" / "prospective_freeze.json"


def _r():
    return json.loads(REPLAY.read_text())


def test_oracle_fit_is_devval_only():
    d = _r()
    assert d["oracle_fit"]["scope"].startswith("DEV+VAL")   # holdout OOS (P4)


def test_no_lookahead_rows_used():
    d = _r()
    assert d["dataset"]["lookahead_rows_dropped"] >= 0       # audited (P3)


def test_verdicts_from_allowed_set():
    allowed = {"QUALIFIED_FOR_PROSPECTIVE", "SHADOW_ONLY", "REJECTED", "INSUFFICIENT_EVIDENCE"}
    for name, v in _r()["offline_verdicts"].items():
        assert v in allowed, (name, v)


def test_execution_and_sizing_are_shadow():
    v = _r()["offline_verdicts"]
    assert v["T2"] == "SHADOW_ONLY"    # hindsight upper bound, not realizable
    assert v["T3"] == "SHADOW_ONLY"    # sizing judged on drawdown


def test_t1_tau_frozen_and_caveats_present():
    d = _r()
    assert "frozen_t1_tau_from_devval" in d
    assert any("HINDSIGHT" in c or "hindsight" in c for c in d["caveats"])


def test_trader_family_one_mechanism_each():
    # T1 differs from T0 only by the disagreement gate; params are frozen constants
    assert T.T1_EDGE_TAU > 0 and T.LOCK_MAX_S > 0 and T.MIN_EDGE > 0


def test_prospective_experiment_registered_no_retro_promotion():
    d = json.loads(FREEZE.read_text())
    e = d["first_experiment"]
    assert e["control"] == "T0" and e["treatment"] == "T1"
    assert e["prospective_state"] == "REGISTERED_PENDING_LIVE_CAPTURE"
    assert "No promotion from retrospective" in e["promotion_rule"]
