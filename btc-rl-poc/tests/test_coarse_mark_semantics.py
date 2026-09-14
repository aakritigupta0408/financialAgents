"""COARSE_MARK_AVAILABILITY_SEMANTICS gate — the coarse settlement-mark chain may
only be frozen as a T0 predictive input if every mark used was AVAILABLE (not
merely dated) by T0. Run scripts/coarse_mark_semantics.py to (re)generate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "research" / "true15m" / "coarse_mark_semantics.json"


def test_coarse_mark_availability_semantics_pass():
    assert ART.exists(), "run scripts/coarse_mark_semantics.py first"
    d = json.loads(ART.read_text())
    assert d["available_after_T0_violations"] == 0, \
        f"{d['available_after_T0_violations']} marks available only after T0"
    assert d["verdict"] == "PASS"
    assert d["marks_evaluated"] > 0
