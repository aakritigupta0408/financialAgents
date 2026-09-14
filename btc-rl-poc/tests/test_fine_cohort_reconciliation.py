"""Gate: the FINE cohort discrepancy (363 vs 365) must be fully accounted and the
three FINE cohorts explicitly frozen (no silent dual definition). Also guards that
FINE_COMPLETE_14 is non-empty (catches an always-null feature)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "research" / "true15m" / "FINE_COHORT_RECONCILIATION.json"


def _doc():
    assert ART.exists(), "run scripts/fine_cohort_reconciliation.py first"
    return json.loads(ART.read_text())


def test_fine_set_difference_accounted():
    d = _doc(); c = d["counts"]; sd = d["set_difference"]
    assert len(sd["factory_only_ids"]) == sd["factory_only_n"]
    assert len(sd["audit_only_ids"]) == sd["audit_only_n"]
    assert c["fine_factory_rows"] == c["fine_5m_audit"] + sd["factory_only_n"] - sd["audit_only_n"]


def test_fine_cohorts_frozen_and_nonempty():
    d = _doc()["cohorts"]
    for name in ("FINE_ROW", "FINE_COMPLETE_14", "FINE_5M_CORE"):
        assert d[name]["n"] > 0 and d[name]["hash"], f"{name} empty/unfrozen"
    assert d["FINE_COMPLETE_14"]["n"] <= d["FINE_ROW"]["n"]
