"""Gate: the COARSE cohort discrepancy (6,189 vs 6,200) must be fully accounted
and the three cohorts explicitly frozen — no silent dual definition of
COARSE_BTC_STATE. Run scripts/coarse_cohort_reconciliation.py to (re)generate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "research" / "true15m" / "COARSE_COHORT_RECONCILIATION.json"


def _doc():
    assert ART.exists(), "run scripts/coarse_cohort_reconciliation.py first"
    return json.loads(ART.read_text())


def test_set_difference_fully_accounted():
    d = _doc()
    c = d["counts"]
    sd = d["set_difference"]
    # every window in exactly one side of the diff is enumerated (no unexplained gap)
    assert len(sd["factory_minus_p11_ids"]) == sd["factory_minus_p11_n"]
    assert len(sd["p11_minus_factory_ids"]) == sd["p11_minus_factory_n"]
    # factory rows == P1.1-core + factory-only  (the diff closes exactly)
    assert c["coarse_factory_rows"] == c["p11_2h_contiguous"] + sd["factory_minus_p11_n"] \
        - sd["p11_minus_factory_n"]


def test_three_cohorts_frozen_and_nested():
    d = _doc()["cohorts"]
    for name in ("COARSE_ROW", "COARSE_COMPLETE_20", "COARSE_2H_CORE"):
        assert d[name]["n"] > 0 and d[name]["hash"], f"{name} not frozen"
    # complete-20 is the strictest, row is the loosest
    assert d["COARSE_COMPLETE_20"]["n"] <= d["COARSE_ROW"]["n"]
    assert d["COARSE_2H_CORE"]["n"] <= d["COARSE_ROW"]["n"]
