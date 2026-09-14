"""Guard tests for the TEST_V2 research firewall (NO_TEST_V2_DEV_ACCESS) and the
APPEND_ONLY membership state. These protect the one-time blind test from accidental
code reuse and from silent membership regressions.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.test_v2_firewall import (assert_no_test_v2_access, membership_ids,  # noqa: E402
                                     cutoff, guard_rows)

T = ROOT / "research" / "true15m"
DEV = T / "TRUE15M_DATASET_V1.jsonl"


def test_dev_dataset_passes_firewall():
    """The frozen A_CORE dev dataset must contain no TEST_V2 window."""
    rows = [json.loads(l) for l in DEV.open() if l.strip()]
    assert guard_rows(rows, "unit-test") is rows          # returns unchanged, no raise


def test_dev_dataset_all_pre_cutoff():
    rows = [json.loads(l) for l in DEV.open() if l.strip()]
    cut = cutoff()
    assert all(r["T0"] <= cut for r in rows), "a dev row has T0 > cutoff (would be TEST_V2)"


def test_membership_id_intersection_raises():
    """A development set that includes a real TEST_V2 id must be rejected."""
    v2 = membership_ids()
    if not v2:
        pytest.skip("no TEST_V2 membership yet")
    poison = next(iter(v2))
    with pytest.raises(AssertionError, match="NO_TEST_V2_DEV_ACCESS"):
        assert_no_test_v2_access(dev_ids=["A", "B", poison], context="unit-test")


def test_post_cutoff_t0_raises():
    """A development row timestamped after the cutoff must be rejected on T0 alone."""
    with pytest.raises(AssertionError, match="NO_TEST_V2_DEV_ACCESS"):
        assert_no_test_v2_access(dev_t0s=[cutoff() + 900], context="unit-test")


def test_clean_dev_ids_pass():
    assert assert_no_test_v2_access(dev_ids=["not-a-real-window"],
                                    dev_t0s=[cutoff() - 900], context="unit-test") is True


def test_membership_state_append_only_and_sealed():
    state = json.loads((T / "TEST_V2_MEMBERSHIP_STATE.json").read_text())
    assert state["mode"] in ("APPEND_ONLY", "CLOSED")
    assert state["append_only_invariant_holds"] is True
    assert state["vanished_windows"] == []
    # final hash is UNSET while accumulating, and frozen (non-UNSET) only once closed
    if state["current_N"] < state["target_N"]:
        assert state["mode"] == "APPEND_ONLY"
        assert state["final_membership_hash"] == "UNSET"
    else:
        assert state["mode"] == "CLOSED"
        assert state["final_membership_hash"] not in (None, "UNSET")
