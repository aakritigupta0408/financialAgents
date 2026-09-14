"""Gates on the TRUE15M offline program: dataset integrity PASS, split frozen with
hashes, sealed test opened exactly once on a pre-frozen finalist, and a verdict
emitted. Guards the sealed-test discipline. Run scripts/true15m_offline_program.py."""
import json
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "research" / "true15m"


def _j(name):
    p = T / name
    assert p.exists(), f"missing {name} — run scripts/true15m_offline_program.py"
    return json.loads(p.read_text())


def test_integrity_all_pass():
    assert _j("DATASET_INTEGRITY_REPORT_V1.json")["all_pass"] is True


def test_split_frozen_with_hashes_and_no_overlap():
    s = _j("SPLIT_SPEC_V1.json")
    for part in ("train", "validation", "sealed_test"):
        assert s[part]["n"] > 0 and s[part]["hash"]
    # chronological, non-overlapping ranges
    assert s["train"]["range"][1] <= s["validation"]["range"][0]
    assert s["validation"]["range"][1] <= s["sealed_test"]["range"][0]


def test_sealed_test_opened_once_and_verdict_present():
    st = _j("SEALED_TEST_REPORT_V1.json")
    assert st["opened_once"] is True
    v = _j("OFFLINE_FINAL_VERDICT_V1.json")
    assert v["verdict"] in ("OFFLINE_QUALIFIED", "NO_OFFLINE_QUALIFIED_MODEL")
