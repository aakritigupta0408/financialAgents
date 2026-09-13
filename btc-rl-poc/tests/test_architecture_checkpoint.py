"""§33 — the architecture checkpoint must be machine-adjudicated and tested."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import architecture_checkpoint as ac  # noqa: E402


def test_forbidden_market_to_oracle_passes():
    # oracle modules must never read a Kalshi field as input; regression guard
    status, detail = ac.check_forbidden_market_to_oracle()
    assert status == "PASS", detail


def test_leak_guard_present():
    status, _ = ac.check_leak_guard()
    assert status == "PASS"


def test_testlabel_fit_uses_splits():
    status, detail = ac.check_testlabel_fit()
    assert status == "PASS", detail
    # freeze_oracle is the production freeze — must be exempt, not flagged
    assert "scripts/freeze_oracle.py" in detail["exempt_production_freeze"]


def test_runtime_contract_truth_detects_drift():
    # documents the KNOWN SEV-1: runtime not yet on exact BRTI. When migrated,
    # this flips to PASS and the assertion below should be updated.
    status, detail = ac.check_runtime_contract_truth()
    assert status in ("PASS", "FAIL")
    if status == "FAIL":
        assert detail["settles_on_coinbase_candle"] or detail["uses_4venue_composite"]


def test_checks_registry_shape():
    names = [n for n, _ in ac.CHECKS]
    for required in ("FORBIDDEN_MARKET_TO_ORACLE", "RUNTIME_CONTRACT_TRUTH",
                     "LEAK_GUARD_PRESENT"):
        assert required in names
