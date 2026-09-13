"""DT-01 §6 golden test + §7 shadow parity. Requires live BRTI access; skips
cleanly when unavailable (offline/no creds) so it never blocks the suite."""
import json
import calendar
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import contract_truth as ct  # noqa: E402

CO = ROOT / "results" / "contract_outcomes.jsonl"


def _epoch(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


def _brti_available():
    try:
        import data.adapters.brti as b
        return b.probe().get("state") == "AVAILABLE"
    except Exception:
        return False


def _recent_windows(n=5):
    rows = [json.loads(l) for l in CO.open() if l.strip()]
    return rows[-n:]


pytestmark = pytest.mark.skipif(not _brti_available(),
                                reason="BRTI live access unavailable")


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("EXACT_BRTI_RUNTIME_ENABLED", raising=False)
    assert ct.runtime_enabled() is False


def test_golden_end_to_end_matches_official():
    """BRTI raw -> target -> settlement -> outcome must equal the official result."""
    checked = 0
    for w in _recent_windows(6):
        r = ct.settle(_epoch(w["open_time"]), _epoch(w["close_time"]))
        if r["contract_truth_quality"] != ct.EXACT_BRTI:
            continue
        checked += 1
        assert r["outcome"] == w["exact_yes"], (w["ticker"], r, w)
        assert abs(r["target"] - w["floor_strike"]) < 5.0
        assert abs(r["settlement_value"] - w["expiration_value"]) < 5.0
    assert checked >= 1, "no window reconstructed at EXACT_BRTI quality"


def test_shadow_parity_exact_beats_or_ties_official():
    """§7 — record legacy(proxy) vs exact vs official; exact must match official."""
    disagreements = []
    for w in _recent_windows(6):
        r = ct.settle(_epoch(w["open_time"]), _epoch(w["close_time"]))
        if r["contract_truth_quality"] != ct.EXACT_BRTI:
            continue
        if r["outcome"] != w["exact_yes"]:
            disagreements.append((w["ticker"], r["outcome"], w["exact_yes"]))
    assert not disagreements, f"exact-BRTI settlement disagreed with official: {disagreements}"


def test_failover_is_explicit_never_silent():
    # a window with an implausible future timestamp cannot be settled from BRTI
    future = int(time.time()) + 3600
    r = ct.settle(future, future + 900)
    assert r["outcome"] is None
    assert r["contract_truth_quality"] in (ct.PROXY_DEGRADED, ct.UNAVAILABLE)


def test_health_shape():
    h = ct.health()
    for k in ("connected", "expected_cadence_hz", "rest_ok", "history_ok",
              "runtime_enabled", "capture_enabled"):
        assert k in h
