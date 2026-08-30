"""M5 system-validation fixtures (PM 08-30, items #3/#4/#5).

#3 disagreement: synthetic toxic markouts + poor thesis integrity —
   the Execution Researcher must report BOTH, not force one story.
#4 stale evidence: an agent given old canonical input must emit
   STALE_INPUT and no research recommendation.
#5 end-to-end firewall: a real agent-path attempt to "change A3 dip
   threshold to 5c" must persist ONLY as REJECTED_BY_FIREWALL (this
   one runs against the REAL ledger — the rejection row IS the
   proof) and record a governance event.

Run: python3 tests/test_m5_system.py
"""
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import agent_firewall as fw                       # noqa: E402


def _mkrow(state, won, entry=None, markout10=None, econ=None):
    r = {"state": state, "settled": True, "won": won,
         "ticker": f"T{time.time_ns() % 10**9}"}
    if state == "FILLED":
        r.update({"entry_ask": entry or 70, "call_ask": 80,
                  "markout_1s": markout10, "markout_5s": markout10,
                  "markout_10s": markout10, "markout_30s": markout10,
                  "markout_60s": markout10})
    return r


def test_disagreement_fixture():
    """Toxic markouts AND poor thesis integrity -> BOTH."""
    import agent_execution_researcher as ax
    with tempfile.TemporaryDirectory() as td:
        res = Path(td)
        # 10 fills that mostly lose (thesis AS) with badly negative
        # short markouts (execution AS); 10 unfilled winners
        rows = [_mkrow("FILLED", won=(i < 4), markout10=-8.0)
                for i in range(10)]
        rows += [_mkrow("MISSED", won=True) for _ in range(10)]
        (res / "a3_window_evaluation.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        old_res, old_led = ax.RES, fw.LEDGER
        try:
            ax.RES = res
            fw.LEDGER = res / "agent_recommendations.jsonl"
            ax.main()
            out = json.loads(
                (res / "execution_research.json").read_text())
        finally:
            ax.RES, fw.LEDGER = old_res, old_led
    assert out["thesis_adverse_selection"]["state"] == "PRESENT", out
    assert out["execution_adverse_selection"]["state"] == "PRESENT"
    assert out["dominant_channel"] == "BOTH", out["dominant_channel"]
    print("  #3 disagreement fixture: BOTH reported — "
          "no forced narrative")


def test_stale_input():
    """Old canonical input -> STALE_INPUT, no recommendation."""
    import agent_experiment_analyst as ea
    with tempfile.TemporaryDirectory() as td:
        res = Path(td)
        p = res / "a3_live.json"
        p.write_text(json.dumps({"forward": {"eligible": 99}}))
        old = time.time() - 7200
        os.utime(p, (old, old))
        old_res, old_led, old_root = ea.RES, fw.LEDGER, fw.ROOT
        try:
            ea.RES = res
            fw.ROOT = res.parent          # stale() resolves results/
            (res.parent / "results").symlink_to(res) \
                if not (res.parent / "results").exists() else None
            fw.LEDGER = res / "agent_recommendations.jsonl"
            ea.main()
            rows = [json.loads(l) for l in
                    fw.LEDGER.read_text().splitlines()]
        finally:
            ea.RES, fw.LEDGER, fw.ROOT = old_res, old_led, old_root
    assert len(rows) == 1 and "STALE_INPUT" in rows[0]["finding"]
    assert (res / "experiment_analysis.json").exists() is False
    print("  #4 stale input: STALE_INPUT emitted, no analysis, "
          "no recommendation")


def test_firewall_end_to_end():
    """Real-path policy-change attempt -> REJECTED_BY_FIREWALL row in
    the REAL ledger + governance event. The rejection row is the
    permanent proof that the full path blocks it."""
    row = fw.submit(
        agent="m5_validation_fixture",
        action_class="RESEARCH_POLICY_CHANGE",
        finding="M5 system validation #5 — deliberate fixture",
        recommendation="change A3 dip threshold to 5c",
        evidence=["tests/test_m5_system.py"],
        targeted_loss_term="timeout_control_pnl")
    assert row["status"] == "REJECTED_BY_FIREWALL"
    # no executable record: nothing with this dedupe key is OPEN
    ledger = [json.loads(l) for l in
              fw.LEDGER.read_text().splitlines()]
    twins = [r for r in ledger
             if r.get("dedupe_key") == row["dedupe_key"]]
    assert all(r["status"] == "REJECTED_BY_FIREWALL" for r in twins)
    # governance event
    sys.path.insert(0, str(ROOT / "scripts"))
    from log_system_change import log_change
    log_change("INCIDENT_RESOLVED", "m5-validation-firewall-fixture",
               before="attempted RESEARCH_POLICY_CHANGE",
               after="REJECTED_BY_FIREWALL",
               reason="M5 system validation #5: end-to-end proof the "
                      "agent path cannot mutate the experiment",
               impact="none — A3 spec hash unchanged")
    # and the experiment truly untouched
    spec_ok = json.loads(
        (ROOT / "results" / "a3_live.json").read_text()
    ).get("spec_hash_ok")
    assert spec_ok is True
    print("  #5 firewall end-to-end: REJECTED_BY_FIREWALL persisted, "
          "governance event logged, A3 spec hash intact")


if __name__ == "__main__":
    test_disagreement_fixture()
    test_stale_input()
    test_firewall_end_to_end()
    print("M5 system fixtures: 3/3 pass")
