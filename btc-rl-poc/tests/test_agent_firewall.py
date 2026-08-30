"""Firewall enforcement tests (M5) — the guarantees are code, so
they get code tests. Run: python3 tests/test_agent_firewall.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "scripts"))
import agent_firewall as fw


def run():
    with tempfile.TemporaryDirectory() as td:
        fw.LEDGER = Path(td) / "agent_recommendations.jsonl"
        ev = ["a3_live.json"]
        # 1. forbidden classes persist only as REJECTED_BY_FIREWALL
        for cls in ("RESEARCH_POLICY_CHANGE", "RISK_CHANGE"):
            r = fw.submit("test", cls, "f", f"change something {cls}",
                          ev, targeted_loss_term="x")
            assert r["status"] == "REJECTED_BY_FIREWALL", cls
        # 2. SAFE_OPS_REPAIR: blocked without a registered+enabled+
        # certified repair_id (M6 ALLOW_IF_REGISTERED_REPAIR
        # semantics; all repairs currently enabled:false)
        r = fw.submit("test", "SAFE_OPS_REPAIR", "f", "restart thing",
                      ev)
        assert r["status"] == "BLOCKED_REPAIR_NOT_ENABLED"
        r = fw.submit("test", "SAFE_OPS_REPAIR", "f", "restart d",
                      ev, repair_id="M6-R1_RESTART_DEAD_DAEMON")
        assert r["status"] == "BLOCKED_REPAIR_NOT_ENABLED", \
            "uncertified repair must stay blocked"
        # 3. PROPOSE without loss term refused, nothing persisted
        n_before = len(fw.LEDGER.read_text().splitlines())
        try:
            fw.submit("test", "PROPOSE", "f", "vague idea", ev)
            raise AssertionError("vague PROPOSE was accepted")
        except fw.FirewallRefusal:
            pass
        assert len(fw.LEDGER.read_text().splitlines()) == n_before
        # 4. PROPOSE with loss term is OPEN
        r = fw.submit("test", "PROPOSE", "f", "smaller threshold",
                      ev, targeted_loss_term="timeout_control_pnl")
        assert r["status"] == "OPEN"
        # 5. identical resubmission collapses to DUPLICATE
        r2 = fw.submit("test", "PROPOSE", "f", "smaller threshold",
                       ev, targeted_loss_term="timeout_control_pnl")
        assert r2["status"] == "DUPLICATE"
        # 6. no-evidence submission refused
        try:
            fw.submit("test", "OBSERVE", "f", "trust me", [])
            raise AssertionError("evidence-free row accepted")
        except fw.FirewallRefusal:
            pass
        # 7. unknown class refused
        try:
            fw.submit("test", "EXECUTE_TRADE", "f", "yolo", ev)
            raise AssertionError("unknown class accepted")
        except fw.FirewallRefusal:
            pass
        rows = [json.loads(l) for l in
                fw.LEDGER.read_text().splitlines()]
        assert all(r["status"] != "OPEN" or r["action_class"]
                   in ("OBSERVE", "DIAGNOSE", "PROPOSE")
                   for r in rows)
    print("firewall tests: 7/7 pass")


if __name__ == "__main__":
    run()
