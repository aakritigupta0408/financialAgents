"""M6.1 chaos-certification matrix for R1 (RESTART_DEAD_DAEMON).

Twelve tests per the PM contract — every one must pass BEFORE
`enabled: true`. Runs against isolated temp dirs and a disposable
fake daemon process; the real daemon/ledgers are never touched.

Run: python3 tests/test_m6_r1.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import meta_monitor                                # noqa: E402
import watchdog as wd                              # noqa: E402

TOKEN = f"m6r1fake{uuid.uuid4().hex[:8]}"
FAKE = [sys.executable, "-c",
        f"import time  # {TOKEN}\ntime.sleep(600)"]
PAT = TOKEN


def mkcfg(td):
    res = Path(td)
    cfg = dict(wd.CFG)
    cfg.update({
        "status": res / "online_status.json",
        "heal_log": res / "self_heal.jsonl",
        "wlog": res / "watchdog_log.jsonl",
        "incidents": res / "incidents.jsonl",
        "maintenance": res / "maintenance.flag",
        "registry": res / "M6_REPAIRS.yaml",
        "daemon_log": res / "daemon.log",
        "pat": PAT, "spawn": FAKE,
        "record_governance": False,
    })
    cfg["registry"].write_text(
        "class_a_allowlist:\n"
        "  - {repair_id: M6-R1_RESTART_DEAD_DAEMON,\n"
        "     enabled: true, certified: true}\n")
    return cfg


def write_hb(cfg, age_s):
    cfg["status"].write_text(json.dumps(
        {"alive_at": time.time() - age_s}))


def heal_states(cfg):
    if not cfg["heal_log"].exists():
        return []
    return [json.loads(l)["state"]
            for l in cfg["heal_log"].read_text().splitlines()]


def kill_fakes():
    subprocess.run(["pkill", "-f", "--", PAT], check=False)
    time.sleep(0.5)


def run():
    passed = 0
    try:
        # --- 1. stale heartbeat detected -> repair attempted
        with tempfile.TemporaryDirectory() as td:
            cfg = mkcfg(td)
            write_hb(cfg, 900)
            assert wd.main(cfg) == "REPAIR_ATTEMPTED"
            st = heal_states(cfg)
            assert "DETECTED" in st and "REPAIR_ATTEMPTED" in st
            passed += 1
            print("  1 stale detected -> repair")

            # --- 4/6/7 on the same live fixture:
            time.sleep(1)
            assert wd.proc_count(cfg) == 1, "restart must spawn"
            passed += 1
            print("  6 restart succeeds (process alive)")
            # 7: immediate re-run is in grace -> no duplicate
            write_hb(cfg, 900)
            assert wd.main(cfg) == "GRACE_NO_TRIGGER"
            assert wd.proc_count(cfg) == 1, "no duplicate process"
            passed += 1
            print("  7 no duplicate on re-run (grace)")
            # 4: singleton precondition recorded
            att = [json.loads(l) for l in
                   cfg["heal_log"].read_text().splitlines()
                   if json.loads(l)["state"] == "REPAIR_ATTEMPTED"]
            assert "procs_before" in att[0]
            passed += 1
            print("  4 singleton precondition checked/recorded")
            # 5: containment precedes repair
            assert st.index("CONTAINED") < st.index("REPAIR_ATTEMPTED")
            passed += 1
            print("  5 containment precedes repair")
            # 10: incident written
            assert cfg["incidents"].exists() and "M6-R1" in \
                cfg["incidents"].read_text()
            passed += 1
            print("  10 incident recorded (governance row covered by "
                  "production path; disabled in fixture)")
            # --- 8/9: independent verifier PASS
            write_hb(cfg, 10)                    # daemon "recovered"
            (Path(td) / "audit_report.json").write_text("{}")
            (Path(td) / "a3_live.json").write_text(
                json.dumps({"spec_hash_ok": True}))
            (Path(td) / "invariants.json").write_text(
                json.dumps({"failed": 0}))
            out = meta_monitor.verify_repairs(
                res_dir=td, min_age_s=0, pat=PAT)
            assert out and out[0]["state"] == "VERIFICATION_PASS"
            assert out[0]["singleton"] and out[0]["heartbeat_fresh"]
            assert out[0]["scientific_unchanged"] is True
            assert "RESTORED" in heal_states(cfg)
            passed += 1
            print("  8 independent verifier passes -> RESTORED")
            assert out[0]["audit_progress"] is True
            passed += 1
            print("  9 audit progress observed")
            kill_fakes()

        # --- 2. healthy daemon: no trigger
        with tempfile.TemporaryDirectory() as td:
            cfg = mkcfg(td)
            write_hb(cfg, 30)
            assert wd.main(cfg) == "HEALTHY_NO_TRIGGER"
            assert not cfg["heal_log"].exists()
            passed += 1
            print("  2 healthy -> no trigger")

        # --- 3. maintenance flag: no trigger
        with tempfile.TemporaryDirectory() as td:
            cfg = mkcfg(td)
            write_hb(cfg, 900)
            cfg["maintenance"].touch()
            assert wd.main(cfg) == "MAINTENANCE_NO_TRIGGER"
            assert not cfg["heal_log"].exists()
            passed += 1
            print("  3 maintenance -> no trigger")

        # --- 11. failed verification stays FAILED_CLOSED
        with tempfile.TemporaryDirectory() as td:
            cfg = mkcfg(td)
            write_hb(cfg, 900)
            wd.main(cfg)
            kill_fakes()                 # daemon dies again pre-verify
            write_hb(cfg, 900)           # heartbeat still stale
            out = meta_monitor.verify_repairs(
                res_dir=td, min_age_s=0, pat=PAT)
            assert out[0]["state"] == "VERIFICATION_FAIL"
            assert heal_states(cfg)[-1] == "FAILED_CLOSED"
            passed += 1
            print("  11 failed verification -> FAILED_CLOSED")

        # --- 12. retry exhaustion: no third attempt, incident open
        with tempfile.TemporaryDirectory() as td:
            cfg = mkcfg(td)
            now = time.time()
            with cfg["heal_log"].open("a") as f:
                for dt in (600, 300):
                    f.write(json.dumps(
                        {"ts": now - dt,
                         "repair_id": wd.REPAIR_ID,
                         "state": "REPAIR_ATTEMPTED"}) + "\n")
            cfg["grace_s"] = 0           # not grace — budget must act
            write_hb(cfg, 900)
            assert wd.main(cfg) == "FAILED_CLOSED"
            st = heal_states(cfg)
            assert "AUTO_REPAIR_EXHAUSTED" in st
            assert st.count("REPAIR_ATTEMPTED") == 2, "no 3rd attempt"
            assert wd.proc_count(cfg) == 0, "no spawn on exhaustion"
            inc = cfg["incidents"].read_text()
            assert "AUTO_REPAIR_EXHAUSTED" in inc and "open" in inc
            passed += 1
            print("  12 retry exhaustion -> FAILED_CLOSED, "
                  "incident open, no third attempt")
    finally:
        kill_fakes()
    print(f"M6-R1 chaos matrix: {passed}/12 pass")
    assert passed == 12


if __name__ == "__main__":
    run()
