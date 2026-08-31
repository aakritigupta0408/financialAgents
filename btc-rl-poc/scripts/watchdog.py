"""Daemon watchdog — M6-R1 executor (RESTART_DEAD_DAEMON).

Runs every 5 min from cron. Two modes:

  * R1 enabled+certified in config/M6_REPAIRS.yaml -> full M6 state
    machine: DETECT -> CLASSIFY -> CONTAIN -> REPAIR -> RECORD, with
    attempt budget (max 2 per 30 min -> AUTO_REPAIR_EXHAUSTED ->
    FAILED_CLOSED) and incident + governance records. VERIFY/RESTORE
    are NOT done here: the independent-verification law says the
    repairing process may never certify itself — meta_monitor.py (a
    separate cron/process) verifies and appends the outcome.
  * otherwise -> legacy protective restart (the daemon is never left
    unguarded during certification), no M6 claims recorded.

Repair ledger: results/self_heal.jsonl (append-only state rows).
Maintenance override: touch results/maintenance.flag to suppress
repairs during intentional work.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# injectable configuration (tests replace these paths/patterns)
CFG = {
    "status": ROOT / "results" / "online_status.json",
    "heal_log": ROOT / "results" / "self_heal.jsonl",
    "wlog": ROOT / "results" / "watchdog_log.jsonl",
    "incidents": ROOT / "results" / "incidents.jsonl",
    "maintenance": ROOT / "results" / "maintenance.flag",
    "registry": ROOT / "config" / "M6_REPAIRS.yaml",
    "daemon_log": ROOT / "results" / "daemon.log",
    # end-anchored: shells that merely MENTION the module carry
    # trailing text (2026-08-21 incident)
    "pat": r"-m btc_rl\.online$",
    "spawn": [sys.executable, "-u", "-m", "btc_rl.online"],
    "stale_s": 300,
    "grace_s": 600,
    "max_attempts": 2,
    "attempt_window_s": 1800,
    "record_governance": True,
}
REPAIR_ID = "M6-R1_RESTART_DEAD_DAEMON"


def heal_rows(cfg):
    p = cfg["heal_log"]
    out = []
    if p.exists():
        for l in p.open():
            if l.strip():
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out


def record(cfg, state, **fields):
    row = {"ts": round(time.time(), 3), "repair_id": REPAIR_ID,
           "state": state, **fields}
    with cfg["heal_log"].open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def r1_enabled(cfg):
    try:
        import yaml
        reg = yaml.safe_load(cfg["registry"].read_text())
        for r in reg.get("class_a_allowlist") or []:
            if r.get("repair_id") == REPAIR_ID:
                return bool(r.get("enabled")) and bool(
                    r.get("certified"))
    except Exception:
        pass
    return False


def heartbeat_age(cfg):
    try:
        return time.time() - json.loads(
            cfg["status"].read_text())["alive_at"]
    except Exception:
        return None


def in_grace(cfg):
    """A fresh restart's cold first poll can exceed the SLO."""
    for r in reversed(heal_rows(cfg)[-10:]):
        if r.get("state") == "REPAIR_ATTEMPTED":
            return time.time() - r["ts"] < cfg["grace_s"]
    try:
        last = json.loads(
            cfg["wlog"].read_text().splitlines()[-1])
        if last.get("event") == "restarted" \
                and time.time() - last["ts"] < cfg["grace_s"]:
            return True
    except (OSError, IndexError, ValueError):
        pass
    return False


def attempts_in_window(cfg):
    cut = time.time() - cfg["attempt_window_s"]
    return sum(1 for r in heal_rows(cfg)
               if r.get("state") == "REPAIR_ATTEMPTED"
               and r.get("ts", 0) >= cut)


def proc_count(cfg):
    r = subprocess.run(["pgrep", "-f", "--", cfg["pat"]],
                       capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def do_restart(cfg):
    """INC 2026-08-30 (sleep-wake duplicate daemons): a fixed
    2s pkill grace raced a SIGTERM'd process still suspended from
    machine sleep — spawn happened while the old daemon lived, giving
    TWO writers. Regression: wait for actual death (up to ~14s),
    escalate to SIGKILL, and never spawn while any old process
    survives."""
    if proc_count(cfg) > 0:
        subprocess.run(["pkill", "-f", "--", cfg["pat"]], check=False)
        for i in range(7):
            time.sleep(2)
            if proc_count(cfg) == 0:
                break
            if i == 3:                      # escalate
                subprocess.run(["pkill", "-9", "-f", "--", cfg["pat"]],
                               check=False)
        if proc_count(cfg) > 0:
            return False                    # refuse to double-spawn
    with cfg["daemon_log"].open("a") as out:
        subprocess.Popen(cfg["spawn"], cwd=ROOT, stdout=out,
                         stderr=subprocess.STDOUT,
                         start_new_session=True)
    return True


def legacy_restart(cfg, age):
    do_restart(cfg)
    with cfg["wlog"].open("a") as f:
        f.write(json.dumps({"ts": int(time.time()),
                            "event": "restarted",
                            "stale_s": None if age is None
                            else round(age)}) + "\n")
    print(f"watchdog: restarted daemon (legacy path, stale "
          f"{'missing' if age is None else round(age)}s)")


def run_r1(cfg):
    """The certified state machine. Returns the terminal state of
    this invocation (VERIFY happens later, in meta_monitor)."""
    age = heartbeat_age(cfg)
    # DETECT
    if age is not None and age < cfg["stale_s"]:
        return "HEALTHY_NO_TRIGGER"
    if in_grace(cfg):
        return "GRACE_NO_TRIGGER"
    # CLASSIFY — intentional maintenance never triggers a repair
    if cfg["maintenance"].exists():
        return "MAINTENANCE_NO_TRIGGER"
    record(cfg, "DETECTED", heartbeat_age_s=None if age is None
           else round(age))
    # attempt budget (retry law)
    attempts = attempts_in_window(cfg)
    if attempts >= cfg["max_attempts"]:
        record(cfg, "AUTO_REPAIR_EXHAUSTED", attempts=attempts)
        record(cfg, "FAILED_CLOSED",
               reason="attempt budget exhausted — incident stays "
                      "open for humans")
        with cfg["incidents"].open("a") as f:
            f.write(json.dumps({
                "sev": 2, "title": "AUTO_REPAIR_EXHAUSTED: daemon "
                "restart budget spent, system failed closed",
                "opened": time.strftime("%Y-%m-%d %H:%M"),
                "status": "open — needs human",
                "detected_by": "watchdog/M6-R1",
                "repair_id": REPAIR_ID,
                "root_cause": "UNKNOWN"}) + "\n")
        return "FAILED_CLOSED"
    # CONTAIN — the record itself marks the Data plane stale for
    # every monitor that reads the heal ledger
    record(cfg, "CONTAINED",
           note="data plane marked stale; no new decisions trusted "
                "until verified")
    # preconditions: singleton semantics — count before repair
    pre_procs = proc_count(cfg)
    # REPAIR
    spawned = do_restart(cfg)
    if not spawned:
        record(cfg, "FAILED_CLOSED",
               reason="old process would not die — refused to "
                      "double-spawn (INC sleep-wake regression)")
        return "FAILED_CLOSED"
    attempt_no = attempts + 1
    record(cfg, "REPAIR_ATTEMPTED", attempt=attempt_no,
           procs_before=pre_procs,
           verification="pending — meta_monitor (independent) "
                        "certifies; this process may not")
    # incident (attach) + governance
    with cfg["incidents"].open("a") as f:
        f.write(json.dumps({
            "sev": 2, "title": f"daemon heartbeat stale "
            f"({'missing' if age is None else round(age)}s) — "
            f"M6-R1 auto-repair attempt {attempt_no}",
            "opened": time.strftime("%Y-%m-%d %H:%M"),
            "status": "auto-repair attempted — verification pending "
                      "(self_heal.jsonl)",
            "detected_by": "watchdog/M6-R1",
            "repair_id": REPAIR_ID,
            "root_cause": "UNKNOWN"}) + "\n")
    if cfg["record_governance"]:
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from log_system_change import log_change
            log_change("SELF_HEAL_OCCURRED", "daemon",
                       before="heartbeat stale",
                       after="restart attempted, verification pending",
                       reason="M6-R1 registered repair; scientific "
                              "state untouched by construction",
                       impact="none")
        except Exception:
            pass
    print(f"watchdog: M6-R1 repair attempt {attempt_no} recorded")
    return "REPAIR_ATTEMPTED"


def main(cfg=CFG):
    # R2 (derived-artifact rebuild) rides the same executor cron;
    # verification stays with meta_monitor (independent)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import repair_r2
        if repair_r2.r2_enabled(repair_r2.CFG):
            r2 = repair_r2.run_r2(repair_r2.CFG)
            acted = {k: v for k, v in r2.items()
                     if v != "HEALTHY_NO_TRIGGER"}
            if acted:
                print(f"watchdog: R2 {acted}")
    except Exception as e:
        print(f"watchdog: R2 pass error {e!r}")
    # R3 (delivery) runs LAST — dependency ordering: its canonical-
    # health gate refuses while compute/state planes are broken
    try:
        import repair_r3
        if repair_r3.r3_enabled(repair_r3.CFG):
            s3 = repair_r3.run_r3(repair_r3.CFG)
            if s3 != "HEALTHY_NO_TRIGGER":
                print(f"watchdog: R3 {s3}")
    except Exception as e:
        print(f"watchdog: R3 pass error {e!r}")
    if r1_enabled(cfg):
        return run_r1(cfg)
    # legacy protective path (certification period): unchanged
    age = heartbeat_age(cfg)
    if age is not None and age < cfg["stale_s"]:
        return "HEALTHY_NO_TRIGGER"
    if in_grace(cfg):
        return "GRACE_NO_TRIGGER"
    legacy_restart(cfg, age)
    return "LEGACY_RESTARTED"


if __name__ == "__main__":
    main()
