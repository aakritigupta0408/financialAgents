"""One-shot: log INC-dup-daemons, bump recovery to R3 (standard
1.0.5). Idempotent."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

inc_p = ROOT / "results" / "incidents.jsonl"
if "INC-2026-08-29-dup-daemons" not in inc_p.read_text():
    inc = {"ts": int(time.time()), "id": "INC-2026-08-29-dup-daemons",
           "sev": "SEV-2",
           "title": "3 concurrent daemon processes from failed-pkill "
                    "manual restarts",
           "status": "closed",
           "detected_by": "chaos drill kill-daemon-mid-poll "
                          "(pre_kill_pids)",
           "blast_radius": "none measurable — atomic full-file "
           "rewrites made writers last-wins; invariants/"
           "reconciliation clean throughout",
           "rca": "an early manual restart used BSD-incompatible "
           "pkill flags, failed silently, then spawned anew; "
           "later restarts compounded",
           "fix": "chaos drill pkill cleaned all; meta-monitor now "
           "checks daemon process count == 1 every 5 min",
           "regression": "meta_monitors 'daemon process count' row"}
    with inc_p.open("a") as f:
        f.write(json.dumps(inc) + "\n")
    print("incident logged")

p = ROOT / "REAL_MONEY_EQUIVALENT_STANDARD.yaml"
s = p.read_text()
if "version: 1.0.4" in s:
    s = s.replace("version: 1.0.4", "version: 1.0.5\n"
                  "# 1.0.5 (2026-08-29): recovery R2->R3 — chaos "
                  "suite 5/5 PASS (live daemon\n# kill/recovery, "
                  "zero dupes, reconcile OK; atomicity/malformed/"
                  "staleness\n# drills); INC-dup-daemons found+closed;"
                  " daemon-singleton monitor added.")
    s = s.replace("""  recovery:
    assessed: R2
    evidence: watchdog restart + append-only ledgers + atomic writes survived real crashes
    blocker_to_next: "R3 gap: no chaos suite (§56), no scripted crash-recovery drills (§57)\"""",
                  """  recovery:
    assessed: R3
    evidence: chaos suite 5/5 (tests/chaos_drills.py) — live kill/recovery with zero dupes + reconcile OK; atomicity/malformed/staleness drills; daemon-singleton monitor
    blocker_to_next: "R4 gap: scheduled rollback drills (§62); FREEZE-branch live test needs a maintenance window (deferred honestly)\"""")
    p.write_text(s)
    print("standard 1.0.5")
