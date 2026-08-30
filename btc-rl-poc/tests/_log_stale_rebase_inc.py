"""One-shot: log INC-stale-rebase (idempotent)."""
import json
import time
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "results" / \
    "incidents.jsonl"
if "INC-2026-08-29-stale-rebase" not in p.read_text():
    inc = {"ts": int(time.time()),
           "id": "INC-2026-08-29-stale-rebase", "sev": "SEV-2",
           "title": "site main-repo sync wedged by crashed rebase; "
                    "domain served stale front end for hours",
           "status": "closed",
           "detected_by": "owner report ('nothing changed on the "
                          "front end')",
           "blast_radius": "front-end staleness only; gh-pages fast "
           "path and all trading unaffected",
           "rca": "publisher rebase --autostash crashed mid-rebase "
           "leaving .git/rebase-merge; every hourly retry failed "
           "with 'already a rebase' under check=False silence",
           "fix": "stale rebase aborted; sync green; live "
           "readiness.json verified 4min fresh on the domain",
           "regression": "meta_monitor 'site main-repo sync' row — "
           "consecutive-failure counter, STALE at >3"}
    with p.open("a") as f:
        f.write(json.dumps(inc) + "\n")
    print("incident logged")
else:
    print("already logged")
