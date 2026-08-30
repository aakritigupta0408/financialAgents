"""Append a semantic system-change event (PM 08-30) — the backend's
"what changed and why?" feed. Not git history: governance history.

    python3 scripts/log_system_change.py COMPONENT_RETIRED pt7 \
        --before ACTIVE --after RETIRED_BENCHMARK \
        --reason "mechanism falsified" --impact none

Types: COMPONENT_RETIRED MODEL_PROMOTED MODEL_REVERTED
EXPERIMENT_REGISTERED EXPERIMENT_CLOSED SCHEMA_CHANGED
INCIDENT_OPENED INCIDENT_RESOLVED SELF_HEAL_OCCURRED
REGISTRY_CORRECTED CLEANUP_EXECUTED
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "system_change_log.jsonl"


def log_change(change_type, entity, before=None, after=None,
               reason=None, impact=None, evidence_preserved=True):
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    row = {"ts": int(time.time()), "change_type": change_type,
           "entity": entity, "before": before, "after": after,
           "reason": reason, "decision_impact": impact,
           "evidence_preserved": evidence_preserved, "git_sha": git}
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("change_type")
    ap.add_argument("entity")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--reason")
    ap.add_argument("--impact")
    print(json.dumps(log_change(**vars(ap.parse_args()))))
