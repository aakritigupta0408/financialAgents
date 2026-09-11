"""Emit results/monitor_health.json — monitor-of-monitors (M6.0 §9).

The system may not claim green because a monitor itself stopped
running. One row per critical monitor: expected cadence, last run,
freshness SLO, state HEALTHY/DEGRADED/STALE/UNKNOWN, owner.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# monitor_id: (artifact, expected_cadence_s, slo_s, owner)
MONITORS = {
    "audit_chain": ("invariants.json", 600, 1800, "cron/auditor"),
    "meta_monitor": ("meta_monitors.json", 300, 900, "observability"),
    "daemon_heartbeat": ("online_status.json", 30, 300, "daemon"),
    "event_capture": ("event_capture.json", 10, 300, "capture"),
    "xvenue_capture": ("xvenue_capture.json", 10, 300, "capture"),
    "micro_capture": ("micro_capture.json", 10, 300, "capture"),
    "parity_check": ("parity.json", 600, 1800, "model_researcher"),
    "data_health": ("data_health.json", 600, 1800,
                    "data_reliability"),
    "pm_snapshot": ("pm_snapshot.json", 600, 1800,
                    "research_manager"),
    "m5_soak": ("m5_soak.json", 600, 1800, "governance"),
    "f1_gate": ("f1_capture_qualification.json", 600, 86400,
                "data_reliability"),
}


def main():
    now = time.time()
    rows = {}
    worst = "HEALTHY"
    order = {"HEALTHY": 0, "DEGRADED": 1, "STALE": 2, "UNKNOWN": 1}
    for mid, (art, cad, slo, owner) in MONITORS.items():
        p = RES / art
        if not p.exists():
            st, age = "UNKNOWN", None
        else:
            age = now - p.stat().st_mtime
            st = ("HEALTHY" if age <= cad * 2 + 60 else
                  "DEGRADED" if age <= slo else "STALE")
        rows[mid] = {"artifact": art, "expected_cadence_s": cad,
                     "freshness_slo_s": slo,
                     "age_s": round(age) if age is not None else None,
                     "state": st, "owner": owner}
        if order.get(st, 1) > order.get(worst, 0):
            worst = st
    doc = {"generated_ts": int(now), "overall": worst,
           "monitors": rows,
           "law": "UNKNOWN is never green; a stopped monitor cannot "
                  "certify the thing it watches"}
    (RES / "monitor_health.json").write_text(json.dumps(doc, indent=1))
    bad = [m for m, r in rows.items() if r["state"] != "HEALTHY"]
    print(f"monitor_health: {worst}"
          + (f" · attention: {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
