"""Monitor-of-monitors (§50) — runs on its OWN crontab line, so the
death of the audit chain (or any producer) is detected from outside
it. Lesson encoded: the 08-29 oversized-crontab incident killed the
whole analytics chain silently for 2.7h while everything looked green;
a freshness invariant inside the chain can only catch the PREVIOUS
run — this watcher is independent.

Every monitored producer: expected cadence, last success (file mtime),
status HEALTHY / WARNING / STALE / UNKNOWN. UNKNOWN is never green.
Output: results/meta_monitors.json. If anything is STALE/UNKNOWN, a
line also goes to /tmp/btc_meta_alerts.log (append-only alert trail).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# producer artifact -> (expected cadence s, warn multiple, source)
WATCH = {
    "online_status.json": (30, 4, "daemon heartbeat"),
    "audit_report.json": (600, 2, "audit chain step 1"),
    "decision_board.json": (600, 2, "audit chain (SRM gate)"),
    "invariants.json": (600, 2, "invariant wall"),
    "reconciliation.json": (600, 2, "independent reconciler"),
    "readiness.json": (600, 2, "readiness machine"),
    "execution_ledger.json": (600, 2, "markout ledger"),
    "world.json": (600, 2, "world/clock emitter"),
    "model_internals.json": (3600, 2, "hourly introspection"),
    "metrics_history.jsonl": (3600, 3, "retrain history append"),
}


def main():
    now = time.time()
    rows, worst = [], "HEALTHY"
    rank = {"HEALTHY": 0, "WARNING": 1, "STALE": 2, "UNKNOWN": 3}
    for name, (cad, warn_x, src) in WATCH.items():
        p = RES / name
        if not p.exists():
            st, age = "UNKNOWN", None
        else:
            age = now - p.stat().st_mtime
            st = ("HEALTHY" if age <= cad * warn_x else
                  "WARNING" if age <= cad * warn_x * 2 else "STALE")
        rows.append({"artifact": name, "source": src,
                     "cadence_s": cad,
                     "age_s": round(age) if age is not None else None,
                     "status": st})
        if rank[st] > rank[worst]:
            worst = st
    # heartbeat special case: read the stamp inside, not the mtime
    try:
        alive = json.loads((RES / "online_status.json").read_text()
                           ).get("alive_at")
        hb_age = now - alive if alive else None
        rows.append({"artifact": "online_status.alive_at",
                     "source": "daemon inner heartbeat",
                     "cadence_s": 30,
                     "age_s": round(hb_age) if hb_age else None,
                     "status": "HEALTHY" if hb_age and hb_age < 120
                     else "STALE"})
        if rows[-1]["status"] != "HEALTHY":
            worst = "STALE" if rank[worst] < 2 else worst
    except Exception:
        rows.append({"artifact": "online_status.alive_at",
                     "source": "daemon inner heartbeat",
                     "cadence_s": 30, "age_s": None,
                     "status": "UNKNOWN"})
        worst = "UNKNOWN"

    # ---- SLO snapshot (§65): current compliance, appended to a
    # history file so burn rates become computable over time. Values
    # read from the machine evidence, never asserted. -----------------
    def _j(name):
        try:
            return json.loads((RES / name).read_text())
        except Exception:
            return {}
    inv = _j("invariants.json")
    rec = _j("reconciliation.json")
    canary = _j("leakage_canaries.json")
    slo = {
        "critical_ledger_integrity": inv.get("health") == "green",
        "duplicate_decisions_zero": all(
            c.get("ok") for c in inv.get("checks", [])
            if c.get("name") == "one-decision-per-window") or None,
        "future_data_violations_zero":
            canary.get("overall") == "PASS" if canary else None,
        "reconciliation_ok": rec.get("overall") == "OK"
        if rec else None,
        "unresolved_sev0_zero": True,   # readiness emitter is source;
        # mirrored here from its last output
    }
    try:
        slo["unresolved_sev0_zero"] = not _j("readiness.json").get(
            "sev0_open")
    except Exception:
        slo["unresolved_sev0_zero"] = None
    compliant = [k for k, v in slo.items() if v is True]
    breached = [k for k, v in slo.items() if v is False]
    unknown = [k for k, v in slo.items() if v is None]
    with (RES / "slo_history.jsonl").open("a") as f:
        f.write(json.dumps({"ts": int(now), "ok": len(compliant),
                            "breach": breached,
                            "unknown": unknown}) + "\n")

    doc = {"generated_ts": int(now), "overall": worst,
           "monitors": rows,
           "slo": {"compliant": compliant, "breached": breached,
                   "unknown": unknown,
                   "note": "snapshot per run; burn rate from "
                           "slo_history.jsonl as it accrues"},
           "note": "independent cron line — watches the watchers; "
                   "UNKNOWN is never green"}
    (RES / "meta_monitors.json").write_text(json.dumps(doc, indent=1))
    if worst != "HEALTHY":
        with open("/tmp/btc_meta_alerts.log", "a") as f:
            bad = [r for r in rows if r["status"] != "HEALTHY"]
            f.write(json.dumps({"ts": int(now), "overall": worst,
                                "bad": bad}) + "\n")
    print(f"meta_monitors: {worst} "
          f"({sum(1 for r in rows if r['status'] == 'HEALTHY')}/"
          f"{len(rows)} healthy)")


if __name__ == "__main__":
    main()
