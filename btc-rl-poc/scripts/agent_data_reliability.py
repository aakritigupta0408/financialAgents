"""M5.3 — Data Reliability (script-backed structured agent).

Mission: feed health, PIT snapshot completeness, timestamp ordering,
missingness, schema drift, parity prerequisites. Emits the canonical
results/data_health.json and reports through the decision firewall.

Authority: DIAGNOSE + PROPOSE only. When it detects a condition with
a deterministic safe repair (stale daemon, stale capture) it submits
a SAFE_OPS_REPAIR — which the firewall correctly lands as
BLOCKED_UNTIL_M6: the diagnosis is recorded, the repair waits for the
whitelist. It cannot touch research policy by construction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import agent_firewall as fw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# feed freshness SLOs (seconds): green < g, amber < a, else red
SLO = {"daemon_heartbeat": (120, 600),
       "event_capture": (120, 600),
       "kb_inference": (180, 900),
       "pit_snapshot": (300, 1200)}
PIT_V = "pit-v1"
PIT_FIELDS = {"prediction_id", "window_id", "variant", "event_ts",
              "receive_ts", "decision_ts", "persist_ts", "close_ts",
              "model", "feature_schema_version", "features",
              "feature_hash", "prediction", "confidence"}
KB_FIELDS = {"ticker", "made_ts", "close_ts", "strike", "variant",
             "p_up", "call", "mins_left"}


def j(name):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return None


def jl(name, tail=None):
    p = RES / name
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    if tail:
        lines = lines[-tail:]
    out = []
    for l in lines:
        if l.strip():
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def grade(age, key):
    if age is None:
        return "UNKNOWN"
    g, a = SLO[key]
    return "HEALTHY" if age < g else "WATCH" if age < a else "CRITICAL"


def main():
    now = time.time()
    onl = j("online_status.json") or {}
    cap = j("event_capture.json") or {}
    kb = jl("kalshi_binary_log.jsonl", tail=3000)
    snaps = jl("feature_snapshots.jsonl")
    par = j("parity.json") or {}

    hb_age = now - onl["alive_at"] if onl.get("alive_at") else None
    cap_age = now - cap["alive_at"] if cap.get("alive_at") else None
    kb_age = now - max((r.get("made_ts") or 0) for r in kb) \
        if kb else None
    snap_age = now - max((r.get("persist_ts") or 0) for r in snaps) \
        if snaps else None
    feeds = {
        "daemon_heartbeat": {"age_s": round(hb_age or -1),
                             "state": grade(hb_age,
                                            "daemon_heartbeat")},
        "event_capture": {"age_s": round(cap_age or -1),
                          "state": grade(cap_age, "event_capture"),
                          "errors": cap.get("errors")},
        "kb_inference": {"age_s": round(kb_age or -1),
                         "state": grade(kb_age, "kb_inference")},
        "pit_snapshot": {"age_s": round(snap_age or -1),
                         "state": grade(snap_age, "pit_snapshot")},
    }

    # PIT completeness: every kb2/kb9 inference row since go-live
    # should have a snapshot with the same (variant, ticker, slot)
    snap_ids = {r["prediction_id"] for r in snaps}
    pit_start = min((r["persist_ts"] for r in snaps), default=now)
    expected = missing = 0
    for r in kb:
        v = r.get("variant")
        if v in ("kb2", "kb9") and (r.get("made_ts") or 0) \
                >= pit_start:
            expected += 1
            if f"{v}-{r['ticker']}-{r['made_ts']}" not in snap_ids:
                missing += 1
    pit = {"snapshots": len(snaps),
           "variants": sorted({r["variant"] for r in snaps}),
           "expected_since_golive": expected,
           "missing": missing,
           "completeness": round(1 - missing / expected, 4)
           if expected else None,
           "state": "HEALTHY" if expected and missing == 0
           else "WATCH" if expected else "UNKNOWN"}

    # timestamp ordering on recent rows (invariant-grade checks run
    # elsewhere; this is the agent's own spot audit)
    ts_bad = sum(1 for r in kb
                 if (r.get("made_ts") or 0) >= (r.get("close_ts")
                                                or 1 << 40))
    ts_bad += sum(1 for r in snaps
                  if r.get("persist_ts", 0) < r.get("decision_ts", 0))
    timestamps = {"violations_in_sample": ts_bad,
                  "state": "HEALTHY" if ts_bad == 0 else "CRITICAL"}

    # missingness on recent kb rows
    recent = kb[-500:]
    miss = {"mkt_p_up_null_rate": round(sum(
        1 for r in recent if r.get("mkt_p_up") is None)
        / len(recent), 4) if recent else None,
        "ask_c_null_rate": round(sum(
            1 for r in recent if r.get("ask_c") is None)
            / len(recent), 4) if recent else None}
    miss["state"] = "HEALTHY" if recent and \
        (miss["mkt_p_up_null_rate"] or 0) < 0.2 else "WATCH"

    # schema drift v1: required fields present on recent rows
    kb_drift = [k for k in KB_FIELDS
                if recent and any(k not in r for r in recent[-50:])]
    pit_drift = [k for k in PIT_FIELDS
                 if snaps and any(k not in r for r in snaps[-50:])]
    wrong_v = sum(1 for r in snaps[-50:]
                  if r.get("feature_schema_version") != PIT_V)
    schema = {"kb_missing_fields": kb_drift,
              "pit_missing_fields": pit_drift,
              "pit_wrong_schema_version": wrong_v,
              "state": "HEALTHY" if not kb_drift and not pit_drift
              and not wrong_v else "CRITICAL"}

    parity_prereq = {"parity_state": par.get("parity_state"),
                     "snapshots_available": len(snaps) > 0,
                     "state": "HEALTHY" if par.get("parity_state")
                     == "PASS" and snaps else "WATCH"}

    order = {"HEALTHY": 0, "UNKNOWN": 1, "WATCH": 1, "CRITICAL": 2}
    blocks = {"feeds": feeds, "pit_store": pit,
              "timestamps": timestamps, "missingness": miss,
              "schema": schema, "parity_prereq": parity_prereq}
    worst = max((s.get("state", "UNKNOWN") for b in blocks.values()
                 for s in ([b] if "state" in b else b.values())
                 if isinstance(s, dict) and "state" in s),
                key=lambda x: order.get(x, 1), default="UNKNOWN")
    doc = {"generated_ts": int(now), "overall": worst, **blocks,
           "slo_seconds": SLO,
           "provenance": ["online_status.json", "event_capture.json",
                          "kalshi_binary_log.jsonl",
                          "feature_snapshots.jsonl", "parity.json"]}
    (RES / "data_health.json").write_text(json.dumps(doc, indent=1))

    problems = [f"{k}:{v['state']}" for k, v in
                {**feeds, "pit": pit, "timestamps": timestamps,
                 "schema": schema}.items()
                if isinstance(v, dict)
                and v.get("state") in ("WATCH", "CRITICAL")]
    if problems:
        fw.submit(agent="data_reliability", action_class="DIAGNOSE",
                  finding="data-plane degradation: "
                          + ", ".join(problems),
                  recommendation="investigate per data_health.json "
                                 "blocks; no repair authority",
                  evidence=["data_health.json"])
    # deterministic safe repairs land BLOCKED_UNTIL_M6 by design
    if feeds["daemon_heartbeat"]["state"] == "CRITICAL":
        fw.submit(agent="data_reliability",
                  action_class="SAFE_OPS_REPAIR",
                  finding="daemon heartbeat stale beyond SLO",
                  recommendation="restart stale daemon "
                                 "(watchdog-equivalent action)",
                  evidence=["online_status.json",
                            "data_health.json"])
    else:
        fw.submit(agent="data_reliability", action_class="OBSERVE",
                  finding=f"data plane {worst}: heartbeat "
                          f"{feeds['daemon_heartbeat']['age_s']}s, "
                          f"capture {feeds['event_capture']['age_s']}s"
                          f", PIT completeness {pit['completeness']}",
                  recommendation="NO_ACTION_REQUIRED"
                  if worst == "HEALTHY" else "monitor",
                  evidence=["data_health.json"])
    print(f"data_reliability: overall {worst} · PIT "
          f"{pit['completeness']} ({pit['missing']} missing of "
          f"{pit['expected_since_golive']}) · ts violations "
          f"{ts_bad}")


if __name__ == "__main__":
    main()
