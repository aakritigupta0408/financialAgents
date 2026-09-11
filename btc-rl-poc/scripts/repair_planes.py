"""Repair-plane orchestration (M6 interaction contract, PM 08-31).

One canonical dependency object — results/repair_dependency_state.json:

    COMPUTE   R1's plane (daemon process + heartbeat)
    STATE     R2's plane (canonical/derived truth fresh + parseable +
              invariants green)
    DELIVERY  R3's plane (published representation current)

Ordering law: repair the EARLIEST broken dependency first —
COMPUTE -> STATE -> DELIVERY. While an upstream plane is unhealthy,
downstream repairs are SUPPRESSED (recorded, not raced). The
per-repair upstream gates remain as defense-in-depth; this object is
the single source the executor consults.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HEALTHY, DEGRADED, FAILED = "HEALTHY", "DEGRADED", "FAILED_CLOSED"


def plane_state(res, pat=r"-m btc_rl\.online$", hb_stale_s=300,
                state_artifact="a3_live.json", state_max_age_s=1800,
                delivery_probe=None):
    """Compute the three plane states. delivery_probe: callable
    returning a trigger string or None (injected: repair_r3.detect
    in production, fixtures in tests)."""
    res = Path(res)
    # COMPUTE
    try:
        hb = time.time() - json.loads(
            (res / "online_status.json").read_text())["alive_at"]
    except Exception:
        hb = None
    try:
        pr = subprocess.run(["pgrep", "-f", "--", pat],
                            capture_output=True, text=True)
        nproc = len([x for x in pr.stdout.splitlines() if x.strip()])
    except Exception:
        nproc = None
    if hb is None or hb > hb_stale_s * 2:
        compute = FAILED
    elif hb > hb_stale_s or (nproc is not None and nproc != 1):
        compute = DEGRADED
    else:
        compute = HEALTHY
    # STATE
    state = HEALTHY
    p = res / state_artifact
    if not p.exists():
        state = FAILED
    else:
        try:
            json.loads(p.read_text())
            if time.time() - p.stat().st_mtime > state_max_age_s:
                state = DEGRADED
        except Exception:
            state = FAILED
    try:
        inv = json.loads((res / "invariants.json").read_text())
        if inv.get("failed"):
            state = FAILED
    except Exception:
        state = DEGRADED if state == HEALTHY else state
    # DELIVERY
    delivery = HEALTHY
    if delivery_probe is not None:
        try:
            trig = delivery_probe()
            if trig == "UPSTREAM_UNHEALTHY":
                delivery = DEGRADED     # judged by upstream planes
            elif trig is not None:
                delivery = FAILED
        except Exception:
            delivery = DEGRADED
    return {"COMPUTE": compute, "STATE": state, "DELIVERY": delivery,
            "heartbeat_age_s": None if hb is None else round(hb),
            "processes": nproc}


def emit(res, planes):
    doc = {"generated_ts": int(time.time()), **planes,
           "ordering_law": "COMPUTE -> STATE -> DELIVERY; downstream "
                           "repairs suppressed while upstream "
                           "unhealthy"}
    (Path(res) / "repair_dependency_state.json").write_text(
        json.dumps(doc, indent=1))
    return doc


def orchestrate(planes, run_r1=None, run_r2=None, run_r3=None,
                record=None):
    """Run at most the single earliest-broken plane's repair.
    Suppressed downstream planes are recorded (visible, not silent).
    Returns {"acted": plane|None, "suppressed": [...], results}."""
    out = {"acted": None, "suppressed": [], "results": {}}

    def _note(plane, reason):
        out["suppressed"].append(plane)
        if record:
            record("REPAIR_SUPPRESSED", plane=plane, reason=reason)
    if planes["COMPUTE"] != HEALTHY:
        if run_r1:
            out["results"]["R1"] = run_r1()
            out["acted"] = "COMPUTE"
        if planes["STATE"] != HEALTHY:
            _note("STATE", "upstream COMPUTE unhealthy — ordering law")
        if planes["DELIVERY"] != HEALTHY:
            _note("DELIVERY", "upstream COMPUTE unhealthy — "
                              "ordering law")
        return out
    if planes["STATE"] != HEALTHY:
        if run_r2:
            out["results"]["R2"] = run_r2()
            out["acted"] = "STATE"
        if planes["DELIVERY"] != HEALTHY:
            _note("DELIVERY", "upstream STATE unhealthy — "
                              "ordering law")
        return out
    if planes["DELIVERY"] != HEALTHY:
        if run_r3:
            out["results"]["R3"] = run_r3()
            out["acted"] = "DELIVERY"
        return out
    return out
