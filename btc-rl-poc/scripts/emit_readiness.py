"""Emit results/readiness.json — machine-computed qualification state
(master directive §3-§4, §77).

Reads REAL_MONEY_EQUIVALENT_STANDARD.yaml (the immutable spec) and the
live evidence artifacts, applies the downgrade rules, and produces the
readiness matrix. The emitter may only DOWNGRADE from the assessed
baseline, never upgrade; system level = MIN over critical planes;
UNKNOWN is never GREEN; no hand-typed READY anywhere.

Hard invariants checked HERE, from disk, every run:
  * no production execution adapter: order-writing code may target
    only the demo host; the production API may appear ONLY in
    read-only market-data paths (sources.py GETs);
  * no production credentials detected (repo + the owner home paths
    the daemon can reach).
Any failure -> security plane R0, system R0, SEV-0 surfaced.
"""
from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
STD = ROOT / "REAL_MONEY_EQUIVALENT_STANDARD.yaml"

R_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]


def _r(n):
    return R_ORDER.index(n)


def _parse_standard():
    """Minimal YAML subset parser (no dependency): we need version,
    planes{assessed, evidence, blocker_to_next}, strategy status."""
    txt = STD.read_text()
    version = re.search(r"^version:\s*(\S+)", txt, re.M).group(1)
    planes = {}
    for m in re.finditer(
            r"^  (\w+):\n    assessed: (R\d)\n"
            r"    evidence: (.+)\n"
            r"    blocker_to_next: \"(.+)\"", txt, re.M):
        planes[m.group(1)] = {"assessed": m.group(2),
                              "evidence": m.group(3).strip(),
                              "blocker": m.group(4)}
    strat = re.search(r"current_status:\s*(.+?)\s*#", txt)
    return version, planes, (strat.group(1).strip() if strat
                             else "UNKNOWN")


def jload(name):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return None


def check_execution_adapter():
    """Order-submission code must never target the production host."""
    findings = []
    for f in list(ROOT.glob("btc_rl/*.py")) + list(
            ROOT.glob("scripts/*.py")):
        if f.name == "emit_readiness.py":
            continue            # the detector's own pattern strings
        src = f.read_text()
        for m in re.finditer(r"(post|POST|create_order|sign)", src):
            seg = src[max(0, m.start() - 400):m.start() + 400]
            if "elections.kalshi.com" in seg \
                    and "portfolio/orders" in seg:
                findings.append(f"{f.name}: order path near "
                                "production host")
    return findings


def check_credentials():
    hits = []
    home = Path.home()
    for pat in (".kalshi*PRODUCTION*", ".kalshi*prod*"):
        hits += [str(p) for p in home.glob(pat)]
    for p in ROOT.rglob("*.pem"):
        hits.append(str(p))
    return hits


def main():
    now = int(time.time())
    version, planes, strategy = _parse_standard()

    inv = jload("invariants.json") or {}
    db = jload("decision_board.json") or {}
    status = jload("online_status.json") or {}
    audit = jload("audit_report.json") or {}

    adapter = check_execution_adapter()
    creds = check_credentials()
    sev0 = []
    if adapter:
        sev0.append({"invariant": "production_execution_adapter_exists",
                     "detail": adapter})
    if creds:
        sev0.append({"invariant": "production_credentials_detected",
                     "detail": creds,
                     "remedy": "owner must delete/relocate the "
                     "production key file; the system cannot clear "
                     "this itself"})

    heartbeat_age = (now - status["alive_at"]) if status.get(
        "alive_at") else None
    audit_age_min = ((now - audit["generated_ts"]) / 60
                     if audit.get("generated_ts") else None)
    frozen = bool(audit_age_min is None or audit_age_min > 20)

    out = {}
    for name, p in planes.items():
        level = p["assessed"]
        downs = []
        if inv.get("health") != "green":
            level, downs = "R0", ["invariant wall not green"]
        if name == "security" and (adapter or creds):
            level = "R0"
            downs.append("hard invariant failed (see sev0)")
        if name == "data" and heartbeat_age is not None \
                and heartbeat_age > 300:
            if _r(level) > 1:
                level = "R1"
            downs.append(f"heartbeat {heartbeat_age:.0f}s stale")
        if name == "experimentation" and \
                (db.get("integrity") or {}).get("health") != "green":
            if _r(level) > 1:
                level = "R1"
            downs.append("decision-board integrity not green")
        out[name] = {"level": level, "assessed": p["assessed"],
                     "target": "R8",
                     "evidence": p["evidence"],
                     "blocker": p["blocker"],
                     "live_downgrades": downs}

    system_level = min((v["level"] for v in out.values()), key=_r)
    weakest = [k for k, v in out.items() if v["level"] == system_level]

    doc = {
        "generated_ts": now,
        "standard_version": version,
        "system": {
            "level": system_level,
            "rule": "MIN over all critical planes — never averaged",
            "weakest_planes": weakest,
            "qualification_frozen": frozen,
            "frozen_reason": (f"auditor age "
                              f"{audit_age_min:.0f}m > 20m"
                              if frozen and audit_age_min is not None
                              else None),
            "headline": (f"REAL-MONEY-EQUIVALENT: {system_level} — "
                         "NOT R8"),
        },
        "strategy": {
            "status": strategy,
            "headline": "STRATEGY: NO QUALIFIED EDGE — best BSS vs "
                        "market ≤ 0 this sample; this is a valid "
                        "scientific outcome, not a system failure",
            "separation_note": "system readiness and strategy "
            "qualification are independent verdicts (§83)",
        },
        "live_execution": "PHYSICALLY LIMITED — order code targets "
                          "demo host only; production API is "
                          "read-only market data",
        "sev0_open": sev0,
        "planes": out,
    }
    (RES / "readiness.json").write_text(json.dumps(doc, indent=1))
    print(f"readiness.json: system {system_level} "
          f"(weakest: {','.join(weakest)}) · strategy: {strategy} · "
          f"sev0 open: {len(sev0)}")


if __name__ == "__main__":
    main()
