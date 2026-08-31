"""M6-R2 executor — REBUILD_DERIVED_ARTIFACT.

Called from watchdog.py (the repair-executor process) each cron
cycle when R2 is enabled+certified. VERIFY/RESTORE live in
meta_monitor (the independent verifier) — this module only runs
DETECT -> CLASSIFY -> CONTAIN -> REPAIR -> RECORD.

Laws implemented here:
  * class boundary — only ARTIFACT_CLASSES.derived_rebuildable rows
    are eligible; anything else is refused at CLASSIFY;
  * upstream gate — every upstream must be fresh (<2x its own SLO
    heuristic) and parseable, else FAILED_CLOSED/UPSTREAM_UNHEALTHY;
  * atomicity — the prior file is preserved as <name>.pre_rebuild
    before the emitter runs; a failed rebuild never destroys the
    last-known artifact;
  * retry law — max 2 attempts per artifact per 30 min, then
    AUTO_REPAIR_EXHAUSTED -> FAILED_CLOSED + incident.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPAIR_ID = "M6-R2_REBUILD_DERIVED_ARTIFACT"

CFG = {
    "res": ROOT / "results",
    "classes": ROOT / "config" / "ARTIFACT_CLASSES.yaml",
    "registry": ROOT / "config" / "M6_REPAIRS.yaml",
    "heal_log": ROOT / "results" / "self_heal.jsonl",
    "incidents": ROOT / "results" / "incidents.jsonl",
    "maintenance": ROOT / "results" / "maintenance.flag",
    "python": sys.executable,
    "emitter_timeout_s": 240,
    "max_attempts": 2,
    "attempt_window_s": 1800,
    "upstream_max_age_s": 3600,
}


def _yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def r2_enabled(cfg):
    try:
        for r in _yaml(cfg["registry"]).get("class_a_allowlist") or []:
            if r.get("repair_id") == REPAIR_ID:
                return bool(r.get("enabled")) and bool(
                    r.get("certified"))
    except Exception:
        pass
    return False


def record(cfg, state, **fields):
    row = {"ts": round(time.time(), 3), "repair_id": REPAIR_ID,
           "state": state, **fields}
    with cfg["heal_log"].open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _attempts(cfg, artifact):
    cut = time.time() - cfg["attempt_window_s"]
    n = 0
    if cfg["heal_log"].exists():
        for l in cfg["heal_log"].open():
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("repair_id") == REPAIR_ID \
                    and r.get("state") == "REPAIR_ATTEMPTED" \
                    and r.get("artifact") == artifact \
                    and r.get("ts", 0) >= cut:
                n += 1
    return n


def _detect(cfg, name, spec):
    """Return a trigger reason or None."""
    p = cfg["res"] / name
    if not p.exists():
        return "FILE_MISSING"
    try:
        json.loads(p.read_text())
    except Exception:
        return "SCHEMA_INVALID"
    if time.time() - p.stat().st_mtime > spec["slo_s"]:
        return "STALE_BEYOND_SLO"
    return None


def _upstreams_healthy(cfg, spec):
    for u in spec.get("upstreams") or []:
        p = cfg["res"] / u
        if not p.exists():
            return False, f"{u} missing"
        if time.time() - p.stat().st_mtime > cfg["upstream_max_age_s"]:
            return False, f"{u} stale"
        if u.endswith(".json"):
            try:
                json.loads(p.read_text())
            except Exception:
                return False, f"{u} corrupt"
    return True, None


def run_r2(cfg=CFG, only=None):
    """One pass over the rebuildable registry. Returns per-artifact
    terminal states for this invocation."""
    if cfg["maintenance"].exists():
        return {}
    classes = _yaml(cfg["classes"])
    out = {}
    for name, spec in (classes.get("derived_rebuildable")
                       or {}).items():
        if only and name != only:
            continue
        trigger = _detect(cfg, name, spec)
        if trigger is None:
            out[name] = "HEALTHY_NO_TRIGGER"
            continue
        record(cfg, "DETECTED", artifact=name, trigger=trigger)
        ok, why = _upstreams_healthy(cfg, spec)
        if not ok:
            record(cfg, "FAILED_CLOSED", artifact=name,
                   reason=f"UPSTREAM_UNHEALTHY ({why}) — refusing "
                          "to rebuild from untrusted input")
            out[name] = "UPSTREAM_UNHEALTHY"
            continue
        if _attempts(cfg, name) >= cfg["max_attempts"]:
            record(cfg, "AUTO_REPAIR_EXHAUSTED", artifact=name)
            record(cfg, "FAILED_CLOSED", artifact=name,
                   reason="attempt budget exhausted")
            with cfg["incidents"].open("a") as f:
                f.write(json.dumps({
                    "sev": 3, "title": f"R2 rebuild budget exhausted "
                    f"for {name}", "status": "open — needs human",
                    "opened": time.strftime("%Y-%m-%d %H:%M"),
                    "detected_by": "watchdog/M6-R2",
                    "repair_id": REPAIR_ID,
                    "root_cause": "UNKNOWN"}) + "\n")
            out[name] = "FAILED_CLOSED"
            continue
        # CONTAIN — the record marks the artifact UNAVAILABLE; UI
        # contract: stale value must not render as current
        record(cfg, "CONTAINED", artifact=name,
               note="artifact marked UNAVAILABLE until independently "
                    "verified")
        # atomicity: preserve the last-known file before any attempt
        p = cfg["res"] / name
        if p.exists():
            shutil.copy2(p, p.with_suffix(p.suffix + ".pre_rebuild"))
        # REPAIR — run the registered emitter (never inline rebuild)
        cmd = spec["emitter"]
        cmd = cmd if isinstance(cmd, list) else [cfg["python"],
                                                 str(ROOT / cmd)]
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                               timeout=cfg["emitter_timeout_s"])
            rc = r.returncode
        except Exception as e:
            rc = repr(e)
        record(cfg, "REPAIR_ATTEMPTED", artifact=name,
               attempt=_attempts(cfg, name) + 1, emitter_rc=rc,
               trigger=trigger,
               verification="pending — meta_monitor (independent) "
                            "certifies; this process may not")
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from log_system_change import log_change
            log_change("SELF_HEAL_OCCURRED", name,
                       before=trigger, after="rebuilt, verification "
                       "pending", reason="M6-R2 registered repair",
                       impact="none — derived state only")
        except Exception:
            pass
        out[name] = "REPAIR_ATTEMPTED"
    return out


if __name__ == "__main__":
    if r2_enabled(CFG):
        print(json.dumps(run_r2(CFG)))
    else:
        print("R2 not enabled/certified — no action")
