"""M6-R3 executor — RETRY_PUBLISHER (delivery plane).

The narrowest repair: canonical local truth is healthy but the
published representation is missing/stale/wrong. Republish the EXACT
canonical state (publish_dashboard.py — idempotent: rides one amended
marker commit) and let the independent verifier fetch the actual
destination. R3 never calculates research metrics and never invokes
R2 implicitly — each repair plane has one owner:

    COMPUTE  R1 · STATE  R2 · DELIVERY  R3

Dependency ordering is by construction: R3's canonical-health gate
refuses when local truth is itself stale/corrupt (that is R1/R2
territory), so multi-fault scenarios repair the earliest broken
dependency first instead of racing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPAIR_ID = "M6-R3_RETRY_PUBLISHER"
PLANE = "DELIVERY"

CFG = {
    "res": ROOT / "results",
    "registry": ROOT / "config" / "M6_REPAIRS.yaml",
    "heal_log": ROOT / "results" / "self_heal.jsonl",
    "incidents": ROOT / "results" / "incidents.jsonl",
    "maintenance": ROOT / "results" / "maintenance.flag",
    # research-sensitive artifact watched end-to-end
    "artifact": "a3_live.json",
    "published_url": "https://www.theaakritigupta.com/btc-oracle/"
                     "results/a3_live.json",
    "publish_cmd": [sys.executable,
                    str(ROOT / "scripts" / "publish_dashboard.py")],
    "publish_timeout_s": 240,
    "canonical_max_age_s": 1800,
    "published_lag_trigger_s": 900,
    "fetch_timeout_s": 15,
    "max_attempts": 2,
    "attempt_window_s": 1800,
}


def _yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def r3_enabled(cfg):
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
           "plane": PLANE, "state": state, **fields}
    with cfg["heal_log"].open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def fetch_published(cfg):
    """(doc, err) from the real destination. Network failure returns
    err='FETCH_FAILED' — treated as transient, never as proof of a
    missing publication."""
    url = cfg["published_url"]
    is_file = url.startswith("file:")
    try:
        req = urllib.request.Request(
            url if is_file else url + f"?t={int(time.time())}",
            headers={"User-Agent": "btc-rl-r3"})
        with urllib.request.urlopen(
                req, timeout=cfg["fetch_timeout_s"]) as r:
            body = r.read()
        try:
            return json.loads(body), None
        except Exception:
            return None, "PUBLISHED_CORRUPT"
    except urllib.error.HTTPError as e:
        return None, ("PUBLISHED_MISSING" if e.code == 404
                      else f"HTTP_{e.code}")
    except Exception:
        # file:// fixtures: a missing local file IS a missing
        # publication; over the network, failure is transient and
        # never proof of absence
        return (None, "PUBLISHED_MISSING") if is_file \
            else (None, "FETCH_FAILED")


def _canonical(cfg):
    p = cfg["res"] / cfg["artifact"]
    if not p.exists() or time.time() - p.stat().st_mtime \
            > cfg["canonical_max_age_s"]:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def detect(cfg):
    """(trigger, canonical_doc) — trigger None when healthy."""
    local = _canonical(cfg)
    if local is None:
        return "UPSTREAM_UNHEALTHY", None
    pub, err = fetch_published(cfg)
    if err == "PUBLISHED_MISSING":
        return "PUBLISHED_MISSING", local
    if err == "FETCH_FAILED":
        return None, local          # transient network ≠ delivery fault
    if err == "PUBLISHED_CORRUPT":
        return "PUBLISHED_HASH_MISMATCH", local
    if err:
        return "PUBLISHER_TRANSIENT_FAILURE", local
    if not isinstance(pub, dict):
        return "PUBLISHED_HASH_MISMATCH", local
    if pub.get("experiment_id") != local.get("experiment_id"):
        return "PUBLISHED_HASH_MISMATCH", local
    lag = (local.get("generated_ts") or 0) \
        - (pub.get("generated_ts") or 0)
    if lag > cfg["published_lag_trigger_s"]:
        return "PUBLISHED_STALE", local
    return None, local


def _attempts(cfg):
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
                    and r.get("ts", 0) >= cut:
                n += 1
    return n


def run_r3(cfg=CFG):
    if cfg["maintenance"].exists():
        return "MAINTENANCE_NO_TRIGGER"
    trigger, local = detect(cfg)
    if trigger is None:
        return "HEALTHY_NO_TRIGGER"
    if trigger == "UPSTREAM_UNHEALTHY":
        record(cfg, "FAILED_CLOSED",
               reason="UPSTREAM_UNHEALTHY — canonical source itself "
                      "stale/corrupt; that is R1/R2 territory, R3 "
                      "refuses (dependency ordering)")
        return "UPSTREAM_UNHEALTHY"
    record(cfg, "DETECTED", trigger=trigger,
           artifact=cfg["artifact"])
    if _attempts(cfg) >= cfg["max_attempts"]:
        record(cfg, "AUTO_REPAIR_EXHAUSTED", attempts=_attempts(cfg))
        record(cfg, "FAILED_CLOSED",
               reason="attempt budget exhausted")
        with cfg["incidents"].open("a") as f:
            f.write(json.dumps({
                "sev": 3, "title": "R3 publish retry budget "
                "exhausted — delivery stays UNAVAILABLE",
                "opened": time.strftime("%Y-%m-%d %H:%M"),
                "status": "open — needs human",
                "detected_by": "watchdog/M6-R3",
                "repair_id": REPAIR_ID,
                "root_cause": "UNKNOWN"}) + "\n")
        return "FAILED_CLOSED"
    record(cfg, "CONTAINED",
           note="published_state = UNAVAILABLE/STALE — the old copy "
                "must not present as current")
    # REPAIR — one idempotent publish pass; command rc is recorded
    # but NEVER trusted as success (the verifier fetches the
    # destination)
    try:
        r = subprocess.run(cfg["publish_cmd"], cwd=ROOT,
                           capture_output=True,
                           timeout=cfg["publish_timeout_s"])
        rc = r.returncode
    except Exception as e:
        rc = repr(e)
    record(cfg, "REPAIR_ATTEMPTED", trigger=trigger,
           artifact=cfg["artifact"], publisher_rc=rc,
           local_generated_ts=(local or {}).get("generated_ts"),
           local_experiment_id=(local or {}).get("experiment_id"),
           local_eligible_n=((local or {}).get("forward")
                             or {}).get("eligible"),
           verification="pending — meta_monitor fetches the REAL "
                        "destination; command rc is not success")
    with cfg["incidents"].open("a") as f:
        f.write(json.dumps({
            "sev": 3, "title": f"delivery fault {trigger} — M6-R3 "
            "republish attempted",
            "opened": time.strftime("%Y-%m-%d %H:%M"),
            "status": "auto-repair attempted — verification pending",
            "detected_by": "watchdog/M6-R3",
            "repair_id": REPAIR_ID, "root_cause": "UNKNOWN"}) + "\n")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from log_system_change import log_change
        log_change("SELF_HEAL_OCCURRED", "publisher",
                   before=trigger, after="republish attempted, "
                   "verification pending",
                   reason="M6-R3 registered repair (delivery plane)",
                   impact="none — exact canonical state republished")
    except Exception:
        pass
    return "REPAIR_ATTEMPTED"


if __name__ == "__main__":
    print(run_r3(CFG) if r3_enabled(CFG)
          else "R3 not enabled/certified — no action")
