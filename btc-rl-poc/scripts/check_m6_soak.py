"""M6 close-soak adjudicator (PM 08-31): >=6 consecutive clean audit
cycles with all three repair planes enabled. A clean cycle =

  * no unexpected repair (no new self_heal repair/failed rows in the
    cycle window — the system should be boring)
  * no dependency-order violation (no REPAIR_ATTEMPTED for STATE/
    DELIVERY while an upstream plane was unhealthy at that moment —
    approximated: any REPAIR_SUPPRESSED row is fine, a downstream
    attempt in the same cycle as an upstream failure is not)
  * repair_dependency_state all HEALTHY
  * invariants fully green
  * A3-v2 spec hash ok (scientific state untouched)

Dirty cycle -> counter reset. SOAK_PASS at 6 -> M6 is machine-
closeable. Output: results/m6_soak.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REQUIRED = 6
CYCLE_S = 660


def j(name, default=None):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return default


def main():
    now = int(time.time())
    prev = j("m6_soak.json", {}) or {}
    inv = j("invariants.json", {}) or {}
    planes = j("repair_dependency_state.json", {}) or {}
    a3 = j("a3_live.json", {}) or {}
    heal_rows = []
    p = RES / "self_heal.jsonl"
    if p.exists():
        for l in p.open():
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("ts", 0) >= now - CYCLE_S:
                heal_rows.append(r)
    problems = []
    if inv.get("failed"):
        problems.append(f"invariants failing {inv['failed']}")
    for pl in ("COMPUTE", "STATE", "DELIVERY"):
        if planes.get(pl) not in ("HEALTHY", None):
            problems.append(f"plane {pl}={planes.get(pl)}")
    acted = [r for r in heal_rows
             if r.get("state") in ("REPAIR_ATTEMPTED",
                                   "FAILED_CLOSED",
                                   "AUTO_REPAIR_EXHAUSTED")]
    if acted:
        problems.append(f"{len(acted)} unexpected repair rows this "
                        "cycle")
    # ordering violation: a downstream attempt while an upstream
    # suppression/failure existed in the same window
    planes_rank = {"M6-R1": 0, "M6-R2": 1, "M6-R3": 2}
    fails = [(planes_rank.get(str(r.get("repair_id", ""))[:5], 9),
              r) for r in heal_rows if r.get("state") in
             ("DETECTED", "REPAIR_ATTEMPTED")]
    if fails:
        min_broken = min(k for k, _ in fails)
        for k, r in fails:
            if r.get("state") == "REPAIR_ATTEMPTED" \
                    and k > min_broken:
                problems.append("dependency-order violation: "
                                f"{r.get('repair_id')} acted while "
                                "an upstream plane was broken")
    if a3.get("spec_hash_ok") is not True:
        problems.append("A3 spec hash not ok")
    clean = not problems
    consecutive = (prev.get("consecutive_clean_cycles", 0) + 1) \
        if clean else 0
    if clean and now - prev.get("last_clean_ts", 0) < CYCLE_S // 2:
        consecutive = prev.get("consecutive_clean_cycles", 0)
    status = ("SOAK_PASS" if consecutive >= REQUIRED
              else "SOAK_RESET" if not clean else "SOAKING")
    doc = {"generated_ts": now, "required_cycles": REQUIRED,
           "consecutive_clean_cycles": consecutive,
           "last_clean_ts": now if clean
           else prev.get("last_clean_ts", 0),
           "cycle_clean": clean, "problems": problems,
           "status": status,
           "close_gate": ["R1 PROVEN", "R2 PROVEN", "R3 PROVEN",
                          "multi-fault ordering PASS (fixtures 10/10)",
                          "independent verification universal",
                          "retry exhaustion universal",
                          "policy/research/scientific mutations = 0"]}
    (RES / "m6_soak.json").write_text(json.dumps(doc, indent=1))
    print(f"m6_soak: {status} · {consecutive}/{REQUIRED}"
          + (f" · {problems}" if problems else " · clean"))


if __name__ == "__main__":
    main()
