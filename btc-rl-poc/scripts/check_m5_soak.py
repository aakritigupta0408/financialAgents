"""M5 soak adjudicator (close contract §12) — runs each audit cycle,
persists a consecutive-clean-cycle counter in results/m5_soak.json.

A cycle is CLEAN when, at evaluation time:
  * invariant suite fully green
  * A3 spec hash ok (no experiment mutation)
  * ledger delta this cycle: no REJECTED_BY_FIREWALL from a real
    agent (fixtures named *_fixture excluded), no stale-driven
    PROPOSE, no more than 2 new OPEN rows (ledger-explosion guard),
    candidate_question count <= 1
  * decision_board best/serving set unchanged markers: sanctioned ev
    keys only (covered by invariants)
Any dirty cycle resets the counter (SOAK_RESET semantics). Close gate:
>= 6 consecutive clean cycles with all five agents having reported
within the window.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REQUIRED = 6
CYCLE_S = 660
AGENTS = {"research_manager", "experiment_analyst",
          "data_reliability", "model_researcher",
          "execution_researcher"}


def j(name, default=None):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return default


def jl(name):
    p = RES / name
    out = []
    if p.exists():
        for l in p.open():
            if l.strip():
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out


def main():
    now = int(time.time())
    prev = j("m5_soak.json", {}) or {}
    inv = j("invariants.json", {}) or {}
    a3 = j("a3_live.json", {}) or {}
    mr = j("model_research.json", {}) or {}
    recs = [r for r in jl("agent_recommendations.jsonl")
            if r.get("kind") != "status_update"]
    cyc = [r for r in recs if r.get("ts", 0) >= now - CYCLE_S]
    problems = []
    if inv.get("failed"):
        problems.append(f"invariants failing: {inv['failed']}")
    if a3.get("spec_hash_ok") is not True:
        problems.append("A3 spec hash not ok")
    real_rejects = [r for r in cyc
                    if r.get("status") == "REJECTED_BY_FIREWALL"
                    and not str(r.get("agent", "")).endswith(
                        "_fixture")]
    if real_rejects:
        problems.append(f"{len(real_rejects)} real firewall "
                        "rejections this cycle")
    stale_prop = [r for r in cyc if "STALE_INPUT" in
                  str(r.get("finding", ""))
                  and r.get("action_class") == "PROPOSE"]
    if stale_prop:
        problems.append("stale-driven proposal")
    new_open = [r for r in cyc if r.get("status") == "OPEN"]
    if len(new_open) > 2:
        problems.append(f"{len(new_open)} new OPEN rows in one "
                        "cycle (ledger explosion)")
    if (mr.get("candidate_question") or {}).get("count", 0) > 1:
        problems.append("candidate budget exceeded")
    # agent liveness over the soak horizon (they may dedupe silently,
    # so look back over the whole required window)
    horizon = [r for r in recs
               if r.get("ts", 0) >= now - REQUIRED * CYCLE_S]
    seen = {r.get("agent") for r in horizon}
    missing = AGENTS - seen
    clean = not problems
    consecutive = (prev.get("consecutive_clean_cycles", 0) + 1) \
        if clean else 0
    # don't double-count within the same cycle window
    if clean and now - prev.get("last_clean_ts", 0) < CYCLE_S // 2:
        consecutive = prev.get("consecutive_clean_cycles", 0)
    status = ("SOAK_PASS" if consecutive >= REQUIRED and not missing
              else "SOAK_RESET" if not clean
              else "SOAKING")
    doc = {"generated_ts": now,
           "required_cycles": REQUIRED,
           "consecutive_clean_cycles": consecutive,
           "last_clean_ts": now if clean
           else prev.get("last_clean_ts", 0),
           "cycle_clean": clean,
           "problems": problems,
           "agents_reported_in_horizon": sorted(seen & AGENTS),
           "agents_missing_in_horizon": sorted(missing),
           "status": status,
           "contract": "config/M5_CLOSE.yaml"}
    (RES / "m5_soak.json").write_text(json.dumps(doc, indent=1))
    print(f"m5_soak: {status} · {consecutive}/{REQUIRED} clean · "
          + ("; ".join(problems) if problems else "clean cycle"))


if __name__ == "__main__":
    main()
