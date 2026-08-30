"""Emit results/agent_performance.json — per-agent metrics (M5 close
contract §10). Agents are never ranked by recommendation volume; the
long-term ratios (validated_helpful/implemented, duplicate_rate) are
what matter, and until enough outcomes exist AGENT_QUALITY stays
UNKNOWN — never green.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
AGENTS = ("research_manager", "experiment_analyst",
          "data_reliability", "model_researcher",
          "execution_researcher")
MIN_OUTCOMES = 10        # METRICS.yaml display gate


def main():
    rows, status_by_rec = [], {}
    p = RES / "agent_recommendations.jsonl"
    if p.exists():
        for l in p.open():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("kind") == "status_update":
                status_by_rec[r["recommendation_id"]] = r["status"]
            else:
                rows.append(r)
    out = {}
    for a in AGENTS:
        mine = [r for r in rows if r.get("agent") == a]
        final = {r["recommendation_id"]:
                 status_by_rec.get(r["recommendation_id"],
                                   r.get("status"))
                 for r in mine}
        vals = list(final.values())
        implemented = sum(1 for s in vals
                          if s in ("IMPLEMENTED", "VALIDATED_HELPFUL",
                                   "VALIDATED_NOT_HELPFUL"))
        helpful = sum(1 for s in vals if s == "VALIDATED_HELPFUL")
        outcomes = sum(1 for s in vals
                       if s in ("VALIDATED_HELPFUL",
                                "VALIDATED_NOT_HELPFUL"))
        n = len(mine)
        out[a] = {
            "submissions": n,
            "new_findings": sum(1 for r in mine
                                if r.get("status") == "OPEN"),
            "duplicates": sum(1 for r in mine
                              if r.get("status") == "DUPLICATE"),
            "duplicate_rate": round(sum(
                1 for r in mine if r.get("status") == "DUPLICATE")
                / n, 3) if n else None,
            "proposals": sum(1 for r in mine
                             if r.get("action_class") == "PROPOSE"),
            "stale_input_count": sum(
                1 for r in mine
                if "STALE_INPUT" in str(r.get("finding", ""))),
            "firewall_rejections": sum(
                1 for r in mine
                if r.get("status") == "REJECTED_BY_FIREWALL"),
            "accepted": sum(1 for s in vals if s == "ACCEPTED"),
            "implemented": implemented,
            "validated_helpful": helpful,
            "validated_not_helpful": sum(
                1 for s in vals if s == "VALIDATED_NOT_HELPFUL"),
            "helpful_per_implemented": round(helpful / implemented, 3)
            if implemented else None,
            "quality": "UNKNOWN — insufficient outcomes "
            f"({outcomes}/{MIN_OUTCOMES} validated)"
            if outcomes < MIN_OUTCOMES else
            ("GOOD" if implemented and helpful / implemented >= 0.5
             else "REVIEW"),
        }
    doc = {"generated_ts": int(time.time()),
           "agents": out,
           "law": "never rank by volume; AGENT_QUALITY stays UNKNOWN "
                  "below the outcome gate; a high duplicate rate on "
                  "unchanged evidence is HEALTHY",
           "provenance": "agent_recommendations.jsonl"}
    (RES / "agent_performance.json").write_text(
        json.dumps(doc, indent=1))
    print("agent_performance: " + " · ".join(
        f"{a}:{m['submissions']}sub/{m['duplicates']}dup"
        for a, m in out.items()))


if __name__ == "__main__":
    main()
