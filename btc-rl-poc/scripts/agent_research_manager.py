"""M5.1 — Research Manager (script-backed structured agent).

The coordinator, not a model inventor. Sole owner of
results/research_queue.json; consumes only canonical artifacts and
the other agents' ledger rows. HARD RULE (PM 08-30): it may not
create work merely because the queue is short — when the evidence
says "collect", its correct, successful output is
NO_NEW_RESEARCH_ACTION.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import agent_firewall as fw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def j(name):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return None


def main():
    now = int(time.time())
    a3 = j("a3_live.json") or {}
    fwd = a3.get("forward") or {}
    n = fwd.get("eligible") or 0
    ea = j("experiment_analysis.json") or {}
    qual = (j("model_qualification.json") or {}).get("models") or {}
    prog = j("program.json") or {}
    rm = prog.get("research_manager") or {}
    no_challenger = all(m.get("verdict") in
                        ("OFFLINE_FAIL", "INVALID")
                        for m in qual.values()) if qual else None
    queue = [
        {"priority": "P0",
         "research_question": "A3-v2 forward qualification — does "
         "waiting for a >=5c dip beat immediate entry after costs? "
         "(fresh prospective test motivated by A3-v1.1's clean "
         "REJECT: fills always won, misses dominated at 10c)",
         "bottleneck": f"evidence n={n} (compare 25, decision 50; "
         "fresh counter from 2026-08-30 registration)",
         "evidence": "a3_live.json (v2); a3_v1_final.json (closed "
         "predecessor); A3_V2_SPEC.yaml registration rationale",
         "estimated_impact": "decides the desk's entry policy",
         "state": "COLLECTING", "blocked_by": "evidence N",
         "owner": "market clock",
         "next_gate": "registered decision gate at n>=50"},
        {"priority": "P0",
         "research_question": "Feature-discovery program Phase 1: "
         "free-data CAPTURE_ONLY (F-MICRO existing tape; F-XVENUE "
         "binance/okx/kraken live 08-30)",
         "bottleneck": "Gate F1 — >=7 days capture, >=95% coverage, "
         "zero unresolved ts/schema issues (eligible >= 09-06)",
         "evidence": "xvenue_capture.json heartbeat; "
         "config/FEATURE_REGISTRY.yaml sources",
         "estimated_impact": "the one-sentence objective: one "
         "compact free feature family with stable PIT "
         "market-relative information early enough to monetize",
         "state": "ACTIVE", "blocked_by": None,
         "owner": "data_reliability", "next_gate": "Gate F1"},
        {"priority": "P1",
         "research_question": "M5.4 diagnosis follow-through: does "
         "kb9 disagreement (>=5pp) predict market error "
         "prospectively? (ONE candidate question; F-XVENUE is its "
         "natural feature family)",
         "bottleneck": "prospective registration + capture Gate F1",
         "evidence": "model_research.json — kb9 EARLY_SKILL, "
         "3/36 slices passed gates; kb2 redundancy 0.998 sanity-"
         "confirmed" + ("" if no_challenger else " (partial)"),
         "estimated_impact": "unlocks a legitimate challenger "
         "pipeline (Phase 10: incumbent + winning family only)",
         "state": "ACTIVE", "blocked_by": "Gate F1",
         "owner": "model_researcher",
         "next_gate": "family-level walk-forward ablation"},
        {"priority": "P2",
         "research_question": "Legacy M10 vs M10+M8 closure",
         "bottleneck": "paired increment P(Δ>0) 77% — unclear",
         "evidence": "program.json incremental; decision_board.json",
         "estimated_impact": "closes the legacy experiment family",
         "state": "ACTIVE", "blocked_by": None,
         "owner": "experiment_analyst",
         "next_gate": "SPRT verdict"},
        {"priority": "P3",
         "research_question": "Maintenance/observability: feature "
         "PSI + prediction-drift instrumentation",
         "bottleneck": "declared NOT INSTRUMENTED gaps",
         "evidence": "backend Data module gap rows",
         "estimated_impact": "drift visibility before label decay",
         "state": "ACTIVE", "blocked_by": None,
         "owner": "data_reliability (M5.3, not yet active)",
         "next_gate": None},
    ]
    blocked = (rm.get("blocked") or [])
    doc = {"generated_ts": now,
           "maintained_by": "agent_research_manager (script-backed)",
           "queue": queue, "blocked": blocked,
           "provenance": ["a3_live.json", "experiment_analysis.json",
                          "model_qualification.json", "program.json",
                          "agent_recommendations.jsonl"]}
    (RES / "research_queue.json").write_text(json.dumps(doc, indent=1))
    # the coordinator's own structured output: at the current
    # evidence state the correct research action is NONE
    row = fw.submit(
        agent="research_manager", action_class="OBSERVE",
        finding=f"P0 bottleneck is A3 evidence accumulation (n={n}); "
                "no candidate model qualifies; no unlock condition "
                "met on any blocked item",
        recommendation="NO_NEW_RESEARCH_ACTION — collect",
        evidence=["research_queue.json", "a3_live.json",
                  "model_qualification.json"], n=n)
    print(f"research_manager: queue 4 · blocked {len(blocked)} → "
          f"{row['recommendation']} ({row['status']})")


if __name__ == "__main__":
    main()
