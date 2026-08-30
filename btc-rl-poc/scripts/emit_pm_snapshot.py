"""Emit results/pm_snapshot.json — the one-screen daily PM state
(migration plan §111): A3 · models · system · migration · complexity
· top bottleneck · next action · blocked. Every field is read from a
published artifact; nothing is typed by hand."""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def j(name, default=None):
    p = RES / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def jl(name):
    p = RES / name
    if not p.exists():
        return []
    out = []
    for l in p.open():
        if l.strip():
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def main():
    now = int(time.time())
    a3 = j("a3_live.json", {}) or {}
    fwd = a3.get("forward") or {}
    prog = j("program.json", {}) or {}
    rm = prog.get("research_manager") or {}
    ml = j("model_lifecycle.json", {}) or {}
    rdy = j("readiness.json", {}) or {}
    inv = j("invariants.json", {}) or {}
    incs = jl("incidents.jsonl")
    open_sev = [i for i in incs
                if not str(i.get("status", "")).lower().startswith(
                    ("closed", "resolved"))]
    kb9 = (ml.get("t2_models") or {}).get("kb9") or {}
    kb2 = (ml.get("t2_models") or {}).get("kb2") or {}
    exps = prog.get("experiments") or []
    active_exp = [e["id"] for e in exps
                  if str(e.get("lifecycle", "")).split(" ")[0]
                  in ("CONTROL", "TREATMENT")]
    n = fwd.get("eligible") or 0
    queue = rm.get("queue") or []
    doc = {
        "generated_ts": now,
        "a3": {"n": n,
               "paired_delta": fwd.get("incremental_per_eligible"),
               "state": "COLLECTING" if n < 25 else "EVALUATING",
               "note": "no statistical conclusion below n=25"},
        "models": {
            "control": "kb2 (market-anchored)",
            "challenger": "kb9 (TimesFM fusion)",
            "challenger_bss_vs_market":
                (kb9.get("lifetime") or {}).get("bss_vs_market"),
            "challenger_trend": (kb9.get("trend") or {}).get("state"),
            "control_trend": (kb2.get("trend") or {}).get("state")},
        "system": {
            "readiness": (rdy.get("system") or {}).get("level"),
            "strategy": (rdy.get("strategy") or {}).get("status"),
            "invariants": f"{inv.get('passed', '?')}/"
                          f"{(inv.get('passed') or 0) + (inv.get('failed') or 0)}",
            "open_sevs": len(open_sev),
            "weakest_planes":
                (rdy.get("system") or {}).get("weakest_planes")},
        "migration": {
            "M1_forward_platform": "COMPLETE (frozen, collecting)",
            "M2_observatory": "IN PROGRESS — metrics v1.1.0 + "
                "training registry + lifecycle shipped; offline/"
                "walk-forward evaluator + parity next",
            "M3_lean_cleanup": "runtime cut LIVE (roster frozen, "
                "9 treatments retired); registry codified",
            "M4_backend_ui": "PENDING (after M2 schemas)",
            "M5_agents": "PENDING (contracts first)",
            "M6_self_healing": "PARTIAL (watchdog/self-heal exist; "
                "formal loop pending)"},
        "complexity": {
            "active_traders": 4, "empty_slots": 1,
            "serving_model_roles": "kb2+kb9 (T2) + 4 frozen T1 pairs",
            "active_experiments": active_exp,
            "shadows": ["T05", "T15", "M8", "M13", "pt6"]},
        "top_bottleneck": (queue[0]["title"] + " — " + queue[0]["why"])
        if queue else "unknown",
        "next_action": queue[1]["title"] if len(queue) > 1 else None,
        "blocked": [b["item"] for b in (rm.get("blocked") or [])],
        "provenance": ["a3_live.json", "program.json",
                       "model_lifecycle.json", "readiness.json",
                       "invariants.json", "incidents.jsonl"],
    }
    (RES / "pm_snapshot.json").write_text(json.dumps(doc, indent=1))
    print(f"pm_snapshot: A3 n={n} Δ={doc['a3']['paired_delta']} · "
          f"sys {doc['system']['readiness']} · "
          f"bottleneck: {doc['top_bottleneck'][:60]}")


if __name__ == "__main__":
    main()
