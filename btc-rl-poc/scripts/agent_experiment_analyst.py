"""M5.2 — Experiment Analyst (script-backed structured agent).

Maintains results/experiment_analysis.json from the canonical A3
artifact and submits ONE structured conclusion through the decision
firewall. It may not suggest parameter changes before the registered
gate — the prohibited_interpretation field is part of its contract,
and its firewall class is DIAGNOSE (never RESEARCH_POLICY_CHANGE).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import agent_firewall as fw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    age = fw.stale("a3_live.json", 3600)
    if age is not None:
        fw.submit(agent="experiment_analyst", action_class="OBSERVE",
                  finding=f"STALE_INPUT — a3_live.json is "
                          f"{age:.0f}s old; refusing to reason from "
                          "old state",
                  recommendation="no research recommendation until "
                                 "the artifact refreshes",
                  evidence=["a3_live.json mtime"])
        print("experiment_analyst: STALE_INPUT — no analysis")
        return
    try:
        a3 = json.loads((RES / "a3_live.json").read_text())
    except Exception:
        print("experiment_analyst: a3_live.json unavailable — no run")
        return
    f = a3.get("forward") or {}
    dec = f.get("decomposition") or {}
    watch = a3.get("failure_watch") or {}
    ti = watch.get("thesis_integrity") or {}
    n = f.get("eligible") or 0
    classes = dec.get("econ_classes") or {}
    firing = [k for k, v in watch.items()
              if k != "note" and (v is True or
                                  (isinstance(v, dict) and v.get("flag")))]
    analysis = {
        "generated_ts": int(time.time()),
        "experiment": "A3-v1.1",
        "n_eligible": n,
        "paired_delta_per_eligible": f.get("incremental_per_eligible"),
        "decomposition": dec,
        "econ_class_distribution": {k: v.get("n")
                                    for k, v in classes.items()},
        "thesis_integrity_gap": {
            "fill_win_rate": ti.get("a3_fill_win"),
            "eligible_win_rate": ti.get("eligible_win"),
            "interpretation": "fills land disproportionately on "
            "windows that go on to lose — dips are partly informative "
            "repricing, not only overreaction"},
        "watch_flags_firing": firing,
        "loss_mechanisms": [
            {"term": "timeout_control_pnl",
             "value": dec.get("timeout_control_pnl"),
             "mechanism": "opportunity cost — winners that never "
                          "offered the dip"},
            {"term": "thesis_integrity_gap",
             "value": (round(ti["eligible_win"] - ti["a3_fill_win"], 3)
                       if ti.get("eligible_win") is not None
                       and ti.get("a3_fill_win") is not None else None),
             "mechanism": "fill-conditioned thesis degradation — "
                          "adverse selection of the dip"},
        ],
        "state": "WATCH" if firing else "COLLECTING",
        "recommended_action": "COLLECT",
        "prohibited_interpretation":
            "do not infer an optimal threshold from the current "
            "sample; a smaller dip could reduce miss cost while "
            "worsening adverse-selection exposure — only the "
            "registered decision gate may adjudicate",
    }
    (RES / "experiment_analysis.json").write_text(
        json.dumps(analysis, indent=1))
    row = fw.submit(
        agent="experiment_analyst", action_class="DIAGNOSE",
        finding=(f"A3 (n={n}) currently loses through BOTH "
                 f"opportunity cost ({dec.get('timeout_control_pnl')}"
                 " control PnL on timed-out winners) and "
                 "fill-conditioned thesis degradation (fill win "
                 f"{ti.get('a3_fill_win')} vs eligible "
                 f"{ti.get('eligible_win')})"),
        recommendation="COLLECT — no parameter response before the "
                       "registered gate",
        evidence=["a3_live.json .forward.decomposition",
                  "a3_live.json .failure_watch"],
        n=n, effect=f.get("incremental_per_eligible"))
    print(f"experiment_analyst: n={n} state={analysis['state']} "
          f"flags={len(firing)} → {row['status']}")


if __name__ == "__main__":
    main()
