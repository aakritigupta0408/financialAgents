"""Emit results/program.json — the ONE experiment registry both the
Program page (lifecycle+causality) and the Analysis page (statistical
evidence) render (TA brief 2026-08-29 §32: "two views of the same
experiment graph").

What this adds beyond decision_board/diagnosis:
  * the full experiment entity per treatment: problem, hypothesis,
    tier path, parents/children, attention reasons, a shared
    recommendation vocabulary (PRIORITY / CONTINUE / HOLD / DIAGNOSE /
    RETIRE / REDESIGN / INVALID / PROMOTE — §19), lifecycle vs
    analysis-recommendation kept SEPARATE (§33: only the owner moves
    lifecycle);
  * PAIRED incremental branch analysis: for each combo, per-window
    d_i = EV_combo,i − EV_parent,i on the SAME windows from
    treatments.jsonl — mean, se, seeded-bootstrap CI, P(inc>0). The
    TA could only gesture at point estimates; the paired log lets us
    do it properly. Cross-basis pairs (real vs model) are refused,
    not fudged;
  * decision inbox (only items where action is useful);
  * system-knowledge statements (supported / developing / under
    pressure / rejected implementation / unknown), each with its
    evidence pointer;
  * retired-entity tombstones (kbf, kb6, M7) — frozen recorded facts.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402

BOOT_N, SEED = 2000, 20260830
BASIS = {"champion_real": "real", "t_exec": "real", "t_exec_reg": "real"}

META = {
    "t_cal": dict(m="M1", name="Calibrated gate", path="T2→T3",
                  problem="probability calibration may be hurting "
                  "decisions",
                  hypothesis="Platt-corrected confidence gates better",
                  parents=[]),
    "t_knife": dict(m="M2", name="Knife-edge veto", path="T3→T5",
                    problem="near-50/50 windows are ambiguous",
                    hypothesis="skipping coin-flips improves EV",
                    parents=[]),
    "t_fshare": dict(m="M3", name="Fixed-Share leader", path="T3→T5",
                     problem="leader churn ~38% — selection unstable",
                     hypothesis="tracking the best SEQUENCE of arms "
                     "beats the win-rate leaderboard",
                     parents=[]),
    "t_regime": dict(m="M8", name="Regime gate", path="T2→T3→T5",
                     problem="the system loses in low-predictability "
                     "regimes",
                     hypothesis="standing down when trailing market "
                     "accuracy < floor improves realized EV",
                     parents=[]),
    "t_cheap": dict(m="M9", name="Underdog (cheap bids)", path="T3→T5",
                    problem="cheap asymmetric contracts may carry "
                    "positive EV",
                    hypothesis="only entering cheap prices wins",
                    parents=[]),
    "t_evlead": dict(m="M12", name="EV-ranked leader", path="T3→T5",
                     problem="win-rate leaderboard optimizes the "
                     "wrong objective (seats market echoes)",
                     hypothesis="ranking leaders by EV fixes selection",
                     parents=[]),
    "t_exec": dict(m="M10", name="Execution guard", path="T3→T4→T5",
                   problem="decision value disappears between quote "
                   "and fill (−5.3c/$1 measured)",
                   hypothesis="refusing fills >3c worse than quote "
                   "preserves decision value",
                   parents=[]),
    "t_limit": dict(m="M11", name="Maker limit", path="T4→T5",
                    problem="crossing the spread pays the book",
                    hypothesis="passive entry below the ask keeps the "
                    "spread",
                    parents=[]),
    "t_both": dict(m="M2+M8", name="Knife × Regime", path="T3→T5",
                   problem="are the two vetoes redundant?",
                   hypothesis="does regime filtering make knife-edge "
                   "filtering redundant?",
                   parents=["t_knife", "t_regime"]),
    "t_fs_reg": dict(m="M3+M8", name="Fixed-Share × Regime",
                     path="T3→T5",
                     problem="does M8 remove the need for leader "
                     "stabilization?",
                     hypothesis="stable leader + regime gate stack",
                     parents=["t_fshare", "t_regime"]),
    "t_exec_reg": dict(m="M10+M8", name="Exec guard × Regime",
                       path="T2→T3→T4→T5",
                       problem="do regime and execution repair "
                       "independent failure modes?",
                       hypothesis="execution protection preserves "
                       "value best after bad regimes are removed",
                       parents=["t_exec", "t_regime"]),
    "t_limit_reg": dict(m="M11+M8", name="Maker × Regime",
                        path="T4→T5",
                        problem="is any maker benefit inherited "
                        "entirely from M8?",
                        hypothesis="passive entry works once bad "
                        "regimes are filtered",
                        parents=["t_limit", "t_regime"]),
    "t_edgeband": dict(m="M13", name="Edge band", path="T2→T3→T5",
                       problem="stated edge is an anti-signal at the "
                       "extremes (kb5 −0.096, pt6 −0.059 weights)",
                       hypothesis="enter only when claimed edge sits "
                       "in [2c, 12c] — enough to pay costs, not so "
                       "much that the model is probably wrong",
                       parents=[]),
}

# Lifecycle rulings — OWNER decisions only (2026-08-29, "all 4"):
# the analysis layer recommends; the human moves lifecycle (§33).
# GREAT SIMPLIFICATION (PM 2026-08-29, docs/RETIREMENT_MANIFEST.md):
# lifecycle vocabulary is now the 6-state machine — CONTROL /
# TREATMENT / SHADOW / QUALIFIED / RETIRED / ARCHIVED. ONE legacy
# experiment remains (M10 control vs M10+M8 treatment); retired
# treatments stopped consuming runtime (no new ev rows) and their
# paired history froze as evidence.
LIFECYCLE = {
    "t_regime": "SHADOW (diagnostic) — standalone question folded "
                "into M10+M8; single-factor evidence stream kept",
    "t_exec": "CONTROL — the legacy experiment incumbent",
    "t_exec_reg": "TREATMENT — the one legacy challenger (does "
                  "regime filtering add value beyond the exec guard?)",
    "t_limit_reg": "ARCHIVED (diagnostic) — causal-explanation "
                   "purpose served (PM §41); no promotion path",
    "t_limit": "RETIRED (standalone) — history frozen 08-29; no "
               "promotion path",
    "t_fshare": "RETIRED (branch) — history frozen 08-29",
    "t_fs_reg": "RETIRED (branch) — history frozen 08-29",
    "t_evlead": "RETIRED (implementation) — objective right, "
                "implementation losing; a redesign would be a NEW "
                "experiment",
    "t_edgeband": "SHADOW / HOLD — anti-signal research, collecting",
    # Phase-5 lean cleanup (master directive §5, 2026-08-29): max 3
    # active treatments per tier; dominated/stale/thin branches close.
    "t_knife": "RETIRED (lean) — dominated by M8, stale at 190 "
               "windows (+0.4c, LLR −0.07)",
    "t_cheap": "ARCHIVED — hypothesis retained inside M13's band "
               "floor; 15% coverage made it a different product",
    "t_both": "RETIRED (lean) — adds nothing over M8 (−3.7c paired "
              "increment, P13%); dominated by M10+M8",
    "t_cal": "RETIRED (lean) — the M1 gate at 13% coverage; the "
             "calibrator itself lives on as the shadow drift "
             "instrument (M1-v3), never a decision input",
}
# LEAN MACHINE (owner 2026-08-29, executive call): ONE control
# (kb2 caller) vs ONE treatment (kb9 caller) on the early Oracle
# product, judged on ML fundamentals (selective acc, Brier, log loss,
# stability). Every M-treatment below is FROZEN — evidence kept.


def jload(name, default=None):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return default


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


def paired_stats(ds):
    n = len(ds)
    if n < 2:
        return None
    m = sum(ds) / n
    var = sum((x - m) ** 2 for x in ds) / (n - 1)
    se = math.sqrt(var / n)
    rng = random.Random(SEED)
    boots = sorted(sum(ds[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(BOOT_N))
    return {"n": n, "mean": round(m, 5), "se": round(se, 5),
            "boot_ci95": [round(boots[int(.025 * BOOT_N)], 5),
                          round(boots[int(.975 * BOOT_N)], 5)],
            "p_gt0_boot": round(sum(1 for b in boots if b > 0)
                                / BOOT_N, 3)}


def recommend(t, inc_map):
    """Shared vocabulary (§19). Analysis recommendation ONLY — the
    lifecycle state stays wherever governance put it (§33)."""
    why = []
    press = t.get("retire_pressure")
    mech = (t.get("mechanism_check") or {}).get("status")
    p = t.get("p_gt0") or 0.5
    cov = t.get("coverage")
    key = t["key"]
    if t.get("state") == "PROMOTE":
        return "PROMOTE", ["SPRT boundary crossed with guardrails"]
    if cov is not None and cov < 0.25:
        return "HOLD", [f"coverage {cov:.0%} — not comparable at "
                        "deployment scale"]
    if key == "t_limit_reg":
        return "DIAGNOSE", ["conditional effect: standalone M11 is "
                            "negative but the M8-filtered variant is "
                            "positive — investigate, don't extend"]
    if mech == "CONTRADICTED" and press == "HIGH":
        return "RETIRE", ["mechanism contradicted by the fill curve",
                          "primary metric negative"]
    if key == "t_evlead" and press == "HIGH":
        return "REDESIGN", ["hypothesis (EV-objective selection) "
                            "remains sound; this implementation "
                            "underperforms"]
    if press == "HIGH":
        inc = inc_map.get(key)
        if inc and (inc.get("p_gt0_boot") or 1) < 0.2:
            why.append("combo adds nothing over its parent")
        why.append("retire-pressure HIGH: " +
                   "; ".join(t.get("retire_flags") or []))
        return "RETIRE", why
    if press == "LOW" and p >= 0.95:
        return "PRIORITY", [f"P(Δ>0) {p:.0%} — spend windows here"]
    if press == "LOW":
        return "CONTINUE", ["evidence developing normally"]
    return "CONTINUE", ["collecting"]


def main():
    now = int(time.time())
    diag = jload("diagnosis.json", {}) or {}
    dts = {t["key"]: t for t in diag.get("treatments", [])}
    twin = rows("treatments.jsonl")

    # paired incremental: combo − each parent, same windows, same basis
    inc_out, inc_by_combo = [], {}
    for key, meta in META.items():
        for par in meta["parents"]:
            b_c = BASIS.get(key, "model")
            b_p = BASIS.get(par, "model")
            if b_c != b_p:
                inc_out.append({"combo": META[key]["m"],
                                "parent": META[par]["m"],
                                "valid": False,
                                "note": "cross-basis (real vs model) — "
                                "refused, not fudged"})
                continue
            ds = [((r["ev"].get(key) or 0.0)
                   - (r["ev"].get(par) or 0.0))
                  for r in twin if r.get("ev")
                  and (key in r["ev"] or par in r["ev"])]
            st = paired_stats(ds)
            if st:
                rec = {"combo": META[key]["m"],
                       "parent": META[par]["m"], "valid": True, **st,
                       "read": ("adds value" if st["p_gt0_boot"] >= .8
                                else "no evidence of useful addition"
                                if st["p_gt0_boot"] <= .35
                                else "unclear")}
                inc_out.append(rec)
                if META[par]["m"].endswith("M8") or par == "t_regime":
                    inc_by_combo[key] = st
                inc_by_combo.setdefault(key, st)

    # experiment entities
    exps = []
    for key, meta in META.items():
        t = dts.get(key, {})
        rec, why = recommend({**t, "key": key}, inc_by_combo)
        attention = []
        p = t.get("p_gt0")
        if p is not None and p <= 0.15:
            attention.append(f"P(Δ>0) {p:.0%} — approaching reject")
        if p is not None and p >= 0.95:
            attention.append(f"strongest positive evidence "
                             f"(P(Δ>0) {p:.0%})")
        cov = t.get("coverage")
        if cov is not None and cov < 0.25:
            attention.append(f"coverage {cov:.0%}")
        mech = (t.get("mechanism_check") or {})
        if mech.get("status") == "CONTRADICTED":
            attention.append("mechanism contradicted")
        children = [META[k]["m"] for k, m2 in META.items()
                    if key in m2["parents"]]
        exps.append({
            "id": meta["m"], "key": key, "name": meta["name"],
            "type": "EXPERIMENT", "tier_path": meta["path"],
            "family": t.get("family"),
            "basis": BASIS.get(key, "model"),
            "problem": meta["problem"],
            "hypothesis": meta["hypothesis"],
            "mechanism": t.get("mechanism"),
            "mechanism_check": t.get("mechanism_check"),
            "parents": [META[p_]["m"] for p_ in meta["parents"]],
            "children": children,
            "effect": t.get("mean"), "ci95_boot": t.get("ci95_boot"),
            "p_gt0": t.get("p_gt0"), "coverage": cov,
            "llr": t.get("llr"), "n": t.get("n"),
            "lifecycle": LIFECYCLE.get(key, "LIVE"),
            "analysis_recommendation": rec,
            "recommendation_why": why,
            "attention": attention,
            "retire_pressure": t.get("retire_pressure"),
            "incremental_vs_parent": inc_by_combo.get(key),
        })

    # decision inbox — only actionable items. The four 08-29 asks were
    # ALL RATIFIED by the owner ("all 4"): M11 retired standalone,
    # M3 branch retired, M12 sent to redesign, M13 Edge Band created.
    inbox = [
        {"kind": "RESOLVED", "id": "M11 · M3 · M12 · M13",
         "ask": "owner ratified all four 08-29 — M11 retired "
         "(standalone), M3 branch retired, M12 in redesign, M13 Edge "
         "Band live and collecting"},
    ]

    knowledge = [
        {"status": "supported",
         "claim": "execution quality materially affects realized "
         "economics",
         "evidence": "measured −5.3c/$1 quote→fill gap; M10 family "
         "positive"},
        {"status": "developing",
         "claim": "regime filtering improves decision quality",
         "evidence": "M8 P(Δ>0)≈97%, mechanism supported (+20c on "
         "claimed windows), boundary not crossed"},
        {"status": "under pressure",
         "claim": "Fixed-Share leader selection helps",
         "evidence": "M3 and M3+M8 negative"},
        {"status": "rejected implementation",
         "claim": "standalone passive maker entry preserves edge",
         "evidence": "M11 negative; win-given-fill falls 68%→47% "
         "with bid depth"},
        {"status": "unknown",
         "claim": "binary predictors contain skill beyond the market "
         "price",
         "evidence": "no arm with BSS>0 this sample (best −0.0016)"},
    ]

    tombstones = [
        {"id": "kbf", "type": "MODEL", "retired": "2026-08-25",
         "looked_good": "best Brier (0.097), AUC 0.93",
         "failed": "−$0.41/$1 at real costs",
         "root_cause": "forecast at T−3 min — value not monetizable",
         "lesson": "accuracy without decision-time monetizability is "
         "not economic skill",
         "do_not_revisit_unless": "decision timing changes"},
        {"id": "kb6", "type": "MODEL", "retired": "2026-08-25",
         "looked_good": "UP-recall 63%",
         "failed": "coverage 37%, calibration slope 0.33 ≈ noise",
         "lesson": "classification recall alone is not skill"},
        {"id": "M11 (standalone)", "type": "EXPERIMENT",
         "retired": "2026-08-29",
         "looked_good": "paying less than the ask sounds free",
         "failed": "−4.7c/$1 paired; win-given-fill falls 68%→47% "
         "with bid depth (adverse selection measured)",
         "root_cause": "a resting bid fills preferentially when the "
         "market has just repriced the call down (Glosten–Milgrom)",
         "lesson": "passive entry alone does not repair execution; "
         "M11+M8 lives on only as a conditional-effect study",
         "do_not_revisit_unless": "fill/queue model changes"},
        {"id": "M3 branch (Fixed-Share)", "type": "EXPERIMENT",
         "retired": "2026-08-29",
         "looked_good": "Herbster–Warmuth tracking beats frozen "
         "winners in theory; leader churn was 38%",
         "failed": "M3 −2.2c standalone; M3+M8 −7.1c vs M8 alone "
         "(P(inc>0)=6%) — it actively interfered with the regime gate",
         "lesson": "stabilizing the leader overrode exactly the arm "
         "switches the regime filter needed",
         "do_not_revisit_unless": "a selection objective change "
         "(M12 redesign) reopens the question"},
        {"id": "M7", "type": "EXPERIMENT", "retired": "2026-08-28",
         "looked_good": "a 'cursed hour' with 64% errors",
         "failed": "p=0.60 under multiplicity — 24 hours searched, "
         "~2 false positives expected, 2 found",
         "lesson": "a plausible slice is not a pre-registered "
         "hypothesis"},
    ]

    # Research Manager block (M2 / Research Command card 3) — the
    # ranked "what should the team work on next" queue, machine-
    # computed from live artifacts. A Research Manager agent will own
    # this later; the page consumes this structure from day one.
    a3 = jload("a3_live.json", {}) or {}
    fwd = a3.get("forward") or {}
    n_elig = fwd.get("eligible") or 0
    mk = [r.get("markout_10s") for r in (a3.get("recent_settled") or [])
          if r.get("state") == "FILLED"
          and isinstance(r.get("markout_10s"), (int, float))]
    mmk = sum(mk) / len(mk) if mk else None
    def _active(e):
        w = str(e["lifecycle"]).split(" ")[0]
        return w in ("CONTROL", "TREATMENT", "LIVE")
    best = max((e for e in exps if _active(e)
                and e.get("p_gt0") is not None),
               key=lambda e: e["p_gt0"], default=None)
    queue = [
        {"rank": 1, "title": "Accumulate A3 forward evidence",
         "why": "biggest current unknown",
         "bottleneck": f"n={n_elig} eligible (compare gate 25, "
                       "decision gate 50)",
         "action": "no engineering — market clock only"},
        {"rank": 2, "title": "Validate A3 execution quality",
         "why": (f"mean 10s markout {mmk:+.1f}c on filled entries"
                 if mmk is not None else "no markout sample yet"),
         "metric": "markout_h / ECR",
         "action": "collect"},
    ]
    if best:
        queue.append(
            {"rank": 3, "title": f"Continue {best['id']}",
             "why": "strongest existing positive legacy branch "
                    f"(P(Δ>0) {best['p_gt0']:.0%})",
             "metric": "incremental paired PnL",
             "action": "continue"})
    queue.append(
        {"rank": 4, "title": "A2 early divergence",
         "why": "+14pp exploratory residual, early windows only (n=28)",
         "metric": "early-window divergence hit rate",
         "action": "shadow collection"})
    research_manager = {
        "what_changed": [
            f"A3 forward: {n_elig} eligible window"
            + ("s" if n_elig != 1 else "") + " settled",
            (f"{best['id']} remains strongest legacy branch "
             f"(P(Δ>0) {best['p_gt0']:.0%})" if best
             else "no live legacy branch"),
            "no new qualified alpha (best BSS ≤ 0 this sample)",
        ],
        "queue": queue,
        "blocked": [
            {"item": "Value-of-Wait ML",
             "reason": "A3 has not demonstrated a timing loss yet "
                       "(needs poor ECR at positive Δ per the frozen "
                       "decision tree)"},
            {"item": "Sizing",
             "reason": "positive durable alpha not established"},
            {"item": "Passive execution",
             "reason": "standalone maker already failed (M11)"},
            {"item": "Champion simplification (CHAMPION-vLean)",
             "reason": "would mutate the frozen control — the "
                       "leader pool (kb3/kb4/kb7/kb8) is a "
                       "CONTROL_DEPENDENCY; requires a registered "
                       "CHAMPION-vCurrent vs CHAMPION-vLean "
                       "experiment, and only after current research "
                       "resolves (PM 08-30)"},
        ],
    }

    # results/research_queue.json — the canonical prioritization
    # artifact (PM 08-30): the future Research Manager agent MAINTAINS
    # this rather than inventing priorities from scratch.
    rq = {"generated_ts": now,
          "queue": [
              {"priority": f"P{i}",
               "research_question": q["title"],
               "bottleneck": q.get("bottleneck") or q.get("metric"),
               "evidence": q["why"],
               "estimated_impact": "unknown — evidence first",
               "state": "COLLECTING" if i == 0 else "ACTIVE",
               "blocked_by": ("evidence N (compare gate 25, "
                              "decision gate 50)") if i == 0 else None,
               "owner": "market clock" if i == 0
               else "research_manager",
               "next_gate": "n>=10 arms the pre-registered "
                            "failure watch" if i == 0 else None}
              for i, q in enumerate(queue)],
          "blocked": research_manager["blocked"],
          "provenance": "emit_program.py research_manager block"}
    (RES / "research_queue.json").write_text(json.dumps(rq, indent=1))

    doc = {"generated_ts": now,
           "counts": {
               "live": len([e for e in exps
                            if not str(e["lifecycle"]).startswith(
                                ("RETIRED", "ARCHIVED"))]),
               "retire_candidates": len([e for e in exps if
                                         e["analysis_recommendation"]
                                         == "RETIRE"]),
               "priority": [e["id"] for e in exps if
                            e["analysis_recommendation"] == "PRIORITY"],
               "decisions_waiting": len(inbox)},
           "governance_note": "analysis_recommendation never changes "
           "lifecycle — only the owner does (§33)",
           "experiments": exps,
           "incremental": inc_out,
           "decision_inbox": inbox,
           "research_manager": research_manager,
           "system_knowledge": knowledge,
           "tombstones": tombstones}
    (RES / "program.json").write_text(json.dumps(doc, indent=1))
    print(f"program.json: {len(exps)} experiments, "
          f"{len(inc_out)} incremental pairs, "
          f"inbox={len(inbox)}")
    for i in inc_out:
        if i.get("valid"):
            print(f"  {i['combo']} − {i['parent']}: "
                  f"{100*i['mean']:+.2f}c (P>0 {i['p_gt0_boot']:.0%}) "
                  f"→ {i['read']}")
        else:
            print(f"  {i['combo']} − {i['parent']}: {i['note']}")


if __name__ == "__main__":
    main()
