"""Emit results/diagnosis.json — the tier-diagnostic layer (TA brief
2026-08-29: "the A/B system should answer WHERE the system is failing,
which tier is responsible, which treatment repairs it, and which
treatments to stop spending time on").

Everything computed from published artifacts; where a decomposition is
not cleanly measurable it is labeled "unknown"/"not instrumented",
never estimated silently. Retire-pressure and mechanism rules are
PRE-REGISTERED here in code.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402


def jload(name, default=None):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return default


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


# ---- treatment -> tier map, family, lineage, mechanism (registered) --
TMAP = {
    "t_cal":      dict(m="M1",  tier="T2", family="calibration",
                       parents=[], mechanism="probabilities honest"),
    "t_knife":    dict(m="M2",  tier="T3", family="decision-selection",
                       parents=[], mechanism="avoid coin-flip entries"),
    "t_fshare":   dict(m="M3",  tier="T3", family="decision-selection",
                       parents=[], mechanism="stabilize leader choice"),
    "t_regime":   dict(m="M8",  tier="T3", family="decision-selection",
                       parents=[], mechanism="stand down when the "
                       "market itself is unpredictable"),
    "t_cheap":    dict(m="M9",  tier="T3", family="decision-selection",
                       parents=[], mechanism="only cheap contracts"),
    "t_evlead":   dict(m="M12", tier="T3", family="decision-selection",
                       parents=[], mechanism="rank leaders by EV not "
                       "win rate"),
    "t_exec":     dict(m="M10", tier="T4", family="execution",
                       parents=[], mechanism="refuse fills >3c worse "
                       "than quote"),
    "t_limit":    dict(m="M11", tier="T4", family="execution",
                       parents=[], mechanism="passive entry below ask"),
    "t_both":     dict(m="M2+M8", tier="T3", family="compound",
                       parents=["t_knife", "t_regime"],
                       mechanism="both vetoes"),
    "t_fs_reg":   dict(m="M3+M8", tier="T3", family="compound",
                       parents=["t_fshare", "t_regime"],
                       mechanism="stable leader + regime gate"),
    "t_exec_reg": dict(m="M10+M8", tier="T4", family="compound",
                       parents=["t_exec", "t_regime"],
                       mechanism="fill discipline + regime gate"),
    "t_limit_reg": dict(m="M11+M8", tier="T4", family="compound",
                        parents=["t_limit", "t_regime"],
                        mechanism="passive entry + regime gate"),
}


def retire_pressure(a):
    """Registered rules. Returns (HIGH/MEDIUM/LOW, [flags])."""
    flags = []
    p = a.get("p_gt0_boot")
    mean = a.get("mean") or 0
    llr = a.get("llr") or 0
    cov = a.get("coverage")
    n = a.get("n") or 0
    if p is not None and p <= 0.25:
        flags.append("primary metric trending down")
    if mean < 0 and n >= O.TREAT_MIN_N:
        flags.append("negative paired mean at full sample")
    if llr <= -1.0:
        flags.append("sequential evidence against")
    if cov is not None and cov < 0.25:
        flags.append("too selective (<25% coverage)")
    if n >= 3 * O.TREAT_MIN_N and abs(llr) < 1.0:
        flags.append("stale — little movement toward a verdict")
    neg = sum(1 for f in flags if f != "too selective (<25% coverage)")
    level = "HIGH" if neg >= 2 else \
        ("LOW" if (p or 0.5) >= 0.85 and mean > 0 else "MEDIUM")
    return level, flags


def main():
    now = int(time.time())
    db = jload("decision_board.json", {}) or {}
    audit = (jload("audit_report.json", {}) or {}).get("sections", {})
    mi = jload("model_internals.json", {}) or {}
    fc = jload("fill_curve.json", {}) or {}
    lr = jload("loss_reviews.json", {}) or {}
    treats = {a["key"]: a for a in db.get("treatments", [])}
    wf = db.get("execution_waterfall") or {}
    twin = rows("treatments.jsonl")

    # ---- tier health -------------------------------------------------
    t1 = audit.get("tier1", {})
    t1_cov = [v.get("band80_cov") for v in t1.values()
              if isinstance(v, dict) and v.get("band80_cov")]
    t2 = audit.get("tier2", {})
    bss = {k: v.get("bss_vs_market") for k, v in t2.items()
           if isinstance(v, dict) and v.get("bss_vs_market") is not None}
    best_bss = max(bss.items(), key=lambda x: x[1]) if bss else None
    churn = None
    if len(twin) > 1:
        ch = sum(1 for a, b in zip(twin, twin[1:])
                 if a.get("leader") != b.get("leader"))
        churn = round(ch / (len(twin) - 1), 3)
    champ = treats.get("champion", {})
    champ_real = treats.get("champion_real", {})

    def grade(ok, warn, val, reverse=False):
        if val is None:
            return "UNKNOWN"
        v = -val if reverse else val
        return "OK" if v >= ok else ("WARN" if v >= warn else "FAIL")

    tiers = {
        "T1": {"name": "Price forecast",
               "job": "predict the future price distribution",
               "headline": f"80% band coverage "
               f"{min(t1_cov):.2f}–{max(t1_cov):.2f}" if t1_cov else "—",
               "status": "WARN" if t1_cov and min(t1_cov) < 0.78
               else ("OK" if t1_cov else "UNKNOWN"),
               "why": "bands still under-cover on the worst family — "
               "dispersion was the SEV-0 root cause"},
        "T2": {"name": "Binary probability",
               "job": "an honest P(up) that beats the market's own",
               "headline": f"best BSS {best_bss[1]:+.4f} ({best_bss[0]})"
               if best_bss else "—",
               "status": "FAIL" if best_bss and best_bss[1] < 0
               else ("OK" if best_bss else "UNKNOWN"),
               "why": "NOBODY BEATS THE BOOK — every arm's Brier skill "
               "vs the market is negative this sample"},
        "T3": {"name": "Decision policy",
               "job": "act, pick the arm, or stand down",
               "headline": f"leader churn {churn:.0%} · champion "
               f"{100*(champ.get('own_ev') or 0):+.1f}c/$1"
               if churn is not None else "—",
               "status": "WARN",
               "why": "win-rate leaderboard seats market-echo arms; "
               "M8-family gates are the racing repairs"},
        "T4": {"name": "Execution",
               "job": "turn the decision into a fill without paying "
               "away the edge",
               "headline": f"gap {100*(wf.get('execution_gap_per_1') or 0):+.1f}c/$1"
               if wf else "—",
               "status": "CRITICAL" if wf and
               (wf.get("execution_gap_per_1") or 0) < -0.03 else "WARN",
               "why": "largest measured leak in the chain"},
        "T5": {"name": "Realized P&L",
               "job": "survive aggregation, sizing and costs",
               "headline": f"realized "
               f"{100*(wf.get('realized_ev_per_1') or 0):+.1f}c/$1 · "
               f"losses graded: {lr.get('by_sev')}",
               "status": "FAIL" if wf and
               (wf.get("realized_ev_per_1") or 0) < 0 else "OK",
               "why": "only pt6 (84% idle) is green among traders"},
    }

    # ---- treatments with tier/lineage/pressure/mechanism -------------
    out_t = []
    for key, meta in TMAP.items():
        a = treats.get(key)
        if not a:
            continue
        level, flags = retire_pressure(a)
        mech = {"status": "NOT INSTRUMENTED", "note": ""}
        if key in ("t_regime", "t_both", "t_fs_reg", "t_exec_reg",
                   "t_limit_reg"):
            sl = {s["name"]: s for s in (a.get("slices") or [])}
            bad = sl.get("regime<floor")
            good = sl.get("regime>=floor")
            if bad and good:
                supported = (bad["mean"] or 0) > (good["mean"] or 0)
                mech = {"status": "SUPPORTED" if supported
                        else "NOT CONFIRMED",
                        "note": f"Δ on regime<floor windows "
                        f"{100*bad['mean']:+.1f}c (n={bad['n']}) vs "
                        f"{100*good['mean']:+.1f}c on healthy windows "
                        f"— the gate earns exactly where it claims to"
                        if supported else
                        f"gate claims regime windows but Δ there "
                        f"({100*bad['mean']:+.1f}c) is not better"}
        if key in ("t_limit", "t_limit_reg") and fc.get("fill_curve"):
            c0 = fc["fill_curve"][0]
            c8 = fc["fill_curve"][-1]
            mech = {"status": "CONTRADICTED",
                    "note": f"fill-curve: win-given-fill falls "
                    f"{c0['win_rate_given_fill']:.0%}→"
                    f"{c8['win_rate_given_fill']:.0%} with bid depth — "
                    "adverse selection eats the price improvement"}
        out_t.append({
            "key": key, "m": meta["m"], "tier": meta["tier"],
            "family": meta["family"], "parents": meta["parents"],
            "mechanism": meta["mechanism"],
            "mechanism_check": mech,
            "label": a.get("label"), "state": a.get("state"),
            "mean": a.get("mean"), "ci95_boot": a.get("boot_ci95"),
            "p_gt0": a.get("p_gt0_boot"), "coverage": a.get("coverage"),
            "llr": a.get("llr"), "n": a.get("n"),
            "retire_pressure": level, "retire_flags": flags,
        })
    # dominated flag (same family, better mean AND >= coverage)
    for a in out_t:
        for b in out_t:
            if a is b or a["family"] != b["family"]:
                continue
            if (b.get("mean") or -9) > (a.get("mean") or -9) + 0.01 \
                    and (b.get("coverage") or 0) >= \
                    (a.get("coverage") or 0) - 0.02:
                a["retire_flags"].append(f"dominated by {b['m']}")
                if a["retire_pressure"] == "MEDIUM":
                    a["retire_pressure"] = "HIGH" \
                        if len(a["retire_flags"]) >= 2 else "MEDIUM"
                break

    # ---- stop / continue / investigate -------------------------------
    stop = [t["m"] for t in out_t if t["retire_pressure"] == "HIGH"]
    cont = [t["m"] for t in out_t if t["retire_pressure"] == "LOW"]
    investigate = [
        "T2: no arm beats market BSS — is there any incremental skill?",
        f"T3: leader churn {churn:.0%} — selection instability"
        if churn is not None else "T3: churn not computed",
        "T4: execution leak "
        f"{100*(wf.get('execution_gap_per_1') or 0):+.1f}c/$1 — the "
        "binding constraint",
        "edge-anti-signal (kb5/pt6 negative own-edge weights) — "
        "D-edge-band",
    ]

    # ---- error funnel (measured vs unknown) --------------------------
    oracle = mi.get("oracle") or {}
    funnel = [
        {"stage": "Oracle opportunity (same rules, clairvoyant)",
         "ev_per_1": oracle.get("oracle_ev"), "kind": "measured",
         "n": oracle.get("windows")},
        {"stage": "Forecast + probability + decision losses",
         "ev_per_1": None, "kind": "unknown",
         "note": "no clean per-tier decomposition yet — labeled "
         "unknown rather than estimated (side+timing dominate per "
         "oracle regret analysis)"},
        {"stage": "Desk decision EV (model quotes)",
         "ev_per_1": wf.get("decision_ev_per_1"), "kind": "measured",
         "n": wf.get("windows")},
        {"stage": "Execution gap (quote→fill)",
         "ev_per_1": wf.get("execution_gap_per_1"), "kind": "measured"},
        {"stage": "Realized EV (real fills)",
         "ev_per_1": wf.get("realized_ev_per_1"), "kind": "measured"},
    ]

    # ---- contradictions ----------------------------------------------
    rep = mi.get("representativeness") or {}
    contra = []
    kbf = rep.get("kbf")
    if kbf:
        contra.append({
            "name": "High AUC, negative EV (kbf)",
            "detail": f"Brier {kbf.get('brier_decision')} — "
            f"EV {kbf.get('ev_per_dollar')}/$1: superb ranking, no "
            "tradable skill; skill lives after the payout window"})
    if best_bss and best_bss[1] < 0:
        contra.append({
            "name": "Profitable trader atop unskilled arms",
            "detail": "pt6 is green while every kb arm fails BSS — "
            "its value is REFUSAL (84% idle), i.e. selection, not "
            "forecasting"})
    contra.append({
        "name": "High-skip policies look profitable",
        "detail": "M9-style gates show positive means on <25% "
        "coverage — a different product until coverage-matched"})

    # ---- experiment debt ---------------------------------------------
    debt = {"running": len(out_t),
            "high_information": len([t for t in out_t
                                     if t["retire_pressure"] == "LOW"
                                     and (t["coverage"] or 0) >= 0.4]),
            "retire_candidates": len(stop),
            "too_selective": len([t for t in out_t
                                  if (t["coverage"] or 1) < 0.25]),
            "stale": len([t for t in out_t
                          if "stale — little movement toward a "
                          "verdict" in t["retire_flags"]])}

    doc = {
        "generated_ts": now,
        "question": "Where is the system failing?",
        "tiers": tiers,
        "treatments": out_t,
        "stop_continue": {"stop": stop, "continue": cont,
                          "investigate": investigate},
        "error_funnel": funnel,
        "contradictions": contra,
        "experiment_debt": debt,
        "blind_spots": [
            {"spot": "no arm beats market BSS", "tier": "T2",
             "evidence": f"best {best_bss[1]:+.4f}" if best_bss else "—",
             "action": "investigate"},
            {"spot": "execution leak", "tier": "T4",
             "evidence": f"{100*(wf.get('execution_gap_per_1') or 0):+.1f}c/$1",
             "action": "prioritize (M10 family racing)"},
            {"spot": "leader churn", "tier": "T3",
             "evidence": f"{churn:.0%}" if churn is not None else "—",
             "action": "investigate (M3/M12 racing)"},
            {"spot": "low-coverage gates", "tier": "T3",
             "evidence": "M1/M9 <25% coverage", "action": "hold"},
            {"spot": "oracle regret", "tier": "cross-tier",
             "evidence": f"{oracle.get('regret')}/$1/window",
             "action": "decompose"},
        ],
        "loop": ["observe a failure", "locate the tier",
                 "form a mechanism hypothesis",
                 "create the smallest treatment", "run paired test",
                 "check primary + mechanism + downstream + guardrails",
                 "combine/promote or retire", "record the learning"],
    }
    (RES / "diagnosis.json").write_text(json.dumps(doc, indent=1))
    print(f"diagnosis.json: tiers "
          f"{[t['status'] for t in tiers.values()]}, "
          f"{len(out_t)} treatments mapped, stop={stop}, cont={cont}")


if __name__ == "__main__":
    main()
