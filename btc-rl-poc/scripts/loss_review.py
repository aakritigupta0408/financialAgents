"""Automated loss review — every losing trade gets an RCA and the
trader gets a response (owner directive 2026-08-29).

Design, pre-registered here:
  * Every settled LOSING row in every trader ledger produces exactly
    one review record (idempotent on (trader, ticker)) with a factor
    attribution computed from the row + its window's treatment record:
      signal      — the leader's side was simply wrong; how confident?
      price       — break-even paid vs the 50c coin-flip line
      regime      — trailing decision-time market accuracy at entry
      sizing      — stake fraction vs the stated-edge Kelly fraction
      execution   — pt7/pt8 only: a filled resting bid that lost
                    (the adverse-selection signature)
  * GRADING (owner's six-level ladder, 2026-08-29 — every losing
    trade gets a SEV class; a loss inside pre-registered risk is the
    cost of being in the game and owes only a response):
      SEV-0  INTEGRITY   — the row itself is impossible (win flag vs
                           pnl sign mismatch, settle inconsistency):
                           results can't be trusted; pages the wall.
      SEV-1  POLICY_BREACH — the row violates its own trader's
                           pre-registered rule (gate, cap, entry
                           ceiling); a bug, not variance.
      SEV-2  CONFIDENT_WRONG — p_arm >= 0.77 and lost: calibration
                           exposure, feeds the drift instrument.
      SEV-3  OVERSIZED   — loss ordinary, size not: stake > 2x
                           stated-edge Kelly (the Gambler SEV-1
                           incident class, now per-row).
      SEV-4  BAD_CONTEXT — knife-edge entry (within 10c of the
                           coin-flip) or regime below floor: known
                           mitigations (M2/M8) are racing to own it.
      SEV-5  EXPECTED_VARIANCE — ordinary loss inside the rules;
                           coach response only, no action owed.
  * The RESPONSE to the trader is a coach note derived from the
    dominant factor, and the aggregate per-trader summary proposes a
    system improvement ONLY as a treatment candidate — the
    champion/challenger law still owns all promotions.

ML-migration hook: the aggregated factors are exactly the labels the
bid-pricing / pacing regressions train on (see emit_fill_curve.py).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402

TRADERS = {
    "pt":  ("Follower",    {"cap_frac": O.PT_FRAC, "tau": 0.62}),
    "pt2": ("Ladder",      {"cap_frac": O.PT_FRAC, "tau": 0.62}),
    "pt3": ("Disciplined", {"cap_frac": O.PT_FRAC, "tau": O.PT3_TAU}),
    "pt4": ("Gambler",     {"cap_frac": O.PT4_FRAC, "tau": O.PT4_TAU,
                            "min_edge_c": O.PT4_MIN_EDGE_C}),
    "pt5": ("Saver",       {"cap_frac": O.PT5_FRAC, "tau": 0.62}),
    "pt6": ("MLE",         {"cap_frac": 0.5, "tau": None}),
    "pt7": ("Patient",     {"cap_frac": O.PT_FRAC, "tau": 0.62,
                            "limit": True}),
    "pt8": ("Ideal",       {"cap_frac": O.PT8_KELLY_CAP, "tau": 0.62,
                            "limit": True}),
}

COACH = {
    "signal": "the leader's read was wrong — nothing you control; "
              "your rule executed as registered",
    "price": "you paid a break-even the window couldn't clear — the "
             "fill-curve model (bid pricing) is the treatment path",
    "regime": "the market itself was guessing when you entered — the "
              "M8 regime gate is racing to own exactly this",
    "sizing": "the loss was ordinary; its SIZE wasn't — stake sat "
              "above stated-edge Kelly",
    "execution": "your resting bid was picked off — filled because "
                 "the market had just repriced you down "
                 "(Glosten-Milgrom); pt8's recheck is the defense",
}


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


def kelly_frac(p, cost_c):
    """Stated-edge Kelly for a binary at total cost per contract."""
    if not p or cost_c <= 0 or cost_c >= 100:
        return None
    b = (100.0 - cost_c) / cost_c            # net odds
    f = (p * b - (1 - p)) / b
    return max(0.0, f)


def review_one(key, label, cfg, t, treat):
    p = t.get("p_arm")
    contracts = t.get("contracts") or 0
    ask = t.get("ask_c")
    fee = t.get("fee_c") or 0
    cost = (ask + fee / max(1, contracts)) if ask is not None else None
    bank_before = (t.get("bankroll_c") or 0) + (t.get("stake_c") or 0)
    frac = (t["stake_c"] / bank_before) if bank_before > 0 else None
    kf = kelly_frac(p, cost) if cost else None
    regime = (treat or {}).get("regime_acc")

    factors, breaches = [], []
    if p is not None:
        factors.append(("signal", round(float(p), 3)))
    if cost is not None and abs(cost - 50) < 10:
        factors.append(("price", round(cost, 1)))
    if regime is not None and regime < O.REGIME_FLOOR:
        factors.append(("regime", regime))
    if frac is not None and kf is not None and kf > 0 \
            and frac > 2 * kf:
        factors.append(("sizing", round(frac / kf, 1)))
    if cfg.get("limit") and not t.get("skipped"):
        factors.append(("execution", 1))

    # policy-breach checks — VERSION-AWARE: a row is judged against
    # the rule in force WHEN IT WAS MADE, never today's rule (first
    # run graded 48 false SEV-1s by judging Gambler-v1 rows against
    # the v2 gate and old Saver rows against the cut cap).
    made = t.get("made_ts") or 0
    tau = cfg.get("tau")
    cap = cfg["cap_frac"]
    if key == "pt4" and made < O.PT4_RESET_TS:
        tau = None                       # v1 had no confidence gate
    if key == "pt5" and made < O.PT4_RESET_TS:
        cap = 0.25                       # pre-08-26 registered stake
        # (cutover epoch proxied by the same-day PT4 reset — both
        # policy changes were ratified 2026-08-26, DECISIONS.md)
    if tau and p is not None and p < tau - 1e-9:
        breaches.append(f"entered below own tau {tau}")
    if ask is not None and ask > 85:
        breaches.append("entry above 85c ceiling")
    if frac is not None and frac > cap * 1.10 + 1e-9:
        breaches.append(f"stake frac {frac:.2f} > cap {cap:.2f}")

    # integrity check first: an impossible row outranks everything
    integrity_bad = (t.get("win") == 1 and (t.get("pnl_c") or 0) < 0
                     and t.get("contracts"))
    oversized = (frac is not None and kf is not None and kf > 0
                 and frac > 2 * kf)
    bad_ctx = ((cost is not None and abs(cost - 50) < 10)
               or (regime is not None and regime < O.REGIME_FLOOR))
    if integrity_bad:
        sev, grade = 0, "INTEGRITY"
    elif breaches:
        sev, grade = 1, "POLICY_BREACH"
    elif p is not None and p >= 0.77:
        sev, grade = 2, "CONFIDENT_WRONG"
    elif oversized:
        sev, grade = 3, "OVERSIZED"
    elif bad_ctx:
        sev, grade = 4, "BAD_CONTEXT"
    else:
        sev, grade = 5, "EXPECTED_VARIANCE"

    dom = max(factors, key=lambda f: {"sizing": 4, "execution": 3,
                                      "price": 2, "regime": 2,
                                      "signal": 1}[f[0]])[0] \
        if factors else "signal"
    return {
        "trader": key, "label": label, "ticker": t["ticker"],
        "close_ts": t.get("close_ts"), "pnl_c": t.get("pnl_c"),
        "stake_c": t.get("stake_c"), "p_arm": p,
        "cost_c": round(cost, 1) if cost is not None else None,
        "regime_acc": regime,
        "stake_frac": round(frac, 3) if frac is not None else None,
        "kelly_frac": round(kf, 3) if kf is not None else None,
        "factors": [f[0] for f in factors],
        "dominant": dom, "sev": sev, "grade": grade,
        "breaches": breaches,
        "response": COACH[dom],
    }


def main():
    now = int(time.time())
    treats = {r["ticker"]: r for r in rows("treatments.jsonl")}
    log_p = RES / "loss_reviews.jsonl"
    seen = {(r["trader"], r["ticker"])
            for r in rows("loss_reviews.jsonl")}
    new = []
    for key, (label, cfg) in TRADERS.items():
        for t in rows(f"{key}_trades.jsonl"
                      if key != "pt" else "pt_trades.jsonl"):
            if t.get("actual") is None or t.get("skipped"):
                continue
            if (t.get("pnl_c") or 0) >= 0 or (key, t["ticker"]) in seen:
                continue
            new.append(review_one(key, label, cfg, t,
                                  treats.get(t["ticker"])))
    if new:
        with log_p.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")

    # per-trader coach summary
    allr = rows("loss_reviews.jsonl")
    summary = {}
    for r in allr:
        s = summary.setdefault(r["trader"], {
            "label": r["label"], "losses": 0, "lost_c": 0,
            "by_grade": {}, "by_dominant": {}, "breach_rows": 0})
        s["losses"] += 1
        s["lost_c"] += -(r["pnl_c"] or 0)
        sk = f"SEV-{r.get('sev', 5)}"
        s.setdefault("by_sev", {})
        s["by_sev"][sk] = s["by_sev"].get(sk, 0) + 1
        s["by_grade"][r["grade"]] = s["by_grade"].get(r["grade"], 0) + 1
        s["by_dominant"][r["dominant"]] = \
            s["by_dominant"].get(r["dominant"], 0) + 1
        s["breach_rows"] += 1 if r["breaches"] else 0
    for s in summary.values():
        dom = max(s["by_dominant"], key=s["by_dominant"].get) \
            if s["by_dominant"] else None
        s["coach_note"] = COACH.get(dom, "")
        s["proposed_treatment_path"] = {
            "price": "M13 learned limit pricing (fill_curve.json)",
            "regime": "M8 regime gate (racing)",
            "sizing": "Kelly-consistent sizing treatment",
            "execution": "pt8 fill-time recheck (racing as M11 twin)",
            "signal": "better arms — tier-1/2 work, not trader work",
        }.get(dom)
    breaches = sum(s["breach_rows"] for s in summary.values())
    by_sev_all = {}
    for r in allr:
        k = f"SEV-{r.get('sev', 5)}"
        by_sev_all[k] = by_sev_all.get(k, 0) + 1
    doc = {"generated_ts": now, "reviews": len(allr),
           "new_this_run": len(new),
           "by_sev": dict(sorted(by_sev_all.items())),
           "policy_breaches_total": breaches,
           "ladder": {"SEV-0": "integrity — row impossible",
                      "SEV-1": "policy breach — own rule violated",
                      "SEV-2": "confident wrong (p>=0.77)",
                      "SEV-3": "oversized (>2x stated-edge Kelly)",
                      "SEV-4": "bad context (knife / regime<floor)",
                      "SEV-5": "expected variance — response only"},
           "traders": summary}
    (RES / "loss_reviews.json").write_text(json.dumps(doc, indent=1))
    print(f"loss reviews: {len(allr)} total (+{len(new)} new), "
          f"breaches={breaches}")


if __name__ == "__main__":
    main()
