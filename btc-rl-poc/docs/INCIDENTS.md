# INCIDENTS — the SEV ladder

Machine-readable mirror: results/incidents.jsonl. An incident closes
only when its verification metric clears, never by declaration.

---

## SEV-0 · 2026-08-26 · systemic error architecture — MITIGATING

Full record: site/sev0.html + docs/SEV0_REMEDIATION.md.
One-line RCA (as corrected 08-28): band under-dispersion on the
decision-feeding path → over-extreme probabilities → uncalibrated
cross-arm comparison + churn → stake multiplication; execution the
binding leak (2.70 pts EV). 20 bugs confirmed by audit; all criticals
fixed and verified. Close-out criteria (each ≥100 windows): herd-loss
share <40%, desk EV/$1 ≥ +3% at sane DD, cross-arm AUC ≥ 0.65 with
|pBias| < 0.02, zero control regression. STATUS: open, mitigations
under SPRT.

---

## SEV-1 · 2026-08-28 · performance degradation — ACTIONS LIVE

**Report**: "the model is performing worse" (owner, 08-28 evening).

**Finding 1 — the model is NOT worse.** kb Brier 0.205 vs market 0.197
(parity, unchanged); market regime normal (acc 69.5% vs 62.1% on the
08-27 stress day); MSE-era retrain gate verified writing val_mse and
functioning. No prediction-quality regression exists.

**Finding 2 — RCA of the losses, two mechanisms:**
(a) GAMBLER VARIANCE: −$4,198 pre-v2.1 (11 bets, 33% stakes, EV −9.5%),
then two post-gate bets with LEGITIMATE stated edge (+6.4¢, +10.1¢)
that both lost (−$3,454). At ~1.6× Kelly two losses ≈ −55% of
bankroll — exactly the MacLean-Thorp-Ziemba over-betting variance the
card advertises. The v2.1 gate is functioning; 33% sizing is the
exposure, and it is a HUMAN-owned policy (curriculum exhibit).
(b) LEADERBOARD MECHANISM: the win-rate ranking seated market-echo
arms (kb4+kb2) 62% of the day; such arms win constantly just below
their own break-even (for kb2, conf ≡ market ⇒ edge ≡ −(spread+fee)).
Desk control: −$76 on 72 bids, win 69.4% vs BE 70.8% — thin,
structural, and part of what the frozen control exists to measure.

**Actions:**
1. ● Gambler v2.1 edge-at-fill gate (shipped before this ticket;
   verification metric: no future entry with stated edge < +2¢).
2. ● M12 "EV-ranked leader" treatment live — the cost-aware
   leaderboard candidate. Backfill +0.01% paired (flat, reported
   honestly); SPRT decides. The Follower stays frozen (control
   integrity, PROGRAM.md §4).
3. ● This ticket + the SEV ladder itself (TA directive).
4. ○ OPEN QUESTION FOR THE OWNER: does the Gambler keep 33% (the
   aggressive-curriculum exhibit, with its drawdowns as the lesson) or
   step down to a Kelly-consistent fraction? Policy call, not ours.

**Verification / close-out:** 7 calendar days with (i) no Gambler
entry below +2¢ stated edge, (ii) desk-control daily win% within 2 pts
of its own break-even or better, (iii) M12 verdict reached or still
fairly collecting. Relationship to SEV-0: same root family (edge vs
price), narrower blast radius; SEV-0's close-out metrics subsume this
ticket's long-run health.
