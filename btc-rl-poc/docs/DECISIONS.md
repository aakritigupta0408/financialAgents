# DECISIONS — the ledger of what exists and why

Every create/retire/policy decision, with its evidence and who decided.
Append-only. Reconstructed 2026-08-28 from NOTES.md and git history;
live entries follow the same shape. "Human" = the project owner; every
live-money policy terminates with them (PROGRAM.md §3).

| date | decision | evidence | decided by | status |
|---|---|---|---|---|
| 08-20..24 | kb2 (market-anchored blend) is THE deliverable; kb stays control | pre-registered design | human | standing |
| 08-25 | retire kbf; retire kb6 from trading | kb6 UP-recall 63%, coverage 37%; later: calib slope 0.33 ≈ noise | human + evidence | retired |
| 08-25 | pt3 Disciplined frozen at pre-registered policy v2 (≥0.77) | pre-registration discipline | human | frozen control |
| 08-26 | Saver stake 0.25→0.10 | ~2.5× Kelly; maxDD $17.6k | human | live |
| 08-26 | Gambler v2: ≥0.77 gate + $10k reset, history preserved | gated tier 79.4% vs 65.6% ungated | human | superseded by v2.1 |
| 08-26 | pt6 MLE launched (supervised edge + half-Kelly) | industry standard vs RL data-appetite analysis | human | live |
| 08-26 | SEV-0 declared; all changes become treatments | full-graph audit | human | standing law |
| 08-28 | M7 hour policy DROPPED (not deferred) | 2 "bad hours" vs 1.98 expected by chance, p=0.60 | evidence | dead |
| 08-28 | RCA corrected: tier-1 defect is dispersion not direction | drift arms never trade; feeding path +$1.76 bias; bands 74–76% | evidence | recorded |
| 08-28 | MAE→MSE as headline error metric everywhere | user directive; MASE kept beside MSSE (fat-tail caveat recorded) | human | live |
| 08-28 | 9 bug fixes incl. trader lockout + treatment stale-quote | each verified vs the metric that exposed it; 3/3 prediction matches | evidence (bug class) | live |
| 08-28 | Gambler v2.1: edge ≥2¢ at ACTUAL fill | −1.7-pt stated-edge entry at $1,911.85; conf≡mkt degeneracy proven | human (flagged row) + evidence | live |
| 08-28 | pt7 Patient + pt8 Ideal launched; M11 twins | naive limit −5.21% paired (adverse selection) → measured, defended | human directive | live, collecting |
| 08-28 | M12 EV-ranked leader as treatment; Follower stays frozen | win-rate leaderboard seats market-echo arms 62%/day; control integrity | evidence + charter | live, flat backfill |
| 08-28 | SEV-1 logged; org charter (PROGRAM.md); this ledger created | TA directive | human | this document |

Proposal queue (PORTFOLIO team writes here; human ratifies):
- **PROPOSE: retire or redesign M1 (Platt calibration layer).**
  Evidence (introspection wave, 08-29 snapshot): prequential log-loss
  is WORSE calibrated than raw for ALL NINE arms (+0.034..+0.164, kb7
  worst) — the layer's own pre-registered success criterion is met by
  nobody, and the fitted direction has flipped since 08-28 (b>1 now vs
  b<1 then): the miscalibration drifts faster than the fit. Options:
  (a) retire M1, keep raw p_up; (b) redesign with shorter memory;
  (c) keep shadow-only as a drift instrument, never as an input.
  Recommendation: (c) — it costs nothing and measures drift.
- **PROPOSE: investigate the edge-anti-signal.** kb5's claimed_edge
  weight −0.096 and pt6's conf_minus_ask −0.059: both edge-gated
  models' own regressions score "stated edge" as predicting LOSSES
  (adverse selection internalized — when you think you know better
  than the price, you usually don't). If it replicates on more data,
  the gates should invert from "edge floor" to "edge band" (too MUCH
  claimed edge is also a red flag).
- (nearest launch candidate: M10+M8 — boundaries now family-wise
  corrected to 5.66, so the bar is properly higher)

| 08-29 | Homepage rebuilt as the 7-scene narrative instrument (Prediction→Decision→Outcome→Evidence→Falsification); classic page preserved at home_classic.html as the audit-depth view | owner's design brief: "live scientific instrument, not a dashboard"; accuracy-is-not-edge as the central visual; normalized returns since policy start (start capitals shown — raw $ comparison was scientifically messy) | human (design brief) | shipping |
| 08-29 | TA metrics review adopted: a DECISION layer now sits above the registry — scripts/emit_decision_board.py computes, per treatment, bootstrap+normal 95% CI, P(Δ>0), P(Δ>edge), MDE and power at the family-wise α, paired-window concentration (top-3 share), evaluability vs activity overlap, veto decomposition (losses avoided / wins forgone on skipped windows), regime/leader/hour effect slices with worst-powered-slice and sign consistency, projected windows to an SPRT verdict, and an automatic PROMOTE / KEEP_TESTING / HOLD / KILL / INVALID state under rules pre-registered in that file. Every number derives from the daemon's own per-window paired log (treatments.jsonl); the SPRT stays the promotion authority — the state machine cannot promote anything the SPRT hasn't passed | TA review (28-section critique); first run: 10 KEEP_TESTING, 2 HOLD (t_cal/t_cheap coverage <25% — "a different product"), 0 KILL; power for the 2¢ pre-registered edge at n=168 is ~2% at α=0.003125, which is precisely why sequential testing, not fixed-horizon, is the desk's law | human (TA directive) + evidence | live, cron 10-min |
| 08-29 | Quant Universe wave 1 shipped: universe.html (explorable world atlas, one geometry × Explore/Engineer lenses), clock.html (every cadence with its proving artifact + dissected retrain + honest cost ledger), agents.html (authority cards + activity stream, agent-quality rows say "not yet metered"), museum.html (six frozen failure exhibits); world.json emitter grounds all of it in real artifacts; docs/OS_BLUEPRINT.md adopted as the engineering constitution (seven planes, four loops, SEV-0..3 taxonomy) | owner's two-worlds brief + TA "Quant Research OS" review; all pages screenshot-verified in both lenses | human (design briefs) | shipping |
| 08-29 | Skip-as-decision formalized: a treatment's stand-down is a COMPLETE scored observation (EV 0 by pre-registration), never "missing data" — pair completeness measures evaluability only, and activity overlap is reported separately without invalidating | first emitter draft wrongly marked all 12 challengers INVALID by counting deliberate skips as missing pairs; the distinction is now in code and on the board | evidence (bug class) | recorded |
