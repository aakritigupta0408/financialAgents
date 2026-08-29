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

| 08-29 | Gambler v3 (D-gambler-sizing RESOLVED): 33% stake KEPT — the aggressive curriculum is the exhibit — but every settle that lifts the bankroll above the $10k start sweeps the excess to a withdrawal ledger (wd_c on the settling row, mirroring Saver's skim_c); withdrawals can never be re-staked; fresh $10k at PT4_RESET2_TS, v1/v2 history preserved. Exposure now bounded at 0.33×$10k forever | owner directive; SEV-1 finding was that sizing, not the gate, was the exposure — the sweep caps compounding risk while keeping the lesson | human | live (daemon restarted) |
| 08-29 | M1 v3 (D-m1-future RESOLVED): redesigned with shorter memory (window 150→50, warm 20, refit 3) AND kept shadow-only as a drift instrument — p_m1 is never a decision input. from_dict made config-sovereign (evidence restored, hyperparameters from code) so the retune actually applies to restored state | owner directive; introspection showed the 150-window fit lagging a miscalibration that flips within a day | human | live (daemon restarted) |

| 08-29 | Loss-review law (owner): every losing trade gets an automated RCA + a graded SEV on the six-level ladder (SEV-0 integrity-impossible row · SEV-1 own-rule breach · SEV-2 confident-wrong ≥0.77 · SEV-3 oversized >2× stated-edge Kelly · SEV-4 bad context knife/regime · SEV-5 expected variance, response only) + a coach response to the trader + a treatment-path proposal when a cause dominates. Version-aware grading: rows judged against the rule in force when made (first run produced 48 false SEV-1s judging v1 rows by today's gates — fixed before shipping). First real catch: one 08-26 window where Gambler staked 42% vs 33% cap and Saver 31% vs 25%, same minute — under investigation | owner directive; scripts/loss_review.py on the 10-min cron; 413 losses graded: 2×SEV-1, 109×SEV-2, 35×SEV-3, 131×SEV-4, 136×SEV-5 | human (law) + evidence | live |
| 08-29 | Trader ML-migration program (owner): traders evolve from rule-based to ML-based regimes, starting with bid PRICING and PACING. Method ruling: regression / full-information counterfactual models now (every candidate limit price and entry minute is scorable on logged quote paths — auditable row-by-row); deep RL deferred to multi-ticker scale (data appetite, no counterfactual audit). First artifact: fill_curve.json — fill-rate/win-given-fill/EV per limit offset δ∈[0,8]¢ + logistic fill_prob(δ, mins_left) + pacing buckets. Headline finding: win-given-fill falls monotonically 67.9%→47.0% with bid depth — adverse selection now measured, not asserted; 9–6min is the least-bad taker entry window. Graduates only as a future M13 treatment via SPRT; nothing changes live | owner directive; scripts/emit_fill_curve.py, model basis (same family as M11/t_limit) | human (program) + evidence | evidence collecting |

| 08-29 | Experiment graph unified (TA brief #3): board.html → "Experiment Program" (lifecycle+causality), metrics_lab decision board → "Experiment Analysis" (evidence) — two views of one registry (program.json). New paired incremental branch analysis on identical windows: M2 adds nothing to M8 (−3.7¢, P13%), M3 INTERFERES with M8 (−7.1¢, P6%), M10 genuinely adds to itself+M8 (+3.6¢, P95%), M11+M8's benefit is inherited from M8 (−2.9¢ vs M8 alone, P19%); cross-basis pairs refused. Governance preserved: analysis recommendations (PRIORITY/CONTINUE/HOLD/DIAGNOSE/RETIRE/REDESIGN/INVALID/PROMOTE) never change lifecycle — only the owner does. Decision inbox raised: retire M11 standalone? retire M3 branch? redesign M12? create M13 Edge Band? | TA directive; emit_program.py + diagnosis layer; all numbers from treatments.jsonl paired windows | human (directive) + evidence | shipping |

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
