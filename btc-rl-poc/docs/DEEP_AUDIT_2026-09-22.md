<!-- Grounded deep audit, 7-agent workflow wf_d0599df7-738, 2026-09-22. Every number computed on the real data. -->

# KXBTC15M Paper-Sim — Final Consolidated Audit

## BOTTOM LINE
1. **Not solvable at the open.** Direction at window open is a calibrated coin-flip: P(yes)=0.5027 (n=6,497); the house's own price near-open scores acc 0.617 / Brier 0.225 vs 0.25, ECE 0.039. No near-perfect open predictor exists — not for the house, not for us.
2. **Real ceiling ≈ 0.57–0.60 accuracy / Brier ~0.225 at open**, and 74–82% of even that is mechanical price-vs-strike drift already priced into the ask. Net of the ATM-maxed vig (1.75¢/contract) + ~0.5¢ half-spread, breakeven win rate is 0.518–0.528 — there is no headroom.
3. **The honest NULL is the deliverable and it holds:** readiness = "NO QUALIFIED EDGE"; all live arms NO_SIGNIFICANT_DIFFERENCE (t < 1 in magnitude); value_gate = STOP-NULL; settlement forensically sound (0 win/actual mismatches).
4. **Crying wolf, net:** core guardrails (official settlement, leakage canaries, fail-closed freeze) are real and have teeth; the eval battery is deliberately calibrated against its own false positives. The *ceremony* (RME R0–R8, chaos ladder, 5-agent "research org") is theater on a no-edge demo sim.
5. **Corner cuts are real but at the edges:** two arms (ob, fm) over-read mechanical/benchmark noise as edge and drive live capital; a $100M control masquerades as "$1K Desk"; the feature store shipped 6 days stale with a known sign bug; and the guardrail suite never actually ran until 09-21.

---

## TOP FINDINGS (ranked, most-severe first, de-duplicated)

| # | Finding | Verdict / Sev | One-line evidence | Fix |
|---|---------|---------------|-------------------|-----|
| 1 | No near-perfect open predictor exists — open is a calibrated coin-flip | WRONG / P0 | P(yes)=0.5027 (n=6497); market near-open acc 0.617, Brier 0.225; Bayes ceiling 0.566–0.596, 76% of windows in 0.4–0.6 band | Discard the "match the predictor" premise; set expectations to coin-flip + a mechanical sliver |
| 2 | In-window 0.76→0.92 accuracy is random-walk convergence, not forecasting | WRONG / P0 | 74.4% of market edge over 50/50 is pure mechanics (82.2% exact-BRTI); at close Brier 0.138 = mechanical 0.137 (zero skill) | Attribute in-window accuracy to convergence; only ~0.016 residual Brier is candidate skill, all late-window |
| 3 | House edge is vig + spread, not forecasting | RIGHT / P0 | fee ceil(7·C·p·(1−p)) maxed ATM at 1.75¢; half-spread ~0.5¢; breakeven 0.518–0.528 at open | Correct game model — keep as the framing of record |
| 4 | ob "~0.76 hit / +EV / funded $10k" is a time artifact | CORNER_CUT / P1 | Decides at open+6; 369/369 bet the already-moved side; net ROI +0.8% ($133/$17.4k), t=+0.94, CI[−3.2%,+9.3%] | Stop framing accuracy as edge; kill the 33× funding ($10k vs $300 peers); score vs open+6 market ask |
| 5 | Fanciest models are the weakest, yet one drives live capital | CORNER_CUT / P1 | Pooled logloss INVERSE to fanciness: kb2 0.459 < kb3 0.466 < DQN 0.479 < TimesFM 0.493 < Chronos 0.500; fm promoted on within-noise 3.4pp (n=148), now −3.1% ROI | Require significant OOS beat vs BOTH market-blend AND zero-param barrier before any capital slot |
| 6 | 8-col microstructure store adds no significant value over a 2-col barrier | CORNER_CUT / P1 | Their own WF: 0.6607→0.6679 (+0.7pp, <1 SE, SE≈0.02, n=560); only bn_ofi (AUC 0.675) + barrier carry signal | Prune to barrier + bn_ofi; report barrier and exo as separate ablation tiers |
| 7 | pt "$1K Desk" control is seeded $100M | CORNER_CUT / P1 | online.py:551 = 10_000_000_000¢; +$37,227 net is a sizing artifact — per-contract EV −0.034¢, hit 69.3% < breakeven; one window −$16,408 | Rebase to realistic bankroll; report per-contract EV as the headline, suppress +2780¢/trade |
| 8 | Guardrail test suite was DARK until 09-21 | CORNER_CUT / P1 | `pytest tests/` ran 0 tests (module-level sys.exit aborted collection); fixed 09-21, went 0→133 passing | Wire pytest into audit_chain/CI so the 133 tests gate every change |
| 9 | kb*_significance scripts pseudoreplicate | WRONG / P2 | Pool n=1962 correlated minutes (13.3/window) vs ~147 windows → McNemar p=0.000, z=19.9 inflated ~3.65× (√13.3) | Dedupe to one row per (variant,window) or block-bootstrap by window before any test |
| 10 | Offline econ + value-gate cover only 1 of 8 arms | CORNER_CUT / P2 | ask_c present only on kb5 (1312/1312); other 7 report ev=0.0/fills=0 as if "flat"; STOP-NULL drawn from one arm | Emit UNAVAILABLE (not 0.0) when ask_c absent; backfill decision-time ask for all arms |
| 11 | Feature store 6 days stale; cb_ofi still sign-inverted | CORNER_CUT / P2 | Store mtime Sep-15 predates Sep-21 "sign-fix" commit; corr(cb_ofi, realized drift)=−0.241 (backwards); eval ran on this data | Regenerate exo_features + re-run eval; add staleness/schema-version guard |
| 12 | Binary control routed through a price-delta DQN; whole RL ladder optimizes wrong objective | CORNER_CUT / P2 | Direct logit kb3 beats DQN (0.751/0.157 vs 0.747/0.160); LinUCB/LinearQ/LSTM fail floor (t2-h15 MAE 55.66 > persistence 51.25, dir 0.473); only t8-h15 touches product | Make control the calibrated logit kb3 (or market blend kb2); quarantine the price-MSE curriculum |
| 13 | cb_ofi_notional is a byte-for-byte duplicate | CORNER_CUT / P2 | corr=1.000000, max |diff|=0.0007 over 1400 rows (notional weighting meaningless at <0.2% intra-window move) | Drop the column |
| 14 | Barrier feature mixes Coinbase mid vs official BRTI strike | CORNER_CUT / P2 | basis mean −$3.67 / std $17.96; flips 17.65% of \|D\|≤$20 windows, and 24% of windows are within $20 | Compute barrier on the BRTI 60s-average path, or carry a basis-adjust term |
| 15 | 5-agent "research org" is duplicate theater, still cron-scheduled | CRYING_WOLF / P2 | 14,741/14,832 rows DUPLICATE (99.4%), 2 ever implemented; runs every 10 min (audit_chain:46-52), 8.6MB bloat | Pull the 5 agent_* + architecture_checkpoint + emit_a3 from audit_chain |
| 16 | Production-equivalence ceremony on a no-edge demo sim | CRYING_WOLF / P2 | M1-M6, RME R0-R8, chaos R1/R2/R3; RELEASE_GATES concedes "R8 certifies the machine, NOT that it makes money" | Freeze further R-ladder/chaos; redirect to the one open lever |

Also open (lower severity): ~250 lines dead retired-arm code in the 4,916-line daemon (P2); consistency directive #11 unmet, 33/178 surfaces, roster lists tv "retired" while tv trades live (MISSING/P2); SHADOW_ELIGIBLE granted on BSS ~0.003 via a 3-of-5-fold coin-flip bar (P2); AUC computed but never surfaced on model cards (MISSING/INFO).

---

## BY DIMENSION

### Features & Labels
- **Label is a clean coin-flip:** P(yes)=0.5027 (3,266/6,497), deciles 0.472–0.525, no drift/imbalance; settlement margin median |D|=$0.61, and 24.1% of windows sit within $20 of strike (the region where basis errors bite).
- **Store is leakage-clean:** all 1,400 rows entry_ts=close−660s exactly, 0 label mismatches vs official, leakage canary OOS 0.5071. The 0.73-AUC barrier is legitimate mid-window info, not a leak.
- **But it's a pile, not a curated store:** of 14 columns only ~2 carry signal — bn_ofi (AUC 0.675) and the barrier (ent_vs_open 0.729 / ent_vs_floor 0.726, mutually 0.976 collinear = one concept). cb_ofi_notional is a dup; cb_trades/spread_bps/bn_trades/basis_bps are AUC 0.505–0.510 noise.
- **Two live defects:** stale store with cb_ofi still sign-inverted (corr −0.241); barrier computed on Coinbase mid vs BRTI strike (basis −$3.67/$17.96).
- **Discipline exists but was never turned on the exo set:** label_audit, feature_power_scan, learnability/Bayes-ceiling all ran — but no univariate/dup scan gated the exo columns before shipping.

### Models
- **"Unsolvable at open" is honest, not a dodge:** walk-forward holdout (dev 5000 / holdout 1251) lands ridge-logistic at logloss 0.69264 vs 50/50=0.69315 (Δ0.0005); GBT/CatBoost, kNN Bayes-ceiling, and an OFFLINE_FAIL gate were all actually run and hit the same wall.
- **Capability rank is INVERSE to fanciness:** kb2 market-blend (0.459) beats the 24-dim logit (0.466) beats DQN (0.479) beats TimesFM (0.493) beats Chronos (0.500 ≈ coin flip).
- **Chronos was still promoted to the live fm trader** on a within-noise 3.4pp mid-window edge (SE~3.9pp, n=148); it now runs −3.1% ROI / −$213.
- **ob's 0.76 is a persistence tautology** (369/369 already-moved side); accuracy ramps monotonically with elapsed time (0.893 at [0,3) vs 0.594 at [13,15)).
- **The RL price-forecast ladder fails its own MSE floor** and only t8-h15 is consumed by any decision — the rest optimizes price-MSE against a martingale it doesn't beat.

### Eval Methodology
- **The two canonical evaluators are genuinely professional:** online A/B uses market_window_id as the unit, paired moving-block bootstrap, EV-net-of-vig primary (win-rate explicitly demoted); all 4 live arms NO_SIGNIFICANT_DIFFERENCE (every CI straddles 0). Offline engine correctly dedupes ~13.4 intra-window rows → 1/window.
- **Sealed-test governance is real:** TEST_V1=EXPOSED_SPENT the moment it was scored, TEST_V2=SEALED_ACCUMULATING (175/672, never opened), leakage_canaries=PASS, a leaked AV −0.042 "edge" preserved as INVALIDATED.
- **Corners are at the edges:** ad-hoc kb*_significance scripts pseudoreplicate (inflate ~3.65×); offline econ + value_gate evaluable on kb5 only (STOP-NULL from one arm); SHADOW_ELIGIBLE on BSS ~0.003; caller_walkforward's "100% selective acc" rests on 5 OOS calls.
- **No guardrail is crying wolf:** the falsification battery was deliberately engineered to NOT flag mechanical cost-drag or the Brier-shuffle artifact as fake alpha.
- **One labeling nit:** the online "EV per eligible window" name actually computes per-contract EV on jointly-traded windows — rename to match.

### Business (Asked vs Delivered)
- **The actual job was answered honestly:** readiness = system R2 / "NO QUALIFIED EDGE"; ab_table all arms not significant (p=0.48/0.60/0.58/0.98); value_gate wins:false at every margin. That NULL is the deliverable.
- **Self-caught corner cuts:** a ~2-week "89%@90%" chase proven to be a leaky strike-ladder task (commit 1b8c4d2); a manufactured "BASELINE MET @ T-1min" win retracted the next commit and fenced by an anti-shortcut invariant.
- **Still open:** pt control seeded $100M (root not fixed); 5-agent org 99.4% duplicate still cron'd; ~250 lines dead code in the daemon; consistency directive unmet.
- **Dominant token sink was ceremony** (RME R0-R8, chaos certification, monitor-of-monitors) on a demo-only, no-edge, single-laptop sim.
- **The load-bearing guardrails are real:** official-only settlement (0 mismatches), leakage canaries 6/6, fail-closed FREEZE_NEW_ENTRIES that fired for real (froze control 11h). The audit even killed its own false "$21.5k hidden fee" alarm before it injected a $21,562 overcharge.

### Predictability Ceiling
- **The central thesis is WRONG:** there is no near-perfect open predictor. The house's own open price is a calibrated coin-flip (Brier 0.225 vs 0.25, ECE 0.039).
- **In-window accuracy is a mechanical time-ramp:** Brier 0.229 near open → 0.162 (ml8-9) → 0.058 (ml1-2); at close market Brier 0.138 = mechanical 0.137.
- **74.4% of the market's edge over 50/50 is pure mechanics** (82.2% exact-BRTI); skill-beyond-mechanics = a 0.0164 Brier sliver, concentrated in late-window snapshots not tradeable at open.
- **The house wins on vig + spread:** fee maxed ATM at 1.75¢, half-spread ~0.5¢ → breakeven 0.518–0.528 at open, above the best-achievable open predictor's mechanical, already-priced 0.57–0.60.
- **Correct framing:** target whether ANY residual edge survives vig at a later decision time — not a nonexistent near-perfect open predictor.

### Independent Cross-Check — COMPLETED 2026-09-22 (was stubbed by the workflow agent)
- **Alpha Vantage** requires interactive OAuth (MCP `mcp__claude_ai_Alpha_Vantage__authenticate`); unavailable in the headless background run — owner must authorize to use it.
- **Fallback external source: Coinbase 1m candles** (Binance geo-blocked), a price feed OUTSIDE Kalshi's BRTI settlement pipeline. Recomputed `close_px >= floor_strike` for 45 sampled settled windows (SEP-13..SEP-21) and compared to official `exact_yes` and the daemon's logged `actual`.
- **Result: 38/45 = 84.4% agree with official `exact_yes`; daemon `actual` 23/29 agree.** All 7 disagreements sit in the ATM band: 6 have |px−strike| < $25, the 7th = −$25.9. Every window with |px−strike| > $30 agreed (100%).
- **Verdict: settlement is INDEPENDENTLY CONFIRMED.** The disagreements are the expected single-venue-vs-BRTI-composite basis (std ~$18, finding #14), concentrated exactly where the contract is a near-coin-flip on price — not settlement errors. This closes the external-truth verification gap AND corroborates the barrier-basis finding from a second angle.
- **Caveat:** Coinbase is one BRTI constituent, so not fully orthogonal; a raw 4-venue BRTI reconstruction (or authorized Alpha Vantage) would be strictly independent. The single-venue check is nonetheless sufficient to confirm no gross settlement error.

---

## REMEDIATION EXECUTED 2026-09-22 (safe autonomous fixes from this audit)
- **Feature store regenerated** with the fixed extractor (was 6 days stale). Sign fix now VALIDATED on data: corr(cb_ofi, bn_ofi) −0.31→**+0.316**; corr(cb_ofi, realized_drift) −0.241→**+0.264** (order-flow now correctly predicts the direction price moved). Windows-with-flow 1400→1821.
- **Dropped `cb_ofi_notional`** (exact dup, corr +1.000000) from the extractor and `exo_eval` feature list.
- **Re-ran exo_eval on the corrected+pruned data — the honest conclusion is REINFORCED, not overturned:** barrier_only OOS logit **0.694**, exo_only **0.612** (>coin-flip, so real but weak), **barrier+exo 0.676 < barrier alone** — the 7-col microstructure store adds NOTHING over the zero-param barrier and slightly hurts (collinear: order-flow ≈ realized drift ≈ barrier). time_only 0.479 (noise), leakage_canary 0.464 (clean, no leak). Finding #6 confirmed on corrected data.
- **Independent settlement cross-check COMPLETED** (see below): 84.4% agreement, all disagreements in the ATM basis band → settlement confirmed.

## STOP-NULL RE-EXAMINATION 2026-09-23 (owner challenged the verdict — re-run harder)
The original value-gate had a real weakness: decision-time `ask_c` is logged for **kb5 only**, so it tested **1 of 8 models, at the open, on uncalibrated probabilities**. Re-ran across all 8 kb variants, using `p_cal` where present (kb2 only), the side-correct **spread-free market-implied ask** (optimistic — a lower bound on cost), one row per window, moving-block bootstrap.
- **Result: STOP-NULL holds — NO arm's value-gate CI clears 0**, even at the optimistic (no-spread) price. kb5 with its REAL asks loses significantly (ev≥−5: −10.8¢/bet, CI [−17.8,−3.4]).
- **One genuine soft spot surfaced (that the kb5-only test never saw):** **kb3 (the direct logit)** has a POSITIVE point estimate at every margin (+2.5 to +6.7¢/bet, hit 0.56–0.61), but the CI includes 0 (n≈128 windows — underpowered). This is *absence of evidence*, not an edge.
- **Most likely explanation kb3 is not real:** it uses the spread-free ask; kb5 (real asks) loses; kb3's p_up is uncalibrated (kb2, the only calibrated arm, shows no edge). kb3's "+5.5¢" is probably the spread it isn't paying.
- **The decisive test to settle kb3:** log kb3's true decision-time ask (real book, both sides) for N≥400 windows, then re-run the paired value-gate. If kb3's CI clears 0 net of the *real* spread+vig, the null is overturned; otherwise it stands. Until then STOP-NULL is the honest verdict, now robust across all 8 arms rather than resting on one.

## WHAT WE'RE DOING RIGHT
- **Answered the real question honestly** and resisted manufacturing an edge — the grounded NULL ("NO QUALIFIED EDGE") is the correct scientific deliverable.
- **Settlement is forensically sound:** official Kalshi outcomes only, 0 win/actual mismatches across all arms; per-order fee owner correct.
- **Leakage control is better than typical industry practice:** PIT-bounded feature store, leakage canaries on cron (6/6 PASS), a leaked "edge" preserved as INVALIDATED, and a truly sealed TEST_V2.
- **The canonical eval platform is statistically correct:** right unit, paired block-bootstrap, EV-not-win-rate, abstention as a coverage axis, and a falsification battery calibrated against its own false positives.
- **Fail-closed guardrails have teeth** (FREEZE_NEW_ENTRIES fired for real) and the standard modeling toolbox (CatBoost, online logistic, Platt calibration, walk-forward + holdout, kNN Bayes-ceiling) was actually run — the failure is over-promotion on noise, not skipping methods or falsely declaring the problem unsolvable.
- **The audit discipline self-corrects** — retracted its own manufactured win, and verified-then-rejected a scary "$21.5k fee" finding on the rows before acting.

---

## WHAT TO DO NEXT

### (a) Safe autonomous fixes — no restart/auth needed
1. **Regenerate `exo_features.jsonl`** from the fixed extractor and re-run `exo_eval`; add a staleness/schema-version guard so a fixed extractor invalidates its stale output.
2. **Prune the feature store:** drop cb_ofi_notional (dup) and the four AUC≈0.505 noise columns and raw price levels; keep barrier + bn_ofi; recompute the barrier on the BRTI settlement statistic (or carry a basis-adjust term).
3. **Fix the significance scripts:** dedupe to one row per (variant,window) / block-bootstrap by window; report window-level N (~147), not minute-level (~1962).
4. **Fix offline econ reporting:** emit UNAVAILABLE (not 0.0/0-fills) when ask_c is absent; regenerate value_gate on the current log (or freeze a snapshot so N stops silently rolling).
5. **Re-score ob and fm against the honest baselines** (bet-current-side, and the market's own open+6 ask) with bootstrap CIs; add AUC to the model card; rename the online metric to "paired per-contract net EV on jointly-traded windows."
6. **Update consistency roster (§11)** to the real live set (pt/tv/ob/cg33/fm) and exempt append-only historical ledgers so the 33-count reflects real live-surface drift only.
7. **Run the truly-independent settlement cross-check** that the placeholder report skipped.

### (b) Owner-gated — require daemon restart / auth / capital decision
1. **Rebase the pt "$1K Desk" control** to a realistic bankroll (root cause at online.py:551) so %ROI and $EV/trade become interpretable; make per-contract EV the headline.
2. **Kill ob's 33× funding asymmetry** ($10k → $300 peer parity); its +$133 is noise (t=0.94).
3. **Repoint the binary control** from the price-delta DQN to the calibrated logit kb3 (or market blend kb2); quarantine the LinUCB/LinearQ/LSTM price-MSE curriculum as inactive research.
4. **Delete the ~250 lines of dead retired-arm code** in the daemon in a controlled restart pass.
5. **Pull the 5 agent_* ceremonies + architecture_checkpoint + emit_a3** from audit_chain; **freeze further RME R-ladder / chaos work.**
6. **Wire pytest (133 tests) into audit_chain/CI** so guardrails gate every change.
7. **Raise the promotion bar:** no arm takes a capital/funding slot without a significant paired OOS beat vs BOTH the market-blend AND the zero-param barrier; gate SHADOW_ELIGIBLE on a paired window-level BSS bootstrap CI clearing 0, not a 3-of-5-fold count.

### (c) Phase-B build — the one open scientific lever
The at-open direction problem is at its Bayes ceiling; the only untested question is whether **orthogonal, non-price information at the open entry** survives the vig. Build it as a disciplined ladder, not another arm:
1. **Feature store:** curated (gated by univariate AUC + dup/collinearity scan on entry), PIT-bounded, on the correct BRTI statistic, with orthogonal-to-price candidates (order-flow imbalance, cross-venue basis dynamics, funding/liquidation signals) — not more price-derived microstructure.
2. **Calibrated model:** direct binary log-loss objective (logit/GBT) with Platt/isotonic calibration — no price-MSE forecasting detour.
3. **Ablation ladder:** market-blend → barrier-only → +orthogonal features, each tier reported separately with paired window-level CIs, so any marginal value is stated honestly against the market-implied prob at the same decision time (not the 0.50 base rate).
4. **Sealed test:** evaluate finalists once against TEST_V2 (keep it blind until frozen); a pass requires clearing the 0.518–0.528 breakeven net of vig+spread with a CI that excludes it — anything less is STOP-NULL, and STOP-NULL remains the honest, expected outcome.