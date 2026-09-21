# Full-Coverage Compliance Audit — Ledger

Goal: audit **every line** of source (code, UI, tests, docs, config) and the code that
produces every data/log/input, then bring it **into accordance** with the audit standard
(honest, no theater, no cheating, grounded numbers, correct task framing, real-not-fancy
guardrails). Track coverage so nothing is missed. Convergence = 100% files audited +
every finding remediated or explicitly deferred with reason.

Standard reference: `Oracle Reality Check` audit (artifact 2b131e09) — the house is a
calibrated coin-flip bookmaker at open; the edge (if any) is value-gating net of the vig.

## Scope
- Tracked files: **627** / **241,007** lines.
- Data blobs (~156k lines: results/*.json*, migration/golden_before_m3.json 79.6k, *.pt):
  audited via the CODE that writes/reads them + sampled integrity (Kalshi 1200/1200 ✓).
- **Code/UI/docs to audit line-by-line: ~85k lines / ~400 files.**

## Coverage by category
| Category | Files | Lines | Status | Wave |
|---|---|---|---|---|
| btc_rl/ (trading core) | 23 | 8,243 | ✅ **DONE** (every line) | W1 |
| scripts/ (emitters, tools, agents) | 182 | 30,893 | ✅ **DONE** (every line) · rolling up | W2 |
| tests/ (guardrails + one-offs) | 154 | 13,401 | ✅ **DONE** (every line) | W3 |
| site/ (UI, 34 files) | 34 | 22,032 | ✅ **DONE** (every line) | W4 |
| docs/ + config/ + architecture/ + adapters | 58 | 8,680 | ✅ **DONE** (every line) | W5 |
| data-producing code + integrity sampling | — | — | ✅ (Kalshi 1200/1200) | done |

**COVERAGE: 451 / 451 code+UI+doc files audited line-by-line = 100%. ✅ Every line accounted for.**

## Verdict codes (per file)
`KEEP` compliant · `FIX` bug/inconsistency to correct · `KILL` dead/theater/duplicate to delete ·
`CHEAT` misleading/corner-cut · `BUG` correctness defect · `HANG` half-built/abandoned.

## Findings ledger
_(appended per wave; each entry: file · verdict · one-line finding · action)_

### Wave 1 — btc_rl/ (trading core) — COVERAGE 23/23 files, every line
**Verdict:** the LIVE prediction/settlement path is genuinely correct & PIT-safe (the integrity core is real — extensive clean leakage-free code confirmed), BUT the core carries ~30-40% dead retired-arm/theater code, several **misleading status numbers**, an **inconsistently-applied freeze**, a few real bugs, and the clean task-aligned EV modules are **shelf-only** while online.py runs inline reimplementations.

**P0/P1 (trust / correctness):**
- P1 BUG `online.py:3436-4287` — fail-closed `_entries_frozen` gates ONLY the pt desk; cg33/tv/fm/ob (+kb5/kb_bets) commit new stake with NO freeze check → published `FREEZE_NEW_ENTRIES` is misleading; 4 of 5 live arms ignore it. (Confirms the earlier incident finding.)
- P1 HANG (structural) — the well-formed EV/vig-gating modules (`traders_v2`, `prospective_capture`, `exec_timing`, `preopen_oracle`, `paper_account`) are imported by NO live path; online.py runs inline reimplementations + legacy point-prediction machinery. Task-aligned code is shelf-only.
- P2 CHEAT `sources.py:292-324` — `fetch_brti_composite` (24h-vol-weighted spot avg, NOT BRTI methodology) feeds synth-bar CLOSE → can set settlement label/PnL when candles lag, while labeled a "BRTI approximation."
- P2 BUG `contract_truth.py:34` — `MIN_SAMPLES=30` lets EXACT_BRTI stamp on ~10% of the 60s window (dark today, flag OFF).
- P2 CHEAT `online.py:1997/4867` — `_kb_bets_summary` reports GROSS pnl (no vig) on the status page; net formula exists but is omitted.
- P2 THEATER `online.py:1966/4861` — status emits retired arm `kbf`'s frozen precision as "the deliverable's headline metric."
- P2 CHEAT `online.py:550` — `PT0_START_BANKROLL_C=$100M`: makes %ROI meaningless & de-facto flat-stakes the control (PT_FRAC 10% never binds), contradicting its stated sizing.

**KILL (dead/theater — delete):** ~250 lines of retired pt2-8 entry/fill/pending (`3438-3906`); dead kbf (`4122`) & kb6 (`3254`) blocks; duplicate settlement loader (`4503-4516`); fshare/evlead maintained+persisted+published but drives 0 decisions; KNIFE_BAND + retired treatment policies re-instantiated each call; agents.py dead `predicted_price` + wasted LSTM MLP; `live.py` (deprecated); config `DECISION_HHMM`/`TARGET_SLOTS` (legacy 7PM).

**FIX (bugs):** `features.py:49` vol_30m silently 0 with <31 bars; `train.py` trains on band-reward but scores exact-int (misleading learning curves); fee canonicalization theater (3+ implementations, multi-contract mispricing risk); settlement coupled to a Coinbase candle it no longer uses for truth (defers on missing candle); LinUCB `lam` not persisted; okx docstring stale + variable momentum lookback; late-settle accounting inconsistent across arms.

**KEEP (genuine bright spots):** online.py prediction/calibration/leakage discipline (extensive PIT-safe code); `features.py` PIT-safe; `agents.py` learning math sound; `treatments.py` SPRT/FixedShare correct; `contract_truth` PIT-safe; `economics`/`metrics` math correct; `test_v2_firewall` (leakage guard, highest-value); `research_events`, `llm_sentiment`, `env`, `history` clean & live.

_Remediation staged to a controlled post-audit pass (online.py is the live daemon; edits need a restart). Full audit coverage first, per owner's "nothing missing."_

### Wave 2 — scripts/ — COVERAGE 182/182 files, every line
**Verdict:** 83/182 fully clean; 107 findings (0 P0, 4 P1, 33 P2, 70 P3). The emitters that build the UI are where the misleading numbers originate; ~24 scripts are dead; the "research agent" cron is duplicate-submission theater.

**P1:**
- BUG `fetch_contract_specs.py:32,66-69` (LIVE) — OVERWRITES `contract_outcomes.jsonl` from the current API pull only; a short/partial Kalshi page truncates ALL historical labels (every score/PnL rests on this file). → upsert-by-ticker, don't overwrite.
- BUG `repair_planes.py:55-61` (LIVE) — self-heal STATE freshness keyed on frozen `a3_live.json` (A3 closed 09-08) → STATE=DEGRADED every 5-min cycle forever; DELIVERY/R3 permanently suppressed. → re-key off a live artifact.
- HANG `rl_treatment_dataset.py` (not scheduled) — froze ~09-14 at 227 windows; RL data store silently not accruing while consumers claim growth. → schedule or mark DEAD.
- CHEAT `emit_coverage_ab.py:66` (LIVE) — Models-Lab "A/B" seeds T0 at $100M vs $10k treatment → %ROI meaningless; columns aren't comparable strategies. → equal bankroll, relabel.

**CHEAT on live surfaces:** `run_audit.py:213` (publishes retired pt2-8 as current), `emit_diagnosis.py` (retired arms cited as current evidence), `emit_experiments_snapshot.py:99` (pt6 stakes $0 → verdict TREATMENT_WINS/PROMOTE), `emit_home_snapshot.py:310` (hardcoded offline strings in a per-minute live snapshot), `emit_modelling_techniques.py:105` (hardcoded card metrics), `emit_pm_snapshot.py:83` (frozen a3, hand-typed status). SHELF research scripts optimistic via test-fold threshold selection + gross-of-vig.

**THEATER (LIVE 10-min cron, no research value):** `agent_experiment_analyst` (14,701 dup submissions / 0 implemented), `agent_execution_researcher`, `agent_research_manager` (hardcoded 5-item list), `architecture_checkpoint` (writes ~10 JSON churn every run), `test_v2_capture_audit` (BUG_FOUND/FIXED every run), `agent_firewall` (dedup bloat); SHELF: `research_intelligence`, `options_probe` (hardcoded verdicts).

**KILL (delete, 19 files):** `_sev_reopen_watch/_sev_repro/_tmp_sev/_sev_idempotence/_sev_zombie_check/_sev_recovery_watch`, `_f1_rerun_watch`, `_gate_watch`, `chrome_ctl`, `build_site`, `train_l3`, `demo_trader`, `mech_fair`/`oracle`/`oracle_residual` (DT-03 superseded), `analyst_deep` (v1), `analyst_foundation`, `av_open6_features`, `official_outcomes_refresh`. **Cron-remove:** `emit_a3` + the 4 `agent_*`/`architecture_checkpoint` ceremonies. **Dead code:** `emit_home_snapshot:328/183`, `mine_legacy:152`, `trackA_exec_timing:81`.

### Wave 3 — tests/ — COVERAGE 154/154, every line
**Junk drawer:** ~20 real guardrails (only 2 actually run — `invariants.py`+`leakage_canaries.py` via audit_chain), 57 dead one-offs, 13 theater/cheat. **Biggest systemic gap: NO CI / pytest runner** → every real leakage/label/economics test sits dark. P1: 2 tautological cohort tests + `paper_account` reconcile-by-construction (never fail); `chaos_drills` pkills the live daemon; `test_m5_system` writes the real ledger. `invariants.py:119` pct==0.0 hole (FIXED). Correction: `introspect_model_internals` is clean, not the div-by-zero hypothesized in W1.

### Wave 4 — site/ — COVERAGE 34/34, every line
7 live pages clean; **23 orphan pages** (pruned at publish) carried the ENTIRE UI hazard surface (frontend-science, $2M theater, $100M-EV, "matches market"). Live fixes: `home.html:373` stale caption, `nav.js` dead search links.

### Wave 5 — docs/config/architecture/adapters — COVERAGE 58/58, every line
Docs **broadly HONEST** (newest canon candid, reality-aligned; 89%@90% explicitly retired). Narrow over-claim risk: headline numbers without market benchmark — `COMBINED_STUDENT_FINDINGS.md` (P1: barrier artifact called "tradeable edge the live T2 arm uses"), `DECISIONS.md` M14 (90.9%@6.5% "productized"), `SYSTEM_AUDIT` chronos 0.73. Adapter bugs (brti/derivatives FIXED, av_news dedup). Stale: MANUAL, COMPONENT_REGISTRY roster, NOTES, requirements.txt (missing deps), OFFLINE_METRICS. KILL: A3_SHADOW_SPEC, A3_CHANGE_CONTROL, golden_snapshot.py. Theater: RME/SEV/cert ladder, PROGRAM org-charter, M6_REPAIRS.

---

## REMEDIATION LOG (post-100%-coverage)
**Done + committed:**
- `607a388` security: redact GitHub token from publish-log (was leaking 13,679×).
- `a08c716` site: **delete 23 orphan pages** + fix nav dead links + home.html honest caption. (tests 9/9)
- `49f4195` cleanup: **delete 67 dead one-off scripts + tests** (14 scripts + 53 tests; footguns removed).
- `<this>` bugs: fetch_contract_specs MERGE (P1 truncation), brti reorder, derivatives missingness, invariants pct==0.0. (invariants 29/29)

**Remaining (next phases):**
- Emitter honesty (~7 LIVE emitters): `emit_coverage_ab` $100M control, `run_audit`/`emit_diagnosis` retired-arm-as-current, `emit_experiments_snapshot` pt6 $0→TREATMENT_WINS, `emit_home_snapshot`/`emit_modelling_techniques` hardcoded literals, `emit_pm_snapshot` frozen a3.
- Cron de-churn: pull `emit_a3` + the 4 `agent_*` ceremonies + `architecture_checkpoint` from audit_chain/cron.
- `online.py` (LIVE — needs daemon restart): delete ~250 lines dead retired-arm entry code + kbf/kb6 blocks + duplicate settlement loader + fshare/evlead; fix misleading status numbers (kbf headline, gross kb_bets pnl, PT0 $100M); make fail-closed cover all live arms.
- Guardrail runner (biggest systemic gap): wire the ~18 dark `test_*.py` into audit_chain/pytest so they actually gate.
- Docs: correct the 3 over-claims; refresh MANUAL/COMPONENT_REGISTRY/NOTES; fix requirements.txt; stamp dead A3 specs.
- Tautological tests: replace the 3 reconcile-by-construction gates with independent recompute.
- `repair_planes` STATE canary off frozen a3_live; `contract_truth` MIN_SAMPLES; `features.py:49` vol guard; fee canonicalization; `rl_treatment_dataset` schedule-or-kill.

---

## SESSION 2026-09-21 — HONESTY HARDENING (grounded in the real ledgers)
Trade-log census (real counts): **pt 1352 windows / 4.06M contracts, tv 169, ob 367, cg33 310, fm 385**; kalshi_binary_log = 8 kb variants × ~195 settled decision-windows (the whole offline universe is only ~195 windows — small, and ~13.7 autocorrelated rows/window shrink effective N further). Retired arms (pt2-pt8/cg5/cg10) confirmed frozen.

**Settlement machinery VERIFIED SOUND** (forensic cross-check vs contract_outcomes.jsonl): 0 `win` mismatches, 0 `actual`-vs-official-`exact_yes` disagreements, 100% pnl internally consistent, 0 dupes — all 5 active arms. The truth layer is trustworthy.

**Real fixes committed this session:**
- `8d3e34c` fix(honesty): `emit_experiments_snapshot` — killed pt6's manufactured `TREATMENT_WINS`. It credited pt6's ~892 ABSTAINED windows as $0 vs the control's forced losses, then compared raw cents across arms whose stakes differ 136×. Now pairs only on JOINTLY-TRADED windows, per-CONTRACT, and forces 0-stake shadow arms to `SHADOW_NO_REALIZED_STAKE` (can never PROMOTE). On the 28 windows pt6 actually traded it is **−3.57¢/contract vs control** — the opposite of the fake win. Also `exo_features` cb_ofi taker-sign inversion (was corr −0.31 with bn_ofi).
- `9aed229` fix(honesty): **cg33 was rendering "LIVE" while ruined** ($1.30 of $300 seed, no trade since 09-19; it is the deliberate 33%-Kelly RUIN-RISK arm and blew up as designed). Liveness was "has any rows", not recency vs the 15-min cadence. Added `_honest_state` → LIVE / STALE (≥8 windows missed) / HALTED-ruined, wired into home_snapshot + the home board badge/chip. Also reframed tv's card, which hardcoded "flips T0 from −$48 to positive" while tv is live −$616.

**CRYING WOLF AVERTED (real-vs-fancy-guardrail):** a forensic pass "found" that treatments understate fees vs `kalshi_fee_c` → "$21.5k hidden on pt, ob's edge vanishes." **FALSE.** Verified on every row: `pnl == per-ORDER fee (ceil(7·C·p·(1-p)) via online._order_fee_c) = 100% (1352/1352, 169/169, 367/367, 310/310, 385/385).` The ledger already charges Kalshi's TRUE per-order fee. The trap: `ceil(Σ) ≠ Σ ceil`, so `C·kalshi_fee_c(price)` rounds up C times and OVERCHARGES (100 lots @80c: true 112c vs 200c). "Fixing" the P&L would have INJECTED a $21,562 overcharge on pt. Hardened the canonical owner: `kalshi_fee_c` docstring now flags C=1-only, and added `metrics.kalshi_order_fee_c(contracts, price_c)` as the named per-order owner. All existing `kalshi_fee_c` call sites verified C=1 (eval_engine per-contract EV, value_gate 1-lot rows, break-even, kb 1-lot bets) → no code bug existed. The DECISIVE value_gate_backtest was already fee-correct (uses kalshi_fee_c on 1-lot rows) → STOP-NULL stands.

**pt "The $1K Desk" is misnamed — seeded $100,000,000, not $1K** (bankroll moved ±0.09%). Its +$69,202 is a **size-concentration artifact**, NOT edge: per-contract EV ≈ **−0.034¢**, hit rate 69.5% *below* breakeven, top-5 largest-size windows = 60% of the "profit", one adverse window −$16,408. To reframe in emitters/UI (task #69).

**Operational:** ~30h system-wide capture outage 09-19→09-20 (all live arms; resumed 09-20 14:xx). Schema drift across arms: `late_settle_ts` only in pt; `p_arm` (pt/tv/cg33) vs `p_up` (ob/fm); ob/fm add model/quantile fields (task #70).
