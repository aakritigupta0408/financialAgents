# RESEARCH PROGRAM — living plan & memory

## 0. PRIME DIRECTIVES (owner, standing — never forget; read every session)
1. **The problem (fixed):** at WINDOW OPEN, predict the CLOSE direction, and buy the winning
   contract while it is still cheap. The oracle's value is being right EARLY. **Entry time is
   fixed at open.**
2. **★ NORTH-STAR = POSITIVE EV (2026-09-15, owner, supersedes the accuracy target).** Optimize
   net $/window, Sharpe, drawdown — NOT hit-rate. Proven why: T0 hits 69% but is −$1,536 / −150¢
   per trade because it buys FAVORITES at ~68¢ (break-even ~70-71% incl fees) — avg loss −$38 vs
   avg win +$15. Accuracy is a TRAP metric here. The lever is a VALUE GATE: bet only when the
   calibrated win-prob beats price+fee. (The 89%@90% target is retired to reference: it was the
   strike-ladder moneyness task; at-the-money direction caps ~0.74 at open+6min — see §3g.)
3. **Forbidden shortcuts (all = cheating, reject on sight):** moving the decision later (T-1min
   etc.), any post-entry information, dropping coverage below 90% to inflate hit, leakage, and
   fabricating/again "artificial numbers wins." A win only counts AT ENTRY, walk-forward,
   canary-clean.
4. **Do the hard analytical work.** Never conclude "unsolvable" to dodge effort; run the Bayes/
   feature/first-principles analysis. But never fake a result either. Both are integrity.
5. **Decision lens:** WWAKD (Karpathy recipe — become one with the data; overfit to prove
   capacity; fix the eval) + Sheldon-grade logic/math/first-principles + engineering math,
   physics of chaos/patterns, probability & statistics, backed by real cited sources.
6. **Exhaust EVERY published method** — GitHub, Kaggle, papers-with-code, blogs, vlogs, forums,
   open research. Reproduce each **EXACTLY as its source specifies** (its data prep, label,
   features, model, params) — a targeted grid over real ideas, NOT a random sweep.
7. **Make the whole system robust, tested, recreatable:** model, data, logging, UI/UX, website,
   retraining, drift, latency, performance — all instrumented, shown in the UI, and reproducible.
8. **Autonomy:** keep assigning myself the next task; do not pause for approval; stop only when
   told, when performance is achieved, or when everything above is robust & tested.
9. **This document is the memory.** Update it every meaningful step; it is authoritative.
10. **Directives are ADDITIVE, never replaced** — every instruction across the whole
    conversation stays live. On direct CONFLICT, the MOST RECENT instruction wins.
11. **Consistency across ALL surfaces.** Every doc / UI page / snapshot / code / website / exec
    summary / Claude memory must stay consistent with the current roster (T0 pt / T1 cg33 /
    T2 fm; retired: cg5 cg10 tv pt2-8) and directives, and be constantly verified for staleness.
    Tool: scripts/consistency_audit.py (R31) — run on cron; distinguish LIVE surfaces (must be
    consistent) from HISTORICAL records (may reference retired arms).



**Purpose.** Single source of truth for the alpha-research program: what's done, in
progress, blocked, remaining, scheduled — with every task carried through the pipeline
**Feasibility → Build → Evaluate → Online-test → Launch**, its dependencies, and its result.
This file is my working memory; I update it every session so context is never lost.

**How I use it.** Read top-to-bottom at session start. §2 = the rules I must not break.
§3 = what's already been proven (don't re-derive). §5 = the task board (the live to-do).
§6 = what's running in background right now. I update statuses + findings as work lands,
and I add a task before starting it. Keep it scannable — terse rows, numbers not prose.

Status legend: ✅ done · 🔄 in-progress · ⏳ scheduled · 🚫 blocked · 🧊 parked · ❌ won't-do.
Pipeline stages per task: **Fe**asibility · **Bu**ild · **Ev**al(offline, walk-forward+canary)
· **On**line(shadow arm) · **La**unch(live treatment).

---

## 1. Mission
Predict whether a Kalshi KXBTC15M 15-minute BTC window closes ≥ its open, well enough to
trade profitably (paper). Improve the model honestly (walk-forward OOS + leakage canary),
maximizing real edge — not in-sample fit.

## 2. Standing constraints (NEVER violate)
- **Paper / simulation only.** Never imply real money.
- **T0 (`pt`) is the frozen A/B control** — never change its policy.
- Git from the worktree; **never push main / force-push / merge**; commit trailer
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Settlement = official Kalshi only** (`_official_outcome`), never the candle proxy.
- **Every model result must carry an OOS number + a label-shuffle leakage canary.**
  Any shuffle-OOS > ~0.55 = a leak; discard the result.
- Two systems stay isolated (live desk vs sealed TRUE15M). Don't change state unnecessarily.
- **Decision lens = "What Would Karpathy Do" (WWAKD):** become one with the data; get a dumb
  baseline; overfit to prove capacity; ONLY THEN regularize — but recognize when the
  generalization gap is irreducible label noise (Bayes error), not fixable regularization.
  Fix the eval before trusting a number (purged CV, canary). Don't be a hero — proven simple
  methods (GBT, calibration, meta-labeling) over novel architectures on noise.
  Ref: karpathy.github.io/2019/04/25/recipe/ .
- **Never conclude "unsolvable" without the Bayes-error test.** Do the hard analysis (nearest-
  neighbour label consistency, conditional entropy, model-vs-ceiling headroom) before deciding
  a lever is exhausted. Never fabricate a win; a >~0.56 true-15-min OOS number is a leak until
  proven otherwise (field ceiling, see §3).
- **FIXED ENTRY TIME (anti-shortcut invariant).** The decision/entry is made ONCE, early, when
  the contract still has value (cheap winning side). NEVER move the decision later to raise
  accuracy — a late call (e.g. T-1min) is near-certain but worthless (contract ~99¢). Any
  accuracy that depends on a later clock, on post-entry data, or on dropping coverage below the
  target is CHEATING and must be rejected. The benchmark is hit@coverage AT ENTRY.

## 3. Scientific anchor — established findings (with numbers; don't re-derive)
- **Capacity is NOT the limit.** High-capacity models reconstruct the labels **100%
  in-sample** (GBM/stacker). Reproducing seen data is trivial.
- **OOS ceiling is information-limited, not data- or model-limited:**
  - Price-path-only OOS ≈ **0.70 mid-window (~11 min left)**, ≈ **0.55 at window open (~15
    min left)**. Entry timing is the single biggest lever (0.55→0.70 as the window fills).
  - Best single OOS signal = zero-shot **chronos-bolt-base (0.696)** / the closed-form
    **barrier** (first-passage z). Every FM basically re-derives the barrier.
  - **Fusion did NOT beat the best single model** (stacker/GBM 0.655 < Chronos 0.696).
  - **Fine-tuning Chronos on all history overfits** (in-sample F1 0.846 → OOS 0.601 < base
    0.683). More training on our data hurt.
  - **Time-of-day/day-of-week alone ≈ 0.52 OOS** (all 6,337 windows) — negligible.
  - **Learning curve flat ~0.51–0.56** from 100→1,025 windows → more data barely helps on
    current features (information-limited).
  - **Concept drift is real**: a frozen model loses ~5.9pp over time; rolling-last-400 ≈
    expanding-all > frozen → retrain periodically, ~400 fresh windows suffice.
- **Sub-bin L2 OFI at the fixed OPEN entry (R19, decisive):** built from 20M L2 messages
  (book reconstruction verified, zero crossed spreads), 1s/5s/30s/60s OFI + queue imbalance +
  Stoikov micro-price over [open, open+180s]. OOS ≈ **0.51** (canary 0.497) — coin flip. Window
  outcomes are serially independent (lag-1 autocorr 0.0065); even the actual 180s micro-drift
  has AUC 0.515. → **At the TRUE open even full microstructure carries ~0 signal.** The ~0.70
  seen elsewhere comes from a slightly LATER regime (price has drifted off the strike →
  distance-to-strike/barrier), NOT from microstructure at the open. results/l2_ofi_eval.json.
- **Order-flow (R7):** carries real leakage-free signal ALONE (~0.63 OOS) but adds only
  **+0.7pp over the barrier** — because aggressive flow drives price, so the barrier already
  reflects it. Lesson: signal *derived from BTC's own price/flow* is largely redundant with
  the barrier. The orthogonality that can actually move OOS must come from data that is NOT a
  function of BTC's price: **cross-asset (COIN/IBIT/SPY/DXY), funding/basis, options skew,
  news/sentiment, LLM reasoning.** That is the forward program (R9/R11/R14).
- **Implication / the only lever:** add **orthogonal, non-price information** + smarter entry
  timing (R13). Fusion/complexity without new information does not help (proven R3/R7).
- **LEARNABILITY / BAYES-ERROR verdict (R21, the crux — settles "is it learnable"):**
  - Near-duplicate feature vectors finish the SAME direction only **52.6% mid / 40% open** →
    P(up|features) ≈ 0.5 for most windows → the label is **near-independent of current
    features** (high Bayes error). This directly refutes "reconstruct-in-sample ⇒ generalize":
    two windows that look identical go opposite ways ~half the time. Memorization ≠ signal.
  - Conditional entropy H(y|features) ≈ **0.91 bits mid / 0.96 open** (max 1.0) → mutual
    information I(y;features) ≈ **0.09 / 0.04 bits**. Almost all label entropy is irreducible
    given these features.
  - Bayes-ceiling(kNN) ≈ **0.645 mid / 0.587 open**; our best model = **0.662 mid / 0.540
    open** → **model is AT the ceiling mid-window (headroom −0.02); a hair below at open
    (+0.05).** No model headroom mid-window with current features.
  - This is NOT "unsolvable": a real ~0.66 mid edge exists and is tradeable. It means **more
    model = fitting noise; the ONLY way up is features that make near-duplicates agree more**
    (lower Bayes error) — i.e. orthogonal information — plus capturing the edge via
    sizing/selectivity (meta-labeling, calibration), per the field (below).
- **FIELD EVIDENCE (web scout, cited — research/... see report):** no clean peer-reviewed
  15-min crypto directional alpha; realistic OOS ≈ **52–56%**; treat any true-15-min backtest
  >56% as a leak. Wins come from **validation discipline, sample weighting, meta-labeling,
  calibration — NOT architecture.** Highest-ROI additions: sub-bin OFI (1s/5s/30s/1m) from raw
  L2; triple-barrier labels + meta-labeling (direction→act/size); queue-imbalance/micro-price;
  cross-venue lead-lag STATE; funding-z/basis/OI + VPIN/DVOL as **sizing/vol conditioners, not
  direction**; probability calibration + purged/embargoed CV + Deflated Sharpe. Free data
  (Binance/Coinbase/Deribit WS) suffices. Refs: Cont-Kukanov-Stoikov 2014
  (arxiv.org/pdf/1011.6402), López de Prado *Advances in Financial ML* 2018 (meta-labeling,
  triple-barrier, purged CV, Deflated Sharpe), Presto funding-rate study (T+1 R²≈0).
- Canary hygiene held throughout (0.500 mid / 0.513 open) — the numbers above are honest.

## 3b. ★ RETRACTED SHORTCUT + the REAL target (89%@90% AT ENTRY)
**RETRACTED (2026-09-15): "baseline met at T-1min" was a CHEAT.** Moving the decision to T-1min
makes the barrier hit 91% — but the contract is priced ~99¢ with 1 min left, so entering has
**zero trading value** (pay 99¢ to win 1¢). Accuracy that only appears by delaying the decision
is a fake win. **The entry time is FIXED by where the contract has value (early / cheap), and
must NEVER be moved later to inflate accuracy.**
- The value-of-waiting curve (research/value_of_waiting.json) is kept only as a DIAGNOSTIC of
  the barrier physics (accuracy ∝ 1/√t), NOT as a solution.
- **THE REAL TARGET: 89% hit @ 90% coverage at the ENTRY time** (early in the window, ~11-12
  min left / at open, when the winning contract is cheap and there is edge to capture).
- At that entry, current-feature ceiling is ~0.66 (§3, learnability). The capstone professor's
  baseline reaches 89% AT ENTRY → **we are missing features/data/method, not time.** Closing
  that gap honestly (orthogonal information + better modelling/labeling) is the whole task.
  Never again "solve" it by choosing a later clock, a confidence gate that drops coverage below
  90%, or any post-entry information.

## 3c. ENTRY-TIME feature-power scan (R21b) — what unlocks 89% at OPEN?
Become-one-with-the-data at the real entry (window open, ~876s left): price sits ON the strike
(median |price−strike| **1.3 bps**; 63% within 2 bps) → contract opens ~50/50, base-rate 0.487.
Standalone feature accuracy for the close (research/feature_power_scan.json):
`brti_distance_to_target 0.61 · k_prob(market) 0.59 · spot/target/current_brti ~0.56 · rvol 0.54`.
**No single feature > 0.61; full model over ALL logged features = 0.696 OOS.** Nothing near 0.89.
- **Verdict:** with current features, entry-time ceiling ≈ 0.70 (matches §3 Bayes + field 52–56%).
  0.89 AT OPEN predicting the close is not in our data. To reach it honestly requires: a
  different **label** (touch-vs-close), a feature in the **capstone dataset** we don't log, or a
  **leak**. Remaining untapped own-lever: **sub-bin L2 OFI** (R19) — field predicts ~0.55.
- **Action if 89% is truly the target: obtain the capstone's exact label/feature spec + dataset**
  — that is the actual blocker, not more modelling on our features.

## 3d. PURE CANDLE-HISTORY test (R21c) — the clean core question, DEFINITIVE
"Given all past 15-min candles + the known open, predict this candle's close?" On the FULL
6,337-candle series (contract_outcomes; leak-free, only past candles + current open),
walk-forward, canary 0.500: **GBM OOS 0.497 (in-sample 0.751), logistic 0.519.** Coin flip.
→ 15-min close-to-open direction is a near-**martingale**; candle history carries ~0 directional
info. **89% at open is NOT in candle history** (definitive). It could only come from: a
different LABEL (touch-vs-close), INTRA-WINDOW/microstructure data (later effective entry), the
capstone's specific dataset, or a LEAK. Artifact: research/candle_history_model.json;
scripts/candle_history_model.py. (Reassurance on latency-leak concern: coinbase_spot alone
predicts only 0.56 — we are NOT living off Coinbase→BRTI latency.)

## 3e. ★★ THE 89%@90% MYSTERY — SOLVED & REPRODUCED (R22, definitive)
Exhaustive published-methods scan (GitHub/Kaggle/papers/blogs/forums, cited in
research/... methods catalog) + our own reproduction settle it:
- **89%@90% is the STRIKE-LADDER / distance-to-strike task, NOT at-the-money-at-open direction.**
  Reproduced on our data (strike_ladder_test.py, 6,340 windows, synthetic ladder from real
  open+settlement): moneyness predictor `sign(open−K)` with ZERO forecasting →
  **coverage 1.0 hit 0.924 · coverage 0.90 hit 0.955 · 0.80 → 0.972 · 0.50 → 0.990.**
  Most ladder strikes are far ITM/OTM so their outcome is near-certain. This is a property of
  the strike distribution, not skill. (Our capture keeps only the 1 at-the-money contract/window
  — base 0.5011 — so the ladder wasn't visible before.)
- **Our real problem (T0 desk): the AT-THE-MONEY contract, decided at OPEN.** base 0.50; ceiling
  ~0.66–0.70 with intra-window features, ~0.52–0.58 at true open. This is the hard, EV-bearing
  task and we are at its ceiling.
- **Field corroboration (all cited):** McNally 52.78% (daily LSTM); Arain&Snudden hourly can't
  beat RW; G-Research live 15-min winning corr ~0.01–0.02; the two honest Kalshi-BTC-15m repos
  (oribarlevco top-confidence bucket caps ~82% on OUR label; SiddhaBasu Brier-only, no accuracy);
  gyusu easier touch-label LSTM only 0.55; selective-classification tops ~63% at low coverage;
  meta-labeling adds ~2–6pp precision by CUTTING coverage. Coinbase LOB-TCN 71% is 2-SECOND
  horizon, doesn't transfer. Every 90%+ paper = leak (shuffled split / full-sample scaling /
  same-bar or settlement-price feature / forward-smoothed label / MAPE-as-accuracy).
- **EV consequence:** ladder 95% accuracy has NO edge (pay ~95¢ for a 95% contract). EV lives
  ONLY in the at-the-money contract where the market is uncertain → the desk (T0/T1/T2) is
  correctly designed. "Beating 89%" for PROFIT ≠ the accuracy benchmark.
- **Reproduce-exactly TODO (targeted, not random):** oribarlevco walk-forward calibration table
  (confirm our top bucket caps ~82%); SiddhaBasu 22-feature + Platt harness (Brier near open);
  Marc-Seger leak as a NEGATIVE control; gyusu touch-label upper bound; selective risk-coverage
  curve on our own model.

## 3f. THE PROBLEM, FINAL & EXACT (owner-confirmed 2026-09-15)
- **Decision = open+6min (~9 min left), using the first 6 minutes of price action.** Fixed,
  tradeable (contract still ~9 min of value). NOT the true open, NOT a late cheat.
- **Contract (verified via live Kalshi API):** 1 at-the-money market/window, strike = prior
  window's settlement (floor_strike[N] = expiration_value[N−1]); no ladder, no round strikes.
- **Target:** 89% hit @ 90% coverage. **Exhaustively-measured honest ceiling = ~0.74@0.90**
  (barrier physics, 370 clean windows, canary 0.50). Gap 15pp. Owner has no professor specifics
  → we maximize the honest number + convert to EV + reproduce public repos.
- **Why 0.74 not 0.89:** at 9 min left `z=(price−strike)/(σ√t)` gives ~0.74; closing to 0.89
  needs 9-min drift-sign forecasting at 0.89, ruled out by ALL evidence (ours + field 0.52-0.58;
  best public Kalshi-15m repos cap ~0.82 at top bucket; every 90%+ = leak).

## 6.5 ★ 7-DAY EXECUTION PLAN (metric-gated; pass→advance, fail→record-truth→next)
Gate everywhere: walk-forward + purged/embargoed CV + label-shuffle canary≈0.50; report
hit@cov≥0.90 AND net EV. No shortcuts (fixed open+6min entry, no post-entry data, coverage≥0.90).
- **D1 — Canonical dataset + eval harness.** Lock the clean open+6min PIT dataset (largest
  reliable set w/ true time_remaining) + harness (walk-forward, purged CV, canary, calibration,
  hit@cov + EV). GATE: reproduces barrier 0.74 & canary 0.50.
- **D2 — Maximize the model.** Tuned GBM/logistic + probability calibration (Platt/isotonic) +
  Chronos + ensemble. GATE: beat barrier 0.74 by a real margin (aim ≥0.76), canary clean.
- **D3 — ★ ALPHA VANTAGE + CLAUDE (the professor's method, owner-confirmed).** AV cross-asset
  (COIN/IBIT/SPY/DXY intraday), AV NEWS_SENTIMENT, AV macro + Claude-as-signal (LLM directional
  read of the 6-min path + cross-asset context). Build @ open+6min with PIT **receipt-latency**
  (av_features.py) AND a NO-latency LEAK-CONTROL to quantify look-ahead — the likely source of a
  reported 0.89. GATE: honest (latency-correct) OOS lift over D2 AND leak-control shows the gap
  is/ isn't look-ahead. If AV bars without latency give ~0.89 but with latency ~0.75, the 0.89
  is a latency leak — report it. Plus order-flow, funding/options as secondary.
- **D4 — EV conversion (the useful goal).** calibrated prob → Kelly sizing on edge-vs-price +
  value-gate + selective; backtest net P&L on OFFICIAL settlement. GATE: positive EV OOS.
- **D5 — Reproduce public repos exactly.** oribarlevco (calibration), SiddhaBasu (22 feats);
  Marc-Seger leak as NEGATIVE control. GATE: match their calibration; leak-control≈0.50.
- **D6 — Robustness.** perf/latency panel, drift monitor (ADWIN/KSWIN), recreatable tests,
  consistency auditor, clean/lean refactor. GATE: all green + reproducible.
- **D7 — Deploy + report.** best model → shadow arm (T3, canary-clean OOS); final report;
  Models Lab updated. GATE: live, documented, reproducible.
  - ✅ **DONE (2026-09-15): open+6min barrier arm `ob` deployed LIVE.** Trades every window at
    open+6min (analytic first-passage barrier from minute candles), logs conf_z=|z|; T0 = control.
    Coverage A/B (results/coverage_ab.json, Models Lab): ob sliced at 90/80/70/60/50 coverage by
    confidence. Offline: cov90~0.74 hit, cov20~0.87. online.py ob arm + emit_coverage_ab.py; wired
    to cron+publish; daemon pid 77858. Fills as trades settle. This is the "filter the bads out"
    live A/B the owner asked for.

## 3g. ★★ AV+CLAUDE (professor's method) TESTED HONESTLY — 0.89 NOT reproducible (D3, definitive)
Reproduced the owner-stated method (Alpha Vantage + Claude) at open+6min, walk-forward, canary
~0.50, WITH receipt-latency (honest) AND a no-latency leak-control (research/av_claude_open6_eval.json):
- barrier only **0.744** hit@90 · +AV(honest) **0.715** (BELOW barrier — AV hurts) · +AV(no-latency
  leak-control) **0.709** (leak_magnitude −0.6pp → NOT a latency leak) · +AV+news **0.666** (worse).
- **Claude-as-signal ties the barrier exactly (0.70=0.70, 60-win sample):** a disciplined LLM read
  re-derives the barrier, adds nothing. Agent's own words: "I am effectively re-deriving the barrier."
- Why AV fails: only ~30% of BTC 15-min windows coincide with a live US-equity session; the rest get
  stale bars. News sparse/noisy at 15-min.
- **VERDICT: 89%@90% at open+6min on the ATM contract is NOT reproducible with any legitimate method,
  including AV+Claude. Honest ceiling = barrier ~0.74. A reported 0.89 is a leak in the professor's
  EVAL (in-sample / overlapping-window label overlap / different label or coverage) — consistent with
  the published record (every 90%+ audits to a leak).** ~12 experiments now converge.
- **Pivot (D4): the useful goal is EV.** 0.74 hit is real & tradeable; convert to positive P&L via
  calibrated sizing + selectivity, NOT chase an unreproducible 0.89.

## 4. Live system state
- Roster: **T0 `pt`** (control, $100M) · **T1 `cg33`** (gated 33% follower) · **T2 `fm`**
  (chronos-bolt-base directional, conf≥0.60, half-Kelly). Daemon `btc_rl.online` pid live.
- T2 live P&L so far: tiny sample, noisy (do not over-read < ~30 settled trades).
- Retired (gated, ledgers frozen): pt2/3/4/5/6/7/8, cg5, cg10, tv.
- Data on hand: ~6,337 settled labels (contract_outcomes); ~1,366 windows with logged PIT
  price paths (feature_snapshots); event tapes Aug30–Sep15 (Coinbase trade/L1, Binance
  xvenue, L2 micro); av_cache (COIN/IBIT intraday, CPI/rates/yields); AV key at
  `~/.alphavantage_key`; news_backfill.py pulls AV NEWS_SENTIMENT (BTC).

## 5. TASK BOARD
Columns: ID · task · stage reached · status · depends-on · finding/next.

### Done ✅
| ID | Task | Stage | Finding |
|---|---|---|---|
| R0 | Great Roster Cut (T0/T1/T2, retire rest, UI, tests) | La ✅ | Live + visible; commit dcee375 |
| R1 | FM benchmark (Chronos/TimesFM/Kronos/barrier/market) | Ev ✅ | chronos-base best (F1 .683, prec .74, 15 FP); no T3/T4 |
| R2 | System audit (files, procs, memory, cleanup, bugs) | Ev ✅ | SYSTEM_AUDIT_2026-09-15.md; disk/token/publisher risks |
| R3 | Combined student (teachers+features+retrieval+GBM) | Ev ✅ | fusion 0.655 < Chronos 0.696; COMBINED_STUDENT_FINDINGS.md |
| R4 | In-sample reconstruction (all models) | Ev ✅ | 100% in-sample, OOS 0.54–0.70; capacity not the limit |
| R5 | Chronos fine-tune on all history | Ev ✅ | overfits; OOS 0.601 < base 0.683 |
| R6 | Data-size + retraining-cadence study | Ev ✅ | data plateaus ~0.55; drift ~6pp; retrain rolling-400 |
| R7 | Order-flow exo features (CB OFI, L1 imb, Binance xvenue) | Ev ✅ | order-flow ALONE OOS ~0.63 (real, canary 0.507); +barrier only +0.7pp (redundant w/ barrier — flow drives price); time-of-day dilutes. results/exo_features.jsonl (1400) |

### In progress 🔄
| ID | Task | Stage | Depends | Next |
|---|---|---|---|---|
| R8 | Web scout: what pro traders/funds/bots/curricula use + data gaps | Fe 🔄 | — | agent running; feeds R9/R11 priorities |

### Scheduled ⏳ — REPRIORITIZED by learnability + field evidence (labeling/sizing beats new models; new features beat bigger models)
| ID | Task | Stage | Depends | Why / notes |
|---|---|---|---|---|
| **R17** | **Meta-labeling + triple-barrier labels** (direction model → LightGBM act/size model; profit-take/stop/time barriers) | Bu | — | **Field #1 ROI, no new data.** Captures the real edge via sizing/selectivity not better direction. LdP 2018. |
| **R18** | **Probability calibration + purged/embargoed CV + Deflated Sharpe** in the eval harness | Bu | — | **No new data, pure rigor.** Our τ-threshold decision needs calibrated P(up); prevents overstated edge. |
| R19 | Sub-bin OFI from raw L2 (events_micro cb_l2): signed Δsize at 1s/5s/30s/1m + queue imbalance + Stoikov micro-price | Bu | events_micro (12GB) | Field top feature gap; orthogonal-ish (but R7 showed flow ~redundant w/ barrier — test increments) |
| R20 | Funding-z / spot-perp basis / OI-quadrant + VPIN + DVOL as **sizing & vol conditioners** (gate/size, NOT direction) | Fe→Bu | Deribit/OKX in sources.py; capture check | Field: no directional edge; real value is sizing/vol regime |
| R21 | Learnability / Bayes-error analysis | Ev ✅ | — | DONE — model AT ceiling mid (headroom −0.02); near-dup 0.53; see §3 |
| R13 | Value-of-waiting / entry-timing (accuracy vs mins-left → optimal entry) | Bu | — | independent; entry timing is the biggest proven lever (0.55→0.70) |
| R9 | Exhaust Alpha Vantage: news-sentiment + COIN/IBIT/SPY/DXY + macro → PIT features, OOS lift | Bu | av_features.py | field: sentiment weak at 15min; test anyway (cross-asset may help sizing) |
| R11 | Claude-as-signal / LLM teacher (reads headline/context → directional prob; "LLMs for gambling") | Bu | R9 | Tauric-style; also world-model research thread |
| R12 | Multi-model stack + custom quant losses (uniqueness×|ret| weighting, Sharpe-utility, focal A/B, rank/AUC) | Bu | R17,R19,R20 | only after ≥1 new feature family / meta-label shows lift |
| R16 | Best model → shadow arm (T3) once it beats base OOS AND canary-clean | On | R17–R20 | gate: beat chronos-base F1 0.683 OOS |
| R15 | Loss-diagnosis harness (attribute losses to missed indicator) | Bu 🧊 | — | staged; TODO(human) attribute_loss in scripts/loss_diagnosis.py |

### Integrity / UI / Ops ⏳
| ID | Task | Status | Notes |
|---|---|---|---|
| R25 | Data-integrity audit (labels, dataset consistency, ledger math, A/B metric correctness, cross-spot) | 🔄 | agent running |
| R26 | UI element + latency inventory (every page/card, data source, writer, update freq, latency) | 🔄 | agent running; feeds R27/R28 |
| R27 | Performance/latency PANEL — new UI section showing end-to-end freshness + per-stage latency (capture→feature→model→ledger→emit→publish) | Bu | R26 | user-requested; pull from receive_ts/decision_ts/persist_ts/publish log/health.age_s |
| R28 | Audit + update ALL pages/subpages/cards (correctness, staleness, dead elements) | Bu | R26 | fix trader_detail (numpy publish failure) etc. |
| R29 | Concept-drift monitor (ADWIN/KSWIN/Page-Hinkley on error stream) → trigger retrain | Bu | R6 | field: rolling refit + drift trigger |
| R30 | Clean/lean refactor: distill winning pipeline into a small, human-usable module + a Skill/automation for the research loop | Bu | R17–R20 | user: "clean, lean, human usable"; workflows/agents/RAG where they earn it |
| O1 | Rotate leaked GitHub token in /tmp/btc_publish.log | ⏳ owner | SECURITY — needs owner |
| O2 | Archive/rotate event tapes (20GB, +2GB/day) — archive, don't delete (R19/R20 raw data) | ⏳ | |
| O3 | Fix publisher silent failures (numpy env, main-sync push) | ⏳ | pre-existing |

### Research understanding (feeds R11/R12/R30) ⏳
| ID | Topic | Notes |
|---|---|---|
| K1 | World models (for market state) | Ha & Schmidhuber; JEPA — is a latent world-model of the book/flow worth it vs GBT? Likely no at our SNR, but scope it. |
| K2 | LLMs for trading/"gambling" | Tauric TradingAgents/Trading-R1 (unsuitable intraday, established); LLM as slow context/news-reasoning signal only (R11). |
| K3 | BTC market microstructure literature | scout report (§3 refs) is the spine; deepen on OFI/lead-lag/meta-labeling. |

## 6. Background jobs running now
- R7 order-flow extraction → results/exo_features.jsonl (slow 7GB scan; then exo_eval.py).
- R8 web-scout agent (trader/fund/feature intelligence report).

## 7. Dependency & parallelization map
- **Independent, run any time:** R7 (order-flow), R8 (scout), R13 (entry-timing), R6 ✅.
- **R9/R10/R11 feed R12** (architecture) and **R16** (shadow arm) — features before fusion.
- **R12 custom losses** only meaningful after ≥1 orthogonal feature family shows OOS lift;
  else it just re-tunes noise.
- **R16 launch** gated on a canary-clean OOS beat of chronos-base — never launch on in-sample.
- Non-conflicting: feature-build tasks (R7/R9/R10/R14) write separate results/*.jsonl and
  don't touch the live daemon; only R16 touches online.py (one arm), after offline proof.

## 8. Update protocol (for me)
1. Before starting work: add/mark the task 🔄 in §5, note deps.
2. On a result: record the number + artifact path in the task row and §3 if it's a finding.
3. On launch: move to La ✅ and note the commit.
4. Keep §3 authoritative — it's what stops me re-running settled questions.
5. Commit this file with every meaningful change.

*Last updated: 2026-09-15, after R7 (order-flow). Commits: dcee375, 83dbf71, 8f91646, 95673ae, 64fc508.*
