# MANUAL — onboarding for the next team

How the car was made, how to drive it, how to fix it. Everything here is
grounded in NOTES.md (the research log), docs/SEV0_REMEDIATION.md, the
git history, and the code. Where a claim has a date, that date is the
NOTES.md entry or commit that recorded it. Decision *process* lives in
docs/DECISIONS.md (companion document).

---

## 1. WHAT THIS IS

An always-on research system that predicts Bitcoin's price minutes ahead
and paper-trades the Kalshi 15-minute BTC binary contract (KXBTC15M).
**No real money trades anywhere.** The only real-exchange contact is a
Kalshi *demo* account mirror (fake balance, real order lifecycle) whose
host is hard-coded to the demo API (scripts/demo_trader.py).

One daemon (`btc_rl/online.py`, ~4,000 lines) runs the whole stack on a
30-second poll loop. The architecture is six tiers (the labels come from
the SEV-0 page, site/sev0.html):

- **T0 — data feed.** Open, no-auth streams: Coinbase 1m bars + trades
  (ticks.jsonl), OKX funding, Deribit mark, order book, RSS news scored
  by a frozen CryptoBERT (llm_sentiment.py), Kalshi quotes/depth/OI.
  Snapshotted every poll into live_snapshots.jsonl.
- **T1 — RL price arms.** Nine-ish arms (control tabular-Q `h*`, replay
  `rp`, LinUCB `t2`/`t6`/`t10`/`t11`, linear-Q `t7`, distributional DQN
  `t8`, LSTM `t9`) commit integer price predictions every 5 minutes at
  +5/+15/+30 horizons into prediction_log.jsonl. Arms never share model
  state, so metric gaps are attributable. Learning is two-speed: an
  immediate update per scored prediction plus an hourly replay retrain
  behind a hold-out no-regression gate (bad retrains revert).
  **Important (leader_audit, 2026-08-28): most T1 arms are logged-only —
  the trading path runs off t8's distribution + the market anchor.**
- **T2 — kb probability arms.** Once per minute per open contract, each
  kb arm writes P(close ≥ strike) into kalshi_binary_log.jsonl:
  `kb` (control: t8's distribution, per-phase calibrated), `kb2` (market
  blend), `kb3` (online logit, 24 features), `kb4` (stack of kb2×kb3),
  `kb5` (train-where-you-trade EV logit), `kb6` (fast-information —
  RETIRED from trading 2026-08-26), `kb7` (Chronos-Bolt-small, frozen
  zero-shot), `kb8` (learned log-opinion pool of kb7 × market, 3
  weights), `kb9` (TimesFM 2.5, frozen zero-shot). Every row also
  carries `p_m1`, the shadow Platt-calibrated probability (M1).
- **T3 — decision layer.** A leaderboard picks the "leader" arm (best
  of last 10 gate-clearing decisions among PT_ARMS = kb2/kb3/kb4/kb7/
  kb8/kb9); entries require confidence ≥ 0.62 (PT_TAU).
- **T4 — traders.** Eight paper traders take the same signals with
  different sizing/entry policies (section 3, roster table).
- **T5 — execution.** Modeled fills at the quoted ask + Kalshi's fee
  `ceil(7·p·(1−p))` cents/contract; measured to be the biggest leak
  (2.70 pts EV real-vs-quote, commit dd29457). pt7/pt8 and treatments
  M10/M11 probe this tier.
- **Observation loop.** Cron: publisher every minute (pushes pages +
  data snapshots to gh-pages on theaakritigupta.com), watchdog every 5
  (restarts a stale daemon), self-audit every 10 (run_audit.py →
  audit_report.json), caffeinate keep-awake every 5.

### Standing constraints (the house rules, all evidenced in NOTES.md)

1. **Additive-only treatments.** Improvements ship as NEW arms/traders/
   treatments beside untouched controls — never as edits to a running
   policy. Controls (kb, kb2, the Follower, the Gambler's sizing, pt3's
   frozen rule) exist to be beaten and stay frozen.
2. **Pre-registration.** Every policy is written down (NOTES.md and/or
   code constants) *before* it trades; threshold changes invalidate the
   track record (see PT3_TAU comment). Policy versions are stamped on
   rows (`pv:2`), cutovers are dated, history is never rewritten.
3. **Window-counted stats.** Effective n = 15-minute windows, never
   minute rows — entries in one window share fate. Origin: the kb7
   "13/14" counting error (NOTES 2026-08-25).
4. **No leakage.** Decision-time inputs only; calibrators train once
   per (arm, window) on the decision row; treatment scoring uses the
   pre-settle `p_m1` stamp, never a calibrator that saw the outcome;
   Fixed-Share updates weights only after a window is used.
5. **Live evidence beats backtests.** Nothing is adopted on offline
   evidence alone — three confident offline conclusions were later
   wrong (tier-1 RCA, the toxic hour, the M1 calibration premise).
   Promotion is by SPRT on paired live windows only.
6. **Honest labeling.** Replay evidence is labeled replay and never
   merged with live streams; model-priced EV is labeled optimistic;
   frozen historical MAE numbers are never re-derived as MSE
   ("recomputing MSE from retained aggregates is impossible and
   altering a dated snapshot would be falsification" — NOTES 08-28).

---

## 2. HOW THE CAR WAS MADE

249 commits, 2026-08-19 → 2026-08-28. Six phases.

### Phase 1 — batch RL POC (08-19/20)

`d6d7030` "BTC integer-price prediction as RL — POC (contextual
bandit)". One-step episodic env (env.py): see features, pick an integer
dollar delta, reward on hitting the future price. Persistence baseline
(L0), tabular Q (L1), 120-day backtest → results/metrics.json →
site/index.html. Key early finding (README): minute-scale BTC is
approximately a martingale — persistence is near-unbeatable on price
*level* (MASE ≈ 1 is the ceiling), so real edges must live in
direction, calibration, and market-relative Brier.

### Phase 2 — online daemon + the arm ladder (08-20 →)

`6eace93` made it always-on; `4a54f37` introduced control/treatment
arms. The ladder grew one capability at a time (t2 LinUCB → t6 live
streams → t7 linear-Q → t8 dist-DQN → t9 LSTM → t10 Kalshi features →
t11 RLHF), each isolating exactly one addition. Discipline hardened
here: an offline gate retired t3/t4/t5 as near-duplicates (`c59cca2`),
a leakage audit found and purged two backfill leaks (`83bdde1`), five
underperforming arm×horizon streams were retired on evidence
(`3bfd350`), the retrain-crash incident produced the watchdog, and the
betting layer was briefly added then removed to keep the objective pure
(`f1625c3`).

### Phase 3 — the kb arms ladder (Kalshi binary)

`8a1004d`→`8a910f7`: the kb arm calls the KXBTC15M contract every
minute, Brier-scored against the market's own odds. Then the binary
treatment set kb2/kb3 (`97eb29d`), stacking kb4 (`33cf2b8`), EV-arm kb5
(`9832d00`), fast-information kb6 (`ad2cc85`), foundation-model kb7
(`c1b646b`), fusion kb8 + the kb9 upgrade gauntlet (`9a78323`).

The gauntlet is the phase's signature negative: **five kb7-upgrade
attempts (4× params, Chronos-2 at 512/2048, covariates, TimesFM,
fine-tuned Bolt) all landed within noise of the 47M zero-shot
Bolt-small** (|t| < 2 on window-clustered paired Brier). Conclusion on
record: at the 15-minute horizon the price path's extractable signal is
the binding constraint, not the model. kb9/TimesFM later launched
anyway — explicitly as a second *family* for decorrelation, not as an
upgrade (NOTES 08-25). Other phase findings: maker execution measured
and rejected (−10.6c/contract adverse selection, `a1a962f`); the TA's
$2M question answered NEVER at any capital (`5bd5223`); the detection
sim's stealth optimum x* ≈ 25% of depth (tests/sneaky_trader.py).

### Phase 4 — the paper desk (08-25 →)

The TA-spec $1K Desk (`996abb2`) grew into an eight-trader sizing
curriculum (roster in section 3). The design principle: same entry
pool, one variable isolated per pair — Follower vs Gambler isolates
sizing; Follower vs Patient isolates execution. Pre-registration became
mechanical: pt3's 0.77 gate frozen before any trade with the thin-n
caveat stated ("17/18 wins at registration, THIN n — that is why this
live test exists"); the Gambler was amended 100%→33% *before
meaningful history* and stamped. Tier evidence was pre-registered
(Method A market-paired scoring, Method B replay; SPRT considered and
NOT adopted at that point), and the replay promptly delivered the
phase's big negative: **the live tier's 20/22 (+21%/$1) did NOT
generalize — 663 replay windows gave +3.6%/$1, clustered t = 0.85, not
significant.** Both numbers are shown on the site, labeled, never
merged. The Kalshi demo mirror proved real order plumbing (201
accepted, `c747300`) with zero financial exposure.

### Phase 5 — the SEV-0 audit era (08-26 → 08-28)

The desk lost money; the response was a full-architecture error audit
(`8e4c6f3`, tests/sev0_error_audit.py) plus an RCA in NOTES and a
cited remediation spec (docs/SEV0_REMEDIATION.md) with mitigations
M1–M7. What makes this era the most instructive is how much of the
first RCA was later *overturned by its own tests*:

- **The toxic hour was noise (M7 dropped).** 09h/14h looked bad, but 24
  hours were searched; at the desk's loss rate the expected number of
  "persistently bad" hours by luck was 1.98, observed 2 —
  **P(≥2 | chance) = 0.60**. Shipping M7 would have been data dredging
  (NOTES 08-28).
- **The tier-1 "bullish drift prior" root cause was wrong.** The biased
  arms (t10, t2, t11) never reach a trading decision; the
  decision-feeding path is near-unbiased (+$1.76 mean). Corrected
  chain: bands under-cover (74–76% vs 80%) → sigma too small →
  probabilities pushed to extremes — a DISPERSION defect, not a
  DIRECTION defect. M4 re-aimed at conformal bands.
- **08/27 was one regime day**, not a defect: three of four days were
  above water; trailing-20 market accuracy < 0.62 flagged 63.6% of
  08/27's windows ex ante → M8 regime gate.
- **The best fixed arm flipped in one day** (kb4 +4.7%/$1 → −3.7%;
  kb9 the only positive) → "freeze the best arm" rejected, Fixed-Share
  (M3) promoted.
- **M1 calibration shipped to SHADOW and its premise was NOT
  confirmed: it helps only kb7 (1 of 9 arms)** and is marginally worse
  on the other eight. It stayed shadow. Two implementation bugs were
  found by checking output (rank-1 per-sample Newton froze b at exactly
  1.000; decayed-accumulator Newton overshot) — fixed with a
  sliding-window batch refit.
- Best positive surprise: **kb5 wins only 44.7% of decisions but
  returns +16.7%/$1** — deep longshots where break-even is far below
  50%. The most-wrong caller is the most profitable one → M9 Underdog.

### Phase 6 — the treatments framework (08-28)

`7ce3475` btc_rl/treatments.py: every improvement runs live as a
champion/challenger TREATMENT, paired same-window (regime cancels in
the difference), window-counted, adjudicated by Wald's SPRT so
continuous monitoring is valid, promotion stamped and reversible with
the loser kept running. Building it surfaced **three near-miss
auto-promotions in quick succession — the framework's best argument
for itself:**

1. **SPRT variance collapse.** Identical early paired differences
   (both policies stand down ⇒ d = 0) drove the running variance to ~0
   and the LLR exploded: M1 hit LLR 123 on 154 windows — a FALSE
   auto-promotion on numerical noise. Fix: variance floor 0.01 + 12-
   window warmup.
2. **The 2.1c pricing artifact.** Champion scored at its REAL ask,
   challengers at the modeled quote — a systematic 2.1c discount that
   manufactured nearly all challenger edge (M3 "+23.52%" → +1.17%
   after the fix). Fix: every policy prices from the same
   decision-time quote; only the paired difference is meaningful.
3. **The stale-quote row.** Treatment evaluation had no ≤12-minute
   envelope filter, so policies were priced ~3.8 min before the desk
   traded; the champion published +11.75%/$1 when decision-time truth
   was −6.5%. Fix verified by *paired prediction*: an independent
   agent predicted −6.5% champion / ~30% knife-edge veto rate / 29.3%
   M10 trip rate; post-fix measured −6.04% / 29% / 29% — three
   predictions, three matches (commit f273006).

Also in this phase: the **p_cal field collision** (`7861625`) — the M1
"shadow" layer stamped `p_cal` onto kb rows, a field the kb2
blend-weight fit already read, so a supposedly non-trading layer was
steering live kb2; fixed by renaming to `p_m1` and scrubbing 390 rows
(values only). M10 execution guard (execution leak 2.70 pts EV — more
than double the best treatment edge). **M11 naive maker limit backfilled
−5.21%/$1 PAIRED** — resting bids are adversely selected
(Glosten-Milgrom picked-off problem); pt7 keeps the naive rule live on
purpose as the measuring stick, pt8 adds the fill-time signal re-check
that M11 lacks. M12 EV-ranked leader (the win-rate leaderboard selects
market-echo arms that win just under their own break-even). MSE became
the headline error metric (user-directed), nine verified bug fixes
shipped, and the 10-minute self-audit cron was installed.

---

## 3. HOW TO DRIVE IT

### Processes

```bash
# the daemon (predict/trade/score/learn — the whole live system)
python3 -m btc_rl.online &            # run forever
python3 -m btc_rl.online --once       # backfill + one pass, then exit
python3 -m btc_rl.ticks &             # tick archiver (separate process)
python3 scripts/demo_trader.py        # optional Kalshi-demo mirror of pt3
tail -f results/online_daemon.log     # watch it think
```

Stop: kill the python process; the watchdog will NOT restart it only if
you also remove/comment its crontab line. Restart: just start it again —
all state is on disk in results/, and the daemon resumes (models, seen
sets, treatment SPRTs, bankrolls are all persisted). The watchdog
restarts it automatically when results/online_status.json's `alive_at`
goes > 5 min stale (grace 10 min after a restart).

### The crons (verify with `crontab -l`)

| Schedule | Script | Job |
|---|---|---|
| every 1 min | scripts/publish_dashboard.py | push pages + trimmed data snapshots straight to gh-pages (site freshness ~1–2 min); hourly slow-path sync into the main site repo |
| every 5 min | scripts/watchdog.py | restart daemon on stale heartbeat → watchdog_log.jsonl |
| every 5 min | caffeinate | keep the laptop awake (single-machine hosting) |
| every 10 min | scripts/run_audit.py | recompute desk/trader/tier1/tier2/treatment health from raw ledgers → audit_report.json (never imports the daemon) |

### Batch/offline tooling

```bash
python3 -m btc_rl.train --days 120     # batch tabular-Q → metrics.json
python3 scripts/train_l2.py|train_l3.py|train_l4.py   # t7/t8/t9 batch
python3 scripts/evaluate_all.py        # every task, every arm
python3 scripts/standings.py           # ranking + DM tests
python3 scripts/offline_gate.py        # gate for NEW feature-bandit arms
python3 scripts/feedback.py up|down    # RLHF vote for t11 (30-min window)
python3 tests/gate_check.py            # step-0 gate PASS/FAIL report
python3 tests/sev0_error_audit.py      # full-architecture audit
python3 -m http.server 8787            # serve site/ locally
```

### Data flow: who writes what, who reads it

| File (results/) | Writer | Read by |
|---|---|---|
| prediction_log.jsonl | daemon (T1 arms) | live_online, experiment_review, sev0, standings/evaluate_all/audit |
| kalshi_binary_log.jsonl | daemon (T2 kb arms) | home, live_online, experiment_review, ab_dashboard, sev0, audits |
| kb_bets.jsonl / kb_bets_sel*.jsonl / pb_bets.jsonl | daemon (paper bet sims / selector A/B / Conviction Book) | home, live_online, ab_dashboard |
| pt_trades.jsonl … pt8_trades.jsonl | daemon (T4 traders) | home, ab_dashboard, sev0, gate_check, run_audit |
| treatments.json / treatments.jsonl | daemon (SPRT registry state / per-window log) | sev0, run_audit |
| kb_calib.json | daemon (M1 shadow Platt state) | published in online_status.kb_calib |
| online_status.json | daemon, every poll (heartbeat + everything live) | home, live_online, ab_dashboard, sev0, watchdog |
| live_snapshots.jsonl | daemon (T0 feature snapshots) | daemon itself (t6/kb features), sev0 audit |
| learning_log.jsonl | daemon (update/retrain counters) | live_online |
| metrics_history.jsonl | history.append_history (hourly gates, batch runs, retirements; git-SHA stamped) | experiment_review, live_training |
| metrics.json | btc_rl.train (batch backtest) | index, experiment_review |
| training_progress.jsonl, live_status.json | batch trainers | live_training |
| ticks.jsonl | btc_rl.ticks | sev0 audit (T0 health) |
| audit_report.json | run_audit.py cron | sev0 |
| watchdog_log.jsonl | watchdog.py | humans |
| demo_orders.jsonl / demo_fills.jsonl / demo_account.json | demo_trader.py / demo_reader.py | home (demo panel) |
| kalshi_history.jsonl, best_bids.jsonl | tests/kalshi_history_miner.py (14-day TRUE-quote mine) | tier_replay, entry-policy studies |
| q_table_online_*, linucb_*, dqn_t8-*, lstm_t9-*, kb*_logit.json, pt6_logit.json | daemon (online model state) | daemon on restart |

### The site (8 pages, all static, published to theaakritigupta.com/btc-oracle)

- **home.html** — the front door: live signal, active bid, desk equity
  race, per-trader cells, demo-account panel. The "answer first" page
  (commit 45ac8ea cut 12.7 screens to 2.5).
- **live_online.html** — the lab bench: ticker, per-horizon predictions
  + bands, kb calls, learning telemetry.
- **experiment_review.html** — the scoreboard: MASE/MSSE, DM, direction,
  pinball/coverage, Brier races, reliability diagram, retrain timeline,
  the disproved wall.
- **ab_dashboard.html** — trader revenue+risk A/B (net P&L, return,
  win%, max drawdown, ranked by net÷DD) + selector/bet ledgers.
- **sev0.html** — the live incident tracker: interactive T0–T5
  architecture explorer, per-tier metrics, treatment board with SPRT
  progress bars, loss attribution.
- **index.html** — the frozen 120-day batch backtest (historical MAE —
  deliberately not recomputed as MSE).
- **live_training.html** — batch training curves + hourly retrain-gate
  strip.

### Treatments: how a change gets in

1. Write the policy as a `decide(ctx) → None | {"side","ask_c"}`
   function in `_treat_policies()` (online.py); standing down is a real
   decision that scores 0.
2. Register it with a key, label, and one-line rationale. Config
   (edge=0.02 EV/$1, alpha=0.05, beta=0.10, min_n=40) comes from code
   constants — persisted state carries only the accumulated evidence
   (that's a deliberate bug fix; see treatments.py `load`).
3. Backfill with tests/backfill_treatments.py (same evaluator the
   daemon runs, so live continues the identical computation).
4. The SPRT accumulates paired EV differences vs the champion *of the
   same pricing family* (model-priced vs real-fill — mixing bases is
   the 2.1c artifact). Verdicts: `collecting` / `promote` (LLR ≥
   log((1−β)/α) ≈ 2.89) / `reject` (LLR ≤ log(β/(1−α)) ≈ −2.25).
   Baselines get n and EV but never a verdict.
5. Promotion is a stamped, reversible event and — per the practiced
   process — a user decision, not an automatic one (see
   docs/DECISIONS.md). Absolute own-EV numbers are model-priced and
   optimistic; **only the paired difference means anything.**

### Trader roster (paper, all fills modeled, fees in)

| Trader | Policy | Role |
|---|---|---|
| pt Follower | leader's call ≥0.62, 10% of funds | the control |
| pt2 Ladder | Follower + banks a level at 11× ($1k→$10k→…) | ratchet baseline |
| pt3 Disciplined | conf ≥ 0.77 (kb7 or leader), 10% — FROZEN, pre-registered | the star; policy untouchable |
| pt4 Gambler | 33% stakes (~1.6× Kelly); v2 adds the 0.77 gate + $10k reset; v2.1 adds edge-at-fill ≥ 2c | ruin/variance demonstration |
| pt5 Saver | 10% stakes (was 25%, bled −31%), skims 25% of wins | profit-ratchet study |
| pt6 MLE | online logit P(win), bet iff edge ≥ 10c, half-Kelly cap 10%, shadow-row learning | meta-labeling; best desk EV (+13.5%/$1, 84% idle) |
| pt7 Patient | Follower signal, naive limit at ask−2c, fill-or-skip | adverse-selection measuring stick (M11 = −5.21% paired) |
| pt8 Ideal | regime gate + edge ≥ 2c at limit + maker limit + half-Kelly ≤10% + 25% depth cap + fill-time signal re-check | the composite of everything verified |

---

## 4. HOW TO FIX IT

### Failure modes already seen — signatures and fixes

| Failure | Signature | Where fixed |
|---|---|---|
| **Field collision** (shadow layer steers live) | a "shadow" output written into a field an existing reader consumes; kb2 blend weights moved without any policy change | `7861625`: renamed p_cal→p_m1; tests/scrub_p_cal.py cleaned 390 rows. Rule: grep every new field name before stamping it |
| **Stale-quote scoring** | treatment EVs implausibly good; champion +11.75% while the desk bleeds; veto fire-rates ~2.5× expectation | ≤12-min envelope filter in `_treat_evaluate` row selection; verified by 3 matched predictions |
| **Variance-collapse SPRT** | LLR in the hundreds on ~150 windows; early paired diffs all identical (d=0) | SPRT.VAR_FLOOR=0.01 + WARMUP=12 (treatments.py) |
| **Pricing-basis mixing** | every challenger beats the champion by a similar few % | one `_ask()` for all policies; pair real-fill policies only against champion_real (FAMILY map) |
| **Per-contract fee ceil** | phantom fee accrual ($2,355) on multi-contract stakes | `_order_fee_c` per-order fee; per-contract ceil kept only for single-contract selector ledgers |
| **Trader lockout** | pt3–pt6 silently never enter; entry-minute guard only covered some traders | entry guard extended to all traders (f273006 fix #1) |
| **Calibrator leakage** | treatment gate reads a calibrator that already trained on the window's outcome | score from the pre-settle `p_m1` stamp; stand down if absent, never peek |
| **SPRT config drift** | retuning TREAT_EDGE/TREAT_MIN_N has no effect | load() restores evidence only; config always from code constants |
| **Degenerate online Newton** | Platt b frozen at exactly 1.000; a drifting to −2.7 | sliding-window (150) batch refit every 5 updates |
| **Wedged connection pool** | endless read-timeouts after a network blip while a fresh session works | `_reset_session()` in sources.py |
| **Silent daemon stall** | predictions flow but online_status.json `alive_at` stale for weeks | watchdog cron + heartbeat during retrains/retries |
| **Retire-revert momentum** | rejected hourly retrain leaves Adam state behind | checkpoints carry optimizer state; revert rebuilds it |

### Invariants to check after ANY change (the practiced routine)

1. **It compiles**: `python3 -m py_compile btc_rl/*.py scripts/*.py`;
   for site edits, syntax-check the inline JS (`node --check` on the
   extracted script) and **render-verify** — actually load the page
   (scripts/cdp.py can screenshot a live Chrome tab); several fixes in
   the log came from screenshot-driven review (`6710379`, `8fd06b8`).
2. **Backfill re-run**: `python3 tests/backfill_treatments.py` — the
   backfill and the daemon share one evaluator; if your change breaks
   the pairing, the backfill numbers move when they shouldn't.
3. **Paired-prediction match**: before trusting a fix, predict what the
   corrected numbers should be from an independent computation, then
   compare (the stale-quote fix was accepted on 3/3 matches). A fix
   whose post-fix numbers you couldn't predict is not yet understood.
4. **Two implementations must agree**: when Python audits and page JS
   compute the same metric, run both (the 08/28 re-triage found them
   disagreeing on loss attribution — gated vs ungated entry sets).
5. **Controls unmoved**: kb/kb2 and the frozen traders must show zero
   regression (the SEV-0 close condition includes this explicitly).
6. **No new leakage**: any new feature/gate must be computable strictly
   at decision time; re-read the leakage rules in constraint list §1.

### Where the diagnostic tooling lives (all in tests/, all re-runnable)

- **tests/sev0_error_audit.py** — the full-architecture DFS: per-tier
  metrics (T0 feed health, T1 MSE/bias/coverage, T2 Brier/AUC/
  calibration slope+intercept, T3 leader churn + fixed-arm
  counterfactual, T4 EV/drawdown) + per-loss root-cause attribution.
- **tests/gate_check.py** — PASS/FAIL of every shipped fix against its
  pre-registered gate, at the break-even actually paid (not a stale
  constant).
- **tests/regime_detect.py** — the ex-ante regime question and the
  toxic-hour luck analysis (the p=0.60 computation lives here).
- **tests/leader_audit.py** — pure-poison-leader check (binomial
  p-values) + whether each T1 arm's output actually reaches a decision.
- Also: tier_replay.py (Method B replay), execution_audit.py (the 2.70
  pt leak), metatrader_baseline.py, sneaky_trader.py (capacity),
  kb8_feature_lab.py / kb8_gauntlet.py (upgrade gates).

---

## 5. THE LEDGERS

All JSONL, append-only, cents-denominated money fields (`*_c`).
Timestamps are unix seconds; site displays PT (and the user's IST).
Snapshot copies on gh-pages may be line-capped by the publisher
(prediction_log 4000, kalshi_binary_log 6000, learning_log 1500,
treatments.jsonl 2000); the local files are the source of truth.

| File | One row = | Key fields | Cap |
|---|---|---|---|
| prediction_log.jsonl | one arm×horizon prediction | variant, made_ts/target_ts, price_now, delta, pred, lo/hi/sigma (80% band), actual, err/abs_err, hit, cal_adj, state/src | 60,000 rows (~2 weeks) |
| kalshi_binary_log.jsonl | one kb-arm call, one minute, one contract | variant, ticker, strike, made_ts/close_ts, mins_left, p_up, p_m1 (shadow calibrated), call, conf_entry, mkt_p_up, ask_c, base, pf (path features), b5x/b8x…, trained, actual, hit, ev_c | 20,000 rows |
| kb_bets.jsonl | one one-bet-per-window paper bet | ticker, side, price_c, edge_c, forced, p_model, sel_* (selector verdict), pnl_c, win | — |
| pb_bets.jsonl | Conviction Book entry (kb5-gated, +EV only) | src, side, price_c, p_win, pnl_c, win | — |
| pt_trades.jsonl (+pt2…pt8) | one trader window (bet or skip) | ticker, leader, p_arm, rec10, side, ask_c, fee_c, contracts, stake_c, depth_cap_c, bankroll_c, pnl_c, win; per-trader extras: pt2 banked_c/level_c, pt3/pt4 pv + src, pt5 savings_c, pt6 p_win/skipped/trained/b6x, pt7/pt8 quoted_c/limit_c (+pt8 skipped) | — |
| treatments.jsonl | one settled desk window scored through the registry | ticker, close_ts, leader, outcome, regime_acc, ev (per-policy map) | full local; 2000 published |
| treatments.json | SPRT registry state | per-key {sprt evidence, n_bet, n_skip, ev_sum, promoted_at}; fshare weights | state file |
| kb_calib.json | M1 shadow Platt state | per-arm (a, b) + window | state file |
| live_snapshots.jsonl | one T0 feature snapshot per poll | spreads, basis, funding, OFI 1/5/15m, tape imbalance/vol, whale flow, Kalshi p_up/depth/OI/spread, sentiment | trimmed with learning_log |
| learning_log.jsonl | per-poll learning counters | ts, updates, retrains | trimmed (1500 published) |
| metrics_history.jsonl | one retrain gate / batch run / retirement verdict | kind (retrain/batch/retire), gate {val_mse_before/after…}, git SHA | 5,000 rows |
| ticks.jsonl | one Coinbase trade | id, ts, size, taker_buy | 500,000 lines, rotated |
| kalshi_history.jsonl | one mined historical quote-minute | ticker, ts, mins_left, yes_bid_c/yes_ask_c, strike, btc_close, outcome | mined snapshot |
| audit_report.json | latest 10-min self-audit | generated_ts, sections {desk, traders, tier1, tier2, treatments} | overwritten atomically |
| watchdog_log.jsonl | one watchdog event | ts, event, stale_s | — |
| demo_orders.jsonl / demo_fills.jsonl | demo-exchange order / fill | ticker, side, price_c, count, result / fill lifecycle fields | — |
| online_status.json | the daemon's whole live status | alive_at (heartbeat), price_now/brti, variants, consensus, kb_*, treatments[], fshare_w, kb_calib, retrain info | overwritten each poll |

Model state files (daemon-owned, restart-safe): q_table_online_h*.json,
linucb_t*-h*.json, dqn_t8-h*.pt, lstm_t9-h*.pt, kb_logit.json (kb3, 24
dims), kb4_logit.json (12), kb5_logit.json (14), kb6_logit.json (12),
kb8_logit.json (3), pt6_logit.json (7). Batch counterparts without the
online suffix (q_table.json, dqn_h*.pt, …) belong to the trainers.

---

## 6. WHO DECIDES WHAT

The process as actually practiced (full write-up: **docs/DECISIONS.md**):

- **The user approves policy changes.** Every sizing/gate/metric change
  in the log is marked user-directed where it was (Gambler v2 "user-
  directed", MSE switch "user-directed", pt7/pt8 quoted verbatim from
  the user's request). Claude proposes with evidence; the user decides.
- **Bug fixes ship with evidence, not permission-per-fix** — but each
  fix is verified against the metric that exposed it and recorded
  (NOTES 08-28 lists nine, each with its verification).
- **Everything else is a treatment vs a control.** No direct edits to
  running policies; controls stay frozen even when they lose (the
  Follower "stays frozen (it is the control)" — commit b62afdf — and
  the Gambler's sizing is kept as the ruin demonstration).
- **Pre-registration in NOTES.md** before results exist (the tier
  evidence protocol was committed before the replay verdict:
  `6815c21` → `3ced34e`/`1311151`). Negative results go on the
  Disproved wall and stay there; a tie is recorded as a tie.
- **Promotion needs live, paired, sequential evidence** (SPRT crossing
  its boundary) — and even then it is a stamped, reversible event with
  the loser kept running.

---

## Appendix — contradictions and staleness found while writing this

Known places where documents lag the code (NOTES.md's newest entries
and the code are authoritative):

1. **docs/SEV0_REMEDIATION.md's header and Tier-1/Tier-3 sections still
   state the original 08/26 RCA** (bias born in tier-1 drift priors;
   M6 asymmetric gate; M7 hour policy as a change spec). NOTES 08-28
   overturned the tier-1 chain (dispersion, not direction), dropped M7
   (p=0.60), and folded M6 into M1. The spec calls itself a living
   document but those sections were not rewritten.
2. **README.md is one era behind**: "Four static pages" (there are 8);
   arm-roster framing predates the kb ladder, the desk, and the
   treatments framework; kb bullet describes only kb/kb2-era behavior.
3. **scripts/standings.py docstring still says "MAE ranking"** —
   MAE-era tool, not updated for the 08-28 MSE switch (deliberately:
   frozen-era numbers stay MAE; new headline surfaces use MSE/MSSE).
4. **Knife-edge veto paired EV**: NOTES says it was refused for pt8 at
   "+1.39% too weak"; the pt8 code comment says "+1.32% paired
   post-fix"; the treatment-stale-quote fix note says "+1.32". Same
   conclusion, inconsistent digit.
5. **kb9 "remains unlaunched"** (NOTES round-2 entry) vs "kb9 LAUNCHED"
   (later 08-25 entry) — resolved by intent: unlaunched *as a kb7
   upgrade*, launched *as a second family*; but a reader scanning
   NOTES top-to-bottom will hit the launch entry first because the
   file is only loosely newest-first.
6. **human_feedback.jsonl** is documented (feedback.py, t11) but does
   not exist yet in results/ — no vote has ever been recorded.
