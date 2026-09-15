# BTC Oracle / Quant Universe — System Audit

**Date:** 2026-09-15 · **Scope:** whole system (code, data, processes, site, tests) ·
**Status of the desk:** LIVE (paper/simulation only) · **Prepared after:** the "Great Roster Cut"

> Paper/simulation only. T0 is the frozen A/B control and was not modified. This audit is
> observation + the one directed change (the roster cut); nothing else in the running system
> was altered.

---

## 0. TL;DR

- **Roster cut is DONE and live.** Three arms trade now: **T0 `pt`** (frozen control),
  **T1 `cg33`** (confidence-gated 33% follower), **T2 `fm`** (Chronos-Bolt **base**,
  directional). Every other treatment (`pt2 pt3 pt4 pt5 pt6 pt7 pt8 cg5 cg10 tv`) is retired —
  gated off in the daemon, removed from the UI, ledgers kept as frozen evidence.
- **Model decision (benchmark):** chronos-bolt-**base** wins on precision (0.74) and false
  positives (15, fewest) with F1 tied for top (0.683). **TimesFM 2.5, Kronos-base, and Tauric all
  fail to beat it → there is NO T3/T4.** Final roster = T0/T1/T2.
- **Two health problems predate this work and need attention:** (1) **20 GB disk**, almost all
  in never-rotated event tapes (~2 GB/day); (2) the site publisher has **two silent-failure
  loops** (missing numpy → trader-detail snapshot never regenerates; hourly main-repo git push
  fails) **and leaks a GitHub token into a world-readable `/tmp` log.**
- **Tests:** guard suites green (settlement integrity + system isolation, 22/22). One
  pre-existing legacy RL test fails; one test file breaks `pytest tests/` collection.

---

## 1. What was done this session

| # | Item | State |
|---|---|---|
| 1 | Finished the foundation-model benchmark (fixed the TimesFM 2.5 backend to the real `TimesFM_2p5_200M_torch` API) | ✅ done, `research/fm_benchmark_report.json` |
| 2 | Decided the model roster from the benchmark (base wins; TimesFM loses) | ✅ done |
| 3 | Built **T2 `fm`** = live chronos-bolt-**base** directional trader (new inference `_chronos_base_p_up`, constants, state-init, entry block, official settlement) | ✅ live, first trade booked |
| 4 | Retired 10 treatments: gated their entry in `online.py`, removed from snapshots + `home.html` + traders board, updated the settlement-guard test roster | ✅ done |
| 5 | Restarted the daemon (pid 43167) with the new code; verified clean startup, no errors | ✅ done |
| 6 | Verified the live public site shows `fm` and no longer shows retired arms | ✅ `theaakritigupta.com/btc-oracle/site/home.html` |
| 7 | Committed + pushed (`worktree-btc-rl-poc`): `dcee375`, `83dbf71` | ✅ done |
| 8 | Updated the stale UI-contract test to the new roster | ✅ green |

## 2. What is pending / hung / not done

- **Kronos (T3 candidate): RESOLVED — does NOT beat base.** Kronos-base (148-win OOS, 30 sampled
  paths/window, `research/kronos_bench_report.json`): F1 0.681, precision 0.658, FP 25, acc 0.696
  — vs base F1 0.683 / precision 0.741 / FP 15. Fails "beat on F1 AND precision/FP" → **no T3.**
  **Tauric** = LLM trading agents, unsuitable for 15-minute intraday → not a candidate. **Final:
  no T3/T4; roster stands at T0/T1/T2.**
- **T2 live vs offline parity (by design, not a bug):** the offline benchmark used a fine
  sub-minute BRTI path + Asian tail-average; the live arm reads minute-candle closes + final step
  (kb7-parity). So the live arm accrues its **own** honest record rather than inheriting the 0.74
  offline precision — the scientifically correct stance. Watch the live number diverge modestly.
- **`emit_trader_detail.py` broken in publish** (pre-existing): fails on `import numpy` under the
  cron's Python → `trader_detail.json` has not regenerated. Trader drill-down data is stale.
- **Hourly main-repo site sync broken** (pre-existing): the git push to `TheAakritiGupta.com`
  fails every hour (bad/expired token). The gh-pages **fast path still works**, so the live site
  is current; only the slow full-rebuild seed is stale.
- **`test_reward_spec`** (legacy RL reward shaping) fails — unrelated to trading; pre-existing.
- **`test_a3_scenarios.py` calls `sys.exit(0)` at import** → aborts `pytest tests/` collection.

## 3. Performance (live, official-BRTI settled)

| Arm | Role | Net P&L | Hit rate | Notes |
|---|---|---|---|---|
| **T0 `pt`** | control | **−$1,536** | 69% | 1018 trades; loses on favorite/longshot payoff asymmetry (69% hits still net-negative buying ~75¢ favorites). Bankroll rebased to $100M so it never busts. |
| **T1 `cg33`** | treatment | **+$131** | 77% | very selective (~13 settled, ~1% coverage); 33% stake is a labelled RUIN-RISK experiment (backtest $300→~$60, 98% drawdown). |
| **T2 `fm`** | treatment | new | — | first trade `KXBTC15M-26SEP151445-45`: read p_up 0.28 → bet NO, half-Kelly $14.10/$300 stake, open. |

**Foundation-model benchmark** (148-window OOS, base-rate 48.6% up):

| model | F1 (best thr) | precision | recall | acc | FP | ms/win |
|---|---|---|---|---|---|---|
| barrier closed-form | 0.687 @.55 | 0.697 | 0.677 | 0.716 | 20 | 0 |
| **chronos-bolt base (T2)** | 0.683 @.60 | **0.741** | 0.632 | **0.730** | **15** | 35 |
| chronos-bolt small (kb7) | 0.677 @.60 | 0.710 | 0.647 | 0.716 | 18 | 12 |
| chronos-bolt local-ft | 0.677 @.55 | 0.692 | 0.662 | 0.710 | 20 | 15 |
| market k_prob | 0.662 @.50 | 0.613 | 0.721 | 0.662 | 31 | 0 |
| TimesFM 2.5 (kb9) | 0.662 @.45 | 0.613 | 0.721 | 0.662 | 31 | 75 |

## 4. Architecture (code)

### `btc_rl/` package — live vs research
**Daemon import set (live):** `config, contract_truth, metrics, history, treatments, agents, env,
features, llm_sentiment, sources`.

| Live modules | Role |
|---|---|
| `online.py` | **The always-on daemon** (~4000 lines): trading loop, all arms, calibration, official settlement. Entry: `python -m btc_rl.online`. |
| `agents.py` | Model classes (LinUCB, DistDQN, LSTM, TabularQ, BinaryLogit, Platt); lazy torch. |
| `contract_truth.py` | Exact CF-BRTI settlement (target = mean BRTI [open−60,open); settle = mean [close−60,close)); flag `EXACT_BRTI_RUNTIME_ENABLED`. |
| `features.py` | Feature/state-vector computation. |
| `sources.py` | No-auth feeds: Coinbase OHLCV, BRTI, Kalshi BTC15, Deribit/OKX, Fear&Greed, mempool. |
| `metrics.py` | Probability/forecast metrics (Brier, pinball, PT, Kalshi fees). |
| `llm_sentiment.py` | CryptoBERT sentiment from RSS. |
| `env.py`,`config.py`,`history.py`,`treatments.py` | bandit env; config; metrics-history append; champion/challenger SPRT routing. |

**Research-only (NOT imported by the daemon — dormant or script-driven):** `economics.py`,
`exec_timing.py`, `preopen_oracle.py`, `prospective_capture.py`, `research_events.py`,
`paper_account.py`, `traders_v2.py`, `train.py`, `ticks.py`, `test_v2_firewall.py`, and
**`live.py` (deprecated, superseded by online.py)**. Several of these have docstrings claiming
daemon involvement that is no longer wired — a documentation-drift flag, not a runtime bug.

**ML models loaded into daemon memory:** chronos-bolt-**small** (kb7), chronos-bolt-**base**
(T2 `fm`), TimesFM-2.5-200M (kb9) — all lazy CPU singletons, fail-safe; plus torch (DQN/LSTM).

### `scripts/` (grouped)
- **Always-on / cron control:** `paper_desk.sh` (daemon start/restart), `watchdog.py` (5 min,
  restart dead daemon, M6 self-heal), `capture_watchdog.py` (5 min, restart stale capture
  daemons), `publish_dashboard.py` (1 min, push to gh-pages), `meta_monitor.py` (monitor-of-
  monitors), `audit_chain.py` (10 min, ~20 independent analytics steps), `official_outcomes_
  refresh.py` (240 s, keep `contract_outcomes.jsonl` fresh), `event_capture.py` /
  `capture_xvenue.py` / `capture_micro.py` (the capture tapes).
- **Snapshot emitters** (`emit_*.py`, ~35): backend → `results/*.json` the site renders
  (`emit_home_snapshot`, `emit_live_desk`, `emit_trader_detail`, `emit_oracle_snapshot`,
  `emit_experiments_snapshot`, `emit_modelling_snapshot`, `emit_research_snapshot`, …).
  `emit_common.py` AST-parses constants from `online.py` (never imports the live module).
- **One-shot research/backtests:** `analyst_*` (the Analyst studies), `fm_benchmark.py`,
  `t0_improvement_backtest.py`, `replay_backtest.py`, `rl_treatment_*`, `open_oracle_ladder.py`,
  `distributional_models.py`, `fetch_contract_specs.py`, `reconcile.py`, `agent_*` autonomous
  research agents.
- **Site build:** `build_site.py` (static `site/index.html`).

### `site/` — published pages (fetch `results/*.json`)
`home.html` (desk + live activity + trader family), `oracle.html` (6 subtabs Input/Features/
Modelling/Output/Sevs/Graveyard), `experiments.html` (A/B paired deltas + bootstrap CI),
`modelling.html` (model ladder), `architecture.html` (DAG), `research.html` (private research
snapshot). Shared: `theme.css`, `nav.js`, `header.js`, `glossary.js/json`.

### Data flow of one 15-minute window
capture (`sources`/`contract_truth` fix official target) → `features` → kb\* **model** arms emit
`p_up` (incl. Chronos small/base, TimesFM) → leader selected → **trader** arms act (T0 `pt`
follows; T1 `cg33` gated-follows; T2 `fm` directional on Chronos-base) → append `*_trades.jsonl`
→ after close `_official_outcome` settles EVERY arm on official Kalshi (defers if missing, never
the candle proxy) → `emit_*` roll into `results/*.json` → `publish_dashboard.py` → live site.

## 5. Processes & memory (live census)

| Process | RSS | Cadence | Role |
|---|---|---|---|
| `btc_rl.online` (pid 43167) | **2.77 GB** | always-on | the trading daemon; big because it holds 3 foundation models (chronos small+base, TimesFM 200M) + torch + sklearn |
| `btc_rl.ticks` | 497 MB | always-on | tick collector → `ticks.jsonl` (no model consumes it yet) |
| `capture_micro.py` | 29 MB | always-on | micro event tape (fastest disk grower) |
| `event_capture.py` | 23 MB | always-on | primary event tape |
| `capture_xvenue.py` | 20 MB | always-on | cross-venue trade tape |
| `http.server` ×2 | ~1 MB | always-on | local static serving |
| cron: watchdog / capture_watchdog / meta_monitor / audit_chain / publish / introspect | transient | 1–10 min | supervision + publish + analytics |

**Total steady RAM ≈ 3.3 GB.** The daemon dominates; the T2 addition added ~200–400 MB
(second Chronos model). If RAM is tight, `btc_rl.ticks` (497 MB for an unused tape) is the first
thing to stop.

## 6. Data layer & what to clean regularly

**`results/` = 20 GB (≈ the entire repo).** ~19.4 GB is never-rotated hourly event tapes:

| Dir/File | Size | Rotation | Action |
|---|---|---|---|
| `results/events_micro/` | **12 GB** (355 hourly files, 60–106 MB each) | none | **archive/compress or prune >N days** — ~1–2 GB/day |
| `results/events_xvenue/` | 4.5 GB | none | same |
| `results/events/` | 2.7 GB | none | same |
| `feature_snapshots.jsonl` | 82 MB | **none (unbounded append)** | add a MAX_ROWS trim like the kb log |
| `ticks.jsonl` | 41 MB | none | trim or stop the tick collector |
| `prediction_log.jsonl` | 30 MB | none | trim |
| `kalshi_binary_log.jsonl` | 9.1 MB | **self-trims (20k rows)** | ok |
| `execution_ledger.jsonl` | 1.1 MB | **self-trims (60k rows)** | ok |
| `chronos_bolt_ft/model.safetensors` | 190 MB | static | keep (local fine-tune) |

**Only two ledgers self-trim.** The event tapes + `feature_snapshots`/`ticks`/`prediction_log`
grow forever. **Junk to remove:** `.wt_tmp/` (25 MB, incl. a cloned Kronos repo), 6 root
`Screenshot 2026-09-11 *.png` (4.1 MB), `catboost_info/`, `artifacts/`. **`/tmp` logs**:
`btc_audit.log` 7.4 MB, `btc_publish.log` 2.1 MB/46k lines (see §8), `btc_introspect.log` 2.2 MB.

**Regular-clean recommendation:** a daily cron that (a) gzips event-tape files older than 2 days
and deletes those older than N; (b) tail-trims `feature_snapshots.jsonl`, `ticks.jsonl`,
`prediction_log.jsonl`; (c) truncates `/tmp/btc_*.log`.

## 7. Tests & manual verification (run this session)

- **Guard suites green:** `test_official_settlement.py` + `test_system_isolation.py` = **22/22
  passed.** (No `resolve_outcome` proxy in the daemon; no active arm carries a PROXY_DEGRADED
  settled row; the two systems stay isolated.)
- **Curated real tests:** 68 passed; UI-contract fixed to the new roster (now green); one
  pre-existing failure: `test_core::test_reward_spec` (legacy RL reward sign — unrelated).
- **Live functional checks (manual):** daemon up & predicting after restart (no errors); **T2
  `fm` booked a well-formed first trade**; official-settlement path present for all kept arms;
  live public site verified to show `fm` and hide retired arms; benchmark reproducible.
- **Test-hygiene debt:** `tests/` holds 140 `.py` but only 33 are real `test_*.py`; the rest are
  diagnostic/research scripts (some run by cron: `invariants.py`, `leakage_canaries.py`).
  `test_a3_scenarios.py`'s import-time `sys.exit(0)` breaks whole-dir collection.

## 8. Risks & recommendations (prioritized)

1. **[SECURITY] Leaked GitHub token** — `/tmp/btc_publish.log` (world-readable) contains a live
   `gho_…` token in the failing push URL, ~7,670 times. **Rotate the token; stop logging the URL;
   restrict the log.** (Not touched here — it's the owner's credential.)
2. **[DISK] 20 GB, +~2 GB/day** with no rotation → will fill the disk. Add the cleanup cron (§6).
3. **[SILENT FAILURE] publisher** — `emit_trader_detail.py` dies on `import numpy` under the cron
   env (fix the cron's interpreter/venv so numpy resolves) and the hourly main-repo push fails.
   Both loop forever without alerting. Wire these into `meta_monitor`.
4. **[MEMORY] 2.77 GB daemon** now holds two Chronos models; fine on this host but worth watching.
5. **[DOC DRIFT] dormant modules** (`preopen_oracle`, `prospective_capture`, `paper_account`,
   `exec_timing`, `live.py`) advertise daemon roles they no longer have — annotate or archive.

## 9. Agents & workflows to keep this healthy (proposed)

The system already has strong supervision (`watchdog`, `capture_watchdog`, `meta_monitor`,
`audit_chain`). Gaps a small set of agents/workflows would close:

- **Disk-janitor agent (cron, daily):** rotate/compress event tapes, trim unbounded ledgers,
  truncate `/tmp` logs; alert if `results/` > threshold. *Closes §6 / §8.2.*
- **Publish-integrity monitor:** parse `/tmp/btc_publish.log` each cycle; alert on any
  `ModuleNotFoundError` / `push failed` / **any `gho_` token in a log**. *Closes §8.1 / §8.3.*
- **Pre-deploy verifier workflow (before every daemon restart):** `py_compile` + guard tests +
  a 1-window dry-run of each active arm's entry/settlement + a snapshot-schema check. Catches the
  exact bug classes below before they reach the live loop.
- **Arm-parity monitor:** nightly, compare each live arm's realized precision/hit vs its offline
  benchmark; flag drift (e.g. T2 live vs the 0.74 offline figure). Keeps live claims honest.
- **Settlement-integrity sentinel:** already guarded by a test; add a daily scan of *all* ledgers
  (incl. retired) for any non-`OFFICIAL_KALSHI` settled row.
- **Test-hygiene workflow:** move the ~100 diagnostic scripts out of `tests/` (or rename off the
  `test_*` pattern) and fix the import-time `sys.exit` so `pytest tests/` runs clean in CI.

### Bug classes that actually recur here (what the verifiers must catch)
1. **Proxy vs official settlement** — the recurring integrity bug (a NO bet on a +$11 up-close
   booked as a win). *Guarded now; keep the sentinel.*
2. **Always-true gates** — e.g. `max(p,1−p) >= τ` is always ≥0.5, so a gate meant to skip
   coin-flips never skips. *Verify first-trade behavior after any gate change.*
3. **Silent-failure loops** — a cron step throws every cycle but the chain keeps running (numpy,
   git push). *Publish-integrity monitor.*
4. **Indentation / block-scope errors** in the 4000-line daemon on edits near interleaved arms.
   *`py_compile` + dry-run in the pre-deploy verifier.*
5. **Model/harness parity drift** — live inference silently differing from the benchmarked one
   (model size, series granularity). *Arm-parity monitor.*
6. **Unbounded append** — new ledgers added without a trim. *Disk-janitor + a lint for `.open("a")`
   without a MAX_ROWS.*
7. **Stale UI contract tests** after roster/schema changes. *Pre-deploy snapshot-schema check.*

---
*Generated during the Great Roster Cut on `worktree-btc-rl-poc`. Benchmark closed: no model beat
Chronos-Bolt base → final roster T0 `pt` / T1 `cg33` / T2 `fm`, no T3/T4.*
