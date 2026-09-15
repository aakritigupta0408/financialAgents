# RESEARCH PROGRAM — living plan & memory

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
- **Order-flow (R7):** carries real leakage-free signal ALONE (~0.63 OOS) but adds only
  **+0.7pp over the barrier** — because aggressive flow drives price, so the barrier already
  reflects it. Lesson: signal *derived from BTC's own price/flow* is largely redundant with
  the barrier. The orthogonality that can actually move OOS must come from data that is NOT a
  function of BTC's price: **cross-asset (COIN/IBIT/SPY/DXY), funding/basis, options skew,
  news/sentiment, LLM reasoning.** That is the forward program (R9/R11/R14).
- **Implication / the only lever:** add **orthogonal, non-price information** + smarter entry
  timing (R13). Fusion/complexity without new information does not help (proven R3/R7).
- Canary hygiene held throughout (0.500 mid / 0.513 open) — the numbers above are honest.

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

### Scheduled ⏳ (ordered; deps noted)
| ID | Task | Stage | Depends | Notes |
|---|---|---|---|---|
| R9 | Exhaust Alpha Vantage: news-sentiment + COIN/IBIT/SPY/DXY cross-asset + macro → PIT features, OOS lift each | Bu | R8 (priorities), av_features.py | infra exists (news_backfill/av_features); vintage-safe receipt-time |
| R10 | Exhaust Kalshi contract data: book depth, volume, OI, quote intensity from events/ (kalshi quotes) | Bu | R7 harness | off-path, cheap |
| R11 | Claude-as-signal teacher (LLM reads headlines/context → directional prob; Tauric-style) | Bu | R9 news | add to ensemble; latency/PIT care |
| R12 | Multi-model architecture (from one-shot) + custom quant losses (payoff-weighted/cost-sensitive, Sharpe-utility, rank/AUC), calibrated selective trading | Bu | R7,R9,R10 features | only worth it once orthogonal features exist |
| R13 | Value-of-waiting / entry-timing model (accuracy vs mins-left → optimal entry) | Bu | — | independent; biggest proven lever |
| R14 | Options/funding features (Deribit skew/gamma, perp funding/basis, OI, liquidations) | Fe | sources.py has Deribit/OKX | check what's captured live |
| R15 | Loss-diagnosis harness — attribute each losing trade to a missed indicator | Bu 🧊 | — | staged w/ TODO(human) attribute_loss in scripts/loss_diagnosis.py |
| R16 | Best-model → new shadow arm (T3) once any beats base OOS AND canary-clean | On | R9–R13 | gate: beat chronos-base F1 0.683 OOS |

### Infra / ops (from audit) ⏳
| ID | Task | Status | Notes |
|---|---|---|---|
| O1 | Rotate leaked GitHub token in /tmp/btc_publish.log | ⏳ owner | security; I can't rotate — needs owner |
| O2 | Disk cleanup cron (event tapes 20GB, +2GB/day) | ⏳ | but tapes are R7/R14 raw data — archive, don't delete |
| O3 | Fix publisher silent failures (numpy env, main-sync push) | ⏳ | pre-existing |

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
