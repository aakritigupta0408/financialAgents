# RL treatment tournament — 3 reward objectives, one architecture

**Goal.** Add three RL-managed paper traders as treatment arms against the T0 control.
Each starts at **$300**, enters **at most once per 15-minute window**, sizes its position
by policy (not gates), may **exit before close** but cannot change side/strike, and
resolves under Kalshi rules. The three arms are **identical except for the reward
objective**, so the tournament isolates *which risk posture the market rewards*.

This lives in the **live trading system** and is kept **completely disconnected** from
the sealed-research TRUE15M program and TEST_V2 (see `SYSTEM_BOUNDARY.md`). **T0 remains
the frozen control** — it is never modified.

## Honest prior (why this is framed as risk/cost management, not alpha)

Both the external literature and our own research say short-horizon binary **direction is
~unlearnable**: an ML battery could not beat a ~54% majority baseline
([arxiv 2511.15960](https://arxiv.org/html/2511.15960v1)), and TRUE15M reached
`NO_OFFLINE_QUALIFIED_MODEL` (base rate ~0.5088, confirmed on this data). So the RL arms
are **not** expected to call direction better than T0. Their only legitimate edge is
(1) **skipping** negative-EV windows, (2) **sizing** (vol-scaled fractional-Kelly), and
(3) **early-exit** (optimal stopping). "Take ≥60% of T0's trades / avoid the bad ones" is
an **expectation to measure, not a target to fit** — training toward known losers is
hindsight and is explicitly disallowed.

## The three arms (differ ONLY in reward)

| Arm | Reward objective |
|---|---|
| **RL-Sharpe** | Differential rolling Sharpe, net of fees, with a drawdown penalty; size capped at fractional-Kelly. ("safe but not too safe") |
| **RL-PnL** | Net realized PnL minus fees. Aggressive, drawdown-blind. |
| **RL-Kelly** | Fractional-Kelly log-growth (geometric bankroll). Growth-optimal, conservative sizing. |

Reward references: differential Sharpe — Moody & Saffell
([NeurIPS 1998](https://proceedings.neurips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf));
cost-in-return + vol scaling — Zhang/Zohren/Roberts
([arxiv 1911.10107](https://arxiv.org/abs/1911.10107)); fractional Kelly caps over-betting.

## Shared MDP

- **Episode** = one window (median ~100 intra-window samples available → early-exit modelable).
- **State** = market features (below) + agent inventory: side (locked once entered), size,
  unrealized PnL, **time-to-close** (fraction remaining).
- **Entry action** (once): `{skip} ∪ {take · size-bucket}` — `skip` is first-class (cost avoidance).
- **Exit action** (each sub-step, side locked): `{hold, exit-now}` — an optimal-stopping head
  ([arxiv 2208.00765](https://arxiv.org/pdf/2208.00765)).
- **Transition** = realized market path; episode ends at exit or close (binary payoff resolves).

## Features (PIT — data ≤ t only)

From the live per-window log (`rl_window_log.jsonl`): oracle `p_up` (per variant),
market `mkt_p_up`, `base` (BRTI) vs `strike` (signed distance + its recent change),
uncertainty band `q80_w`, multi-scale BRTI returns/realized-vol within the window,
`mins_left`; plus agent inventory. **Scalers fit on past-only, refit per walk-forward
fold**; no global normalization (leak); labels resolve at close (drives purge/embargo).

## Algorithm & data

- **Learning:** historical **sim → offline-tune**. Build a per-window simulator from settled
  windows; train (discrete take/skip/size-bucket → Double-Dueling-DQN; continuous fraction →
  PPO). Offline-tune/benchmark vs **behavior-cloning** of the ledger and **always-skip**.
- **Data (the binding constraint):** feature-rich windows live only in rolling buffers.
  **Phase 0 `rl_treatment_dataset.py`** durably accrues them (append-only) so the training set
  grows — same "accrue then decide" posture as TEST_V2.

## Evaluation (must pass before any live promotion)

Walk-forward + **purged/embargoed CV** (labels resolve forward) + **deflated Sharpe**
(penalize configs/seeds tried) + sealed OOS + **placebo/shuffle**. A policy is promotable
only if it **beats BOTH always-skip AND T0 net-of-fees** on OOS with deflated Sharpe > 0 and
a passing placebo. Below that bar, its only honest role is cost/risk management.
([López de Prado purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation))

## Promotion

Add all three as **shadow paper arms** (real windows, no capital risk beyond the $300 paper
stake, not in the treatment slot). Promote a winner into the live treatment slot via the
existing paired-SPRT champion/challenger harness (`btc_rl/treatments.py`) only after it clears
the bar above. The dormant T3 RL arm is the slot a winner would replace.

## Phases

0. **Durable feature store** — `scripts/rl_treatment_dataset.py` (built; accruing). Needs scheduling.
1. **Offline harness** — MDP/env + PIT feature builder + 3 reward variants + purged-CV backtest;
   v0 run on current data with honest wide CIs (pipeline validation, not a verdict).
2. **Gated promotion** — accrue → train → shadow arms → SPRT promote a winner.

## Status

- Phase 0: **DONE** (226 settled windows snapshotted; base rate 0.5088).
- Phase 1: pending.
- Data: **too thin for a trustworthy verdict yet** — accruing.
