# Model Internals — what each model has actually learned, and where it is wrong

Every number below is computed from real files in `results/` by
`tests/introspect_model_internals.py`, which also writes the compact
machine-readable version `results/model_internals.json`. Nothing is estimated.
Snapshot: `generated_ts 1787978537` (2026-08-29 04:42 UTC). Rerun the script to
refresh; the daemon updates these files continuously.

Feature-name maps are transcribed index-by-index from `btc_rl/online.py`
(`_kb_logit_features`, `_kb4_features`, `_kb5_features`, `_kb6_features`,
`_kb8_features`, `_pt6_features`, `_path_features`); the script asserts each
map's length against the checkpoint's `dim`.

---

## 1. Linear / logit arms (fully interpretable)

All are online logistic regressions predicting P(window closes ≥ strike)
(kb5/pt6: P(the taken side wins)). Positive weight = pushes the probability up.

### kb3 — `results/kb_logit.json`, 24 features, 7,379 updates

| # | feature | weight | reading |
|---|---------|-------:|---------|
| 0 | bias | −0.127 | slight down base-rate tilt |
| 1 | market_mid | **+0.854** | 2nd-largest weight: leans on the crowd |
| 2 | quote_present | −0.125 | having a quote at all tilts down a touch |
| 3 | above_strike_z | **+1.041** | the dominant signal: strike geometry |
| 4 | phase (mins_left/15) | −0.009 | dead |
| 5 | z_x_phase | +0.088 | z matters slightly more early |
| 6 | ofi_1m | −0.026 | dead |
| 7 | ofi_5m | +0.064 | near-dead |
| 8 | book_imb | −0.029 | dead |
| 9 | ret_5m | +0.226 | short momentum |
| 10 | ret_15m | +0.391 | 15-min momentum, 3rd-largest signal |
| 11 | log_vol_ratio | +0.044 | near-dead |
| 12 | pf_frac_above | **−0.111** | WRONG SIGN (see flaws) |
| 13 | pf_whipsaw | +0.090 | — |
| 14 | pf_drift3m_z | −0.097 | wrong sign (drift toward strike counts against) |
| 15 | pf_quote_drift3m | +0.033 | dead |
| 16 | rsi14 | +0.063 | near-dead |
| 17 | ema_dist | −0.201 | mean-reversion reading of EMA distance |
| 18 | macd | −0.205 | mean-reversion reading of MACD level |
| 19 | macd_hist | +0.337 | momentum reading of MACD histogram |
| 20 | sma20_gap | +0.233 | trend-following on the SMA gap |
| 21 | bb_z | −0.104 | Bollinger mean-reversion |
| 22 | bb_width | +0.126 | wide bands → up (regime proxy) |
| 23 | vol_1m_ratio_log | −0.000 | dead |

**Flaw analysis.**
- **Echo check:** the market-mid weight (+0.854) is 13–30× larger than every
  microstructure weight kb3 was built to exploit (ofi_1m −0.026, ofi_5m +0.064,
  book_imb −0.029). After 7,379 updates the "information features" are dead and
  the model is ~85% crowd-echo + strike geometry. Its genuinely independent
  content reduces to ret_15m/ret_5m momentum and a technicals cocktail.
- **Economically wrong sign:** `pf_frac_above` −0.111 — a window that has spent
  most of its life ABOVE the strike is read as evidence of closing BELOW it.
  Same for `pf_drift3m_z` −0.097 (price drifting up lowers P(up)). Both are
  small but persistent; conditional on z and the market they act as
  overshoot-correction terms, which is a polite name for fitted noise.
- **Internally split personality:** MACD level −0.205 vs MACD histogram +0.337,
  EMA distance −0.201 vs SMA gap +0.233 — pairs of near-collinear technicals
  carrying opposite signs, the classic signature of correlated-feature weight
  splitting rather than two real effects.

### kb4 (stacker over kb2+kb3) — `results/kb4_logit.json`, 12 features, 7,368 updates

| feature | weight |
|---------|-------:|
| bias | −0.049 |
| p_kb2 | **+0.659** |
| p_kb3 | **−0.457** |
| kb2×kb3 agreement | −0.037 |
| market_mid | +0.802 |
| quote_present | −0.056 |
| above_strike_z | +0.996 |
| phase | −0.100 |
| pf_frac_above | −0.117 |
| pf_whipsaw | +0.050 |
| pf_drift3m_z | +0.022 |
| pf_quote_drift3m | +0.055 |

**Flaw analysis:** the stacker **anti-weights kb3** (−0.457): given kb2, the
market and the strike z, kb3's residual opinion is *negatively* informative —
kb4 profits by fading its own parent. The agreement interaction it was designed
around is dead (−0.037). It also re-derives market (+0.802) and z (+0.996)
itself, so it is closer to "kb2 + market + z, minus kb3" than a blend. The
wrong-sign `pf_frac_above` (−0.117) recurs here.

### kb5 (train-where-you-trade, side-oriented) — `results/kb5_logit.json`, 14 features, 4,439 updates

| feature | weight |
|---------|-------:|
| bias | −0.215 |
| p_kb2_side | +0.496 |
| p_kb3_side | +0.055 |
| p_kb4_side | −0.077 |
| market_side | **+0.695** |
| kb2_vs_market disagreement | **−0.200** |
| ask | **+0.236** |
| claimed_edge | **−0.096** |
| strike_z_toward_side | +0.709 |
| phase | −0.008 |
| hot_hour | +0.087 |
| pf_frac_above_side | −0.155 |
| pf_whipsaw | +0.060 |
| pf_quote_drift_side | +0.052 |

**Flaw analysis — the most self-incriminating weights in the system:**
- `ask` **+0.236**: the more you must pay for a side, the more likely it wins.
  Price is information — correct learning, but it means kb5's p_hat rises with
  the ask, which structurally erodes any "p_hat − ask" edge rule built on it.
- `claimed_edge` **−0.096**: kb5 has learned that its own parents' claimed
  edge (model prob minus price) *predicts losing*. The arm's entry rule
  (`KB5_BE_MARGIN`: bet when p_hat×100 ≥ cost + 3) thresholds on a quantity its
  own regression says is negatively informative.
- `disagreement` −0.200: disagreeing with the market predicts losing — a second
  vote for "the crowd is right".
- Parents: kb2 +0.496, kb3 +0.055 (≈ignored), kb4 −0.077 (mildly faded).

### kb6 (fast-information arm, RETIRED from trader candidacy) — `results/kb6_logit.json`, 12 features, 4,791 updates

| feature | weight |
|---------|-------:|
| bias | −0.006 |
| market_mid | +0.725 |
| quote_present | −0.010 |
| perp_gap_bp | +0.269 |
| perp_mom_bp | +0.084 |
| tape_imb_1m | −0.038 |
| tape_imb_5m | −0.038 |
| whale_net_15m | **−0.165** |
| k_oi_delta | −0.036 |
| above_strike_z | +0.988 |
| phase | −0.092 |
| pf_frac_above | +0.344 |

**Flaw analysis:** built to exploit perp lead-lag, tape aggression, whale flow
and OI — of those only `perp_gap_bp` (+0.269) earned real weight. Its top two
weights (z +0.988, market +0.725) are the same generic pair every arm finds.
`whale_net_15m` −0.165 is economically inverted: 15-min net whale *buying*
lowers its P(up) — either whales here are contrarian liquidity (exit prints) or
this is fitted noise; either way the "whale signal" does the opposite of its
sales pitch. Note kb6 is the one arm whose `pf_frac_above` has the sane
positive sign (+0.344).

### kb8 (log-opinion pool: kb7 × market) — `results/kb8_logit.json`, 3 features, 6,256 updates

| feature | weight |
|---------|-------:|
| bias | −0.216 |
| kb7_log_odds | +0.413 |
| market_log_odds | +0.672 |

**Reading:** the learned answer to "foundation model vs crowd" is
**crowd 1.6× the foundation model** (0.672 vs 0.413). The warm start landed
near 0.4/0.6 (per the code comment); live updates have pushed the market share
further up. Weight sum 1.085 ≈ 1 (a well-behaved opinion pool, slight
sharpening), bias −0.216 = a persistent down-tilt. This arm is *by design*
mostly echo, and it says so honestly.

### pt6 (meta-trader logit) — `results/pt6_logit.json`, 7 features, only 367 updates

| feature | weight |
|---------|-------:|
| bias | +0.256 |
| leader_conf | +0.193 |
| ask | +0.252 |
| conf_minus_ask | **−0.059** |
| market_toward_side | +0.232 |
| phase | +0.239 |
| pf_drift3m_z | −0.133 |

**Flaw analysis:** same disease as kb5, at 1/10th the sample: `ask` +0.252 (its
p_win tracks the price — the exact defect the 2026-08-26 calibration-fix
comment documents) while `conf_minus_ask` — the *edge*, the very quantity its
`PT6_MIN_EDGE_C = 10` gate thresholds on — carries a **negative** weight. The
positive bias (+0.256) plus positive ask weight is why its raw "EV>0" fired on
every window before the 10c margin gate was bolted on. At 367 updates none of
this is settled; it is a model whose gate currently does the work its weights
cannot.

---

## 2. Calibrators — `results/kb_calib.json` (Platt: p_cal = σ(a + b·logit(p)))

a=0, b=1 is identity; b>1 stretches (arm under-confident), a>0 shifts up.
`ll` columns are the decayed prequential mean log-loss (test-then-train), the
calibrator's own shadow scoreboard; **cal − raw > 0 means the calibration layer
is currently making that arm WORSE**.

| arm | a | b | updates | mean ll cal | mean ll raw | cal − raw |
|-----|------:|------:|--------:|------:|------:|------:|
| kb  | +0.670 | 1.549 | 641 | 0.571 | 0.487 | **+0.085** |
| kb2 | +0.292 | 1.437 | 641 | 0.525 | 0.459 | **+0.067** |
| kb3 | +0.205 | 1.243 | 641 | 0.598 | 0.486 | **+0.112** |
| kb4 | +0.332 | 1.456 | 641 | 0.534 | 0.461 | **+0.073** |
| kb5 | +0.551 | 1.369 | 421 | 0.555 | 0.464 | **+0.091** |
| kb6 | −0.388 | 1.325 | 641 | 0.510 | 0.484 | **+0.026** |
| kb7 | +0.282 | 1.125 | 641 | 0.649 | 0.481 | **+0.168** |
| kb8 | +0.534 | 1.382 | 641 | 0.575 | 0.466 | **+0.109** |
| kb9 | +0.236 | 1.574 | 641 | 0.520 | 0.446 | **+0.075** |

**What the (a,b) say about honesty:** every arm fits b>1 (1.12–1.57) — the
window fit reads all nine as under-confident, and eight of nine get an upward
shift (a>0); kb6 alone gets pushed down (a=−0.388, consistent with its
persistent cold streak). kb7 is nominally the most honest (b=1.125).

**The flaw:** on the prequential scoreboard the calibration layer is losing for
**all nine arms** (cal − raw between +0.026 and +0.168). The class docstring's
own success criterion — "calibrated below raw means the layer is earning its
place" — is currently met by nobody. The decayed window (~67 effective samples)
keeps chasing regime-local miscalibration and pays for it out of sample. kb7 is
hurt most (+0.168), i.e. the "most honest" b is applied at the wrong a.

---

## 3. Bandits and networks — what is inspectable without fabrication

Arms are the 19 vol-scaled deltas `K_FACTORS` = [−1.5 … 0 … +1.5]σ (index 9 =
0.0σ). We report **pulls** (real counters in the checkpoints), totals, and the
most-pulled arm. No interpretation of dense weights is offered.

### LinUCB (`results/linucb_t{2,6,10,11}-h*.json`)

| variant | total pulls | most-pulled arm | share |
|---------|------------:|-----------------|------:|
| t2-h1  | 110,548 | −0.2σ (29,847) | 27% |
| t2-h5  | 84,960  | −0.35σ wait—see json; index 9 = 0.0σ (24,703) | 29% |
| t2-h15 | 106,771 | −0.1σ (40,995) | 38% |
| t2-h30 | 151,060 | **+0.5σ (124,562)** | **82%** |
| t6-h1  | 105,752 | +0.2σ (25,430) | 24% |
| t6-h5  | 7,777   | −0.1σ (2,666) | 34% |
| t6-h15 | 107,956 | +0.2σ (55,762) | 52% |
| t6-h30 | 121,810 | +0.2σ (62,226) | 51% |
| t10-h1 | 87,651  | +0.5σ (13,042) | 15% |
| t10-h5 | 92,141  | +0.2σ (27,905) | 30% |
| t10-h15| 110,335 | +0.2σ (48,122) | 44% |
| t10-h30| 134,686 | **+0.65σ (91,605)** | **68%** |
| t11-h1 | 81,628  | −0.1σ (14,765) | 18% |
| t11-h5 | 118,403 | +0.2σ (42,009) | 35% |
| t11-h15| 116,232 | **+0.5σ (74,389)** | **64%** |
| t11-h30| 105,408 | 0.0σ (50,890) | 48% |

(Exact per-arm pull vectors are in `model_internals.json → bandit_pulls`; the
t2-h5 top arm is index 9 = 0.0σ.)

**What the pulls reveal:** the 1-minute bandits hedge around 0 to −0.2σ, but
every 15/30-minute bandit has collapsed onto a single **positive-drift arm**
(+0.2 to +0.65σ), t2-h30 spending 82% of 151k pulls on +0.5σ. The "contextual"
bandits have largely learned an unconditional bullish-drift bet at long
horizons — context moves them little once one arm dominates the ridge prior.

### Linear-Q (t7, live state in `results/linucb_t7-h*.json`; `linear_q.json` is only the 60-day batch warm-start)

| variant | total pulls | most-pulled arm | share |
|---------|------------:|-----------------|------:|
| t7-h1  | 510,943 | 0.0σ (140,469) | 27% |
| t7-h5  | 218,367 | +0.1σ (86,950) | 40% |
| t7-h15 | 263,976 | 0.0σ (86,551) | 33% |
| t7-h30 | 200,192 | −0.35σ (84,922) | 42% |

t7's ε-greedy keeps a visible exploration floor (~1.7–4k pulls on every arm) —
unlike the LinUCB collapse — and t7-h30 is the lone *bearish*-mode learner.

### DQN (t8) and LSTM (t9) — architecture and steps only

| checkpoint | input dim | arms | train steps | architecture |
|------------|----------:|-----:|------------:|--------------|
| dqn_t8-h1  | 13 | 23 | 257,949 | MLP 13→64→64→23 (2 hidden ReLU layers) |
| dqn_t8-h5  | 13 | 23 | 256,101 | same |
| dqn_t8-h15 | 13 | 23 | 251,637 | same |
| dqn_t8-h30 | 13 | 23 | 69,887  | same (¼ the training of its siblings) |
| lstm_t9-h1 | 60 | 23 | 235,795 | 1-dim return seq ×60 → LSTM(32) → head 23 |
| lstm_t9-h5 | 60 | 23 | 55,635  | same (¼ trained) |
| lstm_t9-h15| 60 | 23 | 230,890 | same |
| lstm_t9-h30| 60 | 23 | 225,634 | same |

23 output bins = the 19 action bins + 4 overflow bins (±2σ, ±3σ, targets only).
The h30 DQN and h5 LSTM have roughly a quarter of their siblings' updates —
their scores are not like-for-like with the rest of the family.

---

## 4. Who the meta-layers currently trust — `results/treatments.json`

**Fixed-Share weights** (α=0.08, η=1.2; uniform = 0.1667):

| arm | weight |
|-----|-------:|
| kb2 | 0.1978 |
| kb8 | 0.1741 |
| kb9 | 0.1691 |
| kb4 | 0.1680 |
| kb7 | 0.1480 |
| kb3 | 0.1430 |

Near-uniform by construction (the α-share step floors every arm at ~0.013 and
constantly remixes); the ordering, not the magnitudes, is the signal: kb2
leads, kb3 trails.

**EVLEAD** (mean EV/$1 over each arm's last 20 windows, −1.0 = full loss):

| arm | n | mean EV/$1 |
|-----|--:|-----------:|
| kb7 | 20 | **+0.233** |
| kb2 | 20 | +0.143 |
| kb4 | 20 | +0.143 |
| kb9 | 20 | +0.143 |
| kb3 | 20 | +0.081 |
| kb8 | 20 | +0.064 |

**The two meta-layers disagree about the leader** — Fixed-Share (log-loss
ranked) trusts kb2 most and ranks kb7 5th; EVLEAD (cost-ranked) puts kb7 first
by a wide margin. kb7's two outlier windows (+1.44, +1.56 — cheap longshots
that hit) drive its EV lead while barely moving its log-loss; that one pair of
numbers is the whole "Brier ≠ EV" thesis in miniature.

Treatment ledgers (same file): champion 154 bets, ev_sum −10.66 (−6.9c/$1 per
bet) — the full per-treatment table is in `model_internals.json → trace` runs
and `treatments` keys.

---

## 5. One trade's thought, step by step

Most recent settled desk trade in `results/pt_trades.jsonl`:
window `KXBTC15M-26AUG290030-30` (closes 2026-08-29 04:30 UTC), strike
**$77,585.60**. All numbers below are from the trade row and kb9's logged
per-minute row at the same slot (`made_ts 1787977140/1787977156`).

1. **Who leads and why.** At entry the desk recomputes each arm's last-10
   settled tau-clearing decisions inside the ≤12-min envelope. Recomputed from
   the log: kb2 7/10 (Brier .2002), kb3 8/10 (.1818), kb4 7/10 (.1973), kb7
   8/10 (.1762), kb8 8/10 (.1773), **kb9 8/10 (.1723)** — four arms tie at
   8/10 and kb9 wins the Brier tie-break. The logged `rec10: 8/10` matches.
2. **The leader's opinion.** kb9 (the LSTM-quantile arm) has, at 10.7 min
   left: base price $77,596.04 — $10.44 *above* the strike — and an 80% quantile
   band of [$77,532.80, $77,672.80] (width $140, `q80_w`). Its published
   p_up = **0.6219**. Its previous-minute p was 0.7348, so conviction was
   fading, not building. Path so far: `pf = [0.5, 0.0, −0.18, +0.01]` — the
   window has spent 100% of its bars above the strike, zero crossings, slight
   3-min down-drift.
3. **Gate 1 (confidence):** max(0.6219, 0.3781) = 0.6219 ≥ τ = 0.62. Cleared
   by 0.0019 — this trade exists by two parts in a thousand.
4. **Gate 2 (envelope):** mins_left 10.7 ≤ 12. Cleared.
5. **The market's opinion:** mid 0.615 — kb9 agrees with the crowd (its edge
   over mid is 0.7c; this is a follow-the-market trade, not a fade).
6. **Gate 3 (legality):** side = yes, real ask = **62.0c**, within [5, 80).
   Cleared. Depth cap $15,252.93 — not binding.
7. **Sizing:** per-contract fee 7·0.62·0.38 = 1.6492c → cost/contract 63.6492c.
   10% of bankroll ÷ 63.6492 → **43 contracts**. Stake = 43×62 + order fee
   ⌈7·43·0.62·0.38⌉ = 2,666 + 71 = **2,737c** ($27.37). (Script recomputes
   both: exact match.)
8. **Outcome:** window settled UP (close ≥ strike, `actual = 1`). Payout
   43×100 = 4,300c; **P&L +1,563c** (+$15.63), +57.1c per $1 staked.
   Bankroll after: $291.16.
9. **The counterfactual on the same window** (§6 convention): the cheapest
   modeled yes-ask observed intra-window was also 62.0c — the market never got
   cheaper after entry, so this time the desk's EV (+0.571/$1) *equals* the
   oracle's. A perfectly timed trade — on a window where the arm barely
   scraped past its own gate and simply rode the crowd's price.

---

## 6. Utopian counterfactual — oracle EV and the desk's regret

Convention (same mid→ask model as `tests/pt_replay.py`): logged per-minute rows
carry the market MID; a fill is modeled at **mid + 2.5c** (`SEL_CF_ASK_ADJ`).
The oracle plays the desk's own game — one entry per window, mins_left ≤ 12,
5c ≤ ask < 80c — but knows the outcome and takes the **winning side at the
cheapest modeled ask observed intra-window**. EV per $1 staked =
(100 − a − f)/(a + f), f = 7·(a/100)·(1−a/100). If no legal quote ever exists
on the winning side the oracle stands aside (7 of 150 windows).

Coverage: the rotated per-minute log retains the most recent **150 of 290**
settled desk windows; both sides of the comparison use that same covered set.

| | per window, per $1 staked |
|---|---:|
| Oracle EV | **+1.970** |
| Desk EV (actual, fees included) | **−0.088** |
| **Regret** | **2.058 per $1-window** |

The desk loses ~8.8c per staked dollar per window; perfect foresight with the
same rules would have made ~$1.97 per staked dollar per window. The oracle
number is huge because the cheapest winning-side ask is often a 10–35c quote
printed while the market still disagreed with the eventual outcome — the
regret is dominated by *side and timing*, not by fees or spread. (Caveats:
oracle knows the outcome, so this is an upper bound, and modeled asks are
mid+2.5c, not the historical book.)

Per-window oracle mean, desk mean and regret are in
`model_internals.json → oracle`.

---

## 7. Are the headline metrics representative?

Per arm, over the 168 settled windows in the current log: all-row Brier (the
headline flavor), decision-time Brier (first τ-clearing row at ≤12 min — the
row the desk actually consumes), and the EV/$1 of betting that decision at the
modeled ask.

| arm | Brier (all rows) | Brier (decision) | decisions | EV/$1 at decision | n bets |
|-----|---:|---:|---:|---:|---:|
| kb  | 0.1754 | 0.2131 | 167 | −0.047 | 124 |
| kb2 | 0.1701 | 0.2018 | 168 | −0.074 | 130 |
| kb3 | 0.1754 | 0.2295 | 168 | −0.048 | 130 |
| kb4 | 0.1721 | 0.2047 | 168 | −0.027 | 127 |
| kb5 | 0.1778 | 0.1939 | 166 | **−0.125** | 101 |
| kb6 | 0.1804 | 0.2106 | 168 | −0.053 | 119 |
| kb7 | 0.1770 | 0.2063 | 167 | −0.091 | 123 |
| kb8 | 0.1733 | 0.2095 | 168 | −0.072 | 125 |
| kb9 | 0.1715 | 0.2007 | 167 | **−0.012** | 126 |
| kbf | **0.1094** | **0.0972** | 152 | **−0.409** | 18 |

Verdict, with the counterexamples found in this data:

1. **Headline Brier flatters everyone.** Every arm's decision-time Brier is
   0.02–0.06 worse than its all-row Brier (kb3: 0.175 → 0.230), because the
   all-row average is padded with easy near-close minutes. The number the site
   headlines is not the number the desk trades on.
2. **Brier ranking ≠ EV ranking — live counterexample.** kb2 has the best
   trader-arm all-row Brier (0.1701) and the top Fixed-Share weight, yet its
   decisions monetize at −0.074/$1; kb9, with a *worse* Brier (0.1715), is the
   best monetizer at −0.012/$1, and EVLEAD's favorite kb7 (Brier 0.177) sits at
   −0.091/$1 overall while leading the last-20 EV table. Probability accuracy
   and cost-adjusted value pick different champions in the same data.
3. **kbf is the extreme counterexample.** The flagship one-call-per-window
   arm has by far the best Brier (0.097 at decision) — because it decides at
   T-3 minutes, when the outcome is mostly known. At that point the winning
   side's ask has usually left the legal band (only 18 of 152 calls were even
   biddable), and those residual biddable calls lose −0.409/$1. Its excellent
   headline Brier measures lateness, not value: it is the system's best
   forecaster and its worst trader, by construction.
4. **EV/$1 columns are themselves modeled** (mid+2.5c convention), so treat
   them as ranking evidence, not P&L; the desk's true fees-included number on
   the same covered windows is the −0.088/$1 of §6.
5. **MSE/MAE for the delta arms (t2…t9) is representative of what they do**
   (point prediction) but not of *how* they do it: the pull tables (§3) show
   the long-horizon bandits earning their MAE with a near-unconditional
   +0.2…+0.65σ drift bet — a fact invisible in any error metric and exactly
   what a drift reversal would expose.

---

## Regenerate

```bash
python3 tests/introspect_model_internals.py   # rewrites results/model_internals.json, prints this data
```
