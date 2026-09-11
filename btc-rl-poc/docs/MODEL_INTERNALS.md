# Model Internals — what each model has actually learned, and where it is wrong

Every number below is computed from real files in `results/` by
`tests/introspect_model_internals.py`, which also writes the compact
machine-readable version `results/model_internals.json`. Nothing is estimated.
This document is pinned to snapshot `generated_ts 1787978830` (2026-08-29
04:47 UTC). The daemon keeps learning, so weights drift in the third decimal
and pull counters tick every 5-minute slot; rerun the script to refresh both
files together.

Feature-name maps are transcribed index-by-index from `btc_rl/online.py`
(`_kb_logit_features`, `_kb4_features`, `_kb5_features`, `_kb6_features`,
`_kb8_features`, `_pt6_features`, `_path_features`); the script asserts each
map's length against the checkpoint's `dim`.

---

## 1. Linear / logit arms (fully interpretable)

All are online logistic regressions predicting P(window closes ≥ strike)
(kb5/pt6: P(the taken side wins)). Positive weight = pushes the probability up.

### kb3 — `results/kb_logit.json`, 24 features, 7,393 updates

| # | feature | weight | reading |
|---|---------|-------:|---------|
| 0 | bias | −0.110 | slight down base-rate tilt |
| 1 | market_mid | **+0.855** | 2nd-largest weight: leans on the crowd |
| 2 | quote_present | −0.108 | having a quote at all tilts down a touch |
| 3 | above_strike_z | **+1.041** | the dominant signal: strike geometry |
| 4 | phase (mins_left/15) | +0.001 | dead |
| 5 | z_x_phase | +0.088 | z matters slightly more early |
| 6 | ofi_1m | −0.028 | dead |
| 7 | ofi_5m | +0.057 | near-dead |
| 8 | book_imb | −0.029 | dead |
| 9 | ret_5m | +0.226 | short momentum |
| 10 | ret_15m | +0.391 | 15-min momentum, 3rd-largest signal |
| 11 | log_vol_ratio | +0.039 | near-dead |
| 12 | pf_frac_above | **−0.113** | WRONG SIGN (see flaws) |
| 13 | pf_whipsaw | +0.095 | — |
| 14 | pf_drift3m_z | −0.099 | wrong sign (up-drift counts against up) |
| 15 | pf_quote_drift3m | +0.033 | dead |
| 16 | rsi14 | +0.063 | near-dead |
| 17 | ema_dist | −0.200 | mean-reversion reading of EMA distance |
| 18 | macd | −0.201 | mean-reversion reading of MACD level |
| 19 | macd_hist | +0.335 | momentum reading of MACD histogram |
| 20 | sma20_gap | +0.233 | trend-following on the SMA gap |
| 21 | bb_z | −0.106 | Bollinger mean-reversion |
| 22 | bb_width | +0.132 | wide bands → up (regime proxy) |
| 23 | vol_1m_ratio_log | −0.018 | dead |

**Flaw analysis.**
- **Echo check:** the market-mid weight (+0.855) is 15–30× larger than every
  microstructure weight kb3 was built to exploit (ofi_1m −0.028, ofi_5m +0.057,
  book_imb −0.029). After 7,393 updates the "information features" are dead and
  the model is mostly crowd-echo + strike geometry. Its genuinely independent
  content reduces to ret_15m/ret_5m momentum and a technicals cocktail.
- **Economically wrong sign:** `pf_frac_above` −0.113 — a window that has spent
  most of its life ABOVE the strike is read as evidence of closing BELOW it.
  Same for `pf_drift3m_z` −0.099 (price drifting up lowers P(up)). Both are
  small but persistent; conditional on z and the market they act as
  overshoot-correction terms, which is a polite name for fitted noise.
- **Internally split personality:** MACD level −0.201 vs MACD histogram +0.335,
  EMA distance −0.200 vs SMA gap +0.233 — near-collinear technicals carrying
  opposite signs, the classic signature of correlated-feature weight splitting
  rather than two real effects.

### kb4 (stacker over kb2+kb3) — `results/kb4_logit.json`, 12 features, 7,382 updates

| feature | weight |
|---------|-------:|
| bias | −0.033 |
| p_kb2 | **+0.659** |
| p_kb3 | **−0.460** |
| kb2×kb3 agreement | −0.034 |
| market_mid | +0.803 |
| quote_present | −0.041 |
| above_strike_z | +0.996 |
| phase | −0.091 |
| pf_frac_above | −0.120 |
| pf_whipsaw | +0.055 |
| pf_drift3m_z | +0.019 |
| pf_quote_drift3m | +0.055 |

**Flaw analysis:** the stacker **anti-weights kb3** (−0.460): given kb2, the
market and the strike z, kb3's residual opinion is *negatively* informative —
kb4 profits by fading its own parent. The agreement interaction it was designed
around is dead (−0.034). It also re-derives market (+0.803) and z (+0.996)
itself, so it is closer to "kb2 + market + z, minus kb3" than a blend. The
wrong-sign `pf_frac_above` (−0.120) recurs here.

### kb5 (train-where-you-trade, side-oriented) — `results/kb5_logit.json`, 14 features, 4,447 updates

| feature | weight |
|---------|-------:|
| bias | −0.220 |
| p_kb2_side | +0.497 |
| p_kb3_side | +0.054 |
| p_kb4_side | −0.076 |
| market_side | **+0.697** |
| kb2_vs_market disagreement | **−0.200** |
| ask | **+0.234** |
| claimed_edge | **−0.096** |
| strike_z_toward_side | +0.709 |
| phase | −0.011 |
| hot_hour | +0.087 |
| pf_frac_above_side | −0.159 |
| pf_whipsaw | +0.057 |
| pf_quote_drift_side | +0.055 |

**Flaw analysis — the most self-incriminating weights in the system:**
- `ask` **+0.234**: the more you must pay for a side, the more likely it wins.
  Price is information — correct learning, but it means kb5's p_hat rises with
  the ask, which structurally erodes any "p_hat − ask" edge rule built on it.
- `claimed_edge` **−0.096**: kb5 has learned that its parents' claimed edge
  (model prob minus price) *predicts losing*. The arm's entry rule
  (`KB5_BE_MARGIN`: bet when p_hat×100 ≥ cost + 3) thresholds on a quantity its
  own regression says is negatively informative.
- `disagreement` −0.200: disagreeing with the market predicts losing — a second
  vote for "the crowd is right".
- Parents: kb2 +0.497, kb3 +0.054 (≈ignored), kb4 −0.076 (mildly faded).

### kb6 (fast-information arm, RETIRED from trader candidacy) — `results/kb6_logit.json`, 12 features, 4,805 updates

| feature | weight |
|---------|-------:|
| bias | +0.020 |
| market_mid | +0.727 |
| quote_present | +0.016 |
| perp_gap_bp | +0.249 |
| perp_mom_bp | +0.082 |
| tape_imb_1m | −0.042 |
| tape_imb_5m | −0.050 |
| whale_net_15m | **−0.165** |
| k_oi_delta | +0.023 |
| above_strike_z | +0.988 |
| phase | −0.076 |
| pf_frac_above | +0.340 |

**Flaw analysis:** built to exploit perp lead-lag, tape aggression, whale flow
and contract OI — of those only `perp_gap_bp` (+0.249) earned real weight. Its
top two weights (z +0.988, market +0.727) are the same generic pair every arm
finds. `whale_net_15m` −0.165 is economically inverted: 15-min net whale
*buying* lowers its P(up) — either whales here are contrarian liquidity or this
is fitted noise; either way the "whale signal" does the opposite of its sales
pitch. `k_oi_delta` even flipped sign between two script runs five minutes
apart (−0.036 → +0.023) — weights at that magnitude are noise, live. Note kb6
is the one arm whose `pf_frac_above` has the sane positive sign (+0.340).

### kb8 (log-opinion pool: kb7 × market) — `results/kb8_logit.json`, 3 features, 6,270 updates

| feature | weight |
|---------|-------:|
| bias | −0.198 |
| kb7_log_odds | +0.419 |
| market_log_odds | +0.674 |

**Reading:** the learned answer to "foundation model vs crowd" is
**crowd 1.6× the foundation model** (0.674 vs 0.419). The warm start landed
near 0.4/0.6 (per the code comment); live updates have pushed the market share
further up. Weight sum 1.09 ≈ 1 (a well-behaved opinion pool, slight
sharpening), bias −0.198 = a persistent down-tilt. This arm is *by design*
mostly echo, and its three weights say so honestly — see §5 for a real trade
where that decomposition is the whole story.

### pt6 (meta-trader logit) — `results/pt6_logit.json`, 7 features, only 368 updates

| feature | weight |
|---------|-------:|
| bias | +0.238 |
| leader_conf | +0.181 |
| ask | +0.240 |
| conf_minus_ask | **−0.059** |
| market_toward_side | +0.227 |
| phase | +0.224 |
| pf_drift3m_z | −0.115 |

**Flaw analysis:** same disease as kb5, at 1/12th the sample: `ask` +0.240 (its
p_win tracks the price — the exact defect the 2026-08-26 calibration-fix
comment in `online.py` documents) while `conf_minus_ask` — the *edge*, the very
quantity its `PT6_MIN_EDGE_C = 10` gate thresholds on — carries a **negative**
weight. The positive bias (+0.238) plus positive ask weight is why its raw
"EV>0" fired on every window before the 10c margin gate was bolted on. At 368
updates none of this is settled; it is a model whose gate currently does the
work its weights cannot.

---

## 2. Calibrators — `results/kb_calib.json` (Platt: p_cal = σ(a + b·logit(p)))

a=0, b=1 is identity; b>1 stretches (arm under-confident), a>0 shifts up.
`ll` columns are the decayed prequential mean log-loss (test-then-train), the
calibrator's own shadow scoreboard; **cal − raw > 0 means the calibration layer
is currently making that arm WORSE**.

| arm | a | b | updates | mean ll cal | mean ll raw | cal − raw |
|-----|------:|------:|--------:|------:|------:|------:|
| kb  | +0.670 | 1.549 | 642 | 0.571 | 0.492 | **+0.079** |
| kb2 | +0.292 | 1.437 | 642 | 0.532 | 0.467 | **+0.065** |
| kb3 | +0.205 | 1.243 | 642 | 0.606 | 0.496 | **+0.110** |
| kb4 | +0.332 | 1.456 | 642 | 0.542 | 0.471 | **+0.072** |
| kb5 | +0.551 | 1.369 | 422 | 0.560 | 0.473 | **+0.087** |
| kb6 | −0.388 | 1.325 | 642 | 0.531 | 0.497 | **+0.034** |
| kb7 | +0.282 | 1.125 | 642 | 0.648 | 0.484 | **+0.164** |
| kb8 | +0.534 | 1.382 | 642 | 0.578 | 0.474 | **+0.104** |
| kb9 | +0.236 | 1.574 | 642 | 0.529 | 0.455 | **+0.075** |

**What the (a,b) say about honesty:** every arm fits b>1 (1.12–1.57) — the
window fit reads all nine as under-confident, and eight of nine get an upward
shift (a>0); kb6 alone gets pushed down (a=−0.388, consistent with its
persistent cold streak). kb7 is nominally the most honest (b=1.125).

**The flaw:** on the prequential scoreboard the calibration layer is losing for
**all nine arms** (cal − raw between +0.034 and +0.164). The class docstring's
own success criterion — "calibrated below raw means the layer is earning its
place" — is currently met by nobody. The decayed window (~67 effective samples)
keeps chasing regime-local miscalibration and pays for it out of sample. kb7 is
hurt most (+0.164), i.e. the "most honest" b is applied at the wrong a.

---

## 3. Bandits and networks — what is inspectable without fabrication

Arms are the 19 vol-scaled deltas `K_FACTORS` = [−1.5 … 0 … +1.5]σ (index 9 =
0.0σ). We report **pulls** (real counters in the checkpoints), totals, and the
most-pulled arm. No interpretation of dense weights is offered.

### LinUCB (`results/linucb_t{2,6,10,11}-h*.json`)

| variant | total pulls | most-pulled arm | share |
|---------|------------:|-----------------|------:|
| t2-h1  | 110,549 | −0.2σ (29,847) | 27% |
| t2-h5  | 84,961  | 0.0σ (24,703) | 29% |
| t2-h15 | 106,772 | −0.1σ (40,995) | 38% |
| t2-h30 | 151,061 | **+0.5σ (124,563)** | **82%** |
| t6-h1  | 105,753 | +0.2σ (25,431) | 24% |
| t6-h5  | 7,777   | −0.1σ (2,666) | 34% |
| t6-h15 | 107,957 | +0.2σ (55,762) | 52% |
| t6-h30 | 121,811 | +0.2σ (62,227) | 51% |
| t10-h1 | 87,652  | +0.5σ (13,042) | 15% |
| t10-h5 | 92,142  | +0.2σ (27,905) | 30% |
| t10-h15| 110,336 | +0.2σ (48,123) | 44% |
| t10-h30| 134,687 | **+0.65σ (91,605)** | **68%** |
| t11-h1 | 81,629  | −0.1σ (14,765) | 18% |
| t11-h5 | 118,404 | +0.2σ (42,009) | 35% |
| t11-h15| 116,233 | **+0.5σ (74,390)** | **64%** |
| t11-h30| 105,409 | 0.0σ (50,890) | 48% |

(Exact per-arm pull vectors are in `model_internals.json → bandit_pulls`.)

**What the pulls reveal:** the 1-minute bandits hedge around 0 to −0.2σ, but
every 15/30-minute bandit has collapsed onto a single **positive-drift arm**
(+0.2 to +0.65σ), t2-h30 spending 82% of 151k pulls on +0.5σ. The "contextual"
bandits have largely learned an unconditional bullish-drift bet at long
horizons — context moves them little once one arm dominates the ridge prior.

### Linear-Q (t7; live state in `results/linucb_t7-h*.json` — `linear_q.json` is only the 60-day batch warm-start)

| variant | total pulls | most-pulled arm | share |
|---------|------------:|-----------------|------:|
| t7-h1  | 510,944 | −0.1σ (140,469) | 27% |
| t7-h5  | 218,367 | +0.1σ (86,950) | 40% |
| t7-h15 | 263,977 | −0.1σ (86,551) | 33% |
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
| kb7 | 0.2095 |
| kb2 | 0.1953 |
| kb8 | 0.1694 |
| kb9 | 0.1577 |
| kb4 | 0.1489 |
| kb3 | 0.1193 |

**EVLEAD** (mean EV/$1 over each arm's last 20 windows, −1.0 = full loss):

| arm | n | mean EV/$1 |
|-----|--:|-----------:|
| kb7 | 20 | **+0.281** |
| kb2 | 20 | +0.072 |
| kb4 | 20 | +0.072 |
| kb9 | 20 | +0.072 |
| kb3 | 20 | +0.009 |
| kb8 | 20 | −0.008 |

Both meta-layers currently crown **kb7** — but that agreement is minutes old
and fragile. Two script runs five minutes apart (04:42 → 04:47 UTC, one settled
window between them: the kb8-led loss traced in §5) flipped the Fixed-Share
leader from kb2 (0.1978) to kb7 (0.2095) and dropped kb8's EVLEAD mean from
+0.064 to −0.008. Fixed-Share's α-share step keeps weights near-uniform by
design (floor ≈0.013, remix every window), so the *ordering* is the signal and
it reorders on single windows. kb7's EV lead rests on two cheap-longshot wins
in its last 20 (+1.44, +1.56 per $1) that barely move its log-loss — one pair
of numbers containing the whole "Brier ≠ EV" thesis.

Treatment ledgers (same file): the champion desk policy has 155 scored bets,
ev_sum −11.66 → **−7.5c EV per $1 per bet**; the full per-treatment table is in
`model_internals.json → treatments`.

---

## 5. One trade's thought, step by step

Most recent settled desk trade in `results/pt_trades.jsonl` at snapshot time:
window `KXBTC15M-26AUG290045-45` (closes 2026-08-29 04:45 UTC), strike
**$77,603.12**. Numbers from the trade row and the leader's logged per-minute
row at the same slot (`made_ts 1787977980/1787977999`).

1. **Who leads and why.** The desk recomputes each arm's last-10 settled
   τ-clearing decisions inside the ≤12-min envelope. Recomputed from the log:
   kb2 7/10 (Brier .2008), kb3 8/10 (.1817), kb4 8/10 (.1573), kb7 9/10
   (.1483), kb9 9/10 (.1410), **kb8 9/10 (.1384)** — three arms tie at 9/10
   and kb8 wins the Brier tie-break. Matches the logged `rec10: 9/10`.
2. **The leader's opinion, fully decomposed.** kb8 is a 3-weight logit, so its
   thought is arithmetic. Its logged inputs `b8x = [1.0, +0.0336, −0.5108]`:
   kb7's log-odds +0.034 (kb7 said p=0.508 — a coin flip) and the market's
   log-odds −0.511 (mid 0.375). Contributions to the score: bias −0.198,
   kb7 +0.014, market **−0.344** → z ≈ −0.53 → p_up ≈ 0.37 (logged 0.3671;
   exact reconstruction differs in the 3rd decimal because the weights have
   taken a few updates since). **96% of the non-bias signal in this trade was
   the market's own price.** The "call NO" is the crowd's opinion, re-sold.
3. **Gate 1 (confidence):** max(0.3671, 0.6329) = 0.6329 ≥ τ = 0.62 — cleared
   by 0.013.
4. **Gate 2 (envelope):** 11.7 min left ≤ 12. Cleared — this is the first
   eligible slot of the window.
5. **Context at decision:** base $77,591.85, $11.27 *below* the strike; path
   `pf = [−0.25, 0.25, −0.98, 0.0]` — 75% of elapsed bars below strike, one
   crossing, sharp 3-min down-drift (−0.98σ). Everything pointed down.
6. **Gate 3 (legality):** side = no, ask = **63.0c** (market's no-mid ≈ 62.5c,
   so the stated edge over the crowd was ~0.8c), within [5, 80). Depth cap
   $13,589.54 — not binding.
7. **Sizing:** per-contract fee 7·0.63·0.37 = 1.6317c → cost 64.63c/contract;
   10% of bankroll ÷ 64.63 → **45 contracts**. Stake = 45×63 + order fee
   ⌈7·45·0.63·0.37⌉ = 2,835 + 74 = **2,909c** ($29.09). Script recomputes
   both: exact match. Payoff if right: +54.7c per $1 staked.
8. **Outcome:** the window settled **UP** (`actual = 1`) — price recovered the
   $11 gap in the final minutes. All 45 contracts expired worthless:
   **P&L −2,909c** (−$29.09), bankroll after $262.07.
9. **The oracle on the same window** (§6 convention): the cheapest modeled
   YES-ask observed intra-window was **36.0c** (while the crowd still said
   down); a clairvoyant playing the same rules earns **+1.66 per $1** where the
   desk lost −1.00. One window, regret 2.66 — the aggregate in §6 says this is
   not an outlier.

The uncomfortable summary: the desk's most trusted arm won leadership with a
9/10 streak, then staked 10% of bankroll on a two-cent repackaging of the
market's own mid — and the market was wrong.

---

## 6. Utopian counterfactual — oracle EV and the desk's regret

Convention (same mid→ask model as `tests/pt_replay.py`): logged per-minute rows
carry the market MID; a fill is modeled at **mid + 2.5c** (`SEL_CF_ASK_ADJ`).
The oracle plays the desk's own game — one entry per window, mins_left ≤ 12,
5c ≤ ask < 80c — but knows the outcome and takes the **winning side at the
cheapest modeled ask observed intra-window**. EV per $1 staked =
(100 − a − f)/(a + f), f = 7·(a/100)·(1−a/100). If no legal quote ever exists
on the winning side the oracle stands aside (7 of 151 windows).

Coverage: the rotated per-minute log retains the most recent **151 of 291**
settled desk windows; both sides of the comparison use that same covered set.

| | per window, per $1 staked |
|---|---:|
| Oracle EV | **+1.968** |
| Desk EV (actual, fees included) | **−0.094** |
| **Regret** | **2.062 per $1-window** |

The desk loses ~9.4c per staked dollar per window; perfect foresight with the
same rules would have made ~$1.97 per staked dollar per window. The oracle
number is huge because the cheapest winning-side ask is often a 10–36c quote
printed while the market still disagreed with the eventual outcome — the regret
is dominated by *side and timing*, not by fees or spread. (Caveats: the oracle
knows the outcome, so this is an upper bound; modeled asks are mid+2.5c, not
the historical book.)

Per-window oracle mean, desk mean and regret are in
`model_internals.json → oracle`.

---

## 7. Are the headline metrics representative?

Per arm, over the 169 settled windows in the current log: all-row Brier (the
headline flavor), decision-time Brier (first τ-clearing row at ≤12 min — the
row the desk actually consumes), and the EV/$1 of betting that decision at the
modeled ask.

| arm | Brier (all rows) | Brier (decision) | decisions | EV/$1 at decision | n bets |
|-----|---:|---:|---:|---:|---:|
| kb  | 0.1758 | 0.2124 | 168 | −0.045 | 125 |
| kb2 | 0.1705 | 0.2029 | 169 | −0.081 | 131 |
| kb3 | 0.1763 | 0.2312 | 169 | −0.055 | 131 |
| kb4 | 0.1728 | 0.2061 | 169 | −0.034 | 128 |
| kb5 | 0.1783 | 0.1948 | 167 | **−0.139** | 101 |
| kb6 | 0.1814 | 0.2125 | 169 | −0.061 | 120 |
| kb7 | 0.1771 | 0.2055 | 168 | −0.088 | 124 |
| kb8 | 0.1738 | 0.2106 | 169 | −0.079 | 126 |
| kb9 | 0.1722 | 0.2017 | 168 | **−0.023** | 126 |
| kbf | **0.1088** | **0.0966** | 153 | **−0.409** | 18 |

Verdict, with the counterexamples found in this data:

1. **Headline Brier flatters everyone.** Every arm's decision-time Brier is
   0.02–0.06 worse than its all-row Brier (kb3: 0.176 → 0.231), because the
   all-row average is padded with easy near-close minutes. The number the site
   headlines is not the number the desk trades on.
2. **Brier ranking ≠ EV ranking — live counterexample.** kb2 has the best
   trader-arm all-row Brier (0.1705), yet its decisions monetize at −0.081/$1;
   kb9, with a *worse* Brier (0.1722), is the best monetizer at −0.023/$1, and
   kb7 — the arm both meta-layers currently crown — sits at −0.088/$1 overall
   while leading the last-20 EV table on two longshot hits. Probability
   accuracy and cost-adjusted value pick different champions in the same data.
3. **kbf is the extreme counterexample.** The flagship one-call-per-window arm
   has by far the best Brier (0.097 at decision) — because it decides at T-3
   minutes, when the outcome is mostly known. By then the winning side's ask
   has usually left the legal band (only 18 of 153 calls were even biddable),
   and those residual biddable calls lose −0.409/$1. Its excellent headline
   Brier measures lateness, not value: it is the system's best forecaster and
   its worst trader, by construction.
4. **EV/$1 columns are themselves modeled** (mid+2.5c convention), so treat
   them as ranking evidence, not P&L; the desk's true fees-included number on
   the same covered windows is the −0.094/$1 of §6, and the champion
   treatment ledger's is −0.075/$1 per bet (§4).
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
