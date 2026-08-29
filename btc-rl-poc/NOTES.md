# Research log

## 2026-08-26 — executed: kb6 retired, Saver reworked, kb9 to traders

- kb6 removed from PT_ARMS (trader leader candidacy) — no trader follows
  its calls now; it keeps predicting for the record. Flagged RETIRED in
  the league. Reason: UP recall 63% (worst), coverage 37%, cold.
- kb9 ADDED to PT_ARMS (was missing) — traders can now follow the
  TimesFM arm's calls too.
- Saver (pt5) sizing 0.25 -> 0.10 (25% was ~2.5x Kelly; bled -31% /
  -$15k drawdown). Skim (25% of wins) unchanged. Pre-0826 rows are
  policy v1 (25%); marked in code.


## 2026-08-26 — post-presentation TA feedback: revenue A/B, meta-trader, retirement

**Revenue A/B (Results page):** replaced the dead bidding-vs-selector
A/B with a trader revenue+risk scoreboard — net P&L, return, win%,
avg/trade, avg win/bid, MAX DRAWDOWN, idle%, today, ranked by net÷DD.
Live snapshot: Disciplined +8.3% (net÷DD 0.15, the only real winner);
Follower/Ladder ~flat; Gambler −84.5% (DD $6.4k); Saver −30.6% (DD
$15.2k). The TA's point stands: accuracy without capital discipline
fails — max drawdown is the number the sleeping client wakes to.

**MLE meta-trader — SL vs RL decision.** Industry standard: predict
edge SUPERVISED (like every quant alpha model, like our kb5), size
ANALYTICALLY via fractional Kelly. RL in finance is for EXECUTION
(JPM LOXM, market-making) where tick data is ~infinite; end-to-end RL
for position sizing on ~120 windows would overfit catastrophically.
Decision: supervised edge + half-Kelly baseline now; RL is the
aspirational upgrade gated on far more data. Baseline built
(tests/metatrader_baseline.py): online logit P(win) → bet iff EV>0 →
half-Kelly size capped 10%. Result: 34 bets/190 windows, 62% win,
~break-even EV, but MAX DRAWDOWN $4 vs Gambler $6.4k / Saver $15.2k —
it learned that not-betting (82% idle) is the skill. Next: wire live
as pt6, learning from the rule traders' signals.

**Retirement candidates (evidence-based):**
- RETIRE: kb6 (fast information) — UP recall 63% (worst arm, well below
  80), coverage 37%, persistently cold. Clear underperformer.
- REWORK not retire: kb5 (EV −14% but the cheap-longshot thesis is
  validated — cheap bids ≤51c are +23% EV, just rare); kb8 (young);
  Saver (25% sizing too aggressive — lower it).
- KEEP as controls: kb (null baseline), Gambler (ruin demonstration),
  Follower/Ladder (sizing baselines). Their job is to be beaten.

**Cheap-bid finding:** capping ask ≤51c flips EV positive (+23%, 55%
win, 19 windows) because break-even is ~53c — the kb5 cheap-longshot
thesis. A regime-abstention gate (trailing market acc) only helps at a
LOW threshold (~0.62 over 8 windows); the proposed 0.68 gate zeroes out
trading (market's own accuracy rarely clears 68%).


## 2026-08-25 — Kalshi DEMO mirror for Sagemon (zero money, by design)

scripts/demo_trader.py: a SEPARATE process that mirrors pt3's fresh
paper entries as orders on Kalshi's demo environment (fake balance,
real order lifecycle; KXBTC15M confirmed live on demo). Safety is
structural: the demo host is hard-coded (no production URL exists in
the file), demo-only credentials live outside the repo
(~/.kalshi_demo.pem + env KALSHI_DEMO_KEY_ID), dry-run without them,
one order per window, contract cap. The live daemon is untouched.
Purpose: demonstrate real exchange plumbing (acks, fills, settlement)
for the presentation with zero financial exposure.

## 2026-08-25 — trader 5, the SAVER (registered at launch) + skim-policy finding

Stashmon: starts $10,000, stakes 25% of playing bankroll (desk-wide
live depth cap applies), and skims 25% of every WIN into savings that
never return to play; losses hit the bankroll in full. Same leader
entry pool as the desk (0.62 gate), one bet/window, real asks + fees.
Registered before first trade; rows carry skim_c and savings_c.

Related measured finding (tests, 1,500 paths): holding the bankroll at
exactly K* = depth/fraction ($45k) and withdrawing all overflow is 38%
SLOWER to $2M than letting it ride — the skim removes every buffer, so
each loss pins stakes under the depth cap. Holding at ~2x K* ($90k)
costs only ~1% of time with full safety above. Buffer before skim.

## 2026-08-25 — kb9 LAUNCHED as second foundation family (not an upgrade)

TimesFM 2.5 (200M, frozen zero-shot) goes live as kb9 — explicitly NOT
as a kb7 upgrade: it TIED the gauntlet (t=+0.67) and that negative
stands on the disproved wall. Launch rationale, registered here: a
second model FAMILY gives (a) live decorrelation/disagreement data vs
kb7, (b) native quantile intervals from an independent architecture,
(c) fusion fuel for a future stack (as kb7 fed kb8). Same readout
convention as kb7 (deciles at the strike), frozen, no training, no
Conviction Book stream. kb7 byte-untouched. Also: legacy A/B
bidding-vs-selector section hidden on Results per project decision
(logging continues; the four desk traders carry the trading story).

## 2026-08-25 — kb9 round 2: TimesFM + fine-tuning — the axis is exhausted

Same pre-registered gate as round 1 (window-clustered paired Brier
t < -2 vs live kb7), 233 held-out windows, no leakage (fine-tune
trained strictly before the eval cutoff on 83k of our own BTC minutes):
  TimesFM 2.5 200M zero-shot:  acc 70.0% brier .189  t = +0.67  TIE
  Chronos-Bolt fine-tuned:     acc 70.4% brier .184  t = -0.25  TIE
With round 1 (Bolt-base worse, Chronos-2 tie, +covariates tie), FIVE
upgrade attempts across scale, architecture, covariates, family, and
adaptation all land within noise of the 47M zero-shot Bolt-small.
Conclusion: at the 15-minute BTC horizon the price path's extractable
signal is the binding constraint, not the model. kb9 remains
unlaunched; checkpoint kept local (results/chronos_bolt_ft/,
gitignored). Bench latency note: TimesFM 0.10s/call on CPU — viable
if it ever earns a slot.

## 2026-08-25 — PRE-REGISTRATION: tier evidence acceleration (2 methods)

(SPRT considered and NOT adopted, per project decision.)

METHOD A — market-paired scoring (variance reduction), applied to the
live tier stream and the replay identically: per entered window, the
market's implied win probability is q = entry cost / 100 (ask + fee);
the score is (win - q). Test: mean excess score > 0, t-test CLUSTERED
BY DAY. The market price absorbs shared outcome variance, so this
reaches a verdict with fewer windows than the raw win-rate test.

RESULT (same day): Method B replay, 663 entered windows across 15
days, real asks: 493/663 = 74.4% vs avg cost 71.8c -> EV +3.6%/$1
(Wilson LB 71.5%); Method A clustered t = 0.85, NOT significant. The
live tier's 20/22 (+21%/$1) does NOT generalize — thin-sample
optimism, the shared-outcome lesson at 30x scale. Honest posture: the
tier's durable edge is small-positive and unproven; the live stream
continues under the frozen rule and both numbers are shown, labeled,
never merged.

METHOD B — REPLAY over the mined history (corroboration, labeled
replay, never merged with the live stream): apply the identical frozen
tier rule (kb7 conf >= 0.77, first qualifying minute, real bid/ask
from the mined rows, 5-80c band, fees in) over the ~1,301-window
results/kalshi_history.jsonl. kb7 recomputed per minute from
historical closes with NO look-ahead (context strictly before each
decision minute). Report windows, win rate, Wilson CI, EV per $1 at
real asks + fees, and Method A's clustered excess-score t.

## 2026-08-25 — trader 4, the ALL-IN (registered at launch)

100% of capital on every leader entry (follower's 0.62 gate), stake
capped only by the ~$500 depth-saturation ceiling. The bet-sizing
control group: expected to bust with probability -> 1 (survival =
p^n), and however long he shines first, one settled loss takes
everything staked. Same window pool as the follower so the pair
isolates SIZING as the only variable. Also: page slowness fixed —
"no-store" was re-downloading ~4.5MB of ledgers every 15 s; heavy
files now cache in 60 s buckets.

## 2026-08-25 (later) — pt3 policy v2, stamped

Amendment, same day, before 10 settled trades: the disciplined trader
ALSO takes the follower's leader-based entry when the LEADER's
confidence >= 0.77 (the follower enters at 0.62; the disciplined bar
stays 0.77 everywhere). kb7-source entries unchanged. Rows stamped
src (kb7|leader) + pv:2 — v1 rows are the unstamped ones. Rationale:
same trade pool as the desk twins, stricter admission — selectivity
applied to every stream, not just kb7's.

## 2026-08-25 — PRE-REGISTRATION: the disciplined trader (pt3)

Frozen before any trade: bids ONLY when kb7's same-minute confidence
>= 0.77 (the measured top-44% tier boundary; 17/18 wins at
registration, THIN n — that is why this live test exists), one bid per
window, 10% of a $1,000 paper bankroll, real ask + Kalshi fee, ask
5-80c, mins_left <= 12. Threshold lives in btc_rl/online.py PT3_TAU;
changing it invalidates the track record. Success bar: win rate above
the ~75% biddable break-even over >= 30 WINDOWS.

## 2026-08-25 — capacity, detection, and how real bots live with both

**Measured:** Kalshi near-touch depth now logged per minute
(k_depth_yes/no; first reading ~$6-9k at a 95c late-window quote; the
5-80c tradeable band is thinner). Volume/OI fields were being dropped
by our fetch (the _fp variants DO populate) — fixed.

**Simulated (tests/sneaky_trader.py, Kyle-1985-flavored):** a "loud"
trader taking full depth is quoted out in 1-2 days having extracted a
few hundred dollars; a stealth trader (grid-searched sizing/timing)
converges to taking x* = 1/(2*beta) ~ 25% of depth — the closed form
"harvest half of what detection tolerates" — and survives forever at
~$52/day on kb7's 2.8% edge. Stealth is survival, not acceleration.

**How production systems handle it (the TA's question):**
1. They don't avoid impact — they OPTIMIZE against it. Almgren-Chriss
   (2000) optimal execution; TWAP/VWAP/POV algos cap participation at
   ~5-15% of volume; iceberg orders; randomized slice sizes/timing to
   defeat flow fingerprinting. Our x* ~ 25%-of-depth cap is a POV cap.
2. They hide in volume (Kyle 1985): trade when noise flow is thick.
   Empirical square-root impact law (Bouchaud et al.): cost ~
   sigma * sqrt(Q / V_daily) — taking 1% of volume is cheap, 30% ruinous.
3. BREADTH over depth (Grinold's fundamental law, IR = IC*sqrt(N)):
   tiny edge x thousands of independent markets. The scaling axis is
   number of markets, never size in one. For us: more series (hourly,
   other strikes, ETH), more venues — capacity multiplies linearly.
4. They earn the spread instead of paying it (maker strategies) — but
   that needs speed: our measured -10.6c/contract maker result is what
   passive quoting WITHOUT speed looks like (adverse selection).
5. They measure their own footprint: pre-trade impact models, post-
   trade TCA feedback, capacity-aware sizing (fractional Kelly), and
   funds CLOSE to new capital at strategy capacity.
6. Alpha lifecycle: edges decay as others find them; rotate signals.
No manipulation modeled anywhere (no spoofing/wash/multi-account) —
sizing, timing, venue and breadth discipline only.

Weekly notes on what was tried, what was measured, and what died.
Newest first. Every claim here should be reproducible from a script in
`tests/` against the committed data in `results/`.

## 2026-08-25 — kb8 launched; kb9 killed by its own gate

**kb9 (foundation-model upgrades) — measured negative.** Pre-registered
gate: replace kb7 only if a challenger beats it on window-clustered
paired Brier with |t| > 2 (`tests/kb8_gauntlet.py`, 126 replay windows,
one decision per window so rows = windows):

| candidate | acc | Brier | t vs kb7 |
|---|---|---|---|
| kb7-replay (bolt-small@512, live recipe) | 73.0% | 0.174 | — |
| chronos-bolt-base (4× params) | 69.8% | 0.185 | +1.24 (worse) |
| Chronos-2 univariate @512 | 70.6% | 0.175 | +0.05 |
| Chronos-2 univariate @2048 | 72.2% | 0.180 | +0.55 |
| Chronos-2 + volume/hl-range covariates | 73.0% | 0.174 | 0.00 |

Zero-shot scale and covariates buy nothing at this horizon/sample. The
TA's LLM-time-series direction is fully explored and documented; kb7
stays as-is.

**kb8 (log-opinion pool of kb7 × market) — launched.** The learned
fusion the gauntlet couldn't deliver from bigger models. Three findings
from `tests/kb8_feature_lab.py` on 1,753 replayed decisions / 127
windows:

1. **Probability-space features fail.** Centered probs saturate at ±1;
   the logit can't express confident parent calls. Log-odds features fix
   this ("copy the market" = weight 1). 12-dim centered design: 64.5%
   final-quarter acc vs parents' ~72% — worse than its own inputs.
2. **At n≈127 independent outcomes, every extra feature hurts.** Band
   width w80 (−4.7 pts), path features, time interactions, even a
   market-presence flag (−2.2 pts, 99% collinear with bias). Final arm
   is 3 weights: bias, kb7 log-odds, market log-odds.
3. **The learned pool is 0.38·kb7 + 0.58·market** (log-odds), i.e. a
   measured answer to "how much is a zero-shot foundation model worth
   next to the crowd": ~40/60. Warm-started prequential 74.5% vs 73.9%
   for either parent alone; Brier vs kb7 clustered t = +0.84 (tie, and
   that includes kb8's cold first quarter).

Verified live: first row's p_up reproduces sigmoid(w·b8x) exactly;
`trained` counter carried the 1,753 warm-start updates in.

**Standing discipline reaffirmed:** stream metrics are window-counted
(effective n = windows, not minutes) — the kb7 "13/14" counting error
and the kb8 feature-lab result are the same lesson from two directions.

## 2026-08-26 — Gambler policy v2: the gate + a $10k reset

08/26 diagnosis (ledgers): the desk hit 65.6% of windows vs a ~66.7%
break-even at ~65¢ asks — a −1pp edge that stake size amplified
(Follower −$261, Gambler 33% stakes −$1,092, Saver −$2,354), while the
Disciplined's ≥0.77 gate ran 27/34 = 79.4% and finished +$276, the only
policy in the green.

Change (user-directed): pt4 adopts the same PT3 gate (PT4_TAU = 0.77)
and his funds reset to $10k at cutover ts 1787788353. History is NOT
rewritten — pre-reset rows stay in pt4_trades.jsonl and are excluded
from the v2 bankroll (rows stamped pv:2 going forward); pages filter to
the v2 era. Pre-registered read: v2 Gambler becomes "Disciplined at
3.3× stakes" — same entry set, ~1.6× Kelly sizing. If the gate's edge
is real he compounds ~10× faster; if it's thin-sample optimism the 33%
sizing will surface that within days. Either outcome is informative.

## 2026-08-26 — Metamon calibration bug + fix (margin gate & shadow rows)

Live pt6 bet 7/7 consecutive windows (the baseline said ~82% idle is
the skill). Cause: p_win tracks the ask (weight +0.26 — price IS
information) and sat 4–16 pts above cost on every window, so "EV>0"
always fired. 4W/3L at ~60c = below break-even.

Fix, two parts: (1) PT6_MIN_EDGE_C = 10 — bet only when claimed edge
>= 10c/$1 (of the first 7 live bets only the +15.9c one qualified; it
won). (2) skipped windows log SHADOW rows (stake 0, skipped: true)
that are labeled at settle and still train the logit — a gated trader
that learns only from its own bets would re-learn nothing. Money views
exclude shadows.

## 2026-08-26 — SEV-0 RCA: where the error and bias are born

Full-graph audit (tests/sev0_error_audit.py). Root-cause chain:
T1 RL arms carry a bullish drift prior (t10 bias +$45@h30, t2 +$54@h30,
up-call shares 71-83% vs 53% reality; direction acc of ALL price arms
47-53% = no directional edge) -> T2 kb arms inherit the lean (pBias
+0.03..+0.11, Platt intercepts all negative, kb7 worst a=-0.565;
kb6 slope b=0.33 = near-noise resolution) -> T3 amplifies: leader
compares UNCALIBRATED p_up across arms (mixed-confidence AUC 0.503 =
coin flip) and churns 16x/day; fixed-kb4 counterfactual +4.7%/$1 beat
the churner's +2.0% -> T4 multiplies by stake (Saver -$2.5k at 25%
era) -> losses concentrate: herd whipsaw 75.2% of loss dollars,
knife-edge windows 52.8%, toxic hours 31.2%, idiosyncratic only 5.3%.
Per-tier gates DO work (kb7@0.77 94.3%, desk@0.77 80.8%) but the 0.85
tail collapses (50%, n=6, adverse-selection tail).

Mitigation queue (leverage order): M1 per-arm online Platt layer before
any cross-arm comparison; M2 knife-edge veto |mkt-0.5|<0.10; M3 sticky
leader (evidence-based switching); M4 re-anchor t10/t2/t6/t11 drift
priors; M5 sizing fixes (done: PT5 0.10, PT4 gate, PT6 margin);
M6 asymmetric UP/DOWN gate while pBias>0; M7 09h PT hour policy.

## 2026-08-28 — SEV-0 re-triage on 4 days (112 -> 248 desk bids)

Three findings changed the plan:

1. The loss was ONE DAY. Per-day win% vs the break-even actually paid:
   08/25 76.1 vs 70.7 (+$383) · 08/26 68.5 vs 66.2 (+$331) · 08/27 56.4
   vs 69.3 (-$1,403) · 08/28 77.1 vs 71.3 (+$86). Three of four days
   above water; 08/27 is the entire drawdown. Market regime, not defect.

2. The bias MOVES. Refitting each arm's halves separately: kb
   a+0.03/b0.56 -> a-0.45/b1.34; kb2, kb4, kb9 likewise moved BOTH
   intercept and slope; only kb7/kb8 stable. Family optimism halved
   overnight (kb7 a -0.57 -> -0.35) while slopes fell below 1 — the
   error MODE shifted from systematic optimism to overconfident spread.
   => a one-time correction constant would already be stale; M1 must be
   the online prequential tracker it was specced as. Evidence, not
   assumption, now backs that design.

3. The best fixed arm FLIPPED. Yesterday kb4 was best (+4.7%/$1) and
   kb7 worst; today kb9 is the only positive (+5.2%) and kb4 is -3.7%.
   "Freeze the best arm" is rejected outright; M3 (Fixed-Share, which
   tracks the best SEQUENCE) is promoted.

Loss causes diluted: herd whipsaw 75.2% -> 42.2%, knife-edge 52.8% ->
16.6%, idiosyncratic 5.3% -> 31.4%. M2 re-scoped (smaller payoff).

Step-0 gates: MLE PASS (84% idle of 147 windows, EV +13.5%/$1, the
best on the desk) · Gambler v2 ON TRACK (80.0% win vs 73.8% break-even,
EV +4.8%, +$4,522, n=25/30) · Saver recovering (-$1,968 -> +$28/day).

Fixed a real defect found while cross-checking: the audit's attribution
used the ungated tier-2 set while sev0.html used the desk's own >=0.62
entry gate — two implementations disagreeing on the same metric. Python
now uses the gated set; both agree (herd 42.2%/42%, knife 16.6%/17%).

## 2026-08-28 (later) — plan revision + a correction to our own RCA

Tested two claims before changing the plan; one failed, and a third
finding overturned part of the 08/26 root-cause chain.

Q: was 08/27 detectable ex ante? YES. Trailing-20-window market
accuracy < 0.62 flagged 63.6% of 08/27's windows before entry vs 5.1%
of 08/28's; market Brier 0.224 vs 0.167. -> new M8 regime gate. Not
free: 08/26 flagged 59% and finished green, so it needs a backtest.

Q: is the toxic hour real? NO. 09h/14h look bad on 2 of 3 days, but we
searched 24 hours; at the desk's 32.5% loss rate a 3-4 bid day-cell
looks bad 17.6% of the time => an hour looks "persistently bad" 8.2% of
the time by luck => expected 1.98 of 24, observed 2, P(>=2 | chance) =
0.60. M7 DROPPED, not deferred. Shipping it would be data dredging.

CORRECTION TO THE 08/26 RCA. Asked whether the RL tier is tracked, we
found it is (all arms scored 99.2-100%, zero staleness) AND that most
RL arms never reach a trading decision: t2/t6/t7/t8/t9/t10/t11 are
logged-only; the binary tiers run off consensus/horizon + the market
anchor. So the "bullish drift prior" we named as the root cause lives
in arms that never place a bet (t10 +$29, t11 +$41 bias @h15) while the
decision-feeding path is near-unbiased (+$1.76 mean). Tier-1
DIRECTIONAL drift cannot be the source of tier-2 optimism.

Corrected chain: decision-feeding bands under-cover (74-76% vs 80%
goal) => sigma too small => converting a too-narrow distribution into
P(above strike) pushes probabilities to the extremes => the slope b < 1
measured at tier 2. A DISPERSION defect, not a DIRECTION defect. M4
re-aimed: drop drift-recentering of t10/t2/t11/t6 (cosmetic), promote
the adaptive-conformal band fix on consensus/h15.

Leader audit: no arm is pure poison (none has zero wins). kb8 is the
worst leader (-$494, 58.8% vs 67.1% break-even) but p=0.20 — watch, do
not ban. kb2 -$333 (p=0.13), kb6 -$364 (p=0.46, already retired).
kb9 +$275 and kb4 +$195 are the profitable leaders.

Best finding of the day: at decision level kb5 wins only 44.7% but
returns +16.7%/$1 — the desk's best EV, from deep longshots where
break-even is far below 50%. Our most-wrong caller is our most
profitable one. -> new M9 "Underdog" cheap-bid trader (additive).

Plan deltas: M7 dropped · M4 -> M4' (bands only) · M6 folded into M1 ·
M2 demoted (payoff 52.8% -> 16.6%) · M3 promoted (best fixed arm
flipped) · M8 + M9 added · all gates now regime-stratified, since a
pooled sample mixes regimes and hid all of this.

## 2026-08-28 — M1 shipped to SHADOW; two real bugs found; premise NOT confirmed

PlattCalibrator (agents.py) + per-arm shadow layer in online.py. Every
kb row now carries p_cal beside p_up; NOTHING trades on it. State in
results/kb_calib.json, published in online_status.kb_calib.

Two implementation bugs found by checking output instead of trusting it:
1. Per-sample 2-parameter Newton is degenerate: the 2x2 Hessian from a
   single observation is exactly rank-1 (det = w^2x^2 - (wx)^2 = 0), so
   b is unidentifiable. Symptom: b froze at exactly 1.000 for all nine
   arms while a drifted to -2.7. 
2. Decayed-accumulator Newton overshoots: the gradient keeps
   re-applying residuals computed at stale parameters. Symptom: a =
   -0.64 vs the batch answer -0.13; b swung to 2.4.
Shipped fix: sliding-window (150) batch refit every 5 updates. Bounded
work, cannot overshoot, tracks drift by construction, and reproduces
the offline audit's numbers by design (kb a -0.115/b 0.699 online vs
-0.129/0.746 batch) so daemon and dashboard agree.

SHADOW VERDICT (warm-start replay, prequential decayed log-loss):
calibration HELPS ONLY kb7 (+0.0066), the most miscalibrated arm, and
is marginally worse on the other eight (-0.004 to -0.021). M1's premise
is NOT confirmed. Had we wired it into the decision tier on the
strength of the 08/26 audit we would have shipped a regression. The
shadow week continues; it earns the switch-over on live windows or it
is dropped. This is the process working, not the hypothesis winning.

## 2026-08-28 — champion/challenger routing: everything now proves itself live

btc_rl/treatments.py: every improvement runs as a TREATMENT on real
live windows, paired against the incumbent, promoted only when the live
stream crosses a sequential boundary. Nothing is adopted on backtest
evidence again — three confident offline conclusions of ours have
already been wrong (tier-1 RCA, the 09h "toxic hour", the calibration
premise).

Design: paired same-window scoring (regime cancels out of the
difference — 08/27 showed one day can swamp an unpaired test);
effective n = WINDOWS; Wald SPRT (1945) so continuous monitoring is
valid; standing down is a real decision scoring 0, which is how a veto
can win without betting; promotion is stamped and reversible and the
loser keeps running so a regression stays visible.

Treatments live: M8 regime gate, M2 knife-edge veto, M1 calibrated
gate, M9 Underdog (cheap bids), M2+M8 stacked. Backfilled on 153
settled desk windows, then the daemon continues the same computation.

Backfill (NOT a verdict — all still COLLECTING):
  champion            153 bets            EV -5.47%
  M9 Underdog          10 bets, 143 skips EV +25.72%  (+7.15% paired)
  M1 calibrated gate   55 bets,  98 skips EV  +3.82%  (+6.84% paired)
  M8 regime gate       88 bets,  65 skips EV  +1.21%  (+6.16% paired)
  M2+M8 both vetoes    17 bets, 136 skips EV  -7.72%  (+4.61% paired)
  M2 knife-edge veto   39 bets, 114 skips EV  -6.47%  (+3.82% paired)

TWO BUGS FOUND AND FIXED WHILE BUILDING IT:
1. Regime signal measured at the wrong time. Market accuracy was being
   read from whichever row landed last per ticker — rows near the close
   where the market is trivially right (80-100%), so the gate never
   fired once in 154 windows. Fixed to the decision-time row (earliest
   inside the <=12-minute envelope); it now skips 65 of 153 and lifts
   EV from -6.08% to +1.21%.
2. SPRT divided by a near-zero variance. When early paired differences
   are identical (both policies stand down => d = 0) the running
   variance collapses and the LLR explodes: M1 hit LLR 123 on 154
   windows and tripped a FALSE AUTO-PROMOTION. Fixed with a variance
   floor (0.01, far below any real spread since paired EV differences
   are O(1)) plus a 12-window warmup before scoring. LLRs are now
   0.15-0.41 and nothing promotes on the backfill.

The second bug is the important one: it would have silently promoted a
treatment into live trading on numerical noise.

## 2026-08-28 — M3 Fixed-Share added; a 2.1c pricing bug was faking every win

M3 (Herbster & Warmuth 1998) implemented in treatments.py and routed as
two live treatments: t_fshare (expert-weighted leader) and t_fs_reg
(that leader plus the regime gate). Weights update on each arm's
log-loss AFTER the window was used to decide — no leakage. Chosen over
"freeze the best arm" because the best fixed arm flipped sign in one
day (kb4 +4.7% -> -3.7%), and over contextual bandits because we see
every arm's outcome every window (full information, not bandit
feedback).

THE IMPORTANT FINDING — a 2.1c ask artifact was manufacturing nearly
all challenger edge. The champion was scored at the REAL ask it paid
while challengers priced from the modelled decision-time quote, handing
them a systematic 2.1c discount (mean synthetic - real = -2.10c over
150 paired windows). Under that bug:

  M3 Fixed-Share    +23.52% paired   -> after fix   +1.17%
  M9 Underdog       +10.61%          -> after fix   -5.03%
  M2 knife-edge      +7.98%          -> after fix   -8.59%
  M8 regime gate     +9.09%          -> after fix   +0.38%
  M1 calibrated      -3.74%          -> after fix  -26.09%

Fix: every policy, champion included, now prices from the same
decision-time quote via _ask(side). Absolute EV is therefore
model-priced (optimistic vs real fills); only the PAIRED DIFFERENCE is
apples-to-apples, and that is the only thing the SPRT tests.

After the fix only M8 (+0.38%) and M3 (+1.17%) are positive at all,
both far from the promote boundary. Every mitigation we designed is
now unproven on fair pricing. Had the framework shipped one commit
earlier it would have auto-promoted a pricing artifact into live
trading — the second near-miss auto-promotion in one day (the first
was the SPRT variance collapse).

Fixed-Share weights after backfill: kb2 .303, kb9 .177, kb4 .156,
kb3 .130, kb7 .119, kb8 .115.

## 2026-08-28 (Fable session) — MSE everywhere + the confirmed-bug fix wave

Metric switch (user-directed): headline error metric MAE -> MSE across
daemon, all six pages, audit scripts. Load-bearing switches stamped:
the hourly retrain no-regression gate now accepts/rejects on val MSE
(keys val_mse_before/after; MAE-era history rows left untouched, site
charts start the MSE series at the cutover rather than squaring old
means — mean(|e|)^2 != mean(e^2)), and _winner_variant ranks on
trailing MSE. Caveat on record once: squared loss is outlier-sensitive
in fat-tailed series; MASE kept alongside MSSE for that reason. Frozen
backtest numbers (index.html, offline table) relabeled "historical
MAE" — recomputing MSE from retained aggregates is impossible and
altering a dated snapshot would be falsification.

Bug fixes shipped (each verified against the metric that exposed it):
 1. Trader lockout (critical): entry guard extended to all six traders;
    pt3-pt6 now find their own entry minute.
 2. Treatment stale quote (critical): <=12-min envelope filter added to
    _treat_evaluate row selection. VERIFIED: post-fix champion -6.04%
    vs the independent agent's predicted -6.5%; knife-edge vetoes 29%
    vs predicted ~30% with the predicted sign flip (+1.32%); M10 trips
    29% vs predicted 29.3%. Three predictions, three matches.
 3. t_cal leakage: scores from the pre-settle p_m1 stamp, never a
    calibrator that already saw the outcome (bets 2/150 until p_m1
    history accumulates — honest, not fabricated).
 4. Per-order fee: _order_fee_c helper; per-contract ceil kept ONLY
    for single-contract selector ledgers where the two coincide and
    thetas are frozen. Ends the $2,355 phantom-fee accrual.
 5. _pt_leader ranks only tradeable (<=12 min) decisions.
 6. Bands sigma-conditioned (normalized conformal, Lei et al. 2018;
    0.6x-median floor for the calm bucket; floor/ceil not int()).
 7. Calibrator trains once per (arm, window) on the decision row.
 8. SPRT hygiene: config always from code constants (pre-registration
    restored); load-order fix kills the double-count path; seen-set
    insertion-ordered so the 4000-cap trims chronologically.
 9. Torch state: checkpoints carry Adam state; revert rebuilds the
    optimizer so rejected-replay momentum dies with the revert.

Post-fix treatment board (150 windows, all COLLECTING, honest basis):
M10+M8 +10.06% (LLR 1.10), M8 +8.23% (LLR 1.24) lead; nothing near a
boundary; nothing promoted.

Automation: scripts/run_audit.py recomputes desk/trader/tier1/tier2/
treatment health from raw ledgers every 10 min via cron (installed
alongside the publisher) into results/audit_report.json, shipped by
the publisher. metrics.json now carries mse/msse beside mae/mase
(additive keys, nothing renamed out from under history).

## 2026-08-28 — the 18:33 window: Gambler v2.1 (edge-at-fill gate)

User flagged the 18:33 PT ledger row. Forensics (KXBTC15M-26AUG282145):
pt4 entered at its own >=0.77 minute (18:35:33, conf 78.5%) — but by
then the ask was 79c, break-even 80.2%: a $1,911.85 stake at -1.7 pts
of stated edge. NEGATIVE EV BY CONSTRUCTION: a constant confidence
gate ignores the price paid, and confidence and price co-move — for
the market-anchored kb2 leader they are IDENTICAL (conf 0.685 = mkt
0.685 in the same row), so any constant gate on a kb2 leader is just a
price threshold. Every kb2-led follower entry that window carried edge
= -(spread+fee): pt -2.0, pt5 -2.1, pt4 -1.7 pts. The only trader that
declined was pt6 — the one whose gate compares confidence to price.

Fix: PT4_MIN_EDGE_C = 2.0 — Gambler v2.1 requires conf >= 0.77 AND
100*conf >= ask + fee + 2c at the actual fill. This also neutralizes
the kb2 degeneracy automatically (kb2-led entries can never clear a
positive margin). pt3's frozen pre-registered policy is deliberately
untouched (it cleared +8.3 pts here); pt/pt2/pt5 stay as the
uncorrected controls the curriculum needs. Ledger display now says
"1st entry Xc — each trader fills at its own minute" instead of
implying one desk-wide price.

## 2026-08-28 — pt7 Patient + pt8 Ideal (limit-order execution cohort)

User: "should they not fill at min price for their bid and skip if the
bid is too high — at least a few traders" then "create one ideal
trader from crypto sizing/pricing/entry-time practice."

Shipped, additive (controls untouched):
- pt7 PATIENT: Follower's signal and 10% sizing, but rests a limit at
  quoted ask - 2c; fills only if a later minute's ask reaches it, else
  the window is logged as an unfilled skip. Pure execution isolate —
  and deliberately naive, because...
- ...the backfilled naive limit treatment (M11) scored -5.21%/$1
  PAIRED vs the champion: resting bids are adversely selected (they
  fill when the market reprices the call down — Glosten-Milgrom's
  picked-off problem). pt7 exists to measure that live.
- pt8 IDEAL: regime gate (>=0.62 trailing decision-time market acc) +
  edge >= 2c at the LIMIT price + maker limit + half-Kelly capped 10%
  + 25% depth participation + FILL-TIME re-check of the leader's
  current confidence (refuses fills caused by a falling signal — the
  adverse-selection defense M11 lacks). Components adopted only with
  both literature and our own measurements behind them; hour filters
  and knife-edge veto refused (p=0.60 dredging; +1.39% too weak).
- Treatments M11 / M11+M8 added so the SPRT adjudicates the mechanism
  on live windows; design essays don't promote anything here.
Full grounding with citations in docs/SEV0_REMEDIATION.md.

## 2026-08-29 — TA metrics review → the decision layer
The TA's critique of the Metrics Lab ("research observability, not an
experiment decision system") was adopted as a program. First shipment,
computed entirely from the daemon's own per-window paired log
(results/treatments.jsonl, 168 windows — nothing re-estimated from
aggregates):
- scripts/emit_decision_board.py → results/decision_board.json, on the
  10-min cron: per treatment, normal AND seeded-bootstrap 95% CI,
  median vs mean, top-3 |Δ| concentration (jackpot detector), P(Δ>0),
  P(Δ>2¢), MDE and power at the FAMILY-WISE α (0.003125 one-sided),
  evaluability vs activity overlap, veto decomposition (losses
  avoided / wins forgone / control EV on skipped windows), effect
  slices by regime, ET 6-hour block and leader (n≥15 to count as
  powered), worst powered slice + sign consistency, projected windows
  to an SPRT verdict, and an automatic state:
  PROMOTE / KEEP_TESTING / HOLD / KILL / INVALID — rules
  pre-registered in the emitter, SPRT remains the promotion authority.
- First run: 10 KEEP_TESTING, 2 HOLD (t_cal, t_cheap — coverage <25%,
  "a different product"), 0 KILL/INVALID. M8: Δ +6.0¢/$1,
  boot CI [−0.0¢, +12.0¢], P(Δ>0)=0.97, ~590 windows to a verdict at
  current drift; worst powered slice ET 18-24 at −4.8¢ (n=43).
- The honest headline the TA asked us to surface: power to detect the
  pre-registered 2¢ edge at n=168 under family-wise α is ~2%
  (MDE ≈ 12¢/window). Fixed-horizon testing is hopeless at this
  traffic — the sequential SPRT is not a stylistic choice, it is the
  only viable design. This number now lives on the board.
- Definitional bug caught during the build and recorded in
  DECISIONS.md: a stand-down is a COMPLETE scored observation (EV 0 by
  pre-registration), not a missing pair; the first draft invalidated
  all 12 challengers by conflating the two.
- PT_TAU added to the manifest config export (homepage was falling
  back to REGIME_FLOOR — same 0.62 by coincidence, different constant).
Queued next (needs new capture or more data, not just arithmetic):
post-fill markout (1m/5m from ticks), capacity/depth metrics,
offline→online retention tracker, Brier decomposition + ECE,
risk–coverage curves, redundancy/kill board, factorial interaction
for combo treatments.

## 2026-08-29 — Quant Universe waves 1–2 shipped; Ladder≡Follower explained
Wave 1: universe.html (atlas, Explore/Engineer lenses), clock.html
(cadences + dissected retrain + honest bill), agents.html (authority
cards + activity stream), museum.html (six frozen failure exhibits).
Wave 2: home.html rebuilt as the Playground (play-this-window game,
8 character cards, $1,000 wallet replaying real treatments.jsonl EV
series — verified against an independent recompute to the cent);
old 7-scene home preserved as instrument.html.
Investigated (builder flag): pt_trades and pt2_trades carry identical
(ticker, stake, pnl) series — CORRECT by construction: the Ladder
trades the same PT_FRAC-of-bankroll entries as the Follower and only
diverges when bankroll reaches 11x its level ($11,000) and a rung
banks; both slid to $384 instead, banked_c=0, so the paths coincide
on this sample. The ablation (does rung-banking protect a climb?)
remains untested until a climb happens. No code change; not a bug.
