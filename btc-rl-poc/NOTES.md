# Research log

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
