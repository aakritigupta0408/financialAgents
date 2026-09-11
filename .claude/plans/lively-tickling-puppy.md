# kb8 + kb9: improve on kb7 without touching kb7

## Context

kb7 (Chronos-Bolt-small, zero-shot, univariate, frozen) is our best decorrelated arm — the only one whose confidence survives biddability — but its audit exposed four measured weaknesses:

1. **Univariate & blind to the market.** It sees only minute closes — no volume, perp lead, tape flow, and crucially not the Kalshi market price it must beat. Clustered t vs market = −1.02: not significant.
2. **Zero-shot.** Never adapted to BTC minute dynamics; context ablation plateaus at 512 (more of the *same channel* adds nothing — new information must come from new channels or a learned readout).
3. **Miscalibrated mid-buckets.** Stated 0.6 confidence delivers 52%; top bucket 84%. A learned calibration layer can harvest this.
4. **Stateless.** No settle-time learning at all.

Per the standing law (additive-only), kb7 stays byte-identical. Two new treatments:

- **kb8 — calibrated decorrelation stack** (live within a day): an online BinaryLogit that fuses kb7's signal with the market anchor — the kb4 pattern applied to our best decorrelated source. Attacks weaknesses 1 (market-blindness), 3, 4.
- **kb9 — Chronos-2 with covariates** (research-gated): `chronos-forecasting 2.3.1` is **already installed**, and the Chronos-2 generation accepts covariates natively — exactly the TA's "LLM time-series with more inputs" direction. Attacks weaknesses 1, 2. Goes live only if it beats the kb7 replay baseline on a window-clustered offline gauntlet.

All offline comparisons are **window-clustered** (the lesson from the 13/14 counting error — stream metrics by window, never by minute).

## Phase 1 — Offline gauntlet (`tests/kb8_gauntlet.py`)

Extend the replay mechanics of `tests/kb7_context_ablation.py` (fetch bars once, replay a mid-window decision per settled window) into a shared bench:

- Windows: settled kb2 rows at `6 <= mins_left <= 9`, one per ticker, last ~260 (same selection as the ablation, `tests/kb7_context_ablation.py:20-24`); bars via `fetch_range` (~64h → ~180–260 usable windows).
- Candidates, all scored on the identical window set:
  1. `kb7-replay` (bolt-small @512) — the baseline to beat. **Fix the stale unpack**: `_chronos_p_up` now returns a 4-tuple `(p, w80, lo, hi)`, but the ablation script does `p, _ = out` (`tests/kb7_context_ablation.py:42`) — kb8_gauntlet unpacks 4.
  2. `bolt-base` — same readout, bigger checkpoint (cheap ablation).
  3. `chronos-2 univariate` — new pipeline class from the installed package, same quantile readout at the strike.
  4. `chronos-2 + covariates` — add minute volume (from bars) as the first covariate; perp/tape from `live_snapshots.jsonl` joined ±90s (the `tests/warmstart_kb6.py` join pattern) where coverage allows.
- Report per candidate: n windows, acc, Brier, **window-clustered paired t vs kb7-replay and vs market**, and median CPU latency per predict (live budget: ≤2s per minute loop; try `device_map="mps"` if CPU is too slow — torch 2.7.0 on darwin supports it).

Decision rule (pre-registered): kb9 goes live only if `chronos-2` (either variant) beats `kb7-replay` on clustered Brier with |t| > 2. Otherwise kb9 is reported as a measured negative and only kb8 ships.

## Phase 2 — kb8 live arm (`btc_rl/online.py`, additive)

New 12-dim online stack, template = kb4 (`_kb4_features` at online.py:539-556, commit at 1928-1950, settle at 2069-2114):

- `KB8_DIM = 12`, `_kb8_features(p7, w80, k_pup, bx, pf, mins_left)`:
  `[1.0 (bias), p7−0.5 ×2, mkt−0.5 ×2, (p7−mkt) ×2 (disagreement — the decorrelation harvest term), agreement product, w80 normalized (band width = kb7's own uncertainty), market-presence flag, above-strike z (bx[3]), mins_left/15] + pf[4]` → trim to 12 by folding presence into the market term as kb4 does.
- Commit: inside the existing kb7 block (online.py:1961-1977), immediately after `fm` unpacks — reuse the same `fm` result (one Chronos call per slot, no added latency), guarded by its own `("kb8", ticker, slot1)` key in `kb_made`. Emit row `{**common, variant:"kb8", p_up, call, b8x, trained}`. kb7's rows and Conviction Book logic are untouched; kb8 adds **no** pb_bets stream yet (pre-registration comes later, only after a live track record).
- Settle: in the settle loop, `if r.get("b8x")` → `kb8_logit.update(b8x, outcome)` (label = outcome, like kb4/kb6). Checkpoint `results/kb8_logit.json` with a **unique `.tmp8` suffix** at the online.py:2102-2114 save block.
- Startup load: kb4 pattern at online.py:1549-1556 (dim mismatch → fresh model).
- Status JSON: add `"kb8"` to the arms tuple at online.py:2381.

## Phase 3 — kb8 warm start (`tests/warmstart_kb8.py`)

kb7's own log is too thin to train on (200 settled rows, **15 windows**). Instead, replay-generate the kb7 signal over history:

- For each settled kb2 row (2,826 rows carry decision-time `strike, mins_left, mkt_p_up, bx, pf`), call `_chronos_p_up` on the bar-close prefix strictly before `made_ts` (the `upto[-512:]` mechanic from the ablation script — no leakage; unpack 4 values).
- Build `b8x` from logged decision-time values only, sort by `close_ts` (settle order, the `tests/warmstart_kb4.py:36` discipline), prequential predict-then-update, report full and final-quarter prequential accuracy **and a window-clustered comparison vs kb7-replay and market on the same rows**, save `results/kb8_logit.json`.
- Runtime note: ~2,000+ Chronos calls × 0.05s ≈ 2–3 min — fine. Bars from `fetch_range` limit warm start to ~64h of windows; that's ~180 windows ≈ 12× kb7's own log.

## Phase 4 — kb9 live arm (only if Phase 1 gate passes)

Same additive pattern as kb8: `_chronos2_p_up` with its own lazy singleton (kb7's `_CHRONOS` untouched), covariate assembly from `kbars` volume + `_flow_stats`/OKX fields already computed in the loop, `variant:"kb9"` rows, frozen zero-shot (no checkpoint). If MPS is needed for latency, set device only in the new singleton.

## Phase 5 — Site registration (additive, `site/ab_dashboard.html`)

Six touch-points per new arm (verified locations): `ARMS_META` (~line 449), `LCOL` (~480), `DL_TAU` (~705, start kb8 at 0.62), dlData loop list (~710), ledger chip markup (~line 89), chip wiring (~802-811). Add league-table row + decision-ledger chip for kb8 (and kb9 if promoted). Existing arms' cards/rows untouched. Home page: no change in this pass (kb8 earns a desk-chat mention only after live windows accrue).

## Files

- `tests/kb8_gauntlet.py` — new (offline bench, all candidates, clustered stats)
- `tests/warmstart_kb8.py` — new (replay warm start, prequential report)
- `btc_rl/online.py` — additive: KB8_DIM/`_kb8_features`/kb8 commit-settle-load-save + status tuple; Phase 4 adds `_chronos2_p_up` + kb9 block
- `site/ab_dashboard.html` — additive: 6 registration points
- `tests/kb7_context_ablation.py` — fix stale 2-tuple unpack (test hygiene; kb7 live path untouched)

## Verification

1. Gauntlet table prints with clustered t's; latency per candidate measured; gate decision recorded in the commit message.
2. Warm start reports prequential accuracy (target: beats kb7-replay on the same windows — else ship kb8 cold and say so).
3. Restart daemon; within one 15-min window confirm: kb7 rows byte-identical in shape, new kb8 rows appear with `b8x`/`trained`, settle updates increment `trained`, `kb8_logit.json` written via `.tmp8`.
4. Dashboard: kb8 league row + ledger chip render; all existing rows unchanged; graphs keep titles/legends/axis labels.
5. Report kb8's first live metrics window-counted, with Wilson CIs, in the league table like every other arm.
