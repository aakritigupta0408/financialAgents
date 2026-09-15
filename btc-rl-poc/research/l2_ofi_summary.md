# Sub-bin L2 Order-Flow Microstructure at Fixed OPEN Entry — Result

**Question.** Do sub-bin order-flow-imbalance (OFI) / microstructure features built from the
raw Coinbase L2 tape raise OOS accuracy above the ~0.70 entry-time ceiling for predicting
whether a 15-min BTC window closes ≥ its open, when features are restricted to the first
**180 s after the window OPEN** (fixed-entry, no look-ahead)?

**Verdict: NO — decisively. Sub-bin OFI at the fixed open entry does not push past 0.70; it
does not even beat a coin flip.** Best OOS accuracy across all feature sets and both models
was **51.1%** (GBM, OFI+barrier) vs a 50.2% test base rate and the 70% reference ceiling —
i.e. **~0 edge, roughly −19 points below the 0.70 target.**

## Data / method
- Tape: `results/events_micro/micro-*.jsonl`, 358 hourly files, ~12 GB, **20,028,698** L2
  update messages, spanning 2026-08-30 12:00Z → 2026-09-15 22:00Z (UTC).
- Reconstructed a full-depth top-of-book incrementally per side (SortedDict price→size,
  size 0 = remove). **Book sanity: 0 crossed/negative spreads; spreads 0.01–0.09 bps
  (Coinbase BTC-USD penny tick); `mid_at_entry` tracks the official `floor_strike`/open
  (dist mean ≈ −0.3 on a ~78,700 price).** Reconstruction confirmed correct.
- Entry = window OPEN; features use **only** events in `[open, open+180s)`. No settlement-window
  or post-180s data.
- Features per window: **Cont-Kukanov-Stoikov OFI** at best quote over nested sub-bins
  (1s/5s/30s/60s/180s), top-of-book & depth-5 queue imbalance (time-averaged),
  Stoikov micro-price vs mid + drift, spread (bps) mean/vol, update intensity, realized vol.
- Label = `exact_yes` = 1{close ≥ open}. `floor_strike` = the window's open price.
- 1350 labeled windows had usable in-phase tape (of 1558 in span; 208 dropped to tape gaps).
- Eval: sort by time, walk-forward **60/40** (810 train / 540 test),
  HistGradientBoostingClassifier + LogisticRegression. Selective hit-rate at coverage ≥ 0.90
  by |p−0.5|. Label-shuffle leakage canary.

## OOS results (test span 2026-09-09 17:00Z → 2026-09-15 21:30Z, base rate 50.2%)

| Feature set        | GBM OOS acc | Logit OOS acc | hit@cov≥0.90 | vs 0.70 ceiling |
|--------------------|-------------|---------------|--------------|-----------------|
| OFI/micro only     | 49.3%       | 52.4%         | 48.8%        | −17.6 pts       |
| OFI + barrier      | **51.1%**   | 50.4%         | 51.0%        | **−18.9 pts**   |
| Barrier only (dist+vol) | 50.7%  | 51.7%         | 51.2%        | −18.3 pts       |
| Majority class     | 50.2%       | —             | —            | −19.8 pts       |

- **hit@coverage≥0.90 ≈ 49–51%** everywhere — nowhere near the 89%@90% target. The most
  "confident" predictions are no better than random, i.e. the probabilities are uninformative.
- **Leakage canary** (shuffle train labels, refit): GBM OOS = **49.7%** (5 runs:
  0.472, 0.506, 0.515, 0.469, 0.524). Clean — matches the ~50% real result, confirming the
  real result is genuine no-signal, not a leak.

## Why (diagnostic)
- **Pipeline is not broken — there is simply no generalizable signal.** GBM fits the training
  set (84.6% train acc; a deep unregularized GBM hits 100% train) yet generalizes to ~50.9%
  OOS. Classic overfit-to-noise.
- **Univariate |AUC−0.5| ≤ 0.03 for every feature** (strongest: queue imbalance, dist,
  intensity ≈ 0.53) — none survive OOS.
- **Label lag-1 autocorrelation = 0.0065** — window outcomes are serially independent. Over a
  15-minute horizon, a liquid asset's close-vs-open is effectively a martingale, and 180 s of
  open-time order flow carries no exploitable information about it. Even `micro_drift` (the
  actual price move over the first 180 s) has univariate AUC 0.515.

## Bottom line
Sub-bin OFI/microstructure at the **fixed OPEN entry does NOT beat 0.70 — or 0.50.** OOS ceiling
observed ≈ **51%**, ~19 points short of the 0.70 target and statistically indistinguishable from
chance and from the label-shuffle canary. Any ~0.70 "entry-time ceiling" seen elsewhere must come
from a different entry regime (e.g. features observed closer to settlement, or a distance-to-strike
that is informative only late in the window) — not from order-flow microstructure at the open.

## Artifacts
- Per-window features: `results/l2_ofi_features.jsonl` (1350 rows; features + label; re-runnable)
- Eval metrics: `research/l2_ofi_eval.json`
- Extractor: `.wt_tmp/extract_ofi.py`   Eval: `.wt_tmp/eval_ofi.py`
