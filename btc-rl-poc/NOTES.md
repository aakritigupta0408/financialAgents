# Research log

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
