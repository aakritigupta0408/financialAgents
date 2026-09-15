# Combined-Student & Overfit Study — findings

*Can we (a) overfit/reconstruct the data in-sample, (b) fuse the large models + engineered
features + retrieval + GBM into a student that beats any single model out-of-sample, and
(c) fine-tune Chronos on all history to do better? Full walk-forward, leakage-canaried.*

## Headline

1. **YES, the models reconstruct seen data perfectly.** Every student (logistic+poly², GBM
   stacker, GBM fuse-all) reaches **100% in-sample** accuracy. Capacity is not the limit.
2. **The in-sample skill does NOT generalize.** Out-of-sample, the fused students land at
   **~0.655 (mid-window) / ~0.54 (open)** — the ~0.35–0.46 gap is pure memorization, and the
   **label-shuffle leakage canary returns 0.500 / 0.513**, so those OOS numbers are honest
   (nothing is peeking at the outcome).
3. **Fusion did NOT beat the best single model OOS.** Mid-window, zero-shot **Chronos-base
   alone = 0.696**; the stacker and fuse-all GBM = 0.655 — *worse*. Retrieval, poly features,
   and GBM added variance, not signal. More complexity → more overfit, not more edge.
4. **Fine-tuning Chronos on all history hurts.** In-sample F1 0.846 → OOS F1 0.601, below
   zero-shot base 0.683 (precision 0.516 vs 0.741; 46 FP vs 15). Overfitting, decisively.
5. **The lever that actually moves accuracy is ENTRY TIMING, not the model.** OOS accuracy:
   ~0.54 at window open → ~0.66–0.70 at ~11 min left. The signal accrues as the window fills.

## The numbers (OOS, walk-forward)

### Mid-window entry (~11 min left) — 368 windows, 148 test, base-up 0.486 · canary 0.500
| source | acc | F1 | prec | in-sample acc |
|---|---|---|---|---|
| Chronos-base | 0.696 | 0.662 | 0.677 | — |
| Chronos-small | 0.696 | 0.667 | 0.672 | — |
| TimesFM 2.5 | 0.669 | 0.657 | 0.627 | — |
| barrier closed-form | 0.662 | 0.658 | 0.615 | — |
| retrieval kNN (k=25) | 0.655 | 0.675 | 0.596 | — |
| GBM (features) | 0.662 | 0.675 | 0.605 | — |
| Student A — logistic stacker (teachers+retrieval+gbm) | 0.655 | 0.671 | 0.598 | 1.000 |
| Student B — logistic + degree-2 basis (features) | 0.568 | 0.522 | 0.530 | 1.000 |
| Student C — GBM fuse-everything | 0.655 | 0.679 | 0.593 | 1.000 |

### Window-open entry (~15 min left) — 1,364 windows, 546 test, base-up 0.496 · canary 0.513
All teachers 0.53–0.56 except barrier **0.621**; students OOS 0.53–0.54; every student 0.92–1.00 in-sample.

## What this means (not the null — a quantified, bounded edge)

- The direction **is learnable to a real ~66–70% OOS at the mid-window entry** — leakage-free,
  reproducible. That is a genuine, tradeable edge and is exactly what the live T2 arm uses
  (zero-shot chronos-bolt-base). It is NOT random.
- Near-**100% is achievable only in-sample** (memorization); the honest OOS ceiling here is
  ~0.70 mid-window. No fusion, retrieval, basis expansion, or fine-tune we tried crossed it —
  and the canary proves the gap is memorization, not a fixable modelling miss.
- The single strongest lever is **decision time** (0.54 → 0.70 as the window fills), followed
  by the **first-passage barrier physics** (the best or near-best single OOS signal at both
  regimes). Bigger neural models did not add OOS signal over these.

## Where to push next (to raise the OOS ceiling honestly)
1. **Enter later / model the value-of-waiting** — quantify accuracy vs mins-left and enter at
   the point where edge×payoff is maximal, instead of at open.
2. **Exogenous features not yet in the path** — cross-venue order-flow imbalance, perp funding
   / basis, spot-vs-BRTI lead-lag, options-implied skew — signals *outside* the BRTI path the
   FMs already see. (The event tapes in results/events_* hold this; unused so far.)
3. **Calibrated selective trading** — act only on the high-confidence tail where OOS precision
   is highest, rather than every window.
4. Keep the canary in every experiment; treat any >0.55 shuffle-OOS as a leak, not a discovery.

*Artifacts: scripts/combined_student.py, scripts/insample_reconstruction.py; reports
research/combined_student_report.json (open), research/combined_student_midwindow.json (mid),
research/chronos_finetune_all_history.json (fine-tune). Teacher probs cached in
research/teacher_cache*.json.*
