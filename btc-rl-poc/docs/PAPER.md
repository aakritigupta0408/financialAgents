# The Quant Universe: A Self-Auditing Experimentation System for Short-Horizon Market Prediction

**v1 · 2026-08-29 · numbers as of 177 paired windows.** This is a dated document; the live system supersedes it. NYU Deep Learning course project (paper trading only; nothing here is financial advice).

## Abstract

We report on a live paper-trading research system that predicts 15-minute Kalshi BTC binary settlements. Nine probability models — spanning tabular reinforcement learning, contextual bandits, online logistic regression, a distributional DQN, an LSTM, and two frozen time-series foundation models — all fail to beat the market's own price: the best Brier skill score versus the market is −0.0016 (results/audit_report.json). The durable contribution is therefore not a predictor but a measurement instrument: paired same-window sequential experimentation under family-wise error control, mechanism validation for every treatment, automated root-cause analysis of every losing trade, and nine machine-checked invariants each named for the incident that created it. The instrument caught two would-be false promotions (an SPRT variance collapse and a 2.1¢ pricing-basis artifact) before they shipped. The only profitable trading policy discovered is one whose skill is refusal: it declines to trade 84% of the time and earns +14.8¢ per dollar staked on the remainder (results/audit_report.json).

## 1. Introduction

Every fifteen minutes, Kalshi's KXBTC15M contract settles a single binary question: will the Bitcoin price close at or above a strike? The task is a near-ideal laboratory for online machine learning under real constraints. Outcomes arrive within minutes, roughly 96 times a day, so prequential evaluation (Dawid, 1984) accumulates at a pace no daily-bar strategy can match. The market publishes its own probability continuously — the contract price — so every model faces an observable, adversarial consensus baseline rather than a straw-man climatology. And the costs are real: a bid–ask spread and a fee of ceil(7·p·(1−p)) cents per contract stand between statistical skill and economic skill (docs/MANUAL.md).

The project began as an attempt to build a predictor that beats this market. Section 4 documents, in some detail, that it did not: minute-scale BTC is approximately a martingale on price level (persistence is near-unbeatable, MASE ≈ 1; docs/MANUAL.md), and on event probability no arm in the stable carries positive skill against the book on the current sample (results/audit_report.json). The project's pivot — directed by its TA review — was to treat that negative result as the research object: to build the instrument that can *establish* such a result quickly, cheaply, and without fooling its operator. What follows describes that instrument — a self-auditing experimentation system in the spirit of Arnott, Harvey and Markowitz's (2019) protocol for backtesting — and the empirical record it produced.

## 2. The system

### 2.1 Planes and tiers

The system is organized as seven planes — Data, Prediction, Probability, Decision, Execution, Risk & Capital, and Learning & Governance — mapped onto a historical tier numbering T0–T5 (docs/OS_BLUEPRINT.md). Signal flows upward:

- **T0 — Data.** Open no-auth streams: Coinbase 1-minute bars and trades, OKX funding, Deribit mark and order book, RSS news scored by a frozen CryptoBERT, and Kalshi quotes/depth/open interest, snapshotted every 30-second poll (docs/MANUAL.md).
- **T1 — Price arms.** Independent arms commit integer price predictions every 5 minutes at +5/+15/+30-minute horizons: a control tabular-Q learner, a replay arm, LinUCB contextual bandits (t2, t6, t10, t11; Li et al., 2010), a linear-Q arm (t7), a distributional DQN (t8), and an LSTM (t9). Arms never share model state, so metric gaps are attributable to exactly one design difference. Learning is two-speed: an immediate update per scored prediction plus an hourly replay retrain behind a hold-out no-regression gate that reverts bad retrains (docs/MANUAL.md).
- **z-bridge.** The t8 distributional head supplies the σ that converts a price forecast into a z-score against the strike — the bridge from the price plane to the probability plane, and the site of the SEV-0 defect described in Section 5 (docs/MANUAL.md, docs/SEV0_REMEDIATION.md).
- **T2 — Probability arms.** Once per minute per open contract, each kb arm writes P(close ≥ strike): kb (control: t8's distribution, per-phase calibrated), kb2 (market-anchored blend), kb3 (online logit, 24 features), kb4 (stack of kb2×kb3), kb5 (train-where-you-trade EV logit), kb6 (fast-information; retired), kb7 (Chronos-Bolt-small, frozen zero-shot; Ansari et al., 2024), kb8 (learned log-opinion pool of kb7 × market), kb9 (TimesFM 2.5, frozen zero-shot; Das et al., 2024). Every row also carries p_m1, a shadow Platt-calibrated probability (Platt, 1999) that is never a decision input (docs/MANUAL.md, docs/DECISIONS.md).
- **T3 — Decision.** A leaderboard crowns a leader arm; entries require confidence ≥ 0.62. Treatments layer vetoes and gates on top: knife-edge veto (M2), Fixed-Share leader (M3; Herbster & Warmuth, 1998), regime gate (M8), cheap-contract filter (M9), EV-ranked leader (M12) (results/program.json).
- **T4/T5 — Execution and capital.** Modeled fills at the quoted ask plus fee; eight paper traders take the same signals under different sizing and entry policies, from a full-Kelly-style Gambler to a supervised-edge half-Kelly MLE trader (docs/MANUAL.md; Kelly, 1956).

A notable inventory result predates the pivot: five separate attempts to upgrade the kb7 foundation model (4× parameters, Chronos-2 at two context lengths, covariates, TimesFM, a fine-tuned Bolt) all landed within noise of the 47M-parameter zero-shot original (|t| < 2 on window-clustered paired Brier). At this horizon the extractable signal, not the model, is the binding constraint (docs/MANUAL.md).

### 2.2 The four loops

The running product is four loops (docs/OS_BLUEPRINT.md): a 30-second **runtime** loop (feeds → features → T1 → T2 → policy → execution → ledger → settlement); an hourly-and-on-settlement **learning** loop (evaluation → candidate → holdout gate → keep/revert, with policy changes additionally passing a paired live A/B and a human promotion decision); a **maintenance** loop (monitors → incident → mitigation → postmortem → regression test); and a **research** loop (observation → hypothesis → ticket → treatment → experiment → decision → knowledge base). The human owner is the governance root and the only promote/retire authority.

## 3. Methods

**Prequential, paired, windows-not-rows.** All evaluation is prequential (Dawid, 1984): every prediction is scored on data the model had not seen when it committed. Treatments are scored strictly on paired same-window differences against the champion — both policies see the identical window, quote, and settlement, and only the difference Δ enters the test (results/decision_board.json). Effective n is counted in 15-minute windows, never minute rows, because entries within one window share fate (docs/MANUAL.md).

**Sequential testing under family-wise control.** With 16 concurrent comparisons, the per-test significance level is Bonferroni-corrected to α = 0.05/16 = 0.003125, one-sided, giving SPRT boundaries of +5.663 (promote) and −2.299 (reject) at β = 0.1 for a pre-registered minimum edge of 2¢ per dollar (Wald, 1945; results/audit_report.json, results/decision_board.json). Two hygiene rules exist because their absence caused a near-miss: a variance floor of 0.01 and a 12-window warmup, added after early identical paired differences collapsed the running variance and drove one treatment to LLR 123 on 154 windows — a false auto-promotion caught before shipping (NOTES.md).

**Power honesty.** The board computes, and displays, its own weakness: at n ≈ 170 windows, power to detect the pre-registered 2¢ edge at the family-wise α is roughly 2%, with a minimum detectable effect near 12¢/window (NOTES.md, results/decision_board.json). Fixed-horizon testing is hopeless at this traffic; sequential testing is not a stylistic preference but the only viable design. Uncertainty is reported as both a normal approximation and a seeded 2,000-draw bootstrap CI (Efron, 1979), alongside P(Δ>0), P(Δ>edge), effect slices by regime/hour/leader, and top-3 concentration as a jackpot detector (results/decision_board.json).

**Mechanism validation.** A positive Δ is necessary but not sufficient: each treatment declares the mechanism by which it claims to earn, and the board checks the claim on the windows where the mechanism should bind. M8's gate is SUPPORTED (+20.4¢ on the windows it claims, ~0 elsewhere); M11's passive-entry mechanism is CONTRADICTED by the measured fill curve (results/diagnosis.json).

**Version-aware loss RCA.** Every losing trade receives an automated root-cause review graded on a six-level ladder (SEV-0 integrity-impossible … SEV-5 expected variance), judged against the rule *in force when the trade was made* — the first run produced 48 false policy-breach findings by judging v1 rows under v2 gates, fixed before shipping (docs/DECISIONS.md). Retirement pressure is pre-registered (negative mean at full sample, staleness, domination by a sibling, coverage < 25%), and paired incremental branch analysis asks of every combination whether it adds anything over each parent on identical windows — refusing cross-basis comparisons rather than fudging them (results/program.json).

## 4. Results

### 4.1 Nobody beats the book

Brier skill versus the market price itself, per arm (n = 169 windows; results/audit_report.json):

| arm | Brier | market Brier | BSS vs market |
|---|---|---|---|
| kb2 (market blend) | 0.2025 | 0.2022 | **−0.0016** |
| kb5 (EV logit) | 0.1878 | 0.1858 | −0.0109 |
| kb4 (stack) | 0.2067 | 0.2022 | −0.0222 |
| kb8 (opinion pool) | 0.2089 | 0.2022 | −0.0333 |
| kb3 (online logit) | 0.2097 | 0.2022 | −0.0371 |
| kb9 (TimesFM) | 0.2108 | 0.2022 | −0.0423 |
| kb (control, t8 dist.) | 0.2122 | 0.2022 | −0.0493 |
| kb6 (fast-info, retired) | 0.2129 | 0.2022 | −0.0527 |
| kb7 (Chronos-Bolt) | 0.2172 | 0.2022 | −0.0741 |
| kbf (retired) | 0.1048 | 0.0963 | −0.1014 |

Every skill score is negative. The best arm, kb2, is a market-anchored blend whose near-zero score measures its anchor, not private information. The open scientific question on the board is stated exactly this way: "binary predictors contain skill beyond the market price — status: unknown; no arm with BSS > 0 this sample" (results/program.json).

### 4.2 The execution funnel

Where a clairvoyant policy's value goes (results/diagnosis.json, results/model_internals.json):

| stage | EV per $1 |
|---|---|
| Oracle opportunity (same rules, clairvoyant; n = 150) | **+1.6913** |
| Forecast + probability + decision losses | unknown (not yet decomposed) |
| Desk decision EV, model quotes (n = 177) | −0.0228 |
| Execution gap (quote → fill) | −0.0533 |
| Realized EV, real fills | −0.0761 |

The rules permit ~$1.69 per dollar per window; the desk realizes −7.6¢. The single largest *measured* leak is execution — the −5.3¢ quote-to-fill gap, the chain's CRITICAL tier (results/diagnosis.json) — which is why the two priority treatments are both execution-adjacent.

### 4.3 The regime gate (M8)

M8 stands down when the market's own trailing-20 accuracy falls below 0.62. On 177 paired windows: Δ = +5.7¢/$1, bootstrap 95% CI [+0.0¢, +11.5¢], P(Δ>0) = 0.975, at 72% coverage (results/diagnosis.json). Its mechanism check passes exactly where it claims: +20.4¢ on the 49 regime-below-floor windows versus +0.0¢ on healthy windows (results/diagnosis.json). It has not crossed the SPRT boundary (LLR 1.17 of 5.66) and is therefore not promoted; it is merely the best-evidenced candidate, together with M10+M8 (Δ = +7.8¢, P(Δ>0) = 0.989; results/diagnosis.json).

### 4.4 Branch analysis: what a combination actually adds

Paired incremental analysis on identical windows (n = 179; results/program.json):

| comparison | incremental Δ | P(Δ>0) | reading |
|---|---|---|---|
| M2+M8 − M8 | −3.7¢ | 0.13 | knife-edge veto adds nothing to M8 |
| M3+M8 − M8 | −7.1¢ | 0.06 | Fixed-Share *interferes* with M8 |
| M10+M8 − M10 | +3.6¢ | 0.95 | regime gate genuinely adds to the exec guard |
| M11+M8 − M11 | +7.4¢ | 1.00 | …but M11+M8 − M8 = −2.9¢ (P 0.19): the maker combo's benefit is inherited from M8, not contributed by M11 |

The last row is the method's advertisement: a naive reading ("M11+M8 is positive, +3.0¢!") survives until the branch comparison shows the credit belongs entirely to the parent.

### 4.5 Adverse selection, measured

A full-information counterfactual over logged quote paths scored every candidate limit offset δ ∈ [0, 8]¢ below the ask (results/fill_curve.json): fill rate falls 100% → 59.5%, and — the finding — **win-rate-given-fill falls monotonically 67.9% → 47.0%** as the bid rests deeper. Resting orders fill preferentially when the market has just repriced against them: Glosten and Milgrom's (1985) picked-off problem, now measured on this book rather than asserted. This is the mechanism evidence that contradicts M11 (Section 4.4) and grades pt7's losses ("your resting bid was picked off"; results/loss_reviews.json). The least-bad taker entry bucket is 9–6 minutes before settlement (−1.8¢/$1 vs −12.0¢ at 6–3 minutes; results/fill_curve.json).

### 4.6 The falsification record

The system's negative results are reported as results:

- **The toxic hour (M7).** A "cursed hour" with 64% errors dissolved under multiplicity: 24 hours searched, ~1.98 false positives expected by chance, 2 found; p = 0.60. Dropped, not deferred (docs/DECISIONS.md, results/program.json).
- **kbf, the late genius.** Best Brier in the stable (0.0923 at decision time) and −$0.38/$1 at real costs: its skill lives at T−3 minutes, after anything can be bought. Accuracy without decision-time monetizability is not economic skill (results/diagnosis.json, results/program.json).
- **M1 calibration harms everyone.** The Platt layer's prequential log-loss is *worse* than raw for all nine arms (+0.034 to +0.164, kb7 worst), and the fitted direction flipped within a day — the miscalibration drifts faster than the fit. Redesigned with a 50-window memory and kept shadow-only, as a drift instrument, never a decision input (docs/DECISIONS.md).
- **kb6.** UP-recall of 63% concealed 37% coverage and a calibration slope of 0.33 ≈ noise; recall alone is not skill (docs/DECISIONS.md, results/program.json).
- **The SPRT variance collapse and the 2.1¢ pricing artifact.** Two near-miss false promotions in one day. The second: challengers were priced from modeled quotes while the champion paid real asks — a systematic 2.1¢ subsidy that manufactured +23.5% for M3, +10.6% for M9, +8.0% for M2; after repricing everything from the same decision-time quote, those became +1.2%, −5.0%, −8.6% (NOTES.md).

### 4.7 Traders as ablations

The eight paper traders are a sizing/execution ablation grid over the same signals (results/audit_report.json). The Gambler (pt4) is the aggressive-curriculum exhibit — its v3 ruling keeps the 33% stake but sweeps every dollar above the $10k start to a withdrawal ledger that can never be re-staked, bounding exposure forever (docs/DECISIONS.md). The instructive contrast is pt6, the "MLE" trader (supervised edge model, half-Kelly): +14.8¢ per dollar staked over 26 settled windows — the only green trader on the desk — while sitting idle 84% of the time (results/audit_report.json, results/diagnosis.json). Its edge-model regression even scores its own stated edge *negatively* (conf−ask weight −0.059, echoed by kb5's claimed-edge weight −0.096): when the model most believes it knows better than the price, it is most often wrong — the "edge anti-signal" now queued as a candidate M13 edge-band redesign (docs/DECISIONS.md). On a desk where no arm beats the book, the only monetizable skill found so far is *selection*: knowing when not to trade.

## 5. Incidents and integrity

The defining incident was a SEV-0: the initial root-cause analysis of the desk's losses ("the price arms point the wrong way") was itself falsified by the audit. The corrected RCA: the tier-1 defect is **dispersion, not direction** — drift arms never trade, the feeding path carried a +$1.76 bias, and 80% bands covered only 74–76% (docs/DECISIONS.md). Under-dispersion at T1 poisons the z-bridge σ and every probability downstream (docs/SEV0_REMEDIATION.md). Per the blueprint's rule, the invalidated number is preserved beside the corrected one: the +11.75% "champion" figure stands crossed out next to the true −6.5% (docs/OS_BLUEPRINT.md).

Four recurring bug classes emerged (docs/MANUAL.md, NOTES.md): the **stale quote** (treatment rows evaluated on quotes older than the tradeable envelope — fixed with a ≤12-minute filter, verified when the post-fix champion measured −6.04% against an independently predicted −6.5%); **fee rounding** (per-contract ceil applied per-order accrued $2,355 of phantom fees); the **field collision** (a shadow calibration output written into a field a live reader consumed — p_cal renamed to p_m1, 390 rows scrubbed); and the **trader lockout** (an entry guard that silently prevented four traders from finding their own entry minute).

Each closed incident must yield a fixture, an invariant, or a museum exhibit (docs/OS_BLUEPRINT.md). The invariant suite currently machine-checks nine promises — one-decision-per-window, ledger monotonicity, no duplicate treatment windows, fee-formula parity, the Gambler-v3 sweep, decision-quote age, skip-is-not-missing, frozen-controls-untouched, withdrawals-never-restaked — every one named for the scar that created it, all passing (results/invariants.json). The skip-is-not-missing rule deserves emphasis as an evaluation-methods point: a policy's deliberate stand-down is a complete scored observation with EV 0 by pre-registration, not a missing pair; the first board draft conflated the two and wrongly invalidated all twelve challengers (docs/DECISIONS.md).

## 6. Limitations

**Pricing basis.** Most treatment EVs are computed at model-basis quotes, not real fills; only the paired difference is apples-to-apples, and cross-basis comparisons are refused rather than estimated (NOTES.md, results/program.json). **Liquidity realism.** The venue is a demo-liquidity market; fill and depth behavior may not transfer. **Power ceiling.** The instrument's clock is ~96 windows/day; at family-wise α the desk needs hundreds of windows for a verdict on a 2¢ effect (projected ~590 for M8 at current drift; NOTES.md). **Effective n.** Windows are the counting unit, but adjacent windows share regime; the window-clustering correction to effective n is acknowledged, not solved (docs/MANUAL.md). **Single instrument.** Everything above is one contract family on one underlying; the multi-ticker generalization is future work, and the deep-RL variants of the trader-policy program are explicitly deferred until that scale supplies the data appetite and a counterfactual audit path (docs/DECISIONS.md).

## 7. Conclusion and future work

The honest summary of the prediction program is a well-measured null: against an observable market consensus, at 15-minute horizon and real costs, nine model families extracted no skill the price did not already contain — and five foundation-model upgrades could not move that needle. The honest summary of the *instrument* program is positive: the system now detects its own artifacts faster than it generates them, it has converted every incident into a machine-checked promise, and its two best-evidenced treatments (M8, M10+M8) earn exactly where their mechanisms claim while remaining unpromoted until the sequential boundary is crossed.

Future work follows the decision inbox (results/program.json, docs/DECISIONS.md): an **M13 edge-band gate** (a maximum as well as a minimum on claimed edge, motivated by the replicated edge anti-signal) and learned limit pricing from the fill curve; **post-fill markout capture** at 1m/5m to decompose the execution leak; **multi-ticker seams** to raise the window budget an order of magnitude; and, only at that scale, reinforcement-learned trader policies with a row-by-row counterfactual audit.

## References

- Arnott, R., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the era of machine learning. *Journal of Financial Data Science*.
- Ansari, A. F., et al. (2024). Chronos: Learning the language of time series. arXiv:2403.07815.
- Das, A., et al. (2024). A decoder-only foundation model for time-series forecasting (TimesFM). *ICML*.
- Dawid, A. P. (1984). Present position and potential developments: The prequential approach. *JRSS A*.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*.
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*.
- Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *JFE*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Herbster, M., & Warmuth, M. K. (1998). Tracking the best expert (Fixed-Share). *Machine Learning*.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *IJF*.
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*.
- Lei, J., et al. (2018). Distribution-free predictive inference for regression. *JASA*.
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation (LinUCB). *WWW*.
- Platt, J. (1999). Probabilistic outputs for support vector machines. *Advances in Large Margin Classifiers*.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*.

*Full per-claim source list with URLs: docs/RESEARCH_BASELINE.md and docs/SEV0_REMEDIATION.md §Source list.*
