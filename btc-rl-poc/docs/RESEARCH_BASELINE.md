# Research Baseline — industry-standard evaluation, external landscape, our gaps

Date: 2026-08-28. Method: every external claim below was retrieved this
session via web search/fetch and is cited with title + URL. Internal
"our status" claims are grounded in repo files:
`docs/SEV0_REMEDIATION.md`, `NOTES.md`, `scripts/run_audit.py`,
`btc_rl/treatments.py`. Where a claim could not be verified it is
marked **not verified** or omitted. Format per practice:
**standard / source / our status** (does / partial / lacks).

---

## Tier 1 — price forecasting (point + interval)

**1.1 Scale-free point accuracy: MASE as the cross-series standard.**
- Standard: MASE (error scaled by the in-sample naive forecast) was
  proposed as *the* standard measure for comparing forecast accuracy
  across series because percentage measures degenerate in common
  situations.
- Source: Hyndman & Koehler, "Another look at measures of forecast
  accuracy," IJF 22(4) 679–688, 2006 —
  https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/
- Our status: **does (partial headline).** MASE/MSSE are computed
  alongside MAE/MSE (`NOTES.md` 2026-08-28 metric-switch entry;
  additive keys in metrics.json), but the headline and the retrain
  accept/reject gate are raw MSE (`run_audit.py` tier1 reports MSE,
  bias, band coverage — no scaled or naive-relative number).

**1.2 Competition protocol: OWA = avg(MASE, sMAPE) in M4; live,
pre-registered evaluation.**
- Standard: M4 evaluated 100k series under a fixed protocol with OWA
  (average of relative MASE and relative sMAPE against a naive
  benchmark).
- Source: "A combination-based forecasting method for the
  M4-competition," IJF —
  https://www.sciencedirect.com/science/article/abs/pii/S0169207019301542
- Our status: **partial.** We have the protocol discipline
  (pre-registration, frozen rules, dated cutovers — SEV0_REMEDIATION
  "Evaluation discipline") but no benchmark-relative headline (no OWA
  analogue vs a naive/random-walk floor in the audit).

**1.3 Interval/distributional accuracy: pinball (quantile) loss is the
competition standard.**
- Standard: M5 Uncertainty scored quantile forecasts with the
  (Weighted) Scaled Pinball Loss; GEFCom2014 scored all four energy
  tracks with pinball loss averaged over 99 quantiles.
- Sources: "The M5 uncertainty competition: Results, findings and
  conclusions" —
  https://www.sciencedirect.com/science/article/pii/S0169207021001722 ;
  "Evaluating quantile forecasts in the M5 uncertainty competition" —
  https://www.sciencedirect.com/science/article/abs/pii/S0169207022000449 ;
  "GEFCom2014 probabilistic electric load forecasting" —
  https://www.sciencedirect.com/science/article/abs/pii/S0169207015001405 ;
  "Lasso estimation for GEFCom2014 probabilistic electric load
  forecasting" — https://arxiv.org/pdf/1603.01376
- Our status: **lacks.** Bands are scored by 80% coverage only
  (`run_audit.py` band80_cov). Coverage alone does not penalize width;
  pinball/interval score would let a wide-but-covering band lose to a
  sharp one.

**1.4 CRPS + calibration/sharpness + PIT diagnostics.**
- Standard: proper scoring rules assess calibration and sharpness
  simultaneously; CRPS "might serve as a standard score in evaluating
  probabilistic forecasts of real-valued variables"; the goal is to
  maximize sharpness subject to calibration, with PIT histograms as
  the calibration diagnostic.
- Sources: Gneiting & Raftery, "Strictly Proper Scoring Rules,
  Prediction, and Estimation," JASA 102(477), 2007 —
  https://www.tandfonline.com/doi/abs/10.1198/016214506000001437
  (full text: https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) ;
  "A review of predictive uncertainty estimation with machine
  learning" — https://arxiv.org/pdf/2209.08307
- Our status: **partial.** Conformal, sigma-conditioned bands shipped
  (normalized conformal, NOTES 2026-08-28 fix #6) and coverage is
  tracked at one level (80%). No CRPS, no multi-level quantiles, no
  PIT histogram — miscalibration at other quantile levels is
  invisible.

**1.5 M6: forecast accuracy does not automatically convert to
investment performance.**
- Standard: M6 evaluated forecasting and investment decisions jointly,
  live, on 100 assets, explicitly to test the EMH link between
  accuracy and returns; participants struggled to forecast relative
  performance.
- Source: Makridakis et al., "The M6 forecasting competition: Bridging
  the gap between forecasting and investment decisions" —
  https://arxiv.org/abs/2310.13357
- Our status: **does.** The stack's core finding is the same shape:
  tier-1 direction has no edge (47–53%) while trading EV lives in
  tier-2/3 selection and sizing (SEV0_REMEDIATION Tier 1; NOTES
  kb5 "most-wrong caller is most profitable"). We evaluate tiers
  separately and money at the desk, which is the M6-consistent design.

---

## Tier 2 — probability forecasting / kb arms

**2.1 Brier score with Murphy decomposition
(reliability − resolution + uncertainty).**
- Standard: the decomposition of the Brier score into reliability,
  resolution and uncertainty is "a standard method in forecast
  verification" (Murphy 1973 lineage).
- Source: Siegert, "Simplifying and generalising Murphy's Brier score
  decomposition," QJRMS —
  https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.2985
- Our status: **partial.** Per-arm Brier and market Brier are in the
  audit (`run_audit.py` tier2_section); the decomposition is specced
  as the M1 acceptance instrument (SEV0_REMEDIATION Tier 2) but not
  computed in the automated audit — so we can't see whether a Brier
  move is honesty (reliability) or skill (resolution).

**2.2 Reliability diagrams — use CORP, not ad-hoc binning.**
- Standard: classical binned reliability diagrams are unstable under
  arbitrary binning choices; CORP (isotonic/PAV-based) diagrams are
  optimally binned, reproducible, statistically consistent, and come
  with a miscalibration measure from a revisited score decomposition.
- Sources: Dimitriadis, Gneiting & Jordan, "Stable reliability
  diagrams for probabilistic classifiers," PNAS 2021 —
  https://www.pnas.org/doi/10.1073/pnas.2016191118 ; companion —
  https://arxiv.org/abs/2008.03033
- Our status: **lacks.** No reliability diagrams anywhere in the audit
  pipeline; calibration is summarized only by Platt (a, b) and pBias.

**2.3 ECE — widely used, widely criticized; do not adopt it.**
- Standard (negative result): ECE is sensitive to bin count/edges,
  suffers over/under-confidence cancellation within bins, and can be
  gamed — a constant base-rate forecast achieves perfect ECE.
- Sources: "A comprehensive review of classifier probability
  calibration metrics" — https://arxiv.org/pdf/2504.18278 ;
  "Evaluating Probabilistic Classifiers: The Triptych" —
  https://arxiv.org/pdf/2301.10803 ; "Beyond ECE" —
  https://arxiv.org/html/2605.01796
- Our status: **does (by omission, correctly).** We use Platt
  parameters, pBias, Brier and log-loss rather than ECE — consistent
  with the criticism literature. Keep it that way; if a single
  miscalibration number is wanted, CORP's decomposition-based measure
  is the defensible one (source above).

**2.4 Strictly proper scoring for probability forecasts (Brier / log
loss); prices as probabilities.**
- Standard: proper scoring rules make honest probability reporting
  optimal (Gneiting & Raftery 2007, cited in 1.4). For prediction
  markets, Wolfers & Zitzewitz give conditions under which binary
  contract prices correspond to mean trader beliefs — the basis for
  treating mkt_p_up as a probability benchmark.
- Sources: Wolfers & Zitzewitz, "Interpreting Prediction Market Prices
  as Probabilities," NBER w12200 —
  https://www.nber.org/papers/w12200 ; "Prediction Markets," JEP 2004
  — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=552109
- Our status: **does.** Brier + prequential log-loss are the arm
  metrics; market Brier is reported beside every arm; market-paired
  scoring (win − q) is pre-registered (NOTES Method A). Missing piece:
  no explicit Brier *skill score* vs the market as reference forecast
  — the single number "does the arm beat the market" is derivable but
  not surfaced.

**2.5 Prediction-market calibration is domain- and horizon-dependent.**
- Finding: a study of 353M trades / 429k binary contracts on Kalshi
  and Polymarket finds calibration varies by domain, time-to-
  resolution and trade size (e.g., persistent underconfidence in
  political markets); real-time Kalshi sports prices are well
  calibrated and sharpen toward resolution.
- Sources: "Decomposing Crowd Wisdom: Domain-Specific Calibration
  Dynamics in Prediction Markets" — https://arxiv.org/abs/2602.19520 ;
  "When Do Markets Fully Process Public Information? Evidence from
  Real-Time Prediction Markets" — https://arxiv.org/pdf/2606.07811
- Our status: **partial.** We measure decision-time market accuracy
  and gate on it (M8 regime gate, NOTES 2026-08-28), which is a
  horizon-dependent-calibration response; we do not measure the
  market's own calibration curve by minutes-to-close, which the
  literature says is where its edge concentrates (and matches our
  measured 0.85-tail adverse selection, SEV0_REMEDIATION Tier 3).

---

## Tier 3/4 — trading & execution evaluation

**3.1 Implementation shortfall vs arrival price.**
- Standard: implementation shortfall (Perold, 1988) measures total
  cost vs the paper portfolio at the decision price; arrival price
  (mid at order placement) is the standard benchmark; components are
  delay, impact, and opportunity cost of unfilled orders.
- Source: "Implementation Shortfall: Perold Framework and Transaction
  Cost Analysis" —
  https://ryanoconnellfinance.com/implementation-shortfall/ (secondary;
  primary Perold 1988 J. Portfolio Mgmt not fetched — **not verified
  beyond secondary sources**)
- Our status: **partial.** The 2.1c pricing-bug fix made every policy
  price from the same decision-time quote (NOTES 2026-08-28 M3 entry)
  — that is an arrival-price convention — and the 18:33 forensics
  measured edge at the actual fill. But shortfall is not a first-class
  per-trade metric: fills vs decision-time mid, delay cost, and
  unfilled-skip opportunity cost (pt7's skips) are not aggregated
  anywhere in `run_audit.py`.

**3.2 Optimal execution / participation discipline.**
- Standard: Almgren & Chriss, "Optimal Execution of Portfolio
  Transactions," J. Risk 3(2) 5–39 (2000): trade-off of impact cost vs
  timing risk; participation-capped algos (POV/VWAP/TWAP) are the
  practical descendants.
- Source: https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf ;
  overview https://en.wikipedia.org/wiki/Almgren%E2%80%93Chriss_model
- Our status: **does.** pt8 carries a 25% near-touch participation cap
  from the detection sim; the maker-vs-taker trade-off was measured
  both ways (maker −10.6c without speed; late entries +9.1c)
  (SEV0_REMEDIATION "Ideal trader"; NOTES capacity entry).

**3.3 Multiple testing: deflated Sharpe ratio / probability of
backtest overfitting.**
- Standard: when the reported winner was selected from many trials,
  its performance statistic must be deflated for selection bias and
  non-normality (DSR); the probability of selecting an overfit
  strategy grows rapidly with the number of trials (PBO).
- Sources: Bailey & López de Prado, "The Deflated Sharpe Ratio," SSRN
  — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 ;
  overview https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio ;
  "Statistical Overfitting and Backtest Performance" —
  https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf
- Our status: **lacks.** The treatment board runs ~10 concurrent
  SPRTs each at α=0.05 (`treatments.py`); nothing corrects the
  *family* of tests for selection — at 10 challengers the expected
  count of false "promote" verdicts is materially nonzero, and we
  promote the argmax. This is exactly the DSR/PBO failure mode, and we
  have already logged two near-miss false promotions from other causes
  (variance collapse; 2.1c pricing artifact — NOTES 2026-08-28).

**3.4 Leakage-safe validation: walk-forward and purged/embargoed CV.**
- Standard: financial labels overlap in time, so shuffled k-fold
  leaks; purging + embargo (and combinatorial purged CV for multiple
  paths) is the financial-ML standard; walk-forward remains the
  industry default for realistic simulation.
- Sources: https://en.wikipedia.org/wiki/Purged_cross-validation ;
  López de Prado, "The 10 Reasons Most Machine Learning Funds Fail" —
  https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf ;
  scikit-compatible implementation & comparison —
  https://github.com/eslazarev/purged-cross-validation
- Our status: **does (by a stronger route).** Primary evaluation is
  prequential/live (Dawid-style test-then-train, SEV0_REMEDIATION
  "Evaluation discipline"); replays recompute signals with explicit
  no-look-ahead (NOTES Method B; kb9 fine-tune trained strictly before
  eval cutoff); t_cal scores from pre-settle stamps (fix #3). No
  shuffled CV exists to purge. Residual risk: offline gauntlets
  (kb8/kb9) are single-path walk-forwards — CPCV-style multiple paths
  would strengthen those specific verdicts.

**3.5 Risk-adjusted reporting: drawdown and return per dollar.**
- Standard: TCA + capacity-aware sizing and drawdown discipline are
  the practitioner norm (see 3.1–3.3 sources; Kelly-fraction sizing
  under estimation error per SEV0_REMEDIATION's MacLean–Thorp–Ziemba
  citation — internal citation, not re-verified this session).
- Our status: **does.** `run_audit.py` computes EV/$1, per-day win% vs
  the break-even actually paid, and max drawdown of the bankroll
  curve; the trader scoreboard ranks by net÷DD (NOTES TA-feedback
  entry). No Sharpe-type ratio is reported — at ~4 days of live desk
  data any Sharpe estimate would be noise, and DSR machinery (3.3)
  matters more than the ratio itself.

---

## Tier 5 — online experimentation

**4.1 Sequential testing valid under continuous monitoring.**
- Standard: fixed-horizon p-values are invalid when experimenters peek
  continuously; the mixture SPRT (mSPRT) yields always-valid p-values
  and confidence intervals (deployed at scale in Optimizely's Stats
  Engine since 2015).
- Sources: Johari, Koomen, Pekelis & Walsh, "Peeking at A/B Tests: Why
  It Matters, and What to Do About It," KDD 2017 —
  http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p1517.pdf ; "Always
  Valid Inference: Continuous Monitoring of A/B Tests," Operations
  Research 2021 — https://arxiv.org/abs/1512.04922
- Our status: **partial.** We use Wald SPRT precisely because it is
  valid under continuous monitoring (`treatments.py` header), with a
  pre-registered effect size, variance floor, warmup, and min_n. Two
  honest deviations from the standard: (a) the normal plug-in with
  Welford variance is an approximation, stated in code but with no
  quantified error control — mSPRT is the literature's answer to
  unknown variance; (b) no always-valid confidence intervals, only a
  promote/reject boundary.

**4.2 Guardrails, ramp-up, and auto-shutdown.**
- Standard: launch practice at scale (Google/LinkedIn/Microsoft, >20k
  experiments/yr) starts at low traffic, monitors guardrail metrics
  that must NOT move, auto-shuts-down on guardrail breach, then ramps.
- Sources: Kohavi, Tang & Xu, *Trustworthy Online Controlled
  Experiments* —
  https://books.google.com/books/about/Trustworthy_Online_Controlled_Experiment.html?id=TFjPDwAAQBAJ ;
  Kohavi & Longbotham, "Online Controlled Experiments and A/B Tests"
  — https://exp-platform.com/Documents/2023-03-11EncyclopeiaMLDSABTestingFinal.pdf
- Our status: **partial.** We have the shadow-first rule ("every
  behavior change ships in shadow mode first," SEV0_REMEDIATION),
  reversible stamped promotion with the loser kept running
  (`treatments.py`), and paper-only capital — a strong ramp analogue.
  We lack named guardrail metrics with automatic demotion (e.g., max
  drawdown or skip-rate bounds that auto-revert a promoted treatment).

**4.3 Paired/variance-reduced comparisons.**
- Standard: variance reduction and correct units of analysis are core
  trustworthy-experimentation practice (same sources as 4.2).
- Our status: **does.** Paired same-window scoring so regime cancels;
  effective n = windows, not rows; market-paired excess score with
  day-clustered t (NOTES Method A; `treatments.py` design notes).

---

## Who else works these problems (public record only)

- **Prediction-market research (academic):** Wolfers & Zitzewitz
  (prices-as-probabilities foundations, 4 sources above); large-scale
  Kalshi/Polymarket calibration decomposition
  (https://arxiv.org/abs/2602.19520); real-time information
  processing on Kalshi sports markets
  (https://arxiv.org/pdf/2606.07811); "Information Efficiency Across
  Macroeconomic Prediction Markets: Evidence from Kalshi"
  (https://www.researchgate.net/publication/409472804_Information_Efficiency_Across_Macroeconomic_Prediction_Markets_Evidence_from_Kalshi);
  systematic longshot-style bias in sports prediction markets
  (https://arxiv.org/html/2607.14430); LLM-agent trading/latency
  arbitrage on Polymarket (https://arxiv.org/html/2604.03888v1).
  Kalshi+Polymarket passed ~$150B lifetime volume by April 2026 per
  the retrieved arXiv survey material — an active academic testbed.
- **Short-horizon crypto forecasting (academic):** Jaquart, Dann &
  Weinhardt, "Short-term bitcoin market prediction via machine
  learning" (1–60 min horizons; RNNs/GBMs, technical features
  dominate) —
  https://www.sciencedirect.com/science/article/pii/S2405918821000027 ;
  comparative high-frequency crypto ML studies report horizon-dependent
  predictability, with models struggling to beat a random walk at
  coarser (hourly) frequencies —
  https://www.mdpi.com/2078-2489/16/4/300 and
  https://www.preprints.org/manuscript/202503.0261 . This matches our
  own five-attempt "model axis is exhausted" negative result (NOTES
  2026-08-25 kb9 round 2).
- **Quant firms' PUBLISHED engineering:** Hudson River Trading's tech
  blog covers trading research and engineering practice, including a
  post arguing that "In Trading, Machine Learning Benchmarks Don't
  Track What You Care About" — i.e., money-metric evaluation over
  generic ML metrics, the same lesson as our desk-first scoreboard —
  https://www.hudsonrivertrading.com/hrtbeat/ . Man Group open-sourced
  ArcticDB, the DataFrame database built for its front-office research
  data volumes (Bloomberg BQuant integration) —
  https://www.man.com/man-group-brings-powerful-dataframe-database-product-arcticdb-to-market-with-bloomberg .
- **Private internals:** the actual signals, sizing, and evaluation
  stacks of Jane Street, HRT, Citadel, Jump, etc. are **not publicly
  verifiable**; nothing here speculates about them. Only the published
  artifacts above are claimed.

---

## Prioritized gap list (top 10, by expected value to THIS system)

1. **Family-wise error control on the treatment board.** ~10
   concurrent SPRTs at α=0.05 each with argmax promotion is the
   selection-bias setup DSR/PBO exists to correct (Bailey & López de
   Prado). Cheapest fix: Bonferroni-style α per active treatment, or
   report the expected false-promotion count next to the board. Two
   near-miss false promotions in one day say this is our #1 live risk.
2. **Pinball/interval score for tier-1 bands.** Coverage-only scoring
   (band80_cov) can't distinguish sharp honest bands from wide lazy
   ones; M5/GEFCom standardize pinball. One-line addition to
   `run_audit.py` tier1; directly strengthens the M4′ band-fix gate.
3. **Brier decomposition + market skill score in the automated audit.**
   Murphy decomposition (reliability vs resolution) plus
   BSS = 1 − Brier_arm/Brier_market per arm would have separated the
   M1 shadow verdict ("helps only kb7") into honesty-vs-skill terms
   and gives the one number that answers "does any arm beat the
   market."
4. **CORP reliability diagrams for kb arms and for the market itself
   by minutes-to-close.** Stable, binning-free (PNAS 2021); the
   market's calibration-by-horizon curve is where the literature says
   the adverse-selection tail lives — we measured that tail (0.85-tail
   50%) but never plotted the curve.
5. **Guardrail auto-demotion for promoted treatments.** Promotion is
   reversible and stamped, but nothing *triggers* reversal. Name 2–3
   guardrails (paired EV, max drawdown, skip-rate band) with an
   automatic revert rule — the Kohavi auto-shutdown pattern applied to
   our champion/challenger router.
6. **Implementation-shortfall accounting per trader.** We already
   stamp decision-time quotes; aggregate slippage = fill − decision
   mid, delay cost, and pt7/pt8 unfilled-opportunity cost into the
   audit so execution quality (maker vs taker cohort) is scored by the
   standard TCA decomposition, not just end EV.
7. **Naive-benchmark-relative headline for tier 1.** MSE alone is not
   comparable across regimes; the crypto literature's recurring result
   is "hard to beat random walk." Surface MASE/MSSE (already computed)
   or an OWA-style ratio in `run_audit.py` tier1 as the headline.
8. **PIT / multi-level coverage.** One coverage level (80%) hides
   quantile-shape errors that produced the tier-2 slope<1 defect
   (dispersion RCA, NOTES 2026-08-28). Track 50/80/95 coverage or a
   PIT histogram per family.
9. **mSPRT or documented error bounds for the plug-in SPRT.** The
   normal-plug-in approximation with a variance floor is pragmatic but
   uncharacterized; mixture SPRT gives always-valid p-values with
   unknown variance and would replace two ad-hoc guards (floor,
   warmup) with theory.
10. **Maker-fill markout metrics for pt7/pt8.** The adverse-selection
    experiment currently scores settled EV only; standard practice
    also measures post-fill markouts (price drift after fill) —
    directly testable from ticks.jsonl and would separate "filled
    because signal fell" from benign fills ahead of settlement.

---

## Source count

External sources retrieved and cited this session: **39 distinct
works (41 URLs, counting two same-paper mirrors)** — M-competitions
and forecast-accuracy 6, energy/quantile 2, scoring & calibration 7,
prediction markets 8, execution/TCA 3, backtest-overfitting & CV 6,
online experimentation 4, crypto short-horizon ML 3, quant engineering
blogs 2. Internal grounding: `docs/SEV0_REMEDIATION.md`, `NOTES.md`,
`scripts/run_audit.py`, `btc_rl/treatments.py`.
