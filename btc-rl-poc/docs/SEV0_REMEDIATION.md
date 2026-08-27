# SEV-0 Remediation Spec — per-layer changes, grounded in the literature

Status: living document. Every claim about *our* system is measured by
`tests/sev0_error_audit.py` (re-runnable); every technique carries its
source. For each layer: the measured defect, what the industry standard
is, what the state of the art is, their shortcomings/advantages at our
scale (~200 windows, 15-min horizon), and an explicit **use / don't use**
verdict with an acceptance metric.

The audit that motivates all of this: bias is born in Tier 1 (drift
priors: t10 +$45 bias @h30, up-call share 71–83% vs 53% base rate),
inherited by Tier 2 (all Platt intercepts negative, kb7 a=−0.565),
amplified by Tier 3 (cross-arm confidence AUC 0.503; leader churn
16/day; fixed-kb4 counterfactual +4.7%/$1 vs desk +2.0%), multiplied by
Tier 4 sizing, with 95% of loss dollars carrying a shared taggable
cause (herd whipsaw 75.2%, knife-edge 52.8%).

---

## Tier 0 — Data feed

**Measured state.** Healthy: median inter-tick gap 0.07 s, p99 0.67 s;
5-min prediction cadence, 1 stall. Flag: taker-buy share 75.3% while
windows settle UP only 45.5% — an absorption regime invisible to
the upper tiers because no order-flow feature crosses the kb boundary.

**Industry standard.** Trade-flow imbalance / signed volume as a
short-horizon predictor is standard in microstructure — order flow
carries information (Kyle 1985; Glosten & Milgrom 1985); order-flow
imbalance predicts short-horizon returns (Cont, Kukanov & Stoikov
2014, *J. Financial Econometrics*).

**State of the art.** Deep LOB models (order-book snapshots → CNN/LSTM,
e.g. DeepLOB, Zhang, Zohren & Roberts 2019, *IEEE TSP*). Shortcoming
for us: needs millions of labeled book states, and Kalshi's 15-min
binary is one settle per window — we'd overfit instantly.

**Change spec (M-T0).** Add two features at the kb boundary, computed
from ticks we already store: rolling taker-buy imbalance and its
divergence from price drift. **Use** the simple imbalance (Cont et al.
2014 form), **don't use** deep LOB models (sample-starved).
*Acceptance:* the new features earn non-zero weight in kb3's online
logit under prequential evaluation (Dawid 1984) and don't degrade
window-level Brier over 100 windows.

---

## Tier 1 — RL price arms (the origin of the bias)

**Measured defect.** Directional accuracy of every price arm 47–53% at
every horizon (no edge — value is MAE/bands only); bullish drift
priors: t10 +$45 bias @h30 with 80% up-calls, t2 +$54 @h30 with 83%
up-calls; 80% bands under-cover at h30 (65–73% vs target 80%).

**Why it happens.** Non-stationarity / concept drift: models fit in an
up-trending window carry the trend as a prior (survey: Gama et al.
2014, *ACM Computing Surveys*). Our arms have no drift-adaptation
mechanism except `cal` (trailing recentering).

**Industry standard.**
- *Rolling-window retraining / recentering* — simple, robust, the
  default in production forecasting. Advantage: cheap, transparent.
  Shortcoming: reacts after the fact, window length is a hyperparameter.
- *Explicit drift detection* — Bayesian Online Changepoint Detection
  (Adams & MacKay 2007) or DDM-style detectors (Gama et al. 2004,
  SBIA). Advantage: principled reset signal. Shortcoming: at 15-min
  windows we have few effective regime samples; false alarms churn.

**State of the art.**
- *Time-series foundation models*, zero-shot: Chronos (Ansari et al.
  2024, arXiv:2403.07815) and TimesFM (Das et al. 2024, ICML) — we
  already run both (kb7, kb9). Advantage: no local training to go
  stale. Shortcoming: they extrapolate momentum — kb7 is our *most*
  miscalibrated arm (a=−0.565) — and are frozen, so they can't learn
  the current regime at all.
- *Conformal prediction* for honest bands: split/online conformal
  (Angelopoulos & Bates 2023, *FnTML*) and Conformalized Quantile
  Regression (Romano, Patterson & Candès 2019, NeurIPS); under drift,
  Adaptive Conformal Inference (Gibbs & Candès 2021, NeurIPS).
  Advantage: finite-sample coverage guarantees, distribution-free.
  Shortcoming: none serious at our scale — this is a fit.

**Change spec (M4, M-T1b).**
1. **Use** drift-neutral recentering for the biased arms (t10, t2@h30,
   t11@h15, t6): subtract each arm's trailing signed bias (the `cal`
   mechanism, applied per-arm) — this is rolling-origin recalibration,
   the industry default. *Acceptance:* |bias| < $10 at h30 over a
   200-cycle trailing window; up-call share within ±5 pts of realized.
2. **Use** online adaptive conformal wrappers (Gibbs & Candès 2021)
   for the 80% bands instead of Gaussian σ scaling. *Acceptance:*
   trailing 200-cycle coverage in [77%, 83%] at every horizon.
3. **Don't use** BOCPD-triggered full retrains (regime sample count too
   small; alarm churn would thrash the arms) — revisit at 1,000+
   windows.
4. **Don't use** end-to-end deep RL retraining for direction (Moody &
   Saffell 2001, *IEEE Trans. NN*, is the classic direct-RL trader;
   modern surveys agree data appetite is orders beyond ours). The
   audit shows direction has no edge at tier 1 — direction belongs to
   tier 2 where the market anchor lives.

---

## Tier 2 — kb probability arms (the inheritance)

**Measured defect.** Ranking is real (AUC .705–.754 ex-kb6) but honesty
is not: every arm's recalibration intercept is negative (kb7 −0.565,
kb9 −0.382, kb5 −0.409); pBias +0.03..+0.11; kb6 slope 0.33
(confidence ≈ noise — retired); kb3 slope 0.66 (mushy).

**Industry standard: post-hoc calibration.**
- *Platt scaling* (Platt 1999) — logistic map on the score. Advantage:
  2 parameters, works at n≈150, online-updatable. Shortcoming:
  parametric (sigmoid shape assumed).
- *Isotonic regression* (Zadrozny & Elkan 2002, KDD) — nonparametric.
  Advantage: shape-free. Shortcoming: needs ~1,000+ samples, overfits
  at our n; step-function artifacts.
- *Beta calibration* (Kull, Silva Filho & Flach 2017, AISTATS) —
  3-parameter, better behaved than Platt near 0/1. Reasonable
  alternative; marginal gain at our n.
- Context: modern learned probabilities are systematically
  miscalibrated out of the box (Guo et al. 2017, ICML) — our finding
  is the textbook case, not an anomaly.

**State of the art.** Online/adversarial calibration with guarantees
(calibeating, Foster & Hart 2021+; conformal predictive distributions).
Advantage: worst-case guarantees. Shortcoming: machinery is heavy and
the guarantees are asymptotic — overkill versus prequential Platt at
our scale.

**Change spec (M1 — the highest-leverage fix).**
**Use** per-arm prequential Platt scaling: maintain (a_v, b_v) per arm,
refit each settle on the trailing window (or SGD-update), and expose
`p_cal = sigmoid(a_v + b_v·logit(p_up))` alongside the raw p_up.
Additive: new field, no existing treatment altered; consumers (tier 3)
switch to p_cal. Evaluate prequentially (Dawid 1984) with Brier
decomposition (Murphy 1973) — reliability should shrink while
resolution is preserved; scoring stays strictly proper (Gneiting &
Raftery 2007). **Don't use** isotonic (n too small), **don't use**
temperature-only scaling (fixes slope, not our dominant intercept
error). *Acceptance:* per-arm |pBias| < 0.02 and refit intercept
|a| < 0.1 on trailing 100 windows; cross-arm confidence AUC (tier 3)
recovers from 0.503 to > 0.65.

---

## Tier 3 — decision layer (the amplifier)

**Measured defects.** (1) Leader churn 16 switches/day; fixed-kb4
counterfactual +4.7%/$1 vs churner's +2.0%. (2) Cross-arm raw-p_up
comparison → mixed confidence AUC 0.503. (3) Loss concentration:
knife-edge windows 52.8% of loss dollars; against-market calls 62%
wrong; 09h PT 64% wrong. (4) The ≥0.85 stated-confidence tail wins
only 50% — the adverse-selection tail: the market sells cheapest
exactly what it knows best (Glosten & Milgrom 1985).

**Industry standard.**
- *Prediction with expert advice*: exponential weights / Hedge
  (Cesa-Bianchi & Lugosi 2006, *Prediction, Learning, and Games*).
  Advantage: regret bounds vs best expert; no hard switching.
  Shortcoming: static best-expert benchmark.
- *Fixed-Share* for switching regimes (Herbster & Warmuth 1998,
  *Machine Learning*): tracks the best expert *sequence*; the
  principled version of "sticky leader." Advantage: exactly our
  problem (leaders drift); one share parameter. Shortcoming: needs
  calibrated inputs first — garbage in, garbage weighted.
- *Contextual bandits* (LinUCB — Li et al. 2010, WWW; Thompson
  sampling — Russo et al. 2018, *FnTML*) for gate/arm selection.
  Shortcoming here: we observe every arm's outcome every window
  (full-information, not bandit feedback), so bandit machinery wastes
  information — full-information experts is the right frame.

**State of the art.** Meta-labeling (López de Prado 2018, *Advances in
Financial Machine Learning*): a secondary model that decides
bet/don't-bet and size on top of primary signals — this is exactly what
pt6 is; the margin gate + shadow-row training we shipped is
meta-labeling with off-policy labels.

**Change spec.**
1. **M3 — use Fixed-Share (Herbster & Warmuth 1998)** over calibrated
   arms as a NEW desk treatment (additive; the current Follower stays
   as control): weights w_v updated multiplicatively on log-loss of
   p_cal, small share rate α to allow leader drift. *Acceptance:*
   counterfactual regret vs best fixed arm < 1%/$1 over 100 windows
   (churner's measured gap today: 2.7%/$1).
2. **M2 — use the knife-edge veto**: no desk entry when
   |mkt_p_up − 0.5| < 0.10. Justification: 41.6% wrong rate there,
   52.8% of loss dollars; and theory — near 50/50 the market's
   information advantage is maximal relative to ours
   (Glosten & Milgrom 1985). *Acceptance:* desk EV/$1 improves with
   ≥60% of prior coverage retained.
3. **M6 — asymmetric gate while pBias > 0**: UP entries need
   p_cal ≥ τ+δ (δ≈0.03) vs DOWN at τ. Direct response to FP 33% vs
   FN 24%. Sunset clause: δ→0 when trailing pBias < 0.02.
4. **M7 — hour policy**: stand down 09:00–10:00 PT (64% wrong, market
   itself wrong at 70%+ prices). Revisit monthly — hour effects are
   regime-loaded and we will not hardcode a permanent superstition.
5. **Don't use** against-market overrides anywhere (62% loss rate,
   n=50): arms may only *time* agreement with the market, never
   dispute it, until an arm demonstrates calibrated contrarian skill
   over ≥100 windows (none has).

---

## Tier 4 — traders / sizing (the multiplier)

**Measured state.** Shared-window isolation: Follower +0.8%/$1 vs
Gambler +3.3%/$1 on identical signals = variance at 1.6× Kelly, not
skill. Saver −3.1%/$1 with maxDD $17.6k came from the 25%-stake era
(cut to 10% on 08-26). MLE pre-fix bet 7/7 windows off an optimism
loop (p_win tracked the ask), now margin-gated with shadow-row
training.

**Industry standard.** Kelly criterion (Kelly 1956, *Bell System
Tech. J.*) with fractional (half-)Kelly in practice — full Kelly is
growth-optimal only under exactly-known probabilities; under estimation
error it over-bets, and betting >Kelly is dominated (MacLean, Thorp &
Ziemba 2011, *The Kelly Capital Growth Investment Criterion*). Our
p_win estimates are (measurably) optimistic ⇒ fractional Kelly on
*calibrated* probabilities is the only defensible sizing.

**State of the art.** Meta-labeling for size (López de Prado 2018):
size ∝ secondary-model confidence. pt6 already implements this; its
inputs improve automatically when M1 lands (calibrated features).

**Change spec (M5 — largely shipped).**
- Shipped: PT5_FRAC 0.25→0.10; PT4 ≥0.77 gate + $10k reset (stamped,
  history preserved); PT6_MIN_EDGE_C=10 + shadow rows.
- **Use** half-Kelly everywhere sizes are model-driven, computed from
  p_cal once M1 lands. **Don't use** full Kelly (estimation error;
  MacLean-Thorp-Ziemba), **don't use** fixed-fraction >10% outside the
  explicitly-labeled aggressive-curriculum arm (Gambler).
- *Acceptance:* no trader's realized maxDD exceeds 2× its half-Kelly
  prediction over a rolling 100 windows.

---

## Evaluation discipline (cross-cutting, already policy)

- Prequential (test-then-train) evaluation everywhere (Dawid 1984).
- Window-level effective n; clustered comparisons (same-window entries
  share fate) — Diebold-Mariano tests for forecast comparisons
  (Diebold & Mariano 1995, *JBES*).
- Pre-registration of every new treatment before it trades
  (backtest-protocol discipline: Arnott, Harvey & Markowitz 2019,
  *J. Financial Data Science*); frozen rules, dated cutovers, no
  history rewrites.
- Sequential monitoring: SPRT (Wald 1945) for kill/keep decisions on
  new treatments.

## Source list (everything cited above, one place)

Microstructure & markets: Kyle 1985 (Econometrica) · Glosten & Milgrom
1985 (JFE) · Cont, Kukanov & Stoikov 2014 (J. Fin. Econometrics) ·
Zhang, Zohren & Roberts 2019 DeepLOB (IEEE TSP).
Calibration: Platt 1999 · Zadrozny & Elkan 2002 (KDD) · Kull et al.
2017 Beta calibration (AISTATS) · Guo et al. 2017 (ICML) · Murphy 1973
(J. Appl. Meteorology) · Gneiting & Raftery 2007 (JASA).
Online learning & selection: Cesa-Bianchi & Lugosi 2006 · Herbster &
Warmuth 1998 Fixed-Share (Machine Learning) · Li et al. 2010 LinUCB
(WWW) · Russo et al. 2018 Thompson sampling (FnTML) · Lattimore &
Szepesvári 2020 (Bandit Algorithms).
Drift & conformal: Gama et al. 2014 (ACM CSUR) · Gama et al. 2004 DDM
(SBIA) · Adams & MacKay 2007 BOCPD · Romano et al. 2019 CQR (NeurIPS)
· Gibbs & Candès 2021 ACI (NeurIPS) · Angelopoulos & Bates 2023
(FnTML).
Foundation models: Ansari et al. 2024 Chronos (arXiv:2403.07815) · Das
et al. 2024 TimesFM (ICML).
RL & trading: Sutton & Barto 2018 (2nd ed.) · Bellemare, Dabney &
Munos 2017 distributional RL (ICML) · Moody & Saffell 2001 (IEEE
Trans. NN) · López de Prado 2018 (Wiley).
Sizing & testing: Kelly 1956 · MacLean, Thorp & Ziemba 2011 · Wald
1945 SPRT · Diebold & Mariano 1995 (JBES) · Dawid 1984 (JRSS A) ·
Arnott, Harvey & Markowitz 2019 (JFDS).
