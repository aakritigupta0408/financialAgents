# Reporting Overhaul: Themed Pages, Metric History, Industry-Standard Evaluation

## Context

The live page (`live_online.html`) was redesigned into the "ticker-desk" system
(flat near-black, validated palette, Didot display serif, tabbed IA). The other
three pages — `experiment_review.html`, `live_training.html`, `index.html` —
still wear old skins, don't persist metric history across retrains (every
retrain overwrites; no before/after comparison), and report a metric set that
grew organically. The user wants: (1) all pages in theme, (2) metric history
persisted at every retrain for comparison, (3) the metric set audited against
industry standards for short-horizon financial forecasting, with an honest
verdict on our problem formulation.

## Industry-standards audit (research summary)

| Category | Standard metrics | We have | Missing |
|---|---|---|---|
| Point forecast | MAE, RMSE, **MASE** (scaled to naive/random-walk), DM significance tests | MAE, MAE/floor ratio (≡ MASE vs persistence), DM w/ HAC lags | RMSE; the explicit "MASE" framing |
| Direction | Directional accuracy + **Pesaran–Timmermann test** | dir% | PT significance test |
| Probabilistic | **PICP** (interval coverage), **sharpness** (width), **pinball loss**, CRPS | 80% coverage | sharpness, pinball@10/90 (computable from stored lo/hi for every arm), CRPS (optional) |
| Binary / market | **Brier**, **Brier skill score** vs market & climatology (0.25), reliability diagram | Brier, market Brier | BSS numbers, calibration curve (data already logged) |
| Trading | PnL **after costs**, hit rate, profit factor, max drawdown | raw paper P&L | Kalshi fee model (≈7·P·(1−P)¢), drawdown, per-bet expectancy |

**Formulation verdict** (stated on the review page): literature agrees
minute-scale crypto ≈ random walk — models rarely beat persistence on level at
these horizons; direction is the partially-predictable component. Our design —
persistence-scaled scoring, leakage guards, DM tests, live paper market — is
more rigorous than typical published setups. Real gaps: no cost model in P&L,
no direction significance test, unreported interval sharpness (all fixed by
this plan); the legacy exact-integer hit is already replaced by the vol-scaled
band.

## Implementation plan

### A. Metrics layer — new `btc_rl/metrics.py`
Pure functions reused by evaluator + page builders:
`mase(errs, naive_errs)`, `rmse`, `pinball(actual, lo, hi)` (q10/q90),
`sharpness(lo, hi)`, `pt_test(preds, actuals)` (Pesaran–Timmermann z),
`brier_skill(brier, ref)`, `calibration_bins(p, y, n_bins=10)`,
`kalshi_fee_c(price_c)` ≈ 7·p·(1−p) cents, `max_drawdown(cum_series)`.
Extend `scripts/evaluate_all.py` with the new columns (RMSE, MASE, PT z,
sharpness, pinball, BSS, fee-adjusted bet P&L).

### B. Metric-history persistence — `results/metrics_history.jsonl`
Append-only, one JSON row per event, `kind` + `ts` + git sha; trim ~5k rows.
- `kind:"retrain"` — written by `retrain_all()` (`btc_rl/online.py`) after the
  gate: per arm×horizon `val_mae_before/after/reverted`, plus a trailing-6h
  online snapshot (per arm×h: n, MAE, MASE, dir%, coverage, sharpness; kb:
  brier, mkt_brier, BSS; bets: n, fee-adjusted pnl_c). Fixes the biggest
  confirmed gap: gate outcomes currently live only in
  `online_status.json["last_retrain"]` and are overwritten within one 30s poll.
- `kind:"batch"` — appended by `btc_rl/train.py` (writer of metrics.json —
  which also gains a `generated_at` field), `scripts/train_l2.py` (already
  computes test/persistence MAE), and `scripts/train_l3.py`/`train_l4.py`
  (currently print test MAE to stdout ONLY — capture into a dict and append,
  so L3/L4 batch numbers finally survive somewhere machine-readable; today
  they exist only in the hand-written OFFLINE_METRICS.md).
No append-only metric history exists today (`learning_log.jsonl` is just
session counters that reset per restart).

### C. Shared theme — `site/theme.css`
Extract ticker-desk tokens + components (palette incl. validated series
colors, statusbar/stat/bignum, arm tags, chips/tabs, tables, pills,
details/summary, legend, svg rules) from `live_online.html`; all four pages
link it. `build_site.py` emits the `<link>` instead of its inline CSS.

### D. `experiment_review.html` — rebuild as the online evaluation lab
1. Hero strip: experiment status one-liner (arms, spans, slots scored).
2. **Scoreboard** (centerpiece): per arm×horizon — n, MAE, RMSE, MASE, dir%
   (+PT flag), coverage%, sharpness, pinball — leader highlighted, DM markers.
3. **Across retrains**: timeline from `metrics_history.jsonl` — kept/reverted
   gate heat-strip per retrain, MASE-by-era sparkline, "vs previous retrain"
   delta chips.
4. **Binary & bets**: Brier / market Brier / BSS tiles, SVG reliability
   diagram (10 bins), fee-adjusted bet P&L + max drawdown.
5. **Formulation & methodology** (collapsed): the audit table above, arm
   cards, architecture SVG restyled.
Reuse existing render fns: `onlineStats` (extend with rmse/mase/pinball/
sharpness — it already computes n/mae/rmse/dir/cov/width), `dmStat`,
`groupedChart`, `pctChart`, `offlineBlock` (point it at batch history rows so
L2–L4 stop showing "no batch backtest"); drop the purple hero/bg-fx.

Research sources for the audit: Diebold–Mariano usage and multi-horizon
extensions (Journal of Forecasting 2026), MASE-as-standard (AutoGluon-TS,
M-competitions practice), CRPS/pinball/PICP as probabilistic standards
(energy-forecasting competition literature), crypto-specific evidence that
hourly/minute models rarely beat random walk while direction is partially
predictable (Financial Innovation 2024 crypto DL comparison), and
trading-side standards (Sharpe/PnL-after-costs/max drawdown).

### E. `index.html` via `scripts/build_site.py` — batch report, themed
- theme.css link + ticker-desk hero; MASE column added to per-agent tables
  (agent MAE / persistence MAE — `mae` fields already in metrics.json).
- Fix two confirmed bugs: the "Best exact-int/MAE" stat tiles hardcode the
  persistence baseline (mislabeled as "Best"); `AGENT_LABELS` KeyErrors on any
  new agent key — make labels fall back to the raw key.
- New "previous runs" section from `kind:"batch"` history rows (date, agent,
  per-horizon test MAE, delta vs prior run).
- Batch numbers for L2–L4 (from history rows) join the tables, closing the
  "two disjoint metric worlds" seam (metrics.json knows only L0/L1 today,
  which is why the review page shows "no batch backtest" for most arms).

### F. `live_training.html` — themed telemetry + retrain timeline
- theme.css restyle (keeps epoch charts + EMA smoothing).
- New "Hourly retrains (live)" section from `kind:"retrain"` rows: strip chart
  of kept/reverted per arm over time + latest before→after val-MAE deltas —
  the online-training counterpart to the batch curves above it.

### G. Verification
1. `pytest tests/` (new unit tests for metrics.py: MASE, pinball, PT, fee,
   calibration edges).
2. Headless-render page JS on real ledger dumps (node, as done for
   live_online) + `node --check` each page.
3. Retrain dry-run (tests/check_retrain.py pattern) → confirm a
   `kind:"retrain"` row appends and pages render it.
4. `python scripts/build_site.py` → verify index + history section.
5. curl all four pages via the 8787 server; user eyeballs the theme.
6. Daemon restart after online.py changes (watchdog as backstop).

### Order of work
A → B (daemon restart) → C → D → E → F → G — data layer first so pages have
something to render, then theme, then pages heaviest-first.
