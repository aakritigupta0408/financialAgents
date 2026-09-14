# UI Migration Matrix (DT-07 §38/§39)

| old route | functionality | new location | backend source | status |
|---|---|---|---|---|
| `traders.html` | trader P&L / win-rate / avg W-L / return (computed in JS) | HOME trader cards + comparison | `results/home_snapshot.json` (`btc_rl/economics`) | **DELETED** |
| `perf.html` | mean/trade, total P&L, win rate, bootstrap CI (in JS) | HOME comparison + trader detail | `home_snapshot.json` (`economics.paired_delta`) | **DELETED** |
| `tiers.html` | tiers, offline BSS mean-across-folds (in JS) | ORACLE ▸ Modelling | `results/oracle_snapshot.json` (mean_bss precomputed) | **DELETED** |
| `features.html` | data/feature health, rolling σ band (in JS) | ORACLE ▸ Input Data + Feature Processing | `oracle_snapshot.json` / `feature_monitor.json` | **DELETED** |
| `health.html` | invariants, SEVs, rollup P&L (in JS) | ORACLE ▸ Sevs & Tickets | `oracle_snapshot.json` / `incidents.jsonl` | **DELETED** |
| `home.html` (old) | exec summary + JS mean-BSS | HOME (new, renderer only) | `home_snapshot.json` | **REPLACED** |
| `architecture.html` | DAG drill-down (already clean) | kept as dev drill-down (MORE nav) | `architecture_dag.json` | **KEPT** |

## Published canon (publish_dashboard.PAGES)
`home.html`, `oracle.html`, `architecture.html` (+ theme.css/nav.js/header.js/glossary).

## Parity notes (§39)
- Trader economics: full parity — HOME shows equity, realized P&L, EV/trade, EV/eligible,
  coverage, win-rate, bad-entry, drawdown, return, last-K — all from `btc_rl/economics`.
- ORACLE subtabs are **summary-level** renderers of existing backend snapshots. Deep
  per-model training curves / per-feature distributions (§23/§27) are a follow-up
  enrichment; the data sources exist (`model_offline/online.json`, `feature_monitor.json`)
  and the tabs already read them.

## Enforcement
`tests/test_no_frontend_science.py` fails if any published page implements a canonical
formula in JS. `tests/test_ui_contract.py` validates snapshot shape + referential rules.
