# System Architecture Audit — Initial Run (2026-09-13)

Machine-adjudicated verdict: **FAIL** (1 SEV-1 drift). Recurring checkpoint installed:
`scripts/architecture_checkpoint.py` (6 machine checks, 5 PASS / 1 FAIL).

## A. How the project changed
Six generations (`generations.json`): Gen1 generic BTC price forecasting → Gen2
market-relative multi-model → Gen3 execution research (T0.5) → Gen4 contract-mechanics
(BRTI 60s-avg rule) → Gen5 label correction (Coinbase ≠ settlement, ~4.5%) → Gen6
exact BRTI → Gen7 (current intended) exact-BRTI Oracle pipeline.

## B. Current end-to-end path (actual)
Two disjoint architectures coexist:
- **Runtime (daemon `btc_rl.online`)**: Coinbase + Kalshi + 4-venue BRTI *composite* →
  features → kb/pt arms → traders → **Coinbase-candle settlement** → paper accounts →
  run_audit → publisher → 5-page site. **Gen 4/5.**
- **Research (offline scripts)**: exact BRTI → brti_decision_dataset → MECH_FAIR_BRTI →
  oracle_frozen → prospective_capture (disabled). **Gen 6/7, not runtime-wired.**

## C. Intended vs actual → `architecture_diff.md`
5 changed semantics; **3 edges that should have been removed but still exist** (all
DT-01/02): proxy→settlement, candle→paper-P&L, Kalshi-quote→GateB-label.

## D. Dangling threads → `dangling_threads.json`: 13 total
1× SEV-1 (DT-01 runtime not on exact BRTI), 5× SEV-2 (GateB label, Brier/fee dup,
UI IA, in-browser metrics), 7× SEV-3.

## E. Legacy filters → `inventories.json`
9 active gates (all hard-coded consts), ~8 dormant (retired traders/treatments).
A3 wait-for-dip correctly CLOSED. No filter surviving only "because it used to be here"
in the active set — dormant ones are guarded by RETIRED_* sets.

## F. Market-information audit: **CLEAN**
No forbidden market→Oracle edge. All 6 oracle paths independent (Kalshi = scoring only).
Residual risk DT-13: isolation depends on FEATS lists staying cb_*-only (now checkpoint-guarded).

## G. Future-leakage audit: **PASS** (1 minor)
Active pipeline PIT-clean on splits/σ-calibration/standardization. One SEV-3 nuance
(DT-12): `required_remaining_average` uses realized settlement sample count `n_c`.

## H. Label/target lineage
Canonical = `results/contract_outcomes.jsonl` (official BRTI `exact_yes`). Runtime
settlement + GateB still on proxy labels (DT-01/02).

## I–L. Inventories → `inventories.json`
Features (10 families, 1 orphan `price`); Models (17 registered; kb2 = deliverable;
legacy batch artifacts unregistered); Traders (5 active, 5 retired); Experiments
(specs + treatments, wait-for-dip closed).

## M. UI/backend → DT-07/08/09
5-page console (intended HOME+ORACLE); `perf.html` published-but-unlinked; 4 published
pages compute metrics in-browser; ~24 orphan pages (publisher-pruned) carry stale facts.

## N. Agents
run_audit (independent perf recompute), watchdog (daemon supervision), publisher —
all active, non-duplicated. No stale-metric agents found.

## Q. Current bottleneck (one)
**RUNTIME_CONTRACT_TRUTH** — the live desk is not on exact BRTI (DT-01). Everything
else is WATCH-level debt.

## R. Architecture checkpoint (installed & tested)
`scripts/architecture_checkpoint.py`, governance counter in `governance.json`
(checkpoint every 3 material changes; `--bump "reason"`). FAIL blocks treatment→control
promotion (§34). Snapshots to `architecture/audits/<utc>/`.
