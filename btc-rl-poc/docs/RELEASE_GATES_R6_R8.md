# Release Gates — R6 → R8 (PM ratified 2026-09-10)

Current classification: **R6 candidate** — a credible production-style
autonomous research platform. Reliability/integrity is no longer the
binding constraint; execution and information resolution are.

> **System qualification ≠ strategy qualification.** R8 certifies the
> *machine* (integrity, recovery, lineage, autonomy), NOT that it
> makes money. A trustworthy machine that honestly reports "no edge"
> can still reach R8.

## The six remaining gates

| # | Gate | PASS condition |
|---|------|----------------|
| 1 | **Execution attribution (X1)** | ≥95% of the quoted→realized gap attributable to measured components within reconcile tolerance |
| 2 | **F1 / T1.2 information test** | microstructure gives genuine out-of-sample incremental resolution — *a FAIL is an acceptable completed gate* |
| 3 | **Lean champion** | low-value internal models removable without meaningful degradation (non-inferiority) |
| 4 | **Chaos / recovery** | all critical injected faults reach expected safe states and recover with zero ledger corruption |
| 5 | **Autonomous A/B lifecycle** | a treatment can register → collect → decide → retire/promote → rollback with no manual mutation |
| 6 | **Soak** | sufficient time/windows/regimes/restarts/retrains/settlements with 0 unresolved SEV-0/1, 100% money reconciliation, 100% lineage, all critical monitors live |

## Recovery SLO (from the freeze-convergence incident)

| Event | Target |
|-------|-------:|
| detect critical invariant failure | ≤ 1 runtime loop |
| freeze new entries | ≤ 1 runtime loop |
| auditor evidence generation | ≤ audit cadence |
| safe-state recovery after green evidence | ≤ 2 runtime loops |
| ledger reconciliation after restart | 100% |
| manual override required | 0 |

The headline reliability metric is **MTTR to verified integrity**, not
uptime — because *service alive ≠ system restored*.

## The alpha-taxonomy law (never merge into one "edge")

Four distinct edges, always reported separately:
- **Information alpha** — does the model know something the market
  doesn't? (measured via BSS / resolution)
- **Decision alpha** — does conditional abstention improve selection?
- **Execution alpha** — do we capture more of already-existing
  theoretical EV? (e.g. t_exec_reg's gains are HERE, not information)
- **Capital alpha** — does sizing improve risk-adjusted economics?

A treatment like t_exec_reg that profits by refusing poor executions
is an **execution/utilization edge** and must never be labelled
predictive alpha.

## What does NOT block R8 (strategy facts, not system faults)

kb2 BSS = 0 · no profitable strategy · t_exec_reg failing to promote ·
T1.2 failing · F-AV1 finding zero features · persistence remaining
best · market efficiency.

## What still blocks R8 (genuine production-equivalence faults)

execution economics unreconciled · crash recovery not systematically
exercised · lineage gaps · auditor and runtime sharing critical money
logic · rollback never tested · critical monitors in UNKNOWN · manual
intervention in the treatment lifecycle · stale/missing data reaching
decisions · the source→feature→prediction PIT chain not reproducible ·
unresolved critical incidents · active treatment count over the lean
budget · a change promotable without cross-plane regression evidence.
