# Architecture Diff — Intended vs Actual (§26)

Generated 2026-09-13 by the initial system-archaeology audit. Compares the CURRENT
INTENDED architecture (exact-BRTI Oracle pipeline) against the ACTUAL implemented +
runtime system reconstructed from code.

## CHANGED SEMANTICS (implemented ≠ intended)

| Domain | Intended (current) | Actual (runtime) | Thread |
|---|---|---|---|
| Contract truth | exact CF-BRTI | in-house 4-venue "BRTI composite" | DT-01 |
| Settlement | official BRTI 60s-avg outcome (`contract_outcomes.jsonl`) | `int(coinbase_candle_close ≥ strike)` in daemon | DT-01 |
| Oracle in runtime | recalibrated MECH_FAIR_BRTI feeding locks | offline-only; runtime still uses kb-arm probs | DT-01 |
| Gate B label | official `exact_yes` | Kalshi terminal quote proxy | DT-02 |
| UI top-level | HOME + ORACLE (6 sub-sections) | 5-page research console | DT-07 |

## ADDED NODES (Gen 6/7, present & correct)
`brti_exact`, `brti_decision_dataset`, `mech_fair_brti`, `oracle_frozen`,
`prospective_capture`, `arch_checkpoint`. All **research_reachable only** — none are
on the runtime daemon path yet.

## REMOVED / RETIRED NODES (correctly inactive)
A3 wait-for-dip line (CLOSED 2026-09-08); retired traders pt2/4/5/7/8; retired
treatments t_cal/t_knife/t_cheap/…; per-horizon retired arms (t7-h5, t6-h5, …).

## EDGES THAT SHOULD HAVE BEEN REMOVED BUT STILL EXIST  ← the drift
1. `brti_composite_4venue → runtime_settlement` — should be `brti_exact → runtime_settlement` (DT-01).
2. `coinbase_candle_close → paper_accounts` (settlement) — should be `official BRTI outcome → paper_accounts` (DT-01).
3. `kalshi_terminal_quote → t05_gateb_label` — should be `contract_outcomes.exact_yes → label` (DT-02).

## ADDED EDGES (correct)
`brti_exact → brti_decision_dataset → mech_fair_brti → oracle_frozen → prospective_capture`.
This is the intended spine — built, tested, and **merge-ready but not runtime-wired**.

## PATCH-ON-PATCH (§27)
The exact-BRTI generation was added **alongside** the proxy generation rather than
replacing it: `data/adapters/brti.py` and `btc_rl/prospective_capture.py` are new
files imported only by scripts/tests, while `btc_rl/online.py` still runs the
Gen-4/5 proxy path. This is the canonical drift signature — new science + old
runtime + a merge-ready bridge that was never activated. The migration (DT-01) is
owner-gated because it edits the live desk's settlement/accounting.

## NET
Research architecture = CURRENT (Gen 7). Runtime architecture = Gen 4/5 with a
Gen-6 bridge staged but disabled. The one checkpoint FAIL (`RUNTIME_CONTRACT_TRUTH`)
encodes exactly this gap and will clear when exact BRTI is wired into the daemon.
