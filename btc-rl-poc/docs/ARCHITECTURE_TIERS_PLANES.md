# Quant Universe — Tiers × Planes Architecture

Canonical since the 2026-09-08 architecture directive. This describes
**reality**, not an aspiration; runtime was not reshaped to fit a
diagram. Two orthogonal axes answer two different questions.

- **TIER** — *WHERE in the predictive/decision lifecycle does this
  operate?* (T0…T7)
- **PLANE** — *WHAT responsibility does this carry across the
  system?* (A…F)

A component has one primary plane; it may participate in others.

## Tiers (where)

| Tier | Name | Question |
|------|------|----------|
| T0 | Observation / Data | what happened, point-in-time |
| T1 | Forecasting | what will BTC do (price/return/path) |
| T2 | Probability / market-relative | P(event) — and do we know more than the market |
| T3 | Decision policy | do we act |
| T4 | Execution | how/when we fill, and at what real price |
| T5 | Capital / risk | sizing, bankroll, drawdown |
| T6 | Evaluation / learning | scoring, calibration, resolution, retraining |
| T7 | Experimentation / governance | control/treatment, promotion, incidents, invariants |

Never blur tiers: lower T1 MSE ≠ T2 alpha; T2 BSS gain ≠ profit;
quoted EV ≠ realized EV.

## Planes (what responsibility)

| Plane | Name | One-liner |
|-------|------|-----------|
| A | Data | what happened |
| B | Model | what we believe |
| C | Decision & Execution | what we do |
| D | Research | what we are testing |
| E | Reliability & Governance | whether we can trust any of it |
| F | Teaching / Observation | how we understand and explain it — **must never mutate canonical research state** |

## Tier × Plane map (current real components)

| Component | Tier | Primary plane | Secondary |
|-----------|------|---------------|-----------|
| Binance / Coinbase / Kalshi capture | T0 | A Data | — |
| capture_xvenue + xvenue_sync | T0 | A Data | D Research (F1) |
| AlphaVantageAdapter (planned) | T0 | A Data | D Research (F-AV1) |
| F1 capture program | T0 | D Research | A Data |
| F-AV1 program (planned) | T0 | D Research | A Data |
| kb2 (control), kb4 (challenger), kb, kb3/7/8/9 | T2 | B Model | C (champion-internal), D (kb4 fwd test) |
| t1 serving pairs (consensus, rp, t9…) | T1 | B Model | D (t9 shadow) |
| probability bridge (z-bridge/blend) | T2 | B Model | — |
| champion decision policy | T3 | C Decision | E (frozen control) |
| A3-v2.1 | T3 | D Research | C |
| t_exec / t_exec_reg | T4 | D Research | C |
| bankroll ledger + late-settlement | T5 | C Decision | E |
| reconcile.py | T6/T7 | E Reliability | — |
| eval_engine / model_qualification | T6 | B Model | E |
| meta_monitor | T7 | E Reliability | — |
| invariants suite | T7 | E Reliability | — |
| trajectory board | T7 | F Teaching | — (informational only) |
| CHAMPION-vLean | T3/T7 | D Research | E |
| research agents (RM/EA/DR/MR/XR) | T7 | D Research | E |
| Five Worlds site, localhost Lab, Professor Mode | — | F Teaching | — |

## Why both axes (the professor answer)

> "What is kb4?" → A **T2** probability component in the **Model
> Plane**. Its forward evaluation lives in the **Research Plane**,
> its promotion authority in **Reliability/Governance**, and its
> explanation in the **Teaching Plane**.

Tier says where; plane says what responsibility. The same component
appears in several planes for several purposes — that separation is
what keeps experimentation from silently becoming runtime authority.

## The three system laws

1. **INFORMATION** — a model cannot sustainably discover information
   absent from its observation plane. Improve T0/root information
   before increasing model complexity. (kb2 ≈ 0.9984 corr with
   market; the bottleneck is resolution, not model size.)
2. **EVOLUTION** — a next generation exists only because the previous
   one taught a specific mechanism. Treatments evolve by mechanism,
   never by arbitrary tuning; an immutable generation is never
   mutated in place.
3. **TRUST** — a system is not healthy because everything is green;
   it is trustworthy because it detects when its own evidence cannot
   be trusted, stops itself, proves the mechanism, repairs without
   rewriting history, independently verifies, and only then
   continues. (The zombie-position SEV is the worked example.)

## Load-bearing separations (do not collapse)

- **Trajectory (F/Teaching, informational)** vs **gate/SPRT
  (E/Governance, authority)** — A3-v2.1 may read REGRESSING while its
  frozen gate reads CONTINUE. Not a contradiction.
- **research_status** vs **runtime_reachability** vs
  **removal_authority** — kb9 is RETIRED_FROM_T2_CONTENTION yet
  CONTROL_INTERNAL; removal_authority = CHAMPION_vLean promotion.
- **liveness** vs **state integrity** — RESTORED requires liveness +
  freshness + structural integrity + lifecycle completeness + state
  conservation, not process-alive alone.
