# Quant Universe — Master Redesign Plan (TA directive, 2026-08-29)

Condensed from the full Parts I–XII brief. This is the governing
contract for all site work; page-level briefs (Research Command,
Control Tower, Map, Evidence Explorer) are instances of it.

## Hard laws

- **5 top-level pages maximum**: Home · Map · Models · Research
  (Experiments) · Control Tower (Watchtower). Global Search.
- **3 primary cards per page maximum.** Depth = tabs, drawers,
  modals, filters, search, drill-down — never new pages.
- No permanent "More" nav hiding a second website (retire it once
  absorption is complete).
- The whole site tells one loop: Observe → Predict → Decide →
  Execute → Measure → Learn → Experiment → Monitor → Improve.
- Tier vocabulary used identically everywhere: T0 data · T1 price
  forecast · T2 probability · T3 decision · T4 execution · T5
  capital/risk · T6 evaluation/learning · T7 experimentation/
  governance.

## Shared vocabulary (identical on every page)

- Health: HEALTHY / WATCH / CRITICAL / UNKNOWN (unknown is never
  green; missing monitors show NOT INSTRUMENTED).
- Models: PROPOSED / OFFLINE / SHADOW / ACTIVE / CHAMPION / RETIRED
  / INVALID.
- Experiments: DRAFT / OFFLINE / SHADOW / LIVE / PROMOTE /
  KEEP TESTING / HOLD / KILL / INVALID / COMPLETE — INVALID is never
  collapsed into KILL.
- Incidents: DETECTED → ACKNOWLEDGED → TRIAGED → CONTAINED →
  MITIGATING → VERIFYING → CLOSED.
- Tickets: BACKLOG / READY / ACTIVE / BLOCKED / DONE / CANCELLED.
- Color is semantic only: green validated/healthy, amber
  collecting/warning, red critical/invalid, blue control/baseline,
  purple agent interpretation, gray neutral/unknown. A positive
  number with inconclusive statistics is NOT green (evidence state
  dominates effect size — implemented on Research Command Card 1).

## Page contracts (3 cards each)

1. **Home** — sell + explain + live window. C1 live quant universe
   (make your prediction), C2 executive system state (model edge /
   economic / experiment / reliability / ops; ≤3 visuals: value-chain
   waterfall, top challengers w/ CI, EV×risk×coverage), C3 what the
   system is learning (beliefs / what changed / what failed).
   Built LAST — a projection of the other pages, never a parallel
   data universe.
2. **Map** — C1 system world (regions with LIVE health/metric/
   incidents/cost; lenses Explore/Engineer/Health/Experiments/Cost;
   click = zoom, not navigate), C2 how the system runs (Runtime =
   the System Clock, absorbing clock.html; Learning lifecycle funnel
   with counts; Data lineage trace of a real decision), C3 component
   explorer (search across models/policies/agents/jobs/data/monitors/
   experiments; one standard drawer).
3. **Models** — C1 model quality (4 executive answers: beat
   persistence? beat market? calibrated? economically useful?),
   C2 training & retraining (persistent history; observability gaps
   declared loudly), C3 stability & trust (regime slices, drift,
   monitoring completeness).
4. **Research** (board.html) — SHIPPED 08-29: C1 live priority
   (A3-v1.1, evidence-dominant), C2 portfolio (tabs + rows + causal
   graph drawer), C3 research manager (ranked queue from
   program.json research_manager block). Registry evicted to
   Control Tower.
5. **Control Tower** (watchtower.html) — SHIPPED 08-29: C1 system
   state (7 planes + changed-since-last-check + evidence health),
   C2 incidents & readiness (top blockers, qualification split,
   matrix drawer), C3 operations queue (action required / running
   normally / backlog; agents, jobs, tickets, events, registry,
   cost drawers).

## Ledgers → Evidence Explorer

Not a sixth page. Summary-first (win rate, net P&L, EV/$1, drawdown,
bankroll chart), then a virtualized filtered decision table (never
386 DOM rows), anomaly flags (HIGH-CONF MISS, LARGE LOSS, SEV-
AFFECTED, STALE QUOTE, TREATMENT-CHANGED-DECISION), per-trade trace
(data → features → model → probability → leader → policy → execution
→ sizing → outcome → metric → experiment), era/version stamping,
incident contamination visible, raw .jsonl behind "Raw source".

## Old page → new home

Clock → Map C2 · Agent HQ → Control Tower agents · Museum → Home C3
what-failed (+Research graveyard drawer) · SEV-0 → Control Tower
incidents · Ledgers → Evidence Explorer · Metrics Lab → distributed
(analysis drawers) · Training/Backtest → Models · Analyst → Home
insight feed + drawers · Archive → search + evidence explorer.

## Implementation sequence (TA order)

1. canonical schemas (metric ✅ v1.0.0, model lifecycle, experiment,
   incident, ticket, agent, monitor, job, cost, evidence)
2. shared site-state aggregation (one truth layer; pages stop
   reinterpreting raw files except for audit recomputation)
3. shared UI components
4. Map first (defines the mental model)
5. Experiments ✅ (Research Command)
6. Models
7. Watchtower ✅ (Control Tower)
8. Home last (projection of the rest)

## Quality gates (Consistency Auditor)

- exactly 5 pages, ≤3 cards, no shadow site under More
- every metric: unit, sample, period, benchmark, version; effects
  carry uncertainty
- every treatment: control, hypothesis, success rule, guardrails,
  health, decision status
- every model: version, inputs, training state, health, monitoring
- every subsystem: SLO, monitor, owner, failure action
- open SEVs visible on Home and Control Tower; affected experiments
  marked
- every agent: scope, trigger, permissions, quality metrics, cost
- every executive claim traceable; expert reaches raw evidence in
  ≤3 interactions; new visitor understands without tables

## Acceptance voices

12-year-old: "Different AIs try to predict Bitcoin… sometimes doing
nothing is best." · Investor: "autonomous experimentation platform
with observable quality, health, costs, governance." · Quant: "I can
distinguish forecasting skill, market-relative information,
executable edge, risk, uncertainty." · MLE: "I can see what is
deployed, retrained, drifted, failed, monitored." · Auditor: "every
conclusion traces to observations, versions, policies, metrics,
incidents."
