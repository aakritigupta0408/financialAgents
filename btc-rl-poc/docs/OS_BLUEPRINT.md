# THE QUANT RESEARCH OPERATING SYSTEM — canonical blueprint

Adopted 2026-08-29 from the TA/owner review ("one coherent Quant
Research Operating System"). This document is the binding system
model: everything — models, metrics, agents, experiments, incidents,
tickets, pages — attaches to one or more PLANES, flows through one of
four LOOPS, and answers to the FOUR LEVELS OF TRUTH. PROGRAM.md
remains the org charter; this is the engineering constitution.
Status tags: LIVE (exists, verified) · PARTIAL · QUEUED (tracked task).

## 1. The seven planes

| plane | job | primary question | today's implementation | status |
|---|---|---|---|---|
| Data | acquire + timestamp reality | can we trust what entered? | sources.py, ticks.jsonl, freshness in audit + world.json | LIVE (quality panel as blocking gate: QUEUED) |
| Prediction | forecast price distributions | better than baselines? | tier-1 arms, MSE/MSSE/MASE/pinball vs persistence floor | LIVE (DM test, interval score: QUEUED) |
| Probability | estimate event probability | info beyond the market? | kb arms, BSS-vs-market as primary | LIVE (ECE, Brier decomposition: QUEUED) |
| Decision | select / abstain / gate | when should we act? | leader + τ gate + envelope + treatments | LIVE |
| Execution | decision → fill | does edge survive reality? | real-basis family, waterfall, slip guard | PARTIAL (markout, latency: QUEUED — needs capture) |
| Risk & Capital | sizing, limits, survival | is the policy survivable? | Kelly caps, maxDD, resets | PARTIAL (CVaR, ruin prob, concentration: QUEUED) |
| Learning & Governance | train/test/monitor/promote/retire | should anything change? | gated retrains, SPRT registry, decision board, DECISIONS.md, human veto | LIVE |

Mapping to the historical tier numbering: T0=Data, T1=Prediction,
T2=Probability, T3=Decision, T4/T5=Execution+Risk&Capital; Learning &
Governance is the vertical spine (retrain gate, treatments, ledgers).

## 2. The four loops (the true product)

- **Runtime** (30 s): feeds → features → T1 → T2 → policy → execution
  → capital → paper ledger → settlement. LIVE; drawn on universe.html.
- **Learning** (hourly + on-settlement): settlement → evaluation →
  candidate → holdout gate → keep/revert; policy changes additionally
  → paired live A/B → human promotion. LIVE. (A formal SHADOW stage
  exists only for calibration-class changes today; policy treatments
  go straight to logged-only paired evaluation, which is our shadow.)
- **Maintenance**: telemetry → monitors (watchdog, audit, decision
  board integrity) → incident → mitigation → verification →
  postmortem → regression test. PARTIAL — the loop has run end-to-end
  (SEV-0), but monitors are scattered; Watchtower page QUEUED.
- **Research**: observation → analyst note → hypothesis → ticket
  (board.json) → treatment → experiment → decision → knowledge base
  (NOTES.md, DECISIONS.md, museum). LIVE — e.g. limit-order question
  → M11/pt7/pt8 → "adverse selection measured" → museum exhibit.

## 3. Component contracts

Every region/arm/job must expose the same card: component, plane,
purpose, inputs, outputs, frequency, owner, versions, latency SLO,
cost, consumers, metrics, monitors, failure mode, last deploy, last
incident, open tickets. Today ~60% of these fields exist scattered
across site_manifest.json (config, AST-parsed), world.json (clock,
agents, costs), audit_report.json (metrics), model_internals.json
(weights). model_registry.json + feature_registry.json emitters are
QUEUED (wave 4) to complete the card; the universe map's detail
panels are the rendering surface.

## 4. Metric layers (discipline: never one giant "metrics")

1. **Offline model metrics** — does it learn? (MSE/MSSE/MASE, pinball,
   coverage+width, tail AE; QUEUED: Diebold–Mariano, interval score)
2. **Online probability metrics** — does it know something the market
   doesn't? PRIMARY: BSS vs market. (LIVE; QUEUED: ECE, reliability
   resolution split, risk–coverage curves)
3. **Economic metrics** — can the information be monetized after
   costs? (EV/$1, P&L/window, maxDD; QUEUED: CVaR, capacity, ruin)
4. **Experiment metrics** — should we ship? (decision_board.json:
   paired Δ, bootstrap CI, P(Δ>0), MDE/power at family-wise α, veto
   decomposition, slices, states) — LIVE as of 2026-08-29.

The edge waterfall (model edge → decision-time → fill → after-fee →
realized) is LIVE at 2 stages (decision vs realized, −4.9¢/$1 gap on
168 windows) and QUEUED at full granularity (needs fill-time capture).

## 5. Offline → online retention

Required per treatment: backtest → replay → live-paired → realized
chain, and the meta-number "our offline process overstates effects by
X%". Backfills and live streams already share one evaluator (the
_treat_evaluate contract), so the columns are computable; the tracker
is QUEUED (wave 4). The kb7 counting error and the M3 pricing-basis
artifact are the recorded cautionary instances.

## 6. Experiment validity (live health assertions, not methodology prose)

LIVE assertions: paired same-window scoring, windows-not-rows n,
family-wise α, pre-registered edge/boundaries in code, evaluability
gate in decision_board (INVALID state), config-sovereignty (load
restores evidence only). QUEUED assertions: SRM-analog exposure check,
version-integrity stamp (config hash per window), quote-age invariant
as a runtime monitor. Every treatment carries a decision contract
(hypothesis, primary metric, must-win, must-not-break, current,
decision) — rendered from decision_board + rationale fields; the
three-outcome verdict (statistical / economic / operational) is the
decision-state rule set in emit_decision_board.py.

## 7. SEV taxonomy (adopted; INCIDENTS.md now grades on this)

| level | meaning | triggers (non-exhaustive) |
|---|---|---|
| SEV-0 | results cannot be trusted | leakage; eval sees future data; pairing corrupted; wrong settlement; metric changed mid-experiment; logged policy ≠ running policy; unreproducible result |
| SEV-1 | material performance/risk degradation | maxDD breach; EV collapse beyond threshold; stake-cap breach; execution leak beyond bound |
| SEV-2 | degraded subsystem, safe fallback | stale feed within fallback; repeated retrain failures; drift threshold; coverage collapse; missed monitor run |
| SEV-3 | maintenance, no scientific impact | stale dashboard; publisher failure; delayed non-critical report |

Incident lifecycle: detected → acknowledged → triaged → contained →
root-caused → fix proposed → offline-verified → shadow-verified →
live-verified → closed → postmortem → **regression test added**. The
quant-specific mandatory field: *which historical conclusions became
invalid* — and the invalidated number is preserved next to the
corrected one, never overwritten (LIVE practice: the +11.75% champion
figure stands crossed-out beside the true −6.5% in the SEV record).
Auto-invalidation of affected experiments by an open SEV: QUEUED
(Watchtower linkage).

## 8. Tickets

Taxonomy (research / model / experiment / engineering / reliability /
incident-remediation / data / evaluation / cost / UX / governance /
debt) adopted for board.json entries going forward; every ticket:
impact, urgency, owner, evidence, dependencies, success metric, target
plane. "Needs attention" rollup on Home: QUEUED (homepage wave 2).

## 9. Agents

Operating model is LIVE in world.json: every agent has trigger,
authority, CANNOT list, last-seen evidence; the human is the
governance root and the only promote/retire authority. Quantitative
agent evaluation (evidence-supported claim rate, acceptance rate,
alert precision, cost/run) is QUEUED — the Agent HQ page shows the
empty row honestly ("not yet metered") rather than inventing it. The
Analyst's falsified claims stay on the record (first commentary note
is its own wrong RCA) — that is the evaluation substrate.

## 10. Regression suite from insights (institutional memory)

Adopted rule: every confirmed insight or incident yields either a
fixture (metric_fixtures.json — 27 live parity vectors), an invariant
(tests/invariants.py, QUEUED wave 4: quote-age bound, ledger
monotonicity/no-dupes, evaluability=100%, fee parity, skip≠missing),
or a museum exhibit (LIVE — so no agent rediscovers the toxic hour).

## 11. Evidence graph

Objects (model, feature, policy, trade, experiment, metric, incident,
ticket, note, decision, dataset, code version) + typed edges
("INCIDENT invalidates EXPERIMENT", "NOTE proposes TICKET"...).
QUEUED (wave 4) as an emitted JSON traversed client-side; today the
graph exists implicitly (evidence lines on notes, DECISIONS links,
board references) — the emitter makes it machine-walkable, answering
"why is M11 dead?" by traversal.

## 12. Four levels of truth (design law — never four separate truths)

Story (child) · Decision (investor/PM) · Diagnosis (MLE) · Evidence
(auditor) — one fact, four depths. Rendering: Explore/Engineer lenses
(universe, museum), decision board states above drill-downs, evidence
lines on every claim, ⓘ-popover layer QUEUED (wave 3).

## 13. The §33 reviewer checklist — answer map

Answerable on-site today: now-state (home/live), models running +
configs (manifest, AST-parsed), data freshness (world/clock), retrain
trigger/results/reverts (clock), offline+forward+market-relative+
economic performance (metrics lab, audit), execution leakage
(waterfall), A/Bs+control+hypotheses+uncertainty+power+promote/kill
(decision board), incidents (sev0), learnings (museum, NOTES),
schedules+costs (clock), agent authorities (HQ).
Still "read the source" (each has a queued owner): exact model
versions/checksums (registry), feature lineage (registry), per-slice
risk controls (watchtower), agent quality numbers (metering),
reproduce-any-claim one-click (evidence graph). When that list is
empty, the review passes.
