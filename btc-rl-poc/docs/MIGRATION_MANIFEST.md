# Migration Manifest + Agent Execution Plan
## Master Build Contract → concrete mapping (2026-08-30)

> **SUPERSEDED IN PART (08-30):** the PM issued the authoritative
> 126-section Migration Manifest + Agent Execution Plan. Machine-
> readable dispositions now live in `config/COMPONENT_REGISTRY.yaml`
> (the §85 manifest) and metric contracts in `config/METRICS.yaml`
> v1.1.0. Deltas adopted from the PM plan: M11+M8 → DIAGNOSTIC
> ARCHIVE (runtime stopped, not shadow); M-branches M2/M2+M8/M3/
> M3+M8/M9/M11 → ARCHIVED (not merely retired); kb3 flagged
> INCUMBENT CANDIDATE pending the M2 evaluator; pt7 renamed
> EXEC_BENCH_NAIVE_MAKER; daily PM snapshot (§111) emitted as
> results/pm_snapshot.json on the audit chain. The section below
> remains as the page/script mapping detail.

Companion to docs/RETIREMENT_MANIFEST.md (M3 roster verdicts — already
executing: TX-B runtime cut is LIVE) and docs/REDESIGN_PLAN.md.
**IA supersession**: the master contract replaces the 5-page site with
TWO products — **Home** (public MVP) and **Backend** (one internal
console, sidebar: Overview · Models · Experiments · Data · Operations
· Agents · Registry). The 3-cards-per-view law carries over.

Verdicts: KEEP · CONSOLIDATE (content survives inside a new home) ·
SHADOW · RETIRE (leaves runtime/nav; file stays published for link
integrity) · ARCHIVE (evidence only).

---

## 1. SITE PAGES (26 files)

| Page | Verdict | Destination |
|---|---|---|
| home.html | **KEEP — rebuild as Home MVP** (M4-final): Thesis→Price→Action, principal experiment scoreboard, What-we-learned claims | Home |
| index.html | CONSOLIDATE | redirect → home.html |
| board.html (Research Command) | CONSOLIDATE | Backend → Experiments (its 3 cards become the module; built 08-29, structure already contract-compliant §20-22) |
| watchtower.html (Control Tower) | CONSOLIDATE | Backend → Operations (Card 1/2/3 map §27; registry drawer → Registry module) |
| ledgers.html (Evidence Explorer) | KEEP | Backend → linked evidence drawer (off-nav, per §53 every number explains itself) |
| universe.html (Map) | CONSOLIDATE | Backend → Data (lineage, system graph) + Overview (system clock); leaves primary nav |
| models.html | CONSOLIDATE | Backend → Models (M2 rebuild: four eras, scoreboard §13-14) |
| metrics_lab.html | CONSOLIDATE | Backend → Experiments analysis drawers |
| live_online.html | CONSOLIDATE | Backend → Overview Card 1 (Current Desk) |
| clock.html | RETIRE | absorbed by Map Card 2 → Backend Overview |
| agents.html | CONSOLIDATE | Backend → Agents (roster + §34 performance) |
| museum.html | CONSOLIDATE | Backend → Registry (retired rows carry cause-of-death; Home What-we-learned links here) |
| archive.html | CONSOLIDATE | Backend → Registry (ARCHIVED filter) |
| diagnosis.html | CONSOLIDATE | Backend → Operations drawers |
| sev0.html | CONSOLIDATE | Backend → Operations incidents |
| analyst.html | CONSOLIDATE | Backend → Agents (analyst log) |
| paper.html | KEEP (off-nav) | linked from Home research story §55 |
| home_classic.html | ARCHIVE | published, unlinked (decision ledger view) |
| ab_dashboard.html | RETIRE | superseded by Experiments module |
| experiment_review.html | RETIRE | superseded by Experiments module |
| live_training.html | CONSOLIDATE | Backend → Models training observatory (M2 §15) |
| instrument.html | ARCHIVE | narrative kept, off-nav |
| nav.js | **REBUILD**: `Quant Universe · Home · Backend` + search; drop 5-page primary + More sprawl | global |
| theme.css / glossary.js / glossary.json | KEEP | shared |

## 2. TRADERS / TREATMENTS / MODELS
Verdicts already registered and EXECUTED (TX-B live, invariants
green) in docs/RETIREMENT_MANIFEST.md: roster = Follower(CONTROL) ·
Disciplined(CONTROL-thesis) · MLE(SHADOW, zero-stake) · A3/T10
(TREATMENT) · slot-5 EMPTY; legacy experiment = M10 vs M10+M8 (+3
shadows); serving models = kb2 vs kb9 + 4 frozen T1 horizon pairs;
kb/kb3-kb8/kbf = OFFLINE BENCHMARK. §5 model budget end-state (max 3
roles) still blocked on the owner's keep-all-horizons order — flagged.

## 3. SCRIPTS / SERVICES / CRONS

| Component | Verdict | Note |
|---|---|---|
| btc_rl/online.py daemon | KEEP | roster-frozen build live 08-29; M6 adds evidence-gated retraining (§42) later |
| event_capture.py + capture_watchdog.py | KEEP | Layer-A tape (canonical state feeder §26) |
| audit_chain.py (10-min) | KEEP | the independent-audit spine §31 |
| emit_a3.py | KEEP — FROZEN measurement | A3 change control unchanged |
| watchdog.py, meta_monitor.py, publish_dashboard.py | KEEP | self-healing seeds (§28: restart/retry/rebuild already exist) |
| emit_program / emit_decision_board / emit_diagnosis / emit_oracle_calls / emit_readiness / reconcile / leakage_canaries path | KEEP | feed Backend modules |
| emit_world.py | CONSOLIDATE | becomes Backend Overview/Data state emitter |
| emit_board.py, emit_registry.py | CONSOLIDATE | merge into one registry emitter (§35 schema: state/owner/reason-it-exists/decision-impact/reopen-condition) |
| emit_fill_curve, emit_exec_sensitivity, emit_execution_ledger | KEEP | execution-researcher inputs §43-44 |
| mine_legacy.py, standings.py, debug_arms.py, demo_* | ARCHIVE | one-shot research, keep in repo |
| train_l2/l3/l4.py, evaluate_all.py, offline_gate.py | CONSOLIDATE → M2 | fold into the single offline evaluator + training registry |
| cdp.py, chrome_ctl.py, build_site.py, emit_manifest.py, run_audit.py, feedback.py, loss_review.py | KEEP (support) | loss_review feeds incidents |
| Crons (7 lines) | KEEP as-is | no per-retired-component cron exists; M6 may add a self-healing verifier row |

## 4. M2–M6 BUILD TASKS × AGENT ROLES

The five durable agents (§32) double as the build-work owners; the
Research Manager sequences. All agents consume METRIC_DICTIONARY /
universal metric rows — never private formulas (§36-37).

**M2 — ML & Research Observatory (BUILD NOW, engineering clock)**
1. METRICS.yaml v2: add §36 fields to METRIC_DICTIONARY v1.0.0
   (minimum_n {display/compare/gate}, windows, valid scopes, metric_id
   normalization). *(Model Researcher)*
2. training_runs.jsonl immutable registry (§15 schema; seed from
   metrics_history.jsonl retrain records) + model_lifecycle.json with
   OFFLINE/LAUNCH/CURRENT/LIFETIME blocks (§13, §38). *(Model
   Researcher)*
3. Offline evaluator: one entrypoint, layers A-E (§16) + walk-forward
   folds (§17) + falsification battery (§18 → CANDIDATE INVALID).
   *(Model Researcher + Experiment Analyst)*
4. Online model metrics: rolling windows since-launch/100/50/25,
   UNKNOWN under min-N (§40); prediction-distribution + feature
   observability (§39). *(Data Reliability)*
5. Offline↔online parity replay → MODEL_PARITY_FAIL invariant (§41).
   *(Data Reliability)*
6. Generalization gap + research-optimism tracking per promotion
   (§19). *(Experiment Analyst)*

**M3 — Lean Runtime Cleanup (largely DONE)**
Remaining: TX-C home/live_online headline demotions (folds into M4
Home rebuild), TX-D kb benchmark demotion (folds into M4 Models
module). Registry rows for retired components get §23 preservation
fields.

**M4 — Backend UI (after M2 data model)**
1. backend.html shell: fixed sidebar (7 modules), global status bar
   (§11: SYSTEM/DATA/STRATEGY/A3 n/MODEL/AUDIT/SEV), hash-routed
   modules, ≤3 cards each.
2. Overview = Current Desk + Research + Needs Attention (§12) —
   sources: Control Tower actions queue + Research Command strip.
3. Models module = M2 artifacts (§13-15). Experiments module =
   Research Command cards (§20-22). Data module = feeds/freshness/
   schema + lineage from Map (§24-26). Operations = Control Tower
   (§27-31). Agents (§32-34). Registry (§35).
4. nav.js → Home · Backend. Old URLs keep redirect stubs.
5. Home rebuild LAST (§7-9, M4-final): projection of stable outputs.

**M5 — Agents (after M2 contracts)**
Five structured-output agents (§33 JSON contract, artifact citations
mandatory) + agent-performance ledger (§34). Research Manager emits
the §32 bottleneck report → feeds program.json research_manager block
(structure already live 08-29).

**M6 — Self-Healing Operations**
Codify §28 loop DETECT→CLASSIFY→CONTAIN→REPAIR→VERIFY→RESTORE→RECORD:
wrap existing repairs (watchdog restart, capture restart, publisher
self-heal, fail-closed) in incident records + independent post-repair
verification; add safe additions (artifact rebuild, last-known-good
model artifact restore, verified-dead lock clear). Forbidden-repair
list (§28) enforced by invariant: no automated writer may touch
policy constants, ledgers, or experiment specs.

**M7 — evidence-gated only** (Value-of-Wait / execution challenger /
sizing): stays BLOCKED on the §46 decision tree, unchanged.

## 5. GOVERNANCE CARRY-OVERS
- A3-v1.1 frozen params untouched (§1); change control list (§51)
  identical to A3_CHANGE_CONTROL.md — no drift.
- Claim levels (§56) map to existing system_knowledge statuses;
  Home renders only labeled claims.
- Color law (§52): +16.7¢ at n=1 stays AMBER — already implemented on
  Research Command Card 1 (evidence chip dominant until n≥25).
- Every number self-explains (§53): glossary.js grows metric drawers
  fed by METRICS.yaml (no independent formula definitions).

## 6. SEQUENCE (from here)
1. Commit Map agent output when it lands (content feeds Backend Data/
   Overview; universe.html marked CONSOLIDATE).
2. M2 items 1-2 (metrics v2 + training registry + lifecycle) — the
   shared contract everything else consumes.
3. M4 backend shell + nav switch (Home · Backend), modules embedding
   the already-built 3-card sets.
4. M2 items 3-6, then M5 agents, then M6 formalization.
