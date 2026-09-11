# Retirement Migration Manifest — PM simplification directive (2026-08-29)

The governing rules, then a verdict per component with the exact code,
cron, metric and UI references that change. Retirement ≠ deletion:
every raw row, artifact hash, config, verdict and cause of death is
preserved; components leave *runtime, headline UI, retraining,
default agent context* only.

## Hard architecture rules (new invariants)

```
MAX ACTIVE TRADERS/POLICIES = 5   (one slot deliberately EMPTY)
PER EXPERIMENT LAYER: 1 CONTROL + ≤1 TREATMENT
SHADOWS: observe only — no trades, no selection influence, no
         denominator, not counted as live treatments
ONE question · ONE incumbent · ONE challenger · ONE primary metric ·
ONE statistical denominator
```

Lifecycle state machine (owner-controlled, exactly one per component):
`CONTROL · TREATMENT · SHADOW · QUALIFIED · RETIRED · ARCHIVED`
Recommendations (PRIORITY/CONTINUE/HOLD/DIAGNOSE/REDESIGN/RETIRE)
remain a SEPARATE analysis vocabulary — they never move lifecycle.

Champion/challenger chain law: when a treatment qualifies it BECOMES
the control and the old control archives; the next experiment is
promoted-control vs one new change. Losing treatments retire with
{hypothesis, spec, code hash, forward N, effect, CI, kill reason,
reopen condition} recorded. No strategy forests.

---

## TRADERS (pt*) — target roster: 4 active + 1 empty slot

| Slot | Component | Verdict | Lifecycle |
|---|---|---|---|
| 1 | pt — Follower | **KEEP** | CONTROL (frozen baseline) |
| 2 | pt3 — Disciplined | **KEEP** | CONTROL (core thesis policy) |
| 3 | pt6 — MLE meta-label | **KEEP as SHADOW** | SHADOW (research challenger; earns live status by evidence) |
| 4 | A3/T10 — entry timing | **KEEP** | TREATMENT (primary forward experiment vs C0) |
| 5 | — | **EMPTY by design** | a candidate earns this slot |

### pt2 Ladder — RETIRE (redundant: control twin + banking)
Banking is capital-management, not alpha; if banking matters, test it
later as a treatment against a frozen trader.
- **Runtime**: stop opening new pt2 positions in `btc_rl/online.py`
  (PT2_LOG_NAME block); open positions settle normally, then the log
  freezes.
- **Preserve**: `results/pt2_trades.jsonl` (full history), final
  bankroll, verdict "redundant — control twin".
- **UI**: out of live-desk/home headline; Evidence Explorer moves the
  Ladder tab to an ARCHIVED group.
- **Invariants**: `one-decision-per-window` keeps checking the frozen
  log (historical rows must stay clean).

### pt4 Gambler — ARCHIVE as educational/stress exhibit
Wrong abstraction: 33% staking answers "what if we lever harder", not
"do we have more alpha". Sizing belongs in the risk engine, after
alpha and execution qualify.
- **Runtime**: stop new entries (PT4_* block); the v3
  withdrawal-sweep ledger freezes at its final state.
- **Preserve**: full log incl. wd_c withdrawal ledger; drawdown
  history becomes the stress exhibit (Museum / risk-test suite).
- **Invariants**: `gambler-v3-sweep` and `withdrawals-never-restaked`
  KEEP RUNNING against the frozen history (cash conservation must
  hold forever).
- **Consolidate**: 33%-Kelly-ish sizing recorded as a risk-engine
  configuration, not a trader identity.

### pt5 Saver — RETIRE / CONSOLIDATE into risk layer
10% stake + profit skim = sizing variant, not a strategy.
- **Runtime**: stop new entries (PT5_* block); log freezes.
- **Consolidate**: stake-fraction + skim become risk-engine configs.

### pt7 Patient — RETIRE from live; becomes offline toxicity benchmark
Mechanism falsified: naive maker limit is the measured
adverse-selection stick (win-given-fill 68%→47% with bid depth).
- **Runtime**: stop new entries (PT7_* block).
- **New role**: `benchmarks/naive_maker` — the known-toxic baseline
  future execution treatments must beat. Referenced by A3/execution
  research, never by the live desk.

### pt8 Ideal — FREEZE, ARCHIVE SNAPSHOT, then decompose
Six simultaneous causal changes (regime + edge@fill + maker +
half-Kelly + depth + fill-recheck): cannot attribute incremental P&L
(criterion F). If it wins we don't know why; if it loses we don't
know why.
- **Runtime**: stop new entries (PT8_* block); snapshot config +
  code hash as the reference composite.
- **Follow-up research rule**: its ingredients re-enter only one at a
  time, as single treatments against the current control.

---

## LEGACY TREATMENTS (M*) — one legacy experiment remains

**THE one legacy experiment: CONTROL M10 (t_exec) vs TREATMENT
M10+M8 (t_exec_reg)** — "does regime filtering add value beyond the
execution guard?" (+1.4¢ increment, P 77% — genuinely open).

| Component (ev key) | Verdict | Runtime effect |
|---|---|---|
| M8 t_regime | fold into M10+M8 — RETIRED standalone | stop standalone headline; its question lives inside the combo |
| M10 t_exec | **KEEP — legacy CONTROL** | active |
| M10+M8 t_exec_reg | **KEEP — legacy TREATMENT** | active |
| M11+M8 t_limit_reg | DIAGNOSTIC ONLY | computes as SHADOW; no promotion path |
| M13 t_edgeband | SHADOW / HOLD | computes as SHADOW; anti-signal research |
| M1 t_cal | RETIRED (already) | stop ev computation |
| M2 t_knife, M2+M8 t_both | RETIRE | stop ev computation |
| M3 t_fshare, M3+M8 t_fs_reg | RETIRED (already) | stop ev computation |
| M9 t_cheap | ARCHIVE (hypothesis retained in M13 band floor) | stop ev computation |
| M11 t_limit | RETIRED (already) | stop ev computation |
| M12 t_evlead | RETIRE current implementation | stop ev computation; redesign is a NEW experiment if ever |
| champion / champion_real | **KEEP** | the desk baseline both arms pair against |

After M10-vs-M10+M8 resolves: **one primary strategy experiment
across the entire desk at a time** (plus A3, which is entry-timing on
the same desk, paired and denominated separately).

- **Preserve**: every historical ev pair in `treatments.jsonl`;
  paired incremental verdicts stay in program.json archive; the
  branch kill tree stays reachable in the Research causal-graph
  drawer as history.
- **UI**: decision_board + Research portfolio headline shows only
  M10, M10+M8 (+shadow chips for M11+M8, M13); everything else in
  the RETIRED tab.

## Entry-timing experiment (unchanged — already the clean model)
`C0 LIVE control · T10 LIVE treatment · T05/T15 SHADOW` — shadows
observe the same tape, place nothing, steal no denominator, and
cannot become live treatments until T10 has a verdict.

---

## MODELS — direction: MAX SERVING = 3 (structural control + incumbent + challenger)

Now (kept, per standing owner order "keep all 4 horizons, 1 ctl + 1
treatment each"):
- T1 serving pairs: h1↔t9-h1, h5↔t10-h5, h15↔t7-h15, h30↔t9-h30
  (LEAN_RETRAIN already enforces this; all other t-arms FROZEN —
  serve nothing, never retrain).
- T2 serving: kb2 (CONTROL, market-anchored) vs kb9 (TREATMENT) —
  the early-caller product pair.
- kb, kb3, kb4, kb5, kb6, kb7, kb8, kbf → **OFFLINE BENCHMARK /
  ARCHIVE**: still scored by the 10-min audit for the record, but out
  of headline UI, out of agent default context, no serving role.
  (kbf/kb6 already carry tombstones.)
- The "max 3 serving models" end-state is flagged as the next
  consolidation once a horizon proves dominant — it conflicts with
  the standing keep-all-horizons order, so it waits for an explicit
  owner call. NOTE FOR OWNER: these two directives cannot both hold
  forever.

## CRONS / JOBS — no change required
No cron is per-retired-component. audit_chain (10-min), watchdog,
publisher, meta-monitor, capture-watchdog stay. `introspect_model_
internals` (hourly) keeps scoring all arms — introspection is
archive-grade evidence, not serving. Retirements reduce daemon work
inside the 30s loop (fewer trader blocks, fewer ev keys).

## METRICS — dictionary unchanged
METRIC_DICTIONARY.yaml definitions are component-agnostic; no metric
retires. decision_board.json headline set shrinks to the active
experiment pair(s); retired treatment stats remain in the archive
sections.

## WEBSITE — headline demotions (3-card law already enforced)
- Research Command portfolio: ACTIVE tab = A3 + M10 + M10+M8
  (+shadow-chipped M11+M8, M13); everything else under RETIRED.
- Evidence Explorer: trader chips grouped ACTIVE (Follower,
  Disciplined, MLE·shadow) / ARCHIVED (Ladder, Gambler, Saver,
  Patient, Ideal) — archived ledgers stay fully browsable.
- Control Tower registry: lifecycle chips adopt the 6-state machine.
- Home desk row: game/desk panels show the 4-slot roster + empty
  slot 5.

## AUTOMATED RETIREMENT PROPOSALS (agent contract)
Agents may emit `RETIRE_CANDIDATE` when dominated / redundant /
falsified / stale / no-promotion-path / wrong-abstraction /
not-used-in-any-decision, as structured JSON
(component, reason, replacement, live_decision_impact,
historical_evidence_preserved: true). They never delete history and
never execute — retirement runs as a standard cleanup transaction
ratified by the owner.

## EXECUTION PLAN (staged transactions)
1. **TX-A (this commit)**: manifest + DECISIONS entry. ✔
2. **TX-B runtime**: online.py — halt new entries for pt2/4/5/7/8
   (open positions settle out), trim retired ev keys from the
   per-window treatment evaluation, adopt 6-state lifecycle map in
   emit_program.py, add roster-cap + one-control-one-treatment
   invariants to tests/invariants.py.
3. **TX-C UI**: headline demotions above (after the in-flight Map /
   Evidence Explorer builds land, to avoid conflicts).
4. **TX-D models**: kb benchmark demotion in models/live pages.

Retirement criteria reference (A dominated · B redundant · C wrong
abstraction · D falsified · E no promotion path · F unattributable ·
G not used in any decision) — recorded so future retirements cite a
letter, not a mood.
