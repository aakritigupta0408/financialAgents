# PROGRAM — the operating charter

How this project runs as an organization of agents: who exists, what
each needs, how they interact, what gates every change, and how the
architecture scales without touching the thing that works.

Written 2026-08-28. Status: the org below is partially live (marked ●);
the rest is the build order. This document is the source of truth for
"who decides what"; docs/DECISIONS.md is the ledger of what was
decided; docs/MANUAL.md is how to operate what exists.

---

## 1 · The knowledge substrate (how agents share anything)

Agents do not talk to each other directly. They read and write a
SHARED SUBSTRATE, which makes every interaction inspectable,
replayable, and survivable across agent restarts — the same reason the
daemon logs to jsonl instead of holding state in memory:

| layer | contents | writers | readers |
|---|---|---|---|
| **ledgers** `results/*.jsonl` | every prediction, trade, treatment score, incident | daemon, audit runner | every agent, every page |
| **state** `results/*.json` | model weights, calibrators, SPRT state, fshare/evlead | daemon | introspection, pages |
| **knowledge** `docs/*.md` | research baseline, model internals, manual, this charter, decisions, incidents | teams (one owner-file each) | all teams, the human |
| **narrative** `NOTES.md` | the append-only research log: every result incl. negative | whoever finds something | onboarding, RCA |
| **site** `site/*.html` | the rendered truth — recomputed from ledgers, never stored | build team | the human, the TA |

Ownership rule (already enforced this session): **one writer per file
per work-wave.** Parallel agents never share a file; the conductor
integrates. This is why four teams can run simultaneously with zero
merge conflicts.

Planned upgrade (scale section, §6): a vector index over the knowledge
layer so agents retrieve by meaning, not filename. Not before it pays
for itself — grep over ~10 docs currently works.

## 2 · The teams (specialist agents), their inputs and outputs

Each team is an agent role with a defined contract. ● = instantiated
and has produced output; ○ = defined, spun up on demand.

**● RESEARCH team** — "what does the field know that we don't?"
- needs: WebSearch/WebFetch; our method docs to diff against
  (SEV0_REMEDIATION.md, treatments.py); NO ledger access needed
- produces: docs/RESEARCH_BASELINE.md — standards per tier with real
  citations + a ranked gap list
- hard rule: cite only retrieved sources; "not verified" over guessing
- cadence: on demand + whenever a new tier/mechanism is proposed

**● INTROSPECTION team** — "what did the models actually learn?"
- needs: results/*_logit.json, kb_calib.json, treatments.json, *.pt
  checkpoints, the feature-name maps in btc_rl/online.py, ledgers for
  traces; scripts under tests/introspect_*
- produces: docs/MODEL_INTERNALS.md + results/model_internals.json
  (weights tables, flaw analysis, one-trade thought-flow trace, the
  utopian-counterfactual oracle regret)
- hard rule: every number computed by a committed script
- cadence: after every model-affecting change; report feeds the site

**● BUILD team** — "render the truth"
- needs: the theme system, page conventions, the ledgers' schemas,
  Chrome DevTools for render-verification
- produces: site pages; every panel computed client-side from ledgers
  (tables that derive from the stream cannot go stale — the two tables
  that ever went stale were the two with hand-maintained lists)
- hard rule: node --check + screenshot before "done"; honest
  "insufficient history" notes instead of thin numbers

**● UI/UX team** — "the expression of the work" (the hardest brief)
- mandate: the site is not a dashboard, it is the presentation of the
  research — layered so a 10-second skim, a 2-minute read, and a full
  technical descent are all first-class paths. Interactive diagrams,
  flow animations, comic-strip explainers where a sequence teaches
  better than prose — for the short-attention-span reader WITHOUT
  losing a single technical detail (depth is one click away, never
  deleted). The Jobs bar, made recursive: every panel answers before
  it explains.
- needs: everything the other teams produce (metrics lab, model
  internals, incidents, research baseline) — it renders their truth;
  plus design research (what makes technical storytelling land).
- produces: the story layer over the site; per-wave "expression
  review" of every new panel (tables → visual narratives).
- hard rules: no decoration that lies (animation must encode a real
  process); every visual keeps its numeric source one click away;
  render-verified like all build work.
- cadence: a full pass after each integration wave; owns the
  presentation backlog.

**● DOCS team** — "the car's manual"
- needs: NOTES.md, git log, code docstrings — read-only everywhere
- produces: docs/MANUAL.md (made / fix / drive) for team replacement
- cadence: refresh after each major wave

**● MAINTENANCE team** — currently two cron agents, deliberately dumb:
- publisher (1 min): ships pages + trimmed ledgers to gh-pages
- audit runner (10 min): recomputes cross-tier health →
  results/audit_report.json from raw ledgers, never importing daemon
  code (it must not be able to disturb the thing it measures)
- planned: a watchdog that alerts on daemon staleness >5 min and on
  any SPRT boundary crossing (the launch trigger)

**○ PORTFOLIO team** — "what exists, what retires"
- needs: treatments board, audit_report, DECISIONS.md
- produces: retirement/launch proposals with evidence, written to
  DECISIONS.md as proposals; the human ratifies policy
- the ledger for these decisions: docs/DECISIONS.md (seeded today with
  every past decision reconstructed from NOTES.md)

**○ QUANT-BUILD team** — implements new arms/traders/treatments to the
standing spec: additive, pre-registered in NOTES.md before trading,
SPRT twin created with the mechanism

## 3 · The super-agents (three, as specified)

- **EXECUTION** (conductor — this session's role): decomposes work,
  staffs teams with no file overlap, integrates, verifies end-to-end,
  ships. Owns commit discipline and the "no claim without a check"
  norm. All fixes route through it.
- **TECHNOLOGY**: owns the architecture seams (§6), the invariant list
  (MANUAL §4), tooling (crons, verification harnesses), and vetoes
  anything that touches the hot path's behavior without a treatment.
- **FUNDS & UX** (sales-analog; the single user + LLM-judged UX stand
  in for a market): owns the bankroll ledgers' integrity, the
  launch-pipeline gates (§4), and page quality (the Jobs bar: answer
  first, evidence one click away).

Escalation: teams → Execution → the human. Policy changes (anything
that moves live money behavior) ALWAYS terminate at the human; the
record of that approval lives in DECISIONS.md.

## 4 · The change pipeline (every update is a treatment)

The standing law, now written down:

1. **Propose**: pre-register mechanism + gate metric in NOTES.md.
2. **Offline**: settle-ordered replay (backfill_treatments.py). A lift
   here is NECESSARY but never sufficient — three of our confident
   offline conclusions were wrong (toxic hour p=0.60; naive limit
   −5.21% adverse selection; the tier-1 direction RCA).
3. **Online**: run as a live treatment, paired same-window vs
   champion, Wald SPRT with pre-registered edge (2%/$1) and boundaries
   (α=.05, β=.10, min 40 windows). "New trader with new money" is the
   instantiation for trader-level candidates (pt7/pt8 pattern) — 100%
   traffic is safe because money is per-trader, not shared.
4. **Launch analysis**: when a boundary is crossed, generate the
   launch page: the full case FOR and AGAINST (effect size, CI, regime
   coverage of the sample, what could invalidate it, phased/combo
   options), then the human decides. Losers are retired but keep
   running observationally so regressions stay visible.
5. **Post-launch**: the audit runner's tier metrics are the regression
   watch; any degradation reopens as an incident (SEV ladder below).

Incidents: results/incidents.jsonl + docs/INCIDENTS.md. SEV-0 =
systemic/architectural (2026-08-26, mitigated). SEV-1 = degradation
with mechanism identified (2026-08-28, actions live). Every incident
carries RCA → actions → verification metric, and closes only on the
metric.

## 5 · The LLM-introspection loop (Fable as the analyst)

What "the model reads the model" concretely means here, staying inside
what is real:
- interpretable heads first: every logit/calibrator weight is small
  and named — the introspection team translates them to claims and
  flaw-hunts (wrong-sign weights, market-echo dominance) each wave
- inference-level audit: thought-flow traces (one trade, every number
  from signal to settle) + per-window oracle regret = "what the
  utopian architecture would have earned" — logged, aggregated, and
  rendered, so 'how could this inference have done better' has a
  number, not a vibe
- metric representativeness: each wave, one section asking whether the
  headline metrics still describe behavior (the win-rate leaderboard
  failure — high win%, negative edge — is the canonical example)
- deep nets (t8/t9) get behavioral introspection only (pulls, action
  distributions, val curves) — we do not narrate dense weights,
  because that would be exactly the hallucination we ban

## 6 · Scale architecture (tickers, markets, strategies) — without
touching the hot path

Direction, agreed: everything generalizes; nothing regresses the
current task. The seams, in dependency order:

1. **Namespace the ledgers** (pure rename, zero behavior): every
   ledger/state key gains an instrument prefix (`BTCUSD-15m/...`).
   The current system becomes instance #1 of a general shape.
2. **Config-not-code instruments**: one spec object (tick source,
   contract ladder, session calendar, fee schedule) per market; the
   daemon loop takes the spec. Kalshi-BTC is spec #1.
3. **Shared-nothing hot loops**: one daemon process per instrument
   (isolation = today's performance guaranteed untouched); the
   observation loop, treatments framework, audit runner, and site are
   already instrument-agnostic computations over ledgers — they
   federate by reading more directories.
4. **Knowledge layer goes RAG**: when docs + internals reports span
   instruments, add the vector index over docs/ + model_internals so
   teams retrieve across the fleet. Workflow orchestration (the
   conductor pattern) is already how waves run; formalizing it into
   declarative pipelines happens when wave count demands it.
5. **Strategy library**: arms/traders/treatments as registered plugins
   with the pre-registration metadata machine-readable — the
   DECISIONS.md entry is generated from the registration.

Rule that guards it all: any scale work ships behind the same pipeline
(§4) with the CURRENT instrument's metrics as the guardrail — a scale
refactor that moves any live metric is a failed treatment, reverted.

## 7 · Evolution, not utopia

The environment drifts (measured: calibration (a,b) moved between
halves of a two-day sample; the best fixed arm flipped sign in one
day). Therefore: no terminal architecture. The org's job is a fast,
honest loop — measure, propose, gate, ship, watch — with every step
leaving a record the next team can stand on. Utopia here is a property
of the PROCESS (nothing unverified, nothing unremembered, nothing
unowned), not of any frozen design.
