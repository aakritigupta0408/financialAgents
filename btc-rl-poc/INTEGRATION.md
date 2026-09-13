# LIVE INTEGRATION PACKAGE — main-checkout handoff

This research worktree cannot modify the running checkout. Everything here is
merge-ready and **safe to merge while disabled**. Nothing changes desk
behaviour until the feature flag / scheduler calls are explicitly turned on.

Branch: `worktree-btc-rl-poc` · all modules are new files (no edits to the
live decision path), so a merge cannot alter existing behaviour on its own.

---

## 1. What's in the package (new files only)

| File | Purpose | Tested |
|---|---|---|
| `btc_rl/exec_timing.py` | EXEC-TIMING treatment: pure, feature-flagged decide() | `tests/test_exec_timing.py` 6/6 |
| `research/t05_repricing/exec_timing_model.json` | frozen predictor (β/μ/σ, threshold) | — |
| `EXEC_TIMING_SPEC.yaml` | immutable treatment spec (hash `9c0c89ae`) | — |
| `data/adapters/derivatives.py` | OKX funding/OI/basis/long-short capture | live smoke: CONNECTED |
| `data/adapters/av_options.py` | AV REALTIME_OPTIONS → ATM IV/skew | endpoint verified |
| `data/adapters/av_news.py` | AV NEWS_SENTIMENT structured arrival | endpoint verified |
| `btc_rl/preopen_oracle.py` | pre-open snapshot capture (Kalshi-leak-guarded) | self-test OK |
| `research/research_state.json` | canonical ladder/bottleneck state | — |

## 2. EXEC-TIMING treatment — wiring

Hook point: the paper trader's per-window entry decision (`btc_rl/online.py`,
the block that currently emits ENTER_NOW/SKIP for the champion path).

```python
from btc_rl import exec_timing
# after the frozen control decision (side, size, control_action) is computed:
action, meta = exec_timing.decide(control_action, side, t05_features)
# t05_features = the same microstructure dict used for T0.5
# record BOTH arms for the paired experiment (see spec §4 identifiers)
```

- Flag: env `EXEC_TIMING_TREATMENT_ENABLED` (default `false`).
- When false, `decide()` returns `control_action` unchanged (proved by tests).
- `action == "DEFER"` → re-enter the same side after `defer_horizon_s` (15s),
  still subject to the existing entry cutoff. Never changes side/size/risk.
- Log per window: `market_window_id, experiment_id=EXEC-TIMING, spec_hash,
  control_decision, treatment_decision, frozen_side, T0.5 prediction, action,
  executable price, fill, settlement` → a new `results/exec_timing_ab.jsonl`.

## 3. Prospective capture — scheduler calls

Add to the existing capture cron/loop (all optional; an outage must not touch
core BTC/Kalshi capture — each is wrapped and writes its own `*_health.json`):

```python
from data.adapters import derivatives, av_options, av_news
derivatives.capture_once()          # every 15-30s
av_news.capture_once()              # every 2-5 min (AV 25/day free tier — budget it)
av_options.capture_once()           # a few times per RTH session
```

## 4. Pre-open Oracle — scheduler calls

```python
from btc_rl import preopen_oracle
for expiry, lead in preopen_oracle.due_snapshots(now):
    preopen_oracle.snapshot(expiry, lead, btc_state, deriv_state)  # NO kalshi
# when the contract opens and strike is known:
preopen_oracle.attach_strike(expiry, strike)
# after settlement:
preopen_oracle.attach_settlement(expiry, settlement_0_or_1)
```

## 5. Activation sequence (in the main checkout)

1. Merge/cherry-pick this branch.
2. `python3 tests/invariants.py` and `python3 tests/test_exec_timing.py` → green.
3. Deploy with `EXEC_TIMING_TREATMENT_ENABLED` unset (treatment OFF) and no
   scheduler calls added yet → verify heartbeat + existing capture unchanged.
4. Add the capture scheduler calls (§3, §4); verify `deriv_health.json`,
   `news_health.json`, `options_health.json`, `preopen_oracle.jsonl` accrue.
5. Add the EXEC-TIMING hook (§2), still flag-OFF; verify control path
   decision-for-decision unchanged in a shadow run.
6. Set `EXEC_TIMING_TREATMENT_ENABLED=true`; verify the first paired window in
   `exec_timing_ab.jsonl` (same side both arms, differing only on timing).

## 6. Rollback

- Treatment: unset/`false` the env flag → instant no-op (byte-equivalent).
- Capture: remove the scheduler calls; adapter files are inert if not called.
- Full: revert the merge commit(s). No historical ledgers are mutated; all new
  files write to their own append-only logs.

## 7. Safety invariants preserved

- Real-money execution remains physically disabled (no order-submit path added).
- No edits to existing decision/settlement/bankroll code — new files only.
- Optional sources fail closed (explicit missingness; never forward-filled;
  never crash core capture).
- One formal treatment only (EXEC-TIMING); the Independent Oracle stays DEV.
- Pre-open Oracle rejects Kalshi inputs at the API boundary (assertion).

---

## Exact-BRTI Oracle + Prospective Capture (merge-ready, DISABLED by default)

New files only; inert until env flags are set in the main checkout.

**Contract truth = BRTI.** Credential bundle `~/.kalshi_key_api` (RSA PEM + key-id
UUID, parsed apart by `data/adapters/brti.py`). Endpoints verified:
`cfbenchmarks/values?id=BRTI` (rolling 1h @5Hz) and `cfbenchmarks/history/values`
(`timespan=HOUR`, hour-truncated ISO `timestamp`; full hour/call).

Frozen Independent Oracle: `research/oracle/oracle_frozen.json`
(MECH_FAIR_BRTI sigma + isotonic recalibration; hash `71c3bcffa964`). No Kalshi
input by construction.

### Activation (owner, main checkout)
1. Confirm BRTI health: `python3 data/adapters/brti.py` -> `AVAILABLE`.
2. Wire live capture into the desk's decision loop:
   `from btc_rl.prospective_capture import capture; capture(window_ctx)`
   where `window_ctx` carries decision-time-only state (BRTI, official_target,
   time_remaining_s, k_prob). A leak-guard refuses any settlement/outcome field.
3. Enable: `export PROSPECTIVE_CAPTURE_ENABLED=1` (default off = no-op).
4. After settlement, the desk appends the official `exact_yes` to each record.
5. Records land in `results/prospective_capture.jsonl` (append-only, PIT).

### What it confirms
- The PROMISING_PENDING_PROSPECTIVE disagreement edge (retrospective n=111 only).
- Families B-E (multi-venue / derivatives / options / news) that could not be
  tested historically — captured live for the first true OOS test.

### Degradation (§40-41)
If live BRTI is unhealthy, records are written with
`contract_state_quality=DEGRADED|PROXY`, never a silent Coinbase substitution.

---

## DT-01 — Exact-BRTI runtime contract truth (merge-ready, flag OFF by default)

Closes the split-brain where research used exact CF-BRTI but the daemon settled on
a Coinbase-candle / 4-venue-composite proxy. All 11 settlement sites in
`btc_rl/online.py` now route through `contract_truth.resolve_outcome(...)`:

- flag OFF → returns the legacy Coinbase-candle outcome (**merge is a no-op**)
- flag ON + BRTI EXACT_BRTI → exact 60s-average outcome (YES iff close_avg >= open_avg)
- flag ON + BRTI incomplete → legacy outcome tagged `PROXY_DEGRADED` (never silent)

`contract_truth_quality` is stamped on every settled row; a legacy/exact/official
shadow row is logged to `results/settlement_shadow.jsonl` while the flag is ON.

### Activation (owner, main checkout) — never needs a code edit
1. Deploy code (flag off). Smoke: daemon heartbeat + invariants healthy.
2. `python3 scripts/emit_brti_health.py` → `results/brti_runtime_health.json` shows
   `connected`, cadence, `history_ok`.
3. `export EXACT_BRTI_CAPTURE_ENABLED=1` — verify BRTI state populates.
4. `export EXACT_BRTI_RUNTIME_ENABLED=1` — verify the first window settles at
   `contract_truth_quality=EXACT_BRTI` and `settlement_shadow.jsonl` exact==official.
5. `python3 scripts/architecture_checkpoint.py` → `RUNTIME_CONTRACT_TRUTH=PASS`.

Rollback: unset `EXACT_BRTI_RUNTIME_ENABLED` → legacy settlement, no code change.
Golden + shadow-parity + failover tests: `tests/test_contract_truth.py` (5/5).
Declaration + deploy sequence: `architecture/change_impact.json`.
