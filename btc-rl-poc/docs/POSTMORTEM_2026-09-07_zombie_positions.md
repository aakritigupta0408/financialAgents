# Postmortem — Zombie Positions (bankroll-conservation SEV-1)

**Date opened:** 2026-09-07 (auditor flag; fail-closed active since ~2026-09-03)
**Severity:** SEV-1 (contained; scientific record not invalidated)
**Status:** CLOSED (2026-09-07, live-verified)
**Remediation class:** `FIELD_CORRECTION_PROVABLE / LATE_SETTLEMENT`

## One-sentence lesson

> A rolling market-data window was accidentally being used as an
> implicit bound on how long a position could remain settleable. A
> long machine sleep violated that assumption. The accounting system
> then correctly represented cash, while the lifecycle system failed
> to complete two positions. Reconciliation exposed the
> contradiction.

**Architectural defect:** *settlement completeness must not depend on
routine observation-window length.*

## What happened (timeline)

1. Two paper positions were open (`pt` KXBTC15M-26AUG311800-00, stake
   1690¢; `pt3` KXBTC15M-26SEP062345-45, stake 2750¢).
2. The machine slept — 19.8h (pt) and 12.6h (pt3), both longer than
   the daemon's routine candle-fetch window (`BACKFILL_HOURS`=6 +1h).
3. The positions matured while asleep; on wake the rolling
   `fetch_range(now-7h, now)` could never again contain their settle
   candle `[close_ts-60]`.
4. The rows stayed OPEN forever ("zombies"); their stakes stayed
   debited from cash.
5. Every later `bankroll_c` stamp was a correct **cash** value, lower
   than the reconciliation **settled-P&L** walk by exactly the locked
   stake.
6. `reconcile.py` flagged the two residuals (−1690¢, −2750¢). The
   desk went `FREEZE_NEW_ENTRIES` (fail-closed) — correct behavior.
   Treatments froze at n=374.

## Root cause (proven, with falsifications)

Mechanism: sleep > rolling-window ⇒ permanently unsettleable open
rows. Falsified alternatives:
- **torn writes** — saves are atomic (tmp+rename); ruled out.
- **duplicate writers** — single daemon; no duplicate ticker rows.
- **stale bankroll stamp** — every stamp is cash-exact.
- **double stake debit** — none.
- **same class as the Aug-26 ±11484 pair** — no; that pair is the
  auditor's legitimate in-flight forgiveness (walk reconverges on the
  next row). The zombie offset is permanent and never reconverges.

Both zombies **provably WON** against the authoritative vendor candle
(pt: close 78946.11 < strike 79023.40; pt3: 79624.66 < 79737.04).

## Remediation — LATE SETTLEMENT, not history rewrite

This is authoritative **late settlement of a historically unresolved
position**, NOT an edit of the past:
- entry record and all historical `bankroll_c` stamps: **untouched**;
- the missing lifecycle event is **appended** — outcome filled, and
  the payout credited to cash at `late_settle_ts` (the moment the
  correction became known), by the daemon (single writer);
- `reconcile.py` implements the matching semantics: the preserved
  stamp is entry-time cash; the payout is credited at
  `late_settle_ts`, not at the historical position.

"Aren't you changing the past?" → No. The position was genuinely
unresolved at the time; historical cash stamps stand. Once the
authoritative settlement observation was recovered, the daemon
appended a late settlement dated when it became known.

## The recurrence fix — pointed at the right layer

We did **not** bump `BACKFILL_HOURS` to some arbitrary large number
(a patch, not a fix). Instead: **settlement requirements determine the
backfill, not the other way around** — the daemon detects any matured
unresolved row whose settle candle is outside the routine window and
issues a *targeted* one-shot fetch for exactly that candle, retrying
on transient failure and de-duplicating only once the candle is in
hand (see "second bug" below).

### A second bug, caught by live verification

The first fix marked a `close_ts` as backfilled *before* the targeted
fetch, so a single transient rate-limit permanently skipped it — a
self-inflicted second zombie. Live verification (75-min timeout, not
a test) exposed it. Corrected: fetch first, dedup only on the candle
actually arriving, retry otherwise. This is precisely why the SEV was
not closed on green unit tests.

## Recovery contract strengthened (§26 law)

RESTORED evolved from `process healthy` to
`liveness + freshness + lifecycle completeness + accounting
conservation`. `meta_monitor` now refuses RESTORED while
bankroll-conservation fails or any matured unresolved position exists
(>2h past close). Operational recovery and state recovery are
distinct properties.

## Guards & tests

- `tests/test_bankroll_recovery.py` — 4/4 (detection preserved,
  late-settle reconciles, normal unchanged, audit idempotence).
- Invariant `registered-ledger-never-shrinks` (pre-existing) +
  new `publisher-page-coverage`; suite 26/26.
- Chaos matrix (M6-R1) 12/12.

## The three ideas this incident demonstrates

- **Safety** — the system stopped itself when accounting integrity
  failed, before producing more questionable evidence.
- **Recovery** — it found the missing authoritative information and
  completed the unresolved lifecycle event without rewriting history.
- **Idempotence** — repeating recovery cannot repeat the economic
  effect.

## Live verification chain — CLOSURE GATE (all proven live)

- [x] both zombies late-settle — pt WON pnl 810¢, pt3 WON pnl 1150¢
- [x] matured-unresolved staked positions = 0
- [x] bankroll residual = 0 · live reconcile = PASS (overall OK,
      bankroll-conservation 0/1941)
- [x] verification restart → rows byte-identical, reconcile OK — NO
      second payout, NO replay, NO bankroll change
- [x] machine-controlled reopen — FREEZE cleared automatically when
      reconciliation went OK; no human flipped the switch
- [x] first reopened window opens/settles/reconciles → champion
      n=374 → 375, reconcile OK (SEV interval recorded as a
      zero-exposure pause; no backfill, no reset)
- [x] SEV-1 CLOSED + REGRESSION_INVARIANT (`no-zombie-positions`,
      suite 27/27; `test_bankroll_recovery` 4/4)

## Root cause was found in FOUR layers (verification discipline)

Three hypotheses were wrong before instrumentation found the truth,
and the SEV never closed on green unit tests:
1. fail-closed gate skipping settlement — FALSIFIED (gate blocks
   only entries, not settles);
2. dedup-before-fetch permanently skipping a transient failure —
   real bug, fixed, but not the whole story;
3. `_merge_synth` rebuilding `by_ts` from `bars` AFTER the backfill,
   discarding recovered candles — THE root cause, found by a
   one-loop debug print after the standalone repro kept passing
   (it never exercised the `_merge_synth` rebuild the live loop did);
4. a fourth, 16-day-old zombie in a *different* ledger (kb_bets),
   surfaced by the new permanent invariant — turned out non-staked
   (economically inert), which sharpened the invariant's scope.

Lesson: a standalone reproduction that omits a production code path
is not a live verification. Instrument the real loop early.
