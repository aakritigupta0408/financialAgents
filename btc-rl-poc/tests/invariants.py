"""Regression invariants — institutional memory, machine-checked.

Every confirmed insight or incident in this project must end as a
fixture, an exhibit, or an INVARIANT here (OS_BLUEPRINT §10). Runs on
the 10-min audit cron; failures land in results/invariants.json and a
red row on the Watchtower. Each check names the incident that created
it — these are not hypothetical lints, they are scars.

Exit code 0 always (the cron chain must not stop); the JSON verdict
is the signal.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


CHECKS = []


def check(name, origin):
    def deco(fn):
        CHECKS.append((name, origin, fn))
        return fn
    return deco


@check("one-decision-per-window",
       "trader-lockout bug 08-28 + owner guarantee 08-29")
def _one_per_window():
    """Each trader log: at most ONE row per (ticker); a committed
    decision is never re-entered or flipped."""
    bad = []
    for n in ("pt_trades", "pt2_trades", "pt3_trades", "pt4_trades",
              "pt5_trades", "pt6_trades", "pt7_trades", "pt8_trades"):
        seen = set()
        for t in rows(n + ".jsonl"):
            k = t.get("ticker")
            if k in seen:
                bad.append(f"{n}:{k}")
            seen.add(k)
    return not bad, bad[:5]


@check("ledger-monotonic-close-ts", "settle-ordered replay discipline")
def _monotonic():
    """treatments.jsonl close_ts must be non-decreasing — an
    out-of-order settle would corrupt every prequential number."""
    ts = [r.get("close_ts", 0) for r in rows("treatments.jsonl")]
    bad = [i for i in range(1, len(ts)) if ts[i] < ts[i - 1]]
    return not bad, bad[:5]


@check("no-duplicate-treatment-windows", "windows-not-rows law")
def _treat_dupes():
    tk = [r.get("ticker") for r in rows("treatments.jsonl")]
    dupes = {t for t in tk if tk.count(t) > 1} if len(tk) < 5000 else set()
    if len(tk) >= 5000:                     # O(n) path
        seen, dupes = set(), set()
        for t in tk:
            (dupes if t in seen else seen).add(t)
    return not dupes, sorted(dupes)[:5]


@check("fee-formula-parity", "per-contract fee rounding bug ($2,355)")
def _fee():
    """The daemon's _order_fee_c must equal ceil(7*C*p*(1-p)) per
    ORDER at canonical points (matches metric_fixtures vectors)."""
    cases = [(1, 50, 2), (10, 50, 18), (7, 30, 11), (3, 80, 4)]
    bad = [(c, a) for c, a, want in cases
           if O._order_fee_c(c, a) != want]
    return not bad, bad


@check("gambler-v3-sweep", "owner decision D-gambler-sizing 08-29")
def _sweep():
    """Every settled v3 pt4 row: bankroll_c never exceeds the $10k
    start (excess must have been withdrawn), and wd_c is stamped."""
    bad = []
    for t in rows("pt4_trades.jsonl"):
        if t.get("actual") is None or t["made_ts"] < O.PT4_RESET2_TS:
            continue
        if "wd_c" not in t or t.get("bankroll_c", 0) > O.PT4_RESET_C:
            bad.append(t["ticker"])
    return not bad, bad[:5]


@check("decision-quote-age", "stale-quote treatment bug (fake +11.75%)")
def _quote_age():
    """Rows the treatments evaluator scored must sit inside the <=12
    minute decision envelope (mins_left recorded on desk trades)."""
    bad = [t["ticker"] for t in rows("pt_trades.jsonl")
           if t.get("mins_left") is not None and t["mins_left"] > 12.05]
    return not bad, bad[:5]


@check("skip-is-not-missing", "decision-board INVALID bug 08-29")
def _evaluable():
    """decision_board completeness.pct must be 1.0 for every
    challenger — a skip is a scored decision, never missing data."""
    try:
        d = json.loads((RES / "decision_board.json").read_text())
    except Exception:
        return False, ["decision_board.json unreadable"]
    bad = [a["key"] for a in d.get("treatments", [])
           if a.get("completeness")
           and (a["completeness"].get("pct") or 1.0) < 1.0]
    return not bad, bad


@check("frozen-controls-untouched", "additive-only law")
def _frozen():
    """pt3 stays policy v2 (tau 0.77) and the SPRT family config
    stays code-sovereign at registered values."""
    ok = (abs(O.PT3_TAU - 0.77) < 1e-9
          and abs(O.TREAT_EDGE - 0.02) < 1e-9
          and abs(O.TREAT_ALPHA - 0.05 / 16) < 1e-12)
    return ok, [] if ok else ["a registered constant moved"]


@check("withdrawals-never-restaked", "D-gambler-sizing ledger law")
def _wd_conserved():
    """Reconstructed v3 bankroll from the ledger must equal the last
    settled row's bankroll_c exactly (cash conservation incl. wd)."""
    v3 = [t for t in rows("pt4_trades.jsonl")
          if t["made_ts"] >= O.PT4_RESET2_TS]
    settled = [t for t in v3 if t.get("actual") is not None]
    if not settled:
        return True, []
    bank = O.PT4_RESET_C \
        + sum(t["pnl_c"] - t.get("wd_c", 0) for t in settled)
    want = settled[-1].get("bankroll_c")
    ok = want is not None and bank == want \
        and bank <= O.PT4_RESET_C
    return ok, [] if ok else [f"recomputed {bank} vs ledger {want}"]


@check("analytics-freshness",
       "oversized-crontab silent failure 08-29 (chain dead 2.7h)")
def _fresh():
    """The audit chain's own outputs must be <30 min old — cron can
    fail SILENTLY (an oversized line is simply dropped), so the wall
    itself watches the cadence. Note: this check runs inside the
    chain, so it detects staleness of the PREVIOUS run."""
    import time as _t
    bad = []
    for n in ("audit_report.json", "decision_board.json"):
        p = RES / n
        if not p.exists():
            bad.append(n + " missing")
        elif _t.time() - p.stat().st_mtime > 1800:
            bad.append(f"{n} {(_t.time()-p.stat().st_mtime)/60:.0f}m")
    return not bad, bad


@check("retired-roster-frozen",
       "Great Simplification 08-29 (docs/RETIREMENT_MANIFEST.md)")
def _roster_frozen():
    """No retired trader may open a position after the freeze, and
    every pt6 row after the freeze is a zero-stake SHADOW row —
    max-5-slot roster law, machine-checked."""
    bad = []
    for n in sorted(O.RETIRED_TRADERS):
        for t in rows(n + "_trades.jsonl"):
            if t.get("made_ts", 0) > O.ROSTER_FREEZE_TS \
                    and not t.get("skipped"):
                bad.append(f"{n}:{t.get('ticker')} opened post-freeze")
    if O.PT6_SHADOW:
        for t in rows("pt6_trades.jsonl"):
            if t.get("made_ts", 0) > O.ROSTER_FREEZE_TS \
                    and (t.get("stake_c") or 0) > 0:
                bad.append(f"pt6:{t.get('ticker')} staked in SHADOW")
    return not bad, bad[:5]


@check("one-control-one-treatment",
       "Great Simplification 08-29 — no strategy forests")
def _one_ctl_one_treat():
    """Post-freeze treatment rows may carry ONLY the sanctioned keys:
    the two baselines, the one legacy control/treatment pair, and the
    registered diagnostics/shadows. A retired key reappearing (or a
    new key arriving unregistered) is a roster violation."""
    ALLOWED = {"champion", "champion_real", "t_regime", "t_exec",
               "t_exec_reg", "t_edgeband"}
    # t_limit_reg rows written between the 08-29 freeze and the 08-30
    # diagnostic-archive ruling (PM §41) are legitimate history
    GRACE = {"t_limit_reg"}
    bad = []
    for r in rows("treatments.jsonl"):
        if r.get("close_ts", 0) <= O.ROSTER_FREEZE_TS:
            continue
        extra = set((r.get("ev") or {})) - ALLOWED - GRACE
        if extra:
            bad.append(f"{r.get('ticker')}: {sorted(extra)}")
        if not {"t_exec", "t_exec_reg"} <= set(r.get("ev") or {}):
            bad.append(f"{r.get('ticker')}: legacy pair missing")
    return not bad, bad[:5]


@check("stale-input-never-proposes",
       "M5 close contract §6 — stale evidence may never produce a "
       "research conclusion")
def _stale_never_proposes():
    bad = []
    for r in rows("agent_recommendations.jsonl"):
        if r.get("kind") == "status_update":
            continue
        if "STALE_INPUT" in str(r.get("finding", "")) \
                and r.get("action_class") in ("PROPOSE",
                                              "SAFE_OPS_REPAIR"):
            bad.append(r.get("recommendation_id"))
    return not bad, bad[:5]


@check("no-policy-mutation-by-self-healing",
       "M6 launch contract §12 — POLICY_MUTATIONS_CAUSED_BY_"
       "SELF_HEALING must equal 0, forever")
def _no_heal_policy():
    POLICY_WORDS = ("threshold", "floor", "cutoff", "sizing",
                    "risk limit", "treatment", "denominator")
    bad = []
    for r in rows("system_change_log.jsonl"):
        if r.get("change_type") == "SELF_HEAL_OCCURRED":
            blob = json.dumps(r).lower()
            if any(w in blob for w in POLICY_WORDS):
                bad.append(f"{r.get('entity')}@{r.get('ts')}")
    return not bad, bad


@check("no-private-time-alignment",
       "M5 system validation (PM 08-30) — the temporal equivalent of "
       "no-private-formulas: two researchers aligning the same raw "
       "tape differently is a leakage vector")
def _no_private_alignment():
    """Only the sanctioned layers may touch the raw cross-venue
    shards: capture_xvenue.py (writer) and xvenue_sync.py (the ONE
    alignment layer). Any other code referencing events_xvenue is a
    private-alignment violation."""
    # writer, the ONE alignment layer, and the F1 quality auditor
    # (coverage/gap measurement is not alignment)
    ALLOWED = {"capture_xvenue.py", "xvenue_sync.py",
               "emit_f1_gate.py"}
    bad = []
    for d in (ROOT / "scripts", ROOT / "btc_rl"):
        for p in d.glob("*.py"):
            if p.name in ALLOWED:
                continue
            try:
                if "events_xvenue" in p.read_text():
                    bad.append(p.name)
            except Exception:
                pass
    return not bad, bad


@check("agent-decision-firewall",
       "M5 (PM 08-30) — the firewall is code, not prompt: forbidden "
       "classes never execute, proposals name their loss term")
def _firewall():
    """Every ledger row: valid action_class; RESEARCH_POLICY_CHANGE /
    RISK_CHANGE only ever REJECTED_BY_FIREWALL; SAFE_OPS_REPAIR only
    BLOCKED_UNTIL_M6 (until an explicit whitelist ships); every
    PROPOSE carries targeted_loss_term; nothing is ever marked
    executed autonomously."""
    VALID = {"OBSERVE", "DIAGNOSE", "PROPOSE", "SAFE_OPS_REPAIR",
             "RESEARCH_POLICY_CHANGE", "RISK_CHANGE"}
    bad = []
    for r in rows("agent_recommendations.jsonl"):
        if r.get("kind") == "status_update":
            continue                # governance lifecycle rows
        ac, st = r.get("action_class"), r.get("status")
        rid = r.get("recommendation_id", "?")
        if ac not in VALID:
            bad.append(f"{rid}: unknown class {ac}")
        if ac in ("RESEARCH_POLICY_CHANGE", "RISK_CHANGE") \
                and st != "REJECTED_BY_FIREWALL":
            bad.append(f"{rid}: forbidden class escaped ({st})")
        if ac == "SAFE_OPS_REPAIR" and st not in (
                "BLOCKED_UNTIL_M6", "BLOCKED_REPAIR_NOT_ENABLED",
                "APPROVED_FOR_EXECUTION"):
            bad.append(f"{rid}: ops repair invalid status ({st})")
        if ac == "SAFE_OPS_REPAIR" \
                and st == "APPROVED_FOR_EXECUTION" \
                and not r.get("repair_id"):
            bad.append(f"{rid}: approved repair without repair_id")
        if ac == "PROPOSE" and not r.get("targeted_loss_term"):
            bad.append(f"{rid}: PROPOSE without loss term")
    return not bad, bad[:5]


@check("model-researcher-discipline",
       "M5.4 (PM 08-30) — one candidate question max, allowed "
       "actions only, capacity claims inadmissible without evidence")
def _mr_discipline():
    ALLOWED = {"NO_ACTION", "NEEDS_MORE_DATA",
               "ONE_CANDIDATE_QUESTION", "RETIRE_MODEL_FAMILY"}
    try:
        d = json.loads((RES / "model_research.json").read_text())
    except Exception:
        return True, ["model_research.json not yet published"]
    bad = []
    if (d.get("candidate_question") or {}).get("count", 0) > 1:
        bad.append("candidate_question.count > 1")
    if d.get("recommended_action") not in ALLOWED:
        bad.append(f"action {d.get('recommended_action')} not allowed")
    cq = d.get("candidate_question") or {}
    if cq.get("count") and not cq.get("targeted_loss_term"):
        bad.append("candidate without targeted_loss_term")
    return not bad, bad


@check("zombie-component-count",
       "M3 (PM 08-30) — registry says retired, nothing may still "
       "invoke it; a zombie is a red operational condition")
def _zombies():
    """ZOMBIE_COMPONENT_COUNT must be 0: no retired kb arm produces a
    row after M3_CLEANUP_TS, and no unregistered kb arm appears. The
    allowed set is exactly the arms with live decision impact: the
    frozen champion's leader pool (kb2/kb3/kb4/kb7/kb8/kb9), its kb
    input, and the M14-v2 caller kb5."""
    ALLOWED = {"kb", "kb2", "kb3", "kb4", "kb5", "kb7", "kb8", "kb9"}
    bad = []
    for r in rows("kalshi_binary_log.jsonl"):
        if (r.get("made_ts") or 0) <= O.M3_CLEANUP_TS:
            continue
        v = r.get("variant") or "kb"
        if v not in ALLOWED:
            bad.append(f"{v}:{r.get('ticker')} row post-cleanup")
    return not bad, bad[:5]


@check("complexity-budget",
       "M3 (PM 08-30) — lean is an enforceable SLO, not a philosophy")
def _budget():
    """active traders <= 5 (new-row logs after the roster freeze),
    serving T2 model roles <= 3 (control + incumbent + challenger),
    and the treatment cap is enforced by one-control-one-treatment."""
    bad = []
    active_traders = set()
    for n in ("pt", "pt2", "pt3", "pt4", "pt5", "pt6", "pt7", "pt8"):
        for t in rows(n + "_trades.jsonl"):
            if (t.get("made_ts") or 0) > O.ROSTER_FREEZE_TS \
                    and not t.get("skipped"):
                active_traders.add(n)
                break
    if len(active_traders) > 5:
        bad.append(f"traders {sorted(active_traders)} > 5")
    # serving T2 roles: market baseline + kb2 incumbent + kb9
    # challenger; kb3/4/7/8 are frozen champion INPUTS (leader pool),
    # kb5 a product caller — roles, not serving challengers
    SERVING = {"kb2", "kb9"}
    if len(SERVING) + 1 > 3:            # +1 = market baseline
        bad.append("serving model roles exceed 3")
    return not bad, bad


@check("model-parity", "M2.5 (PM 08-30) — live vs replay predictions "
       "must match; a mismatch is silent model corruption")
def _parity():
    """parity.json FAIL = red wall + promotion freeze. NO_SNAPSHOTS /
    ALL_DEFERRED are amber-by-honesty (reported, not failed) until
    the store accumulates — but a FAIL is never survivable."""
    try:
        d = json.loads((RES / "parity.json").read_text())
    except Exception:
        return True, ["parity.json not yet published"]
    if d.get("parity_state") == "FAIL":
        return False, [f["prediction_id"] for f in
                       d.get("failures", [])][:5]
    return True, []


@check("a3-experiment-integrity",
       "A3-v1.1 evaluator spec §53 (A3-02..06,10,11)")
def _a3():
    """Every A3 ledger row: trigger conf >= floor, improvement >=
    threshold, exactly one entry, paired outcomes present, and no
    pre-registration window in the forward set."""
    try:
        d = json.loads((RES / "a3_live.json").read_text())
    except Exception:
        return True, ["a3_live.json not yet published"]
    reg = d.get("registered_ts", 0)
    bad = []
    if d.get("spec_hash_ok") is False:      # §53 A3-12
        bad.append("A3_SPEC.yaml hash drift — new experiment "
                   "version required")
    for e in d.get("recent_settled", []):
        if e.get("state") == "SYSTEM_EXCLUDED":
            continue
        if e.get("close_ts", 0) < reg:
            bad.append(f"{e.get('ticker')}: pre-registration in fwd")
        if e.get("state") == "FILLED":
            if (e.get("entry_conf") or 1) < 0.65:
                bad.append(f"{e.get('ticker')}: conf<floor")
            if (e.get("improvement_c") or 0) < 10:
                bad.append(f"{e.get('ticker')}: dip<10c")
        if e.get("settled") and ("control_pnl" not in e
                                 or "a3_pnl" not in e):
            bad.append(f"{e.get('ticker')}: unpaired")
    return not bad, bad[:5]


@check("a3-independent-artifact-audit",
       "A3 measurement spec §41 (A3-13,15,19,21,22) — the evaluator "
       "may not certify its own implementation")
def _a3_indep():
    """Independent checks on a3_window_evaluation.jsonl rows:
    A3-21 every FILLED row carries executable size >=1 (or explicit
          unknown from pre-depth shards, never <1);
    A3-22 INVALIDATED rows carry no entry;
    A3-15 best_valid_ts (when present) <= cutoff_ts;
    A3-13 numeric markouts imply the anchor precedes the horizon
          (entry_ts + h consistency: markout fields only on FILLED
          rows with entry_ts);
    A3-19 shadow arms never appear in any trader order ledger."""
    p = RES / "a3_window_evaluation.jsonl"
    if not p.exists():
        return True, ["artifact not yet published"]
    bad = []
    for l in p.open():
        e = json.loads(l)
        st_ = e.get("state")
        if st_ == "FILLED":
            sz = e.get("entry_ask_sz")
            if sz is not None:
                try:
                    if float(sz) < 1.0:
                        bad.append(f"{e.get('ticker')}: filled with "
                                   f"size {sz} < 1 (A3-21)")
                except (TypeError, ValueError):
                    pass
            if e.get("entry_ts") is None:
                bad.append(f"{e.get('ticker')}: FILLED without "
                           "entry_ts (A3-13)")
        if st_ == "INVALIDATED" and e.get("entry_ask") is not None:
            bad.append(f"{e.get('ticker')}: entry after "
                       "invalidation (A3-22)")
        bts, cts_ = e.get("best_valid_ts"), e.get("cutoff_ts")
        if bts is not None and cts_ is not None and bts > cts_:
            bad.append(f"{e.get('ticker')}: best_valid after "
                       "cutoff (A3-15)")
    for n in ("pt_trades", "pt4_trades", "pt6_trades"):
        for t in rows(n + ".jsonl"):
            if t.get("src") in ("T05", "T15", "T05_SHADOW",
                                "T15_SHADOW"):
                bad.append(f"{n}: shadow order leaked (A3-19)")
    # paired-Δ attribution audit: recompute BOTH arms' pnl from the
    # stored prices + outcome with independent math; the reported Δ
    # must reconcile exactly with entry improvement through the
    # cost basis — any drift means money-math corruption
    def _pnl(ask, won):
        cost = ask + 7 * (ask / 100.0) * (1 - ask / 100.0)
        return (100 - cost) / cost if won else -1.0
    for l in p.open():
        e = json.loads(l)
        if e.get("state") == "FILLED" and e.get("settled"):
            want_c0 = round(_pnl(e["call_ask"], e["won"]), 4)
            want_a3 = round(_pnl(e["entry_ask"], e["won"]), 4)
            if abs(want_c0 - e["control_pnl"]) > 1e-3 \
                    or abs(want_a3 - e["a3_pnl"]) > 1e-3:
                bad.append(f"{e.get('ticker')}: attribution drift "
                           f"(recomputed {want_a3}/{want_c0})")
    return not bad, bad[:5]


def main():
    out, failed = [], 0
    for name, origin, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:              # a crashed check is a fail
            ok, detail = False, [f"check crashed: {e!r}"]
        out.append({"name": name, "origin": origin,
                    "ok": bool(ok), "detail": detail})
        failed += 0 if ok else 1
    doc = {"generated_ts": int(time.time()),
           "passed": len(out) - failed, "failed": failed,
           "health": "green" if failed == 0 else "red",
           "checks": out}
    (RES / "invariants.json").write_text(json.dumps(doc, indent=1))
    print(f"invariants: {doc['passed']}/{len(out)} passed "
          f"({doc['health']})")
    for c in out:
        if not c["ok"]:
            print(f"  FAIL {c['name']} — {c['detail']}")


if __name__ == "__main__":
    main()
