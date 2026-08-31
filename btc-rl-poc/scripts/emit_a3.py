"""A3 forward evaluator — ACTIVE: A3-v2 (registered 2026-08-30).

A3-v1.1 (10c dip) ran 2026-08-29..30, was REJECTED at its registered
gate and is CLOSED; its frozen evidence lives in a3_v1_final.json /
a3_v1_window_evaluation.jsonl / a3_v1_decision.json. v2 changes ONE
variable (dip 10c -> 5c) per A3_V2_SPEC.yaml.

v1.1 supersedes the v0 shortcut REGISTERED same evening: v0 used
modeled asks (100p+2.5), which §4 forbids ("never model-derived
synthetic price"). Per §48 a price-source change creates a new
registration with a clean forward counter. v0's ledger is retained
as exploratory replay only (§53 A3-11).

Hybrid tape (both point-in-time):
  * THESIS/CONFIDENCE: kb2 minute rows (the model updates per
    minute — using its step function at finer resolution is honest);
  * EXECUTABLE PRICES: the Layer-A event tape (results/events/,
    ~1s Kalshi quotes with receive_ts) — real side asks, never mid,
    never modeled. Windows whose event coverage is missing are
    SYSTEM_EXCLUDED (§7), not guessed.

Frozen v1.1 parameters (== A3_SPEC.yaml; invariant A3-12 checks):
  CALL_CONF 0.75 · envelope mins_left [6,13] · FLOOR 0.65
  DIP_C 10 · ENTRY_CUTOFF_SECONDS 60 · one primary entry ·
  unit size 1 contract · quote freshness gate 10s.
State machine (§7): REGISTERED -> INVALIDATED | TRIGGERED->FILLED
  (v1: fill == take the ask, qty 1 — no depth data yet, recorded as
  convention) | MISSED | SYSTEM_EXCLUDED.
Primary (§10): sum(PnL_A3 - PnL_C0) / N_eligible, every eligible
window counted, misses = 0 trading P&L. Control fills at the real
call-time ask (§9, same realism).
"""
from __future__ import annotations

import glob
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
# ================= ACTIVE EXPERIMENT: A3-v2 =========================
# A3-v1.1 was REJECTED at its registered gate (n=56, CI below zero)
# and CLOSED by PM ratification 2026-08-30. Its evidence is frozen
# forever in results/a3_v1_final.json (sha 22014cee8ebe2b82) +
# a3_v1_window_evaluation.jsonl (sha 669bf952ace24ceb) +
# a3_v1_decision.json. Reopen: NEVER as v1.1.
# v2 changes EXACTLY ONE thing: dip threshold 10c -> 5c. Everything
# else identical; fresh forward counter; no historical shadow row
# counts toward the v2 statistic (A3_V2_SPEC.yaml).
EXPERIMENT_ID = "A3-v2"
CALL_CONF, FLOOR, DIP_C = 0.75, 0.65, 5.0
ENV_LO, ENV_HI = 6.0, 13.0
CUTOFF_S = 60
FRESH_S = 10.0
V2_REGISTERED_TS = 1788149400    # 2026-08-30 — PM ratification
SPEC_FILE = "A3_V2_SPEC.yaml"
SPEC_HASH_FROZEN = "ab0168b48c6ba794"   # A3_V2_SPEC.yaml at freeze
# Shadow: T10 only — the rejected v1.1 rule as a diagnostic
# continuity benchmark. T15 retired (deep-wait mechanism understood;
# no threshold ladder zoo).
SHADOWS = {"T10": 10.0}
SHADOW_REGISTERED_TS = V2_REGISTERED_TS
MARKOUT_H = (1, 5, 10, 30, 60)
LEDGER_NAME = "a3v2_window_evaluation.jsonl"


def fee(a):
    return 7 * (a / 100.0) * (1 - a / 100.0)


def pnl(ask, won):
    cost = ask + fee(ask)
    return (100 - cost) / cost if won else -1.0


def load_events():
    """ticker -> sorted [(receive_ts, yes_ask, no_ask...)] from the
    event tape (kalshi quotes only)."""
    ev = {}
    for f in sorted(glob.glob(str(RES / "events" / "*.jsonl"))):
        for l in open(f):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("src") != "kalshi" or r.get("yes_ask") is None:
                continue
            ev.setdefault(r["ticker"], []).append(
                (r["receive_ts"], r["yes_ask"], r.get("no_ask"),
                 r.get("close_time"), r.get("yes_bid"),
                 r.get("no_bid"), r.get("yes_ask_sz"),
                 r.get("yes_bid_sz")))
    for tk in ev:
        ev[tk].sort()
    return ev


def side_ask(q, up):
    return q[1] if up else (q[2] if q[2] is not None else 100 - q[1])


def side_bid(q, up):
    b = q[4] if up else q[5]
    return b


def side_depth(q, up):
    """Executable size at the SIDE ask. NO-side ruling
    (A3_CHANGE_CONTROL.md): no_ask ≡ 100−yes_bid natively, so its
    depth is yes_bid_size. Returns float or None (unknown ≠ zero)."""
    raw = q[6] if up else (q[7] if len(q) > 7 else None)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def depth_ok(q, up):
    d = side_depth(q, up)
    return d is None or d >= 1.0   # unknown passes (old shards); <1 fails


def markouts(q, up, t0, price):
    """§9-10: conservative convention — mark against the EXECUTABLE
    SIDE BID at t0+h (if we had to exit, what would the market pay);
    lookup = first quote AT OR AFTER target within MARKOUT_MAX_DELAY
    (2s); never forward-filled; UNAVAILABLE otherwise."""
    out = {}
    for h in MARKOUT_H:
        tgt = t0 + h
        near = next((x for x in q
                     if tgt <= x[0] <= tgt + 2.0
                     and side_bid(x, up) is not None), None)
        out[f"markout_{h}s"] = (round(side_bid(near, up) - price, 1)
                                if near else "UNAVAILABLE")
    return out


def run_shadow(dip_c, q, up, reg_ts, call_ask, conf_at, cutoff_ts):
    """Same machine at a different dip threshold — observation only."""
    for x in q:
        if x[0] <= reg_ts or x[0] > cutoff_ts:
            continue
        if conf_at(x[0]) < FLOOR:
            return {"state": "INVALIDATED"}
        a = side_ask(x, up)
        if call_ask - a >= dip_c and depth_ok(x, up):
            return {"state": "FILLED", "entry_ask": round(a, 1),
                    "entry_ts": x[0],
                    "entry_ask_sz": side_depth(x, up),
                    "improvement_c": round(call_ask - a, 1)}
    return {"state": "MISSED"}


def main():
    now = int(time.time())
    rows = [json.loads(l) for l in
            (RES / "kalshi_binary_log.jsonl").open() if l.strip()]
    kb2 = {}
    for r in rows:
        if (r.get("variant") or "kb") != "kb2":
            continue
        if r.get("p_up") is None or r.get("mins_left") is None:
            continue
        kb2.setdefault(r["ticker"], []).append(r)
    events = load_events()

    ledger, live = [], None
    for tk, rs in kb2.items():
        rs.sort(key=lambda r: -r["mins_left"])
        cts = max(r.get("close_ts") or 0 for r in rs)
        if cts < V2_REGISTERED_TS:
            continue                      # clean forward counter
        env = [r for r in rs if ENV_LO <= r["mins_left"] <= ENV_HI]
        call = next((r for r in env
                     if max(r["p_up"], 1 - r["p_up"]) >= CALL_CONF),
                    None)
        if call is None:
            st = {"state": "NO_THESIS", "ticker": tk,
                  "close_ts": cts,
                  "settled": any(r.get("actual") is not None
                                 for r in rs)}
            if not st["settled"] and (live is None
                                      or cts > (live.get("close_ts")
                                                or 0)):
                live = st
            continue
        up = call["p_up"] >= 0.5
        reg_ts = cts - call["mins_left"] * 60   # registration wall-time
        q = events.get(tk, [])
        call_q = next((x for x in reversed(q)
                       if x[0] <= reg_ts + FRESH_S
                       and x[0] >= reg_ts - FRESH_S * 6), None)
        if call_q is None:
            ledger_or_live = {"state": "SYSTEM_EXCLUDED",
                              "reason": "no fresh executable quote "
                              "at registration (event tape)",
                              "ticker": tk, "close_ts": cts,
                              "settled": any(
                                  r.get("actual") is not None
                                  for r in rs)}
            if ledger_or_live["settled"]:
                ledger.append(ledger_or_live)
            continue
        call_ask = side_ask(call_q, up)
        st = {"state": "REGISTERED", "ticker": tk, "close_ts": cts,
              "side": "UP" if up else "DOWN",
              "call_conf": round(max(call["p_up"],
                                     1 - call["p_up"]), 3),
              "call_ask": round(call_ask, 1),
              "call_mins_left": call["mins_left"],
              "trigger_ask": round(call_ask - DIP_C, 1)}
        # confidence step function (minute rows after registration)
        conf_steps = [(cts - r["mins_left"] * 60,
                       (r["p_up"] if up else 1 - r["p_up"]))
                      for r in rs if r["mins_left"] < call["mins_left"]]
        conf_steps.sort()

        def conf_at(t):
            c = st["call_conf"]
            for ts_, cv in conf_steps:
                if ts_ <= t:
                    c = cv
                else:
                    break
            return c
        cutoff_ts = cts - CUTOFF_S
        for x in q:
            if x[0] <= reg_ts or x[0] > cutoff_ts:
                continue
            c = conf_at(x[0])
            if c < FLOOR:
                st["state"] = "INVALIDATED"
                st["invalidated_conf"] = round(c, 3)
                break
            a = side_ask(x, up)
            if call_ask - a >= DIP_C and depth_ok(x, up):
                st["state"] = "TRIGGERED"
                st["entry_ask"] = round(a, 1)
                st["entry_ts"] = x[0]
                st["entry_ask_sz"] = side_depth(x, up)
                st["improvement_c"] = round(call_ask - a, 1)
                st["entry_conf"] = round(c, 3)
                break
        # §17-19 diagnostics (research only, never decisions): best
        # valid executable price before cutoff while thesis alive
        valid_asks = [side_ask(x, up) for x in q
                      if reg_ts < x[0] <= cutoff_ts
                      and conf_at(x[0]) >= FLOOR]
        if valid_asks:
            bv = min(valid_asks)
            st["best_valid_price"] = round(bv, 1)
            st["best_valid_improvement_c"] = round(call_ask - bv, 1)
            st["best_valid_ts"] = round(next(
                x[0] for x in q if reg_ts < x[0] <= cutoff_ts
                and conf_at(x[0]) >= FLOOR
                and abs(side_ask(x, up) - bv) < 0.05), 1)
        st["cutoff_ts"] = round(cutoff_ts, 1)
        # prospective threshold shadows (observation only)
        st["shadows"] = {name: run_shadow(dc, q, up, reg_ts,
                                          call_ask, conf_at,
                                          cutoff_ts)
                         for name, dc in SHADOWS.items()}
        # §35-37 dip ladder + competing risks (DERIVED_EX_POST)
        ladder = {}
        for d in (5, 10, 15, 20):
            hit = next((x for x in q
                        if reg_ts < x[0] <= cutoff_ts
                        and conf_at(x[0]) >= FLOOR
                        and call_ask - side_ask(x, up) >= d), None)
            ladder[f"first_{d:02d}"] = (
                {"ts": round(hit[0], 1),
                 "price": round(side_ask(hit, up), 1),
                 "conf": round(conf_at(hit[0]), 3)} if hit else None)
        st["dip_ladder"] = ladder
        inval = next((x[0] for x in q
                      if reg_ts < x[0] <= cutoff_ts
                      and conf_at(x[0]) < FLOOR), None)
        first10 = (ladder["first_10"] or {}).get("ts")
        if first10 and (inval is None or first10 < inval):
            st["event_first"] = "DIP"
            st["time_to_first_event_s"] = round(first10 - reg_ts, 1)
        elif inval is not None:
            st["event_first"] = "INVALIDATION"
            st["time_to_first_event_s"] = round(inval - reg_ts, 1)
        else:
            st["event_first"] = "TIMEOUT"
            st["time_to_first_event_s"] = round(cutoff_ts - reg_ts, 1)
        settled = [r for r in rs if r.get("actual") is not None]
        if settled:
            won = bool(settled[-1]["actual"]) == up
            st["won"] = won
            st["settled"] = True
            st["control_pnl"] = round(pnl(call_ask, won), 4)
            st.update(**{f"c0_{k}": v for k, v in
                         markouts(q, up, reg_ts, call_ask).items()})
            if st["state"] == "TRIGGERED":
                st["state"] = "FILLED"      # v1 fill = take ask, qty 1
                st["a3_pnl"] = round(pnl(st["entry_ask"], won), 4)
                st.update(**markouts(q, up, st["entry_ts"],
                                     st["entry_ask"]))
                if st.get("best_valid_improvement_c", 0) > 0:
                    st["ecr"] = round(st["improvement_c"]
                                      / st["best_valid_improvement_c"],
                                      3)
                    st["wait_regret_c"] = round(
                        st["best_valid_improvement_c"]
                        - st["improvement_c"], 1)
            else:
                if st["state"] == "REGISTERED":
                    st["state"] = "MISSED"
                st["a3_pnl"] = 0.0
            for name in SHADOWS:
                sh = st["shadows"][name]
                sh["pnl"] = (round(pnl(sh["entry_ask"], won), 4)
                             if sh["state"] == "FILLED" else 0.0)
            # PM 08-30: canonical A-F economic class per eligible
            # window (measurement only — no thresholds move)
            if st["state"] == "FILLED":
                st["econ_class"] = ("A_FILLED_GOOD"
                                    if st["a3_pnl"] >= st["control_pnl"]
                                    else "B_FILLED_BAD")
            elif st["state"] == "MISSED":
                st["econ_class"] = ("C_MISSED_WINNER"
                                    if st["control_pnl"] > 0
                                    else "D_MISSED_LOSER")
            elif st["state"] == "INVALIDATED":
                st["econ_class"] = ("E_INVALIDATED_WINNER"
                                    if st["control_pnl"] > 0
                                    else "F_INVALIDATED_LOSER")
            ledger.append(st)
        else:
            if live is None or cts > (live.get("close_ts") or 0):
                live = st

    el = [e for e in ledger if e["state"] != "SYSTEM_EXCLUDED"]
    filled = [e for e in el if e["state"] == "FILLED"]
    agg = {"eligible": len(el),
           "filled": len(filled),
           "missed": sum(1 for e in el if e["state"] == "MISSED"),
           "invalidated": sum(1 for e in el
                              if e["state"] == "INVALIDATED"),
           "system_excluded": sum(1 for e in ledger
                                  if e["state"] == "SYSTEM_EXCLUDED"),
           "miss_rate": round(sum(1 for e in el if e["state"]
                                  in ("MISSED", "INVALIDATED"))
                              / len(el), 3) if el else None,
           "control_pnl_per_eligible": round(
               sum(e["control_pnl"] for e in el) / len(el), 4)
           if el else None,
           "a3_pnl_per_eligible": round(
               sum(e["a3_pnl"] for e in el) / len(el), 4)
           if el else None,
           "incremental_per_eligible": round(
               sum(e["a3_pnl"] - e["control_pnl"] for e in el)
               / len(el), 4) if el else None,
           "mean_entry_improvement_c": round(
               sum(e["improvement_c"] for e in filled)
               / len(filled), 1) if filled else None,
           "a3_fill_win_rate": round(
               sum(1 for e in filled if e["won"]) / len(filled), 3)
           if filled else None,
           "control_win_rate": round(
               sum(1 for e in el if e["won"]) / len(el), 3)
           if el else None,
           "missed_control_pnl_total": round(
               sum(e["control_pnl"] for e in el
                   if e["state"] in ("MISSED", "INVALIDATED")), 3)
           if el else None}
    # PM 08-30 canonical economic decomposition — the identity
    #   paired Δ (total) = fill-window incremental gain
    #                      − opportunity cost of no-entry windows
    # holds EXACTLY under the frozen convention (no-entry a3_pnl = 0),
    # and the residual is emitted so any drift is self-evident.
    gross_fill_gain = sum(e["a3_pnl"] - e["control_pnl"]
                          for e in filled)
    missed_rows = [e for e in el if e["state"] == "MISSED"]
    inval_rows = [e for e in el if e["state"] == "INVALIDATED"]
    timeout_pnl = sum(e["control_pnl"] for e in missed_rows
                      if e.get("event_first") == "TIMEOUT")
    missed_other_pnl = sum(e["control_pnl"] for e in missed_rows
                           if e.get("event_first") != "TIMEOUT")
    inval_pnl = sum(e["control_pnl"] for e in inval_rows)
    total_delta_raw = sum(e["a3_pnl"] - e["control_pnl"] for e in el)
    net_wait = gross_fill_gain - timeout_pnl - missed_other_pnl \
        - inval_pnl
    classes = {}
    for e in el:
        c = e.get("econ_class")
        if c:
            d = classes.setdefault(c, {"n": 0, "delta_contribution": 0.0})
            d["n"] += 1
            d["delta_contribution"] = round(
                d["delta_contribution"]
                + e["a3_pnl"] - e["control_pnl"], 4)
    agg["decomposition"] = {
        "gross_fill_gain": round(gross_fill_gain, 4),
        "timeout_control_pnl": round(timeout_pnl, 4),
        "missed_control_pnl": round(missed_other_pnl, 4),
        "invalidation_control_pnl": round(inval_pnl, 4),
        "net_waiting_value": round(net_wait, 4),
        "identity_residual": round(total_delta_raw - net_wait, 6),
        "note": "under A3-v1.1 MISSED ≡ timed-out (dip never arrived "
                "by cutoff), so missed_control_pnl is 0 unless a new "
                "state path appears; differential execution costs are "
                "inside gross_fill_gain (both arms priced at their "
                "own executable ask)",
        "econ_classes": classes,
    }
    # ---- registered decision artifact (PM 08-30): at n>=50 the
    # MACHINE emits the decision evidence under the frozen
    # registration — never an engineering reaction. Below 50 the
    # artifact exists with state INSUFFICIENT_EVIDENCE so the gate's
    # arrival changes a field, not the architecture.
    import random as _rnd
    deltas_all = [e["a3_pnl"] - e["control_pnl"] for e in el]
    n_el = len(deltas_all)
    ci_lo = ci_hi = None
    if n_el >= 2:
        rng = _rnd.Random(20260830)
        means = []
        for _ in range(2000):
            s = [deltas_all[rng.randrange(n_el)] for _ in range(n_el)]
            means.append(sum(s) / n_el)
        means.sort()
        ci_lo, ci_hi = means[50], means[1949]     # 95% bootstrap
    conc_ok = None
    if deltas_all:
        srt = sorted(deltas_all, reverse=True)
        tot = sum(srt)
        conc_ok = not (tot > 0 and tot - srt[0] <= 0)  # §39: no
        # promotion if the effect dies without the single best window
    if n_el < 50:
        decision = "INSUFFICIENT_EVIDENCE"
        why = f"n={n_el} < registered decision gate 50"
    elif ci_lo is not None and ci_lo > 0 and conc_ok:
        decision = "QUALIFY"
        why = "95% bootstrap CI excludes 0 from above AND survives " \
              "ex-best-1 concentration — registered promotion gate met"
    elif ci_hi is not None and ci_hi < 0:
        decision = "REJECT"
        why = "95% bootstrap CI excludes 0 from below — waiting " \
              "demonstrably loses under the frozen rule"
    else:
        decision = "CONTINUE"
        why = "CI spans 0 at the gate — the registered rule says " \
              "keep collecting; no threshold change is admissible"
    (RES / "a3_decision.json").write_text(json.dumps({
        "generated_ts": int(time.time()),
        "experiment": EXPERIMENT_ID, "registered_gate_n": 50,
        "eligible_n": n_el,
        "primary_effect_per_eligible": round(sum(deltas_all) / n_el, 4)
        if n_el else None,
        "ci95_bootstrap": [round(ci_lo, 4), round(ci_hi, 4)]
        if ci_lo is not None else None,
        "concentration_survives_ex_best1": conc_ok,
        "decision": decision, "why": why,
        "allowed_outcomes": ["CONTINUE", "REJECT", "QUALIFY",
                             "INSUFFICIENT_EVIDENCE"],
        "law": "no interpretation-induced threshold change in the "
               "same cycle; a clean failure is a successful research "
               "outcome; full mechanism evidence lives in a3_live "
               "(decomposition, watches, shadows, markouts)",
    }, indent=1))
    # outlier-dependence view (fat-tail discipline)
    deltas = sorted((e["a3_pnl"] - e["control_pnl"] for e in el),
                    reverse=True)
    outlier = None
    if deltas:
        tot = sum(deltas)
        outlier = {"total_delta": round(tot, 3),
                   "delta_ex_best1": round(tot - deltas[0], 3),
                   "delta_ex_best3": round(tot - sum(deltas[:3]), 3),
                   "top10pct_share": round(
                       sum(deltas[:max(1, len(deltas) // 10)])
                       / tot, 3) if tot > 0 else None}
    # shadow aggregates on the same eligible set
    shadow_agg = {}
    for name in SHADOWS:
        sf = [e["shadows"][name] for e in el if "shadows" in e]
        if sf:
            filled_s = [s for s in sf if s["state"] == "FILLED"]
            shadow_agg[name] = {
                "registered_ts": SHADOW_REGISTERED_TS,
                "dip_c": SHADOWS[name], "eligible": len(sf),
                "filled": len(filled_s),
                "pnl_per_eligible": round(
                    sum(s["pnl"] for s in sf) / len(sf), 4)}
    # exclusion monitor
    excl = [e for e in ledger if e["state"] == "SYSTEM_EXCLUDED"]
    import hashlib
    spec_hash = hashlib.sha256(
        (ROOT / SPEC_FILE).read_bytes()).hexdigest()[:16]
    # the five failure modes the research manager watches (TA order);
    # thresholds are WATCH flags, never actions
    fw = None
    if el:
        inc = sum(e["a3_pnl"] - e["control_pnl"] for e in el) / len(el)
        missed_cost = sum(e["control_pnl"] for e in el
                          if e["state"] in ("MISSED", "INVALIDATED"))
        fill_gain = sum(e["a3_pnl"] - e["control_pnl"] for e in el
                        if e["state"] == "FILLED")
        mo10 = [e["markout_10s"] for e in el
                if e["state"] == "FILLED"
                and isinstance(e.get("markout_10s"), (int, float))]
        fw = {
            "paired_delta_negative": inc < 0,
            "miss_cost_overwhelms": (missed_cost > fill_gain
                                     and len(el) >= 10),
            "thesis_integrity": {
                "a3_fill_win": agg.get("a3_fill_win_rate"),
                "eligible_win": agg.get("control_win_rate"),
                "flag": (agg.get("a3_fill_win_rate") is not None
                         and agg.get("control_win_rate") is not None
                         and agg["a3_fill_win_rate"]
                         < agg["control_win_rate"] - 0.10
                         and len(el) >= 10)},
            "toxic_markouts": (sum(1 for m in mo10 if m < 0)
                               / len(mo10) > 0.7
                               if len(mo10) >= 10 else None),
            "outlier_concentration": (outlier or {}).get(
                "top10pct_share"),
            "note": "WATCH flags only — thresholds informative at "
                    "n>=10; no automatic action"}
    doc = {"generated_ts": now, "spec": SPEC_FILE,
           "dip_c": DIP_C,
           "experiment_id": EXPERIMENT_ID,
           "predecessor": {"experiment": "A3-v1.1",
                           "outcome": "CLOSED_REJECTED",
                           "final": "a3_v1_final.json"},
           "failure_watch": fw,
           "spec_hash": spec_hash,
           "spec_hash_frozen": SPEC_HASH_FROZEN,
           "spec_hash_ok": spec_hash == SPEC_HASH_FROZEN,
           "exclusion_monitor": {
               "potential_registrations": len(el) + len(excl),
               "system_excluded": len(excl),
               "reasons": {"QUOTE_MISSING": len(excl)},
               "note": "high exclusion rate = operationally unable "
                       "to capture the opportunity set"},
           "outlier_dependence": outlier,
           "shadows": shadow_agg,
           "registered_ts": V2_REGISTERED_TS,
           "price_source": "EXECUTABLE event-tape asks (1s) — "
           "modeled asks forbidden (spec §4); v0 ledger retained as "
           "exploratory replay only",
           "conventions": "fill = take ask qty 1 (no depth data yet, "
           "recorded); entry cutoff 60s; quote freshness 10s",
           "live": live or {"state": "NO_THESIS"},
           "forward": agg,
           "recent_settled": sorted(
               ledger, key=lambda e: -e["close_ts"])[:10]}
    # §2 derived artifacts (reproducible; DERIVED_EX_POST provenance)
    import hashlib as _h
    sh_hash = _h.sha256((ROOT / "A3_SHADOW_SPEC.yaml").read_bytes()
                        ).hexdigest()[:16] \
        if (ROOT / "A3_SHADOW_SPEC.yaml").exists() else None
    common = {"a3_version": "1.1", "a3_config_hash": spec_hash,
              "shadow_config_hash": sh_hash,
              "provenance": "DERIVED_EX_POST",
              "fill_source": "V1_TAKE_ASK_CONVENTION"}
    with (RES / LEDGER_NAME).open("w") as f:
        for e in ledger:
            row = {**common, **{k: v for k, v in e.items()
                                if k != "shadows"}}
            for name in SHADOWS:
                sh = e.get("shadows", {}).get(name, {})
                row[f"{name}_state"] = sh.get("state")
                row[f"{name}_net_pnl"] = sh.get("pnl")
            if "a3_pnl" in row and "control_pnl" in row:
                row["delta_pnl"] = round(
                    row["a3_pnl"] - row["control_pnl"], 4)
            f.write(json.dumps(row) + "\n")
    (RES / "a3_live.json").write_text(json.dumps(doc, indent=1))
    print(f"{EXPERIMENT_ID}: live={doc['live']['state']} · eligible "
          f"{agg['eligible']} filled {agg['filled']} excluded "
          f"{agg['system_excluded']} · incr "
          f"{agg['incremental_per_eligible']}")


if __name__ == "__main__":
    main()
