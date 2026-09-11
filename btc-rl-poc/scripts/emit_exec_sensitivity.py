#!/usr/bin/env python3
"""Quote-age / latency sensitivity + execution models E1/E2/E3 (directive §28/§30, Phase 3).

Re-prices every settled champion (pt) decision against the OBSERVED per-minute
kb control market path at quotes Δ minutes EARLIER (staleness) and Δ minutes
LATER (delayed execution), and applies the three qualification execution models.

Data granularity is per-minute: ms-scale latency is honestly UNAVAILABLE, so
the sub-minute (0-60s) band is labeled UNAVAILABLE and only minute-scale aging
(Δ ∈ {0,1,2,3,5} min) is modeled.

Conventions (documented, deterministic, stdlib-only):
  modeled ask (cents/contract) = 100*mkt_p_up + 2.5   (yes side)
                               = 100*(1-mkt_p_up) + 2.5 (no side)
  asks are clipped to [1.0, 99.0] cents after any adjustment.
  fee (cents/contract)         = 7 * (a/100) * (1 - a/100)  where a = executed ask
      -- the ceil-per-order Kalshi convention, kept CEIL-FREE FRACTIONAL because
         all quantities here are per-$1 (order size normalized away).
  EV per $1 staked, one trade  = (100*win - a - fee(a)) / a
      win = actual if side==yes else 1-actual   (settled outcome, OBSERVED)
  curve/model EV = mean of per-trade EV/$1 over the covered decision set.

Provenance labels: OBSERVED / DERIVED / MODELED / STRESS / UNAVAILABLE (§74).

Inputs : results/pt_trades.jsonl        (champion decisions, OBSERVED)
         results/kalshi_binary_log.jsonl (variant "kb" per-minute path, OBSERVED)
         results/execution_ledger.json   (measured mean slippage -> E2 calibration)
Output : results/exec_sensitivity.json
Exit 0 always on success; deterministic (E3 no-fill uses seed 20260829 over the
decision set sorted by (made_ts, ticker)).
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
KB_LOG = os.path.join(RES, "kalshi_binary_log.jsonl")
PT_LOG = os.path.join(RES, "pt_trades.jsonl")
LEDGER = os.path.join(RES, "execution_ledger.json")
OUT = os.path.join(RES, "exec_sensitivity.json")

DELTAS = [0, 1, 2, 3, 5]
ANCHOR_TOL_S = 120          # decision joined to kb minute row at-or-before, <=120s (ledger convention)
E3_EXTRA_SPREAD_C = 2.5     # stress: double the modeled half-spread
E3_NOFILL_RATE = 0.10
E3_SEED = 20260829
E2_FALLBACK_C = 2.0         # only if ledger value absent

FEE_FORMULA = ("fee_c = 7 * (a/100) * (1 - a/100) per contract at executed ask a; "
               "ceil-at-order-level convention kept ceil-free fractional because "
               "everything is per-$1 (1-contract order, fractional fee)")


def clip_ask(a):
    return max(1.0, min(99.0, a))


def modeled_ask(p_up, side):
    p = p_up if side == "yes" else 1.0 - p_up
    return clip_ask(100.0 * p + 2.5)


def fee_c(a):
    return 7.0 * (a / 100.0) * (1.0 - a / 100.0)


def ev_per_dollar(win, a):
    return (100.0 * win - a - fee_c(a)) / a


def load_kb_path():
    """(ticker -> {minute_ts: mkt_p_up}) from variant 'kb' control rows. OBSERVED."""
    path = {}
    with open(KB_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("variant") != "kb":
                continue
            if r.get("mkt_p_up") is None:
                continue  # quote UNAVAILABLE for that minute; never interpolated
            path.setdefault(r["ticker"], {})[int(r["made_ts"])] = float(r["mkt_p_up"])
    return path


def load_decisions():
    rows = []
    with open(PT_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("actual") not in (0, 1):
                continue  # unsettled
            rows.append(r)
    rows.sort(key=lambda r: (r["made_ts"], r["ticker"]))  # deterministic order
    return rows


def load_e2_correction():
    """Measured mean model-vs-real ask gap (cents). Real asks sit INSIDE the model
    (negative mean), so E2 ask = modeled ask + mean_slippage. DERIVED from ledger."""
    try:
        with open(LEDGER) as f:
            led = json.load(f)
        mean_slip = led["overall"]["slippage_vs_model_c"]["mean"]
        n = led["overall"]["slippage_vs_model_c"].get("n")
        return float(mean_slip), ("DERIVED — measured mean slippage_vs_model_c from "
                                  "execution_ledger.json (n=%s)" % n)
    except (OSError, KeyError, ValueError, TypeError):
        return -E2_FALLBACK_C, ("DERIVED (FALLBACK) — execution_ledger.json value absent; "
                                "using the documented -2.0c measured overstatement")


def anchor(kb_path, trade):
    """Join a decision to its kb minute row: latest minute ts <= made_ts, within tolerance."""
    ts_map = kb_path.get(trade["ticker"])
    if not ts_map:
        return None
    made = int(trade["made_ts"])
    cand = [t for t in ts_map if t <= made and made - t <= ANCHOR_TOL_S]
    return max(cand) if cand else None


def main():
    kb_path = load_kb_path()
    decisions = load_decisions()
    e2_mean_slip, e2_prov = load_e2_correction()

    # ---- join champion decisions to the observed path ----------------------
    covered = []          # (trade, t0, win)
    n_no_ticker = 0
    n_no_anchor = 0
    for tr in decisions:
        if tr["ticker"] not in kb_path:
            n_no_ticker += 1
            continue
        t0 = anchor(kb_path, tr)
        if t0 is None:
            n_no_anchor += 1
            continue
        win = tr["actual"] if tr["side"] == "yes" else 1 - tr["actual"]
        covered.append((tr, t0, win))
    n_excluded = n_no_ticker + n_no_anchor

    def curve_point(delta, direction):
        """direction=-1: quote Δ min EARLIER (staleness); +1: Δ min LATER (delay)."""
        evs = []
        for tr, t0, win in covered:
            ts = t0 + direction * 60 * delta
            p = kb_path[tr["ticker"]].get(ts)
            if p is None:
                continue  # path does not cover that minute for this window
            evs.append(ev_per_dollar(win, modeled_ask(p, tr["side"])))
        n = len(evs)
        return {
            "delta_min": delta,
            "ev_per_dollar": round(sum(evs) / n, 6) if n else None,
            "n": n,
            "excluded_no_path_minute": len(covered) - n,
        }

    ev_by_staleness = [curve_point(d, -1) for d in DELTAS]
    ev_by_delay = [curve_point(d, +1) for d in DELTAS]

    # ---- execution models E1/E2/E3 + oracle, on the same covered set -------
    def model_ev(adjust_c):
        evs = []
        for tr, t0, win in covered:
            a = clip_ask(modeled_ask(kb_path[tr["ticker"]][t0], tr["side"]) + adjust_c)
            evs.append(ev_per_dollar(win, a))
        return round(sum(evs) / len(evs), 6) if evs else None, len(evs)

    e1_ev, e1_n = model_ev(0.0)
    e2_ev, e2_n = model_ev(e2_mean_slip)

    rng = random.Random(E3_SEED)
    e3_evs = []
    e3_nofills = 0
    for tr, t0, win in covered:  # covered is in deterministic (made_ts, ticker) order
        if rng.random() < E3_NOFILL_RATE:
            e3_nofills += 1
            continue
        a = clip_ask(modeled_ask(kb_path[tr["ticker"]][t0], tr["side"]) + E3_EXTRA_SPREAD_C)
        e3_evs.append(ev_per_dollar(win, a))
    e3_ev = round(sum(e3_evs) / len(e3_evs), 6) if e3_evs else None

    oracle_evs = [ev_per_dollar(win,
                                clip_ask(modeled_ask(kb_path[tr["ticker"]][t0], tr["side"]) - 2.5))
                  for tr, t0, win in covered]  # mid = 100p = modeled ask - 2.5
    oracle_ev = round(sum(oracle_evs) / len(oracle_evs), 6) if oracle_evs else None

    # ---- verdict ------------------------------------------------------------
    threshold = (e1_ev - 0.01) if e1_ev is not None else None
    cross_stale = next((pt["delta_min"] for pt in ev_by_staleness
                        if pt["delta_min"] > 0 and pt["ev_per_dollar"] is not None
                        and threshold is not None and pt["ev_per_dollar"] < threshold), None)
    cross_delay = next((pt["delta_min"] for pt in ev_by_delay
                        if pt["delta_min"] > 0 and pt["ev_per_dollar"] is not None
                        and threshold is not None and pt["ev_per_dollar"] < threshold), None)
    survives_e2 = e2_ev is not None and e2_ev > 0
    survives_e3 = e3_ev is not None and e3_ev > 0
    summary = ("Champion edge %s under E2 (%.4f/$1) and %s under E3 stress (%.4f/$1); "
               "staleness EV first drops >1c/$1 below the E1 baseline (%.4f/$1) at %s." % (
                   "survives" if survives_e2 else "FAILS",
                   e2_ev if e2_ev is not None else float("nan"),
                   "survives" if survives_e3 else "FAILS",
                   e3_ev if e3_ev is not None else float("nan"),
                   e1_ev if e1_ev is not None else float("nan"),
                   ("Delta=%d min" % cross_stale) if cross_stale is not None
                   else "no tested staleness (<=5 min)"))

    out = {
        "generated_by": "scripts/emit_exec_sensitivity.py",
        "directive": "master qualification §28/§30 (Phase 3) — quote-age sensitivity + execution models",
        "determinism": {"e3_seed": E3_SEED,
                        "e3_draw_order": "settled covered decisions sorted by (made_ts, ticker)"},
        "conventions": {
            "modeled_ask": "100*mkt_p_up + 2.5c (yes) / 100*(1-mkt_p_up) + 2.5c (no), clipped to [1,99]c — MODELED",
            "fee": FEE_FORMULA,
            "ev_per_dollar": "(100*win - ask - fee)/ask per trade, averaged; win is the OBSERVED settled outcome",
        },
        "decision_set": {
            "source": "results/pt_trades.jsonl (champion) joined to results/kalshi_binary_log.jsonl variant=kb — OBSERVED",
            "settled_decisions": len(decisions),
            "covered": len(covered),
            "excluded_no_kb_path": n_excluded,
            "excluded_breakdown": {"ticker_absent_from_kb_log": n_no_ticker,
                                   "no_minute_row_within_120s_before_decision": n_no_anchor},
            "note": ("Windows without kb path coverage are excluded — the kb log retains only "
                     "recent windows; excluded decisions are counted, never interpolated."),
        },
        "quote_age": {
            "provenance": "DERIVED — repricing of OBSERVED decisions against the OBSERVED per-minute quote path at MODELED asks",
            "sub_minute": {"band": "0-60s", "status": "UNAVAILABLE — per-minute data"},
            "ev_by_staleness": ev_by_staleness,
            "ev_by_delay": ev_by_delay,
            "definition": ("staleness Δ: the SAME side priced at the quote Δ minutes BEFORE the decision minute "
                           "(older/staler quote); delay Δ: at the quote Δ minutes AFTER (delayed execution). "
                           "Δ=0 is the E1 baseline. Points with no minute row are excluded per point."),
        },
        "execution_models": {
            "E1_taker_at_book": {"ev_per_dollar": e1_ev, "n": e1_n,
                                 "label": "MODELED — current registry convention, ask = 100p + 2.5c"},
            "E2_calibrated": {"ev_per_dollar": e2_ev, "n": e2_n,
                              "ask_correction_c": round(e2_mean_slip, 4),
                              "label": e2_prov + " — modeled ask minus the measured ~2.0c overstatement"},
            "E3_stress": {"ev_per_dollar": e3_ev, "n": len(e3_evs), "no_fills": e3_nofills,
                          "extra_spread_c": E3_EXTRA_SPREAD_C, "no_fill_rate": E3_NOFILL_RATE,
                          "label": "STRESS — modeled ask + 2.5c (double spread) and 10% seeded random no-fill"},
            "oracle_mid_fill": {"ev_per_dollar": oracle_ev, "n": len(oracle_evs),
                                "label": "NON-REALISTIC — diagnostic only, never for qualification (fill at mid = 100p)"},
        },
        "verdict": {
            "edge_survives_E2": survives_e2,
            "edge_survives_E3": survives_e3,
            "staleness_crossing_min": cross_stale,
            "delay_crossing_min": cross_delay,
            "crossing_definition": "first tested Δ where curve EV < E1 baseline EV minus 0.01 $/1 (1c per $1)",
            "summary": summary,
        },
    }

    os.makedirs(RES, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("wrote %s" % OUT)
    print("decision set: %d settled, %d covered, %d excluded (no kb path)"
          % (len(decisions), len(covered), n_excluded))
    print("staleness curve (EV/$1):")
    for pt in ev_by_staleness:
        print("  d=-%dmin  ev=%s  n=%d" % (pt["delta_min"], pt["ev_per_dollar"], pt["n"]))
    print("delay curve (EV/$1):")
    for pt in ev_by_delay:
        print("  d=+%dmin  ev=%s  n=%d" % (pt["delta_min"], pt["ev_per_dollar"], pt["n"]))
    print("E1=%s  E2=%s (corr %.4fc)  E3=%s (%d no-fills)  oracle=%s"
          % (e1_ev, e2_ev, e2_mean_slip, e3_ev, e3_nofills, oracle_ev))
    print("VERDICT: " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
