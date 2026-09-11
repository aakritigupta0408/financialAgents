#!/usr/bin/env python3
"""Emit the desk execution ledger with markouts (qualification directive §26/§29).

Reads the desk's real-basis paper-trade logs (pt/pt4/pt6/pt7/pt8) and the
per-minute market-state path (kalshi_binary_log.jsonl, variant "kb"), and
emits:

  results/execution_ledger.jsonl  — one row per settled order
  results/execution_ledger.json   — aggregates + provenance legend + coverage

Integrity laws (§74): every metric is labeled OBSERVED / DERIVED / MODELED /
UNAVAILABLE. Nothing is interpolated across data gaps: markout horizons the
per-minute data cannot support (+1s/+5s/+15s/+30s) are emitted as the literal
string "UNAVAILABLE" so the gap stays visible. ticks.jsonl is checked for a
price field; if (as recorded) it carries none, tick-based price markouts are
UNAVAILABLE and price markouts are derived from the kb path's per-minute
`base` BTC price instead, with that substitution declared in the legend.

Pure stdlib, deterministic (no wall-clock, no randomness).
"""

import bisect
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

TRADE_LOGS = [  # (trader label, filename) — pt primary, then pt4/pt6/pt7/pt8
    ("pt", "pt_trades.jsonl"),
    ("pt4", "pt4_trades.jsonl"),
    ("pt6", "pt6_trades.jsonl"),
    ("pt7", "pt7_trades.jsonl"),
    ("pt8", "pt8_trades.jsonl"),
]

KB_LOG = os.path.join(RESULTS, "kalshi_binary_log.jsonl")
TICKS = os.path.join(RESULTS, "ticks.jsonl")

HALF_SPREAD_C = 2.5          # registry model-basis convention (MODELED)
MODELED_SPREAD_C = 2 * HALF_SPREAD_C
QUOTE_AGE_MAX_S = 120        # kb grid is 60s (occasional 120s gap); older ⇒ UNAVAILABLE
MARKOUT_TOL_S = 30           # nearest kb row within ±30s of the target minute
SUB_MINUTE_HORIZONS = ["+1s", "+5s", "+15s", "+30s"]  # unsupported by 30-60s data
PROB_HORIZONS = [("+1m", 60), ("+3m", 180), ("+5m", 300)]
PRICE_HORIZONS = [("+1m", 60), ("+5m", 300)]

UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------- data loading

def read_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_kb_paths():
    """kalshi_binary_log variant=='kb' (control) → per-ticker sorted minute path.

    Returns {ticker: (sorted_ts_list, {ts: (mkt_p_up, base)})}.
    """
    raw = {}
    for r in read_jsonl(KB_LOG):
        if r.get("variant") != "kb":
            continue
        t = r.get("ticker")
        ts = r.get("made_ts")
        p = r.get("mkt_p_up")
        if t is None or ts is None or p is None:
            continue
        raw.setdefault(t, {})[ts] = (p, r.get("base"))
    paths = {}
    for t, d in raw.items():
        keys = sorted(d)
        paths[t] = (keys, d)
    return paths


def probe_ticks_for_price():
    """Single streaming pass over ticks.jsonl looking for any price-like key.

    Cheap substring scan (no full JSON parse of 500k rows). Returns
    (n_lines, has_price). Deterministic; reads the file exactly once.
    """
    n = 0
    has_price = False
    needles = (b'"price"', b'"px"', b'"last"', b'"close"', b'"mid"')
    if not os.path.exists(TICKS):
        return 0, False
    with open(TICKS, "rb") as f:
        for chunk_line in f:
            n += 1
            if not has_price and any(k in chunk_line for k in needles):
                has_price = True
    return n, has_price


# ------------------------------------------------------------------- kb lookup

def kb_at_or_before(path, ts, max_age):
    """Latest kb row at or before ts within max_age seconds, else None."""
    keys, d = path
    i = bisect.bisect_right(keys, ts) - 1
    if i < 0:
        return None
    k = keys[i]
    if ts - k > max_age:
        return None
    return k, d[k]


def kb_nearest(path, target, tol):
    """kb row nearest to target ts within ±tol seconds, else None."""
    keys, d = path
    i = bisect.bisect_left(keys, target)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(keys):
            dist = abs(keys[j] - target)
            if dist <= tol and (best is None or dist < best[0]):
                best = (dist, keys[j])
    if best is None:
        return None
    k = best[1]
    return k, d[k]


# ------------------------------------------------------------------ statistics

def percentile(sorted_vals, q):
    """Linear-interpolation percentile on a pre-sorted list (deterministic)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def dist_stats(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 4),
        "median": round(statistics.median(s), 4),
        "p10": round(percentile(s, 0.10), 4),
        "p90": round(percentile(s, 0.90), 4),
    }


# ------------------------------------------------------------------ band logic

def conf_band(p_arm):
    if p_arm is None:
        return "unknown"
    if p_arm < 0.7:
        return "<0.7"
    if p_arm <= 0.8:
        return "0.7-0.8"
    return ">0.8"


def price_band(ask_c):
    if ask_c is None:
        return "unknown"
    if ask_c < 40:
        return "<40c"
    if ask_c <= 60:
        return "40-60c"
    return ">60c"


def mins_band(m):
    if m is None:
        return "unknown"
    if m < 8:
        return "<8m"
    if m < 11:
        return "8-11m"
    return ">=11m"


# --------------------------------------------------------------------- ledger


def build_order(trader, r, kb_paths, base_price_available):
    """One per-order ledger row. Returns None for skipped/unsettled rows."""
    if r.get("skipped"):
        return None
    win = r.get("win")
    actual = r.get("actual")
    if win not in (0, 1) or actual not in (0, 1):
        return None  # unsettled / voided

    ticker = r.get("ticker")
    made_ts = r.get("made_ts")
    side = r.get("side")
    ask_c = r.get("ask_c")
    stake_c = r.get("stake_c")
    pnl_c = r.get("pnl_c")

    row = {
        "trader": trader,
        "ticker": ticker,
        "decision_ts": made_ts,
        "close_ts": r.get("close_ts"),
        "side": side,
        "contracts": r.get("contracts"),
        "stake_c": stake_c,
        "ask_c": ask_c,                       # OBSERVED fill/decision ask
        "fee_c": r.get("fee_c"),              # OBSERVED
        "limit_c": r.get("limit_c"),          # OBSERVED where logged (pt7/pt8)
        "quoted_c": r.get("quoted_c"),        # OBSERVED where logged (pt7/pt8)
        "p_arm": r.get("p_arm"),
        "mins_left": r.get("mins_left"),
        "spread_at_decision_c": MODELED_SPREAD_C,  # MODELED (no real bid/ask in row)
        "settlement": {"actual": actual, "win": win, "pnl_c": pnl_c},
        "realized_ev_per_dollar": (
            round(pnl_c / stake_c, 6)
            if isinstance(pnl_c, (int, float)) and stake_c else None
        ),
    }

    # --- decision quote from the kb per-minute path -------------------------
    path = kb_paths.get(ticker)
    anchor = None
    if path is not None and made_ts is not None:
        anchor = kb_at_or_before(path, made_ts, QUOTE_AGE_MAX_S)

    if anchor is None:
        reason = ("ticker_not_in_kb_log" if path is None
                  else "no_kb_row_within_%ds_before_decision" % QUOTE_AGE_MAX_S)
        row["quote_age_s"] = UNAVAILABLE
        row["mkt_p_up_at_decision"] = UNAVAILABLE
        row["modeled_ask_at_decision_c"] = UNAVAILABLE
        row["slippage_vs_model_c"] = UNAVAILABLE
        row["markout_prob_pp"] = {h: UNAVAILABLE for h in SUB_MINUTE_HORIZONS}
        row["markout_prob_pp"].update({h: UNAVAILABLE for h, _ in PROB_HORIZONS})
        row["markout_price_usd"] = {h: UNAVAILABLE for h, _ in PRICE_HORIZONS}
        row["markout_unavailable_reason"] = reason
        return row

    t0, (p0, base0) = anchor
    sign = 1.0 if side == "yes" else -1.0
    row["quote_age_s"] = made_ts - t0                          # DERIVED
    row["mkt_p_up_at_decision"] = p0                           # OBSERVED (kb path)
    modeled_ask = (100.0 * p0 + HALF_SPREAD_C if side == "yes"
                   else 100.0 * (1.0 - p0) + HALF_SPREAD_C)    # MODELED
    row["modeled_ask_at_decision_c"] = round(modeled_ask, 2)
    row["slippage_vs_model_c"] = (
        round(ask_c - modeled_ask, 2)
        if isinstance(ask_c, (int, float)) else UNAVAILABLE)   # DERIVED

    # --- markouts -----------------------------------------------------------
    prob_mo = {h: UNAVAILABLE for h in SUB_MINUTE_HORIZONS}    # data granularity
    for h, dt in PROB_HORIZONS:
        hit = kb_nearest(path, t0 + dt, MARKOUT_TOL_S)
        if hit is None:
            prob_mo[h] = UNAVAILABLE
        else:
            _, (p1, _) = hit
            prob_mo[h] = round(sign * (p1 - p0) * 100.0, 4)    # DERIVED, prob pts
    row["markout_prob_pp"] = prob_mo

    price_mo = {}
    for h, dt in PRICE_HORIZONS:
        if not base_price_available or base0 is None:
            price_mo[h] = UNAVAILABLE
            continue
        hit = kb_nearest(path, t0 + dt, MARKOUT_TOL_S)
        if hit is None or hit[1][1] is None:
            price_mo[h] = UNAVAILABLE
        else:
            _, (_, b1) = hit
            price_mo[h] = round(sign * (b1 - base0), 2)        # DERIVED, USD
    row["markout_price_usd"] = price_mo
    return row


def markout_vals(orders, horizon):
    return [o["markout_prob_pp"][horizon] for o in orders
            if isinstance(o["markout_prob_pp"].get(horizon), (int, float))]


def split_stats(orders, keyfn, horizons=("+1m", "+5m")):
    out = {}
    groups = {}
    for o in orders:
        groups.setdefault(keyfn(o), []).append(o)
    for k in sorted(groups):
        out[k] = {"orders": len(groups[k])}
        for h in horizons:
            out[k]["markout_%s_pp" % h.strip("+")] = dist_stats(
                markout_vals(groups[k], h))
    return out


def aggregate(orders):
    agg = {"orders": len(orders)}
    for h in ("+1m", "+3m", "+5m"):
        agg["markout_%s_pp" % h.strip("+")] = dist_stats(markout_vals(orders, h))
    for h in ("+1m", "+5m"):
        vals = [o["markout_price_usd"][h] for o in orders
                if isinstance(o["markout_price_usd"].get(h), (int, float))]
        agg["price_markout_%s_usd" % h.strip("+")] = dist_stats(vals)
    agg["by_side"] = split_stats(orders, lambda o: o.get("side") or "unknown")
    agg["by_confidence_band"] = split_stats(orders, lambda o: conf_band(o.get("p_arm")))
    agg["by_entry_price_band"] = split_stats(orders, lambda o: price_band(o.get("ask_c")))
    agg["by_mins_left_band"] = split_stats(orders, lambda o: mins_band(o.get("mins_left")))
    agg["adverse_selection"] = {
        "winning_entries": split_stats(
            [o for o in orders if o["settlement"]["win"] == 1], lambda o: "all")
        .get("all"),
        "losing_entries": split_stats(
            [o for o in orders if o["settlement"]["win"] == 0], lambda o: "all")
        .get("all"),
    }
    slip = [o["slippage_vs_model_c"] for o in orders
            if isinstance(o.get("slippage_vs_model_c"), (int, float))]
    agg["slippage_vs_model_c"] = dist_stats(slip)
    ev = [o["realized_ev_per_dollar"] for o in orders
          if isinstance(o.get("realized_ev_per_dollar"), (int, float))]
    agg["realized_ev_per_dollar"] = dist_stats(ev)
    return agg


PROVENANCE_LEGEND = {
    "trader/ticker/decision_ts/close_ts/side/contracts/stake_c": "OBSERVED (trade log row)",
    "ask_c": "OBSERVED (decision ask recorded at entry in the trade log)",
    "fee_c": "OBSERVED (trade log row)",
    "limit_c/quoted_c": "OBSERVED where the trader logs them (pt7/pt8 only); absent (null) otherwise",
    "p_arm/mins_left": "OBSERVED (trade log row)",
    "quote_age_s": "DERIVED (decision_ts minus ts of nearest kb control row at-or-before it, <=120s; else UNAVAILABLE)",
    "mkt_p_up_at_decision": "OBSERVED (kalshi_binary_log variant=kb market-probability path)",
    "spread_at_decision_c": "MODELED (registry 2 x 2.5c half-spread convention; pt rows carry only ask_c, no real bid/ask was recorded)",
    "modeled_ask_at_decision_c": "MODELED (100*mkt_p_up + 2.5 for yes; 100*(1-mkt_p_up) + 2.5 for no)",
    "slippage_vs_model_c": "DERIVED (OBSERVED ask_c minus MODELED same-minute ask; the model-vs-real gap per order)",
    "markout_prob_pp[+1m/+3m/+5m]": "DERIVED (kb mkt_p_up path: (p(t0+dt) - p(t0)) * 100, signed + for yes / - for no; nearest kb row within +/-30s of target, else UNAVAILABLE)",
    "markout_prob_pp[+1s/+5s/+15s/+30s]": "UNAVAILABLE (market path is per-minute and ticks are 30-60s without price; sub-minute markouts cannot be supported and are NOT interpolated)",
    "markout_price_usd[+1m/+5m]": "DERIVED (kb path per-minute `base` BTC price, signed by side; ticks.jsonl carries NO price field - only id/ts/size/taker_buy - so tick-based price markouts are impossible and the kb base path is used instead, declared here)",
    "settlement.actual/win/pnl_c": "OBSERVED (trade log row)",
    "realized_ev_per_dollar": "DERIVED (pnl_c / stake_c)",
    "aggregates (mean/median/p10/p90, bands, adverse-selection split)": "DERIVED (computed from the per-order rows above; nothing sampled, nothing simulated)",
}


def main():
    kb_paths = load_kb_paths()
    kb_min_ts = min((k[0] for k, _ in kb_paths.values() if k), default=None)
    kb_max_ts = max((k[-1] for k, _ in kb_paths.values() if k), default=None)
    tick_lines, ticks_have_price = probe_ticks_for_price()

    orders = []
    per_log_counts = {}
    skipped_rows = 0
    unsettled_rows = 0
    for trader, fname in TRADE_LOGS:
        path = os.path.join(RESULTS, fname)
        n = 0
        for r in read_jsonl(path):
            if r.get("skipped"):
                skipped_rows += 1
                continue
            o = build_order(trader, r, kb_paths, base_price_available=True)
            if o is None:
                unsettled_rows += 1
                continue
            orders.append(o)
            n += 1
        per_log_counts[trader] = n

    # deterministic ordering
    orders.sort(key=lambda o: (o["decision_ts"] or 0, o["trader"], o["ticker"] or ""))

    # ------------------------------------------------------------- coverage
    def n_avail(h):
        return sum(1 for o in orders
                   if isinstance(o["markout_prob_pp"].get(h), (int, float)))

    pre_kb = sum(1 for o in orders
                 if kb_min_ts is not None and (o["decision_ts"] or 0) < kb_min_ts)
    no_anchor = sum(1 for o in orders if o.get("quote_age_s") == UNAVAILABLE)
    coverage = {
        "settled_orders": len(orders),
        "orders_per_trader": per_log_counts,
        "skipped_rows_excluded": skipped_rows,
        "unsettled_rows_excluded": unsettled_rows,
        "kb_control_path": {
            "tickers": len(kb_paths),
            "ts_range": [kb_min_ts, kb_max_ts],
            "note": "kalshi_binary_log.jsonl retains only this window (rotated); "
                    "orders decided before it have no market path and their "
                    "markouts are UNAVAILABLE, not backfilled.",
        },
        "orders_before_kb_window": pre_kb,
        "orders_without_decision_anchor": no_anchor,
        "markout_derivable": {h: n_avail(h) for h in ("+1m", "+3m", "+5m")},
        "markout_unavailable": {h: len(orders) - n_avail(h)
                                for h in ("+1m", "+3m", "+5m")},
        "sub_minute_markouts": "UNAVAILABLE for all orders (data granularity)",
        "ticks_jsonl": {
            "lines_scanned": tick_lines,
            "has_price_field": ticks_have_price,
            "note": ("ticks.jsonl rows carry only id/ts/size/taker_buy - no price - "
                     "so tick-based price markouts are UNAVAILABLE; price markouts "
                     "use the kb path per-minute `base` BTC price instead."
                     if not ticks_have_price else
                     "price-like key detected; review before relying on kb base substitution."),
        },
    }

    # ------------------------------------------------------------ aggregates
    overall = aggregate(orders)
    per_trader = {t: aggregate([o for o in orders if o["trader"] == t])
                  for t, _ in TRADE_LOGS}

    m1 = overall["markout_1m_pp"]["mean"]
    m5 = overall["markout_5m_pp"]["mean"]
    adverse = (isinstance(m1, (int, float)) and m1 < 0) or \
              (isinstance(m5, (int, float)) and m5 < 0)
    markout_summary = {
        "orders": len(orders),
        "orders_with_derivable_markouts": coverage["markout_derivable"]["+5m"],
        "mean_markout_1m_pp": m1,
        "mean_markout_5m_pp": m5,
        "adverse_selection_signature": adverse,
        "definition": ("markout = signed post-entry move of the market probability "
                       "in the trade's direction, in probability points; a negative "
                       "mean at +1m or +5m is the adverse-selection signature."),
    }

    ledger = {
        "generated_from": {
            "trade_logs": [f for _, f in TRADE_LOGS],
            "market_path": "kalshi_binary_log.jsonl (variant=kb control)",
            "ticks": "ticks.jsonl (probed; no price field)",
        },
        "basis_note": ("Real-basis desk entries joined to the per-minute OBSERVED "
                       "market-state path. Modeled quantities use the registry "
                       "convention ask = 100*mkt_p_up + 2.5c. No field is "
                       "interpolated across gaps; gaps are UNAVAILABLE."),
        "markout_summary": markout_summary,
        "field_provenance": PROVENANCE_LEGEND,
        "coverage": coverage,
        "overall": overall,
        "per_trader": per_trader,
    }

    out_json = os.path.join(RESULTS, "execution_ledger.json")
    out_jsonl = os.path.join(RESULTS, "execution_ledger.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for o in orders:
            f.write(json.dumps(o, sort_keys=True) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, sort_keys=False)
        f.write("\n")

    print(json.dumps(markout_summary, indent=2))
    aw = overall["adverse_selection"]["winning_entries"]
    al = overall["adverse_selection"]["losing_entries"]
    if aw and al:
        print("winners  markout_1m mean=%s  markout_5m mean=%s (n=%s)"
              % (aw["markout_1m_pp"]["mean"], aw["markout_5m_pp"]["mean"], aw["orders"]))
        print("losers   markout_1m mean=%s  markout_5m mean=%s (n=%s)"
              % (al["markout_1m_pp"]["mean"], al["markout_5m_pp"]["mean"], al["orders"]))
    print("wrote %s (%d orders) and %s" % (out_jsonl, len(orders), out_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
