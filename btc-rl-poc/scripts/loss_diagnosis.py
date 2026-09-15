"""LOSS DIAGNOSIS — why were the control (T0 pt) and the live treatments (cg33, fm)
wrong at market close, and WHICH entry-time indicator would have caught it.

For every SETTLED LOSING trade (official Kalshi outcome, win==0) this reconstructs the
point-in-time state at entry by joining the trade to the model log (kalshi_binary_log.jsonl,
same ticker, nearest made_ts) and computes the indicators that WERE observable at decision
time — never the close:

  * barrier position   : base (BRTI at entry) vs strike  -> which side the spot already favored
  * dist_bps           : (base-strike)/strike*1e4        -> how far, in bps, from the barrier
  * ask_c              : price paid                       -> payoff asymmetry (favorite trap)
  * confidence         : leader p_arm / model |p_up-.5|   -> was it a near-coin-flip we over-trusted
  * mkt_p_up           : Kalshi market prob               -> did the crowd price the winning side
  * consensus          : fraction of kb arms calling the WINNING side -> was the leader an outlier

It then attributes each loss to a primary MISSED INDICATOR (see attribute_loss / TODO(human)),
tallies the failure modes per arm and overall with dollars lost, and writes:
  research/loss_diagnosis.json  (per-trade rows + aggregates)
  research/LOSS_DIAGNOSIS.md    (human-readable lessons for the next model)

PAPER / SIMULATION ONLY. Read-only w.r.t. the live system.
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT_JSON = ROOT / "research" / "loss_diagnosis.json"
OUT_MD = ROOT / "research" / "LOSS_DIAGNOSIS.md"

ARMS = [("pt", "T0 control (Follower)", "pt_trades.jsonl"),
        ("cg33", "T1 (Confidence-Gated 33%)", "cg33_trades.jsonl"),
        ("fm", "T2 (Chronos-Bolt base)", "fm_trades.jsonl")]

FAVORITE_ASK_C = 65.0    # >= this ask -> a "favorite"; a loss forfeits the full rich stake
NEAR_STRIKE_BPS = 8.0    # |dist_bps| <= this at entry -> effectively a coin-flip barrier
MKT_CONF = 0.60          # market priced the winning side at >= this -> crowd disagreed with us
CONSENSUS = 0.60         # >= this fraction of model arms called the winning side -> leader outlier


def _load(name):
    p = RES / name
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def entry_snapshots():
    """ticker -> entry snapshot {base, strike, mkt_p_up, calls:{variant:p_up}} taken at the
    made_ts with the MOST time left (closest to the ~12-min entry) available in the model log."""
    rows = _load("kalshi_binary_log.jsonl")
    by_ticker_ts = defaultdict(lambda: defaultdict(dict))   # ticker -> made_ts -> variant -> row
    for r in rows:
        tk, ts, v = r.get("ticker"), r.get("made_ts"), r.get("variant")
        if tk and ts is not None and v:
            by_ticker_ts[tk][ts][v] = r
    snaps = {}
    for tk, byts in by_ticker_ts.items():
        # entry = the earliest decision instant we logged for this window (most mins_left)
        ent_ts = max(byts, key=lambda ts: max((rr.get("mins_left") or 0)
                                              for rr in byts[ts].values()))
        variants = byts[ent_ts]
        any_row = next(iter(variants.values()))
        snaps[tk] = {
            "base": any_row.get("base"), "strike": any_row.get("strike"),
            "mkt_p_up": any_row.get("mkt_p_up"),
            "mins_left": max((rr.get("mins_left") or 0) for rr in variants.values()),
            "calls": {v: rr.get("p_up") for v, rr in variants.items() if rr.get("p_up") is not None},
        }
    return snaps


def indicators(trade, snap):
    """All entry-time (PIT) indicators for one losing trade. Outcome used ONLY to name the
    winning side (never fed into any indicator)."""
    winning_side = "yes" if trade.get("actual") == 1 else "no"
    our_side = trade.get("side")
    ask_c = trade.get("ask_c")
    conf = trade.get("p_arm") or trade.get("p_conf")
    strike = trade.get("strike") or (snap or {}).get("strike")
    base = (snap or {}).get("base")
    dist_bps = ((base - strike) / strike * 1e4) if (base and strike) else None
    barrier_side = None if dist_bps is None else ("yes" if base >= strike else "no")
    mkt = (snap or {}).get("mkt_p_up")
    mkt_win_conf = None if mkt is None else (mkt if winning_side == "yes" else 1 - mkt)
    calls = (snap or {}).get("calls") or {}
    if calls:
        agree = sum(1 for p in calls.values()
                    if ("yes" if p >= 0.5 else "no") == winning_side)
        consensus = agree / len(calls)
    else:
        consensus = None
    return {
        "winning_side": winning_side, "our_side": our_side, "ask_c": ask_c,
        "confidence": round(conf, 4) if conf is not None else None,
        "dist_bps": round(dist_bps, 2) if dist_bps is not None else None,
        "barrier_side": barrier_side, "mkt_win_conf": round(mkt_win_conf, 4) if mkt_win_conf is not None else None,
        "consensus_for_winner": round(consensus, 3) if consensus is not None else None,
        "n_model_arms": len(calls) or None,
    }


def attribute_loss(ind):
    """Decide the PRIMARY missed indicator for one losing trade, given its entry-time
    indicators `ind` (dict from indicators()). Return (cause_code, missed_indicator, note).

    This is the modeling judgment: when several indicators could each explain the loss,
    which one do we "blame" (and therefore prioritise in the next model)? The order and
    thresholds encode that priority. Available fields on `ind`:
        winning_side, our_side            ('yes'/'no')
        ask_c            (price paid, cents)                        -> favorite trap if high
        confidence       (our stated conviction 0..1)               -> over-trust if high near a flip
        dist_bps         (base-strike in bps; sign = barrier side)  -> may be None if unjoined
        barrier_side     ('yes'/'no' the spot already favored)      -> may be None
        mkt_win_conf     (market prob assigned to the WINNING side) -> crowd disagreement if high
        consensus_for_winner (fraction of model arms that called the winner) -> leader outlier
        n_model_arms
    Module-level thresholds you can tune: FAVORITE_ASK_C, NEAR_STRIKE_BPS, MKT_CONF, CONSENSUS.

    TODO(human): implement the attribution priority. Suggested cause codes to return:
      "FAVORITE_TRAP", "WRONG_SIDE_BARRIER", "COINFLIP_NEAR_STRIKE",
      "MARKET_DISAGREE", "MODEL_CONSENSUS_MISS", "NOISE_UNAVOIDABLE".
    """
    raise NotImplementedError("attribute_loss: implement the attribution priority (TODO human)")


def run():
    snaps = entry_snapshots()
    per_arm = {}
    all_rows = []
    for arm_id, arm_name, log in ARMS:
        trades = _load(log)
        losses = [t for t in trades if t.get("actual") is not None and t.get("win") == 0]
        settled = [t for t in trades if t.get("actual") is not None]
        cats = Counter()
        dollars = defaultdict(float)
        rows = []
        for t in losses:
            snap = snaps.get(t.get("ticker"))
            ind = indicators(t, snap)
            cause, missed, note = attribute_loss(ind)
            cats[cause] += 1
            dollars[cause] += -(t.get("pnl_c") or 0) / 100.0
            rows.append({"ticker": t.get("ticker"), "arm": arm_id, "cause": cause,
                         "missed_indicator": missed, "note": note,
                         "loss_usd": round(-(t.get("pnl_c") or 0) / 100.0, 2),
                         "joined": snap is not None, **ind})
        all_rows.extend(rows)
        per_arm[arm_id] = {
            "name": arm_name, "settled": len(settled), "losses": len(losses),
            "loss_rate": round(len(losses) / len(settled), 4) if settled else None,
            "total_loss_usd": round(sum(-(t.get("pnl_c") or 0) / 100.0 for t in losses), 2),
            "by_cause_count": dict(cats.most_common()),
            "by_cause_usd": {k: round(v, 2) for k, v in sorted(dollars.items(),
                                                               key=lambda kv: -kv[1])},
        }
    agg = Counter(r["cause"] for r in all_rows)
    agg_usd = defaultdict(float)
    for r in all_rows:
        agg_usd[r["cause"]] += r["loss_usd"]
    report = {"schema": "loss-diagnosis-1", "thresholds": {
        "FAVORITE_ASK_C": FAVORITE_ASK_C, "NEAR_STRIKE_BPS": NEAR_STRIKE_BPS,
        "MKT_CONF": MKT_CONF, "CONSENSUS": CONSENSUS},
        "per_arm": per_arm,
        "overall_by_cause_count": dict(agg.most_common()),
        "overall_by_cause_usd": {k: round(v, 2) for k, v in sorted(agg_usd.items(), key=lambda kv: -kv[1])},
        "trades": all_rows}
    OUT_JSON.write_text(json.dumps(report, indent=1))
    _write_md(report)
    print(f"loss_diagnosis: {sum(v['losses'] for v in per_arm.values())} losses across "
          f"{len(per_arm)} arms -> {OUT_JSON.name}, {OUT_MD.name}")
    for aid, v in per_arm.items():
        print(f"  {aid:5} losses {v['losses']:4}/{v['settled']:<4} ${v['total_loss_usd']:>10.2f}  {v['by_cause_count']}")
    print("  OVERALL by cause:", dict(agg.most_common()))


def _write_md(rep):
    MISSED = {
        "FAVORITE_TRAP": "Price/EV ceiling — decline favorites priced above ~65c unless edge >> price.",
        "WRONG_SIDE_BARRIER": "Distance-to-strike / barrier z — trade WITH the spot's side of the strike.",
        "COINFLIP_NEAR_STRIKE": "Barrier-confidence gate — skip windows within ~a fraction of a sigma of the strike.",
        "MARKET_DISAGREE": "Market prob / disagreement gate — respect the crowd when it strongly prices the other side.",
        "MODEL_CONSENSUS_MISS": "Model ensemble/consensus — the leader was an outlier; blend or vote.",
        "NOISE_UNAVOIDABLE": "No entry-time indicator flagged it — genuine variance (irreducible).",
    }
    L = ["# Loss Diagnosis — why the control & treatments were wrong at close",
         "", "*Point-in-time attribution of every settled losing trade to the entry-time indicator "
         "that would have caught it. Read this before building the next model.*", "",
         "## Overall — losses by missed indicator", "",
         "| cause | count | $ lost | the indicator to add/weight next |", "|---|---|---|---|"]
    for cause, n in rep["overall_by_cause_count"].items():
        L.append(f"| {cause} | {n} | ${rep['overall_by_cause_usd'].get(cause,0):,.2f} | {MISSED.get(cause,'')} |")
    L += ["", "## Per arm", ""]
    for aid, v in rep["per_arm"].items():
        L.append(f"### {aid} — {v['name']}")
        L.append(f"- settled {v['settled']}, losses {v['losses']} "
                 f"(loss rate {v['loss_rate']}), total lost ${v['total_loss_usd']:,.2f}")
        L.append(f"- by cause (count): {v['by_cause_count']}")
        L.append(f"- by cause ($): {v['by_cause_usd']}")
        L.append("")
    L += ["## Lessons for the next model", "",
          "Ranked by dollars bled, the indicators the next model must weight more heavily:"]
    for cause, usd in rep["overall_by_cause_usd"].items():
        if cause == "NOISE_UNAVOIDABLE":
            continue
        L.append(f"- **{cause}** (${usd:,.2f}): {MISSED.get(cause,'')}")
    L += ["", f"*Thresholds used: {rep['thresholds']}. Tune in scripts/loss_diagnosis.py.*"]
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
