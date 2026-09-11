"""DECISION FRONTIER (PM 09-11) — the selective-decision objective.

Reframes the trader as a selective decision system with ASYMMETRIC
error costs: maximize profitable-trade RECALL subject to a hard upper
bound on bad-entry rate. Not win-rate, not EV-at-one-threshold.

For every settled window we score the ex-post EV of entering the
model's side at the MARKET-IMPLIED price (the honest counterfactual
cost), so we can measure BOTH errors the TA named:
  BAD_ENTRY               — entered, realized EV < 0
  MISSED_PROFITABLE_TRADE — abstained, but ex-post EV > 0
Then sweep the opportunity signal (model edge) to trace the
profit-recall vs bad-entry Pareto frontier and pick the operating
point under a frozen risk budget epsilon.

Observation-only (Research plane). Uses the serving arm kb2.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS_BAD_ENTRY = 0.10          # frozen risk budget: <=10% bad entries
ARM = "kb2"


def fee_c(price_c):
    p = price_c / 100.0
    return 7 * p * (1 - p)


def windows():
    rows = [json.loads(l) for l in
            (RES / "kalshi_binary_log.jsonl").open() if l.strip()]
    out = []
    for r in rows:
        if r.get("variant") != ARM or r.get("actual") is None:
            continue
        p_up = r.get("p_up")
        mkt = r.get("mkt_p_up")
        if p_up is None or mkt is None:
            continue
        side_yes = p_up >= 0.5
        p_side = p_up if side_yes else 1 - p_up          # model P(side)
        price = 100 * (mkt if side_yes else 1 - mkt)     # market cost c
        price = min(99.0, max(1.0, price))
        won = (int(r["actual"]) == 1) == side_yes
        cost = price + fee_c(price)
        ev_c = (100 - cost) if won else -cost            # per contract
        edge_c = p_side * 100 - price                    # model edge
        out.append({"ts": r.get("made_ts"), "edge_c": edge_c,
                    "ev_c": ev_c, "profitable": ev_c > 0,
                    "ev_per_$1": ev_c / cost})
    out.sort(key=lambda w: w["ts"] or 0)
    return out


def metrics(entered, allw):
    prof_all = [w for w in allw if w["profitable"]]
    ent_prof = [w for w in entered if w["profitable"]]
    ent_bad = [w for w in entered if not w["profitable"]]
    return {
        "coverage": round(len(entered) / max(1, len(allw)), 3),
        "profitable_trade_recall": round(len(ent_prof)
                                         / max(1, len(prof_all)), 3),
        "bad_entry_rate": round(len(ent_bad)
                                / max(1, len(entered)), 3),
        "precision": round(len(ent_prof) / max(1, len(entered)), 3),
        "captured_profit_ev_c": round(sum(w["ev_c"] for w in ent_prof), 0),
        "dicey_entry_ev_c": round(sum(w["ev_c"] for w in ent_bad), 0),
        "missed_profit_ev_c": round(sum(w["ev_c"] for w in prof_all
                                        if w not in entered), 0),
        "mean_ev_per_$1_c": round(100 * (sum(w["ev_per_$1"]
                                  for w in entered) / len(entered))
                                  if entered else 0.0, 2),
        "n_entered": len(entered)}


def main():
    allw = windows()
    prof = [w for w in allw if w["profitable"]]
    # sweep the opportunity signal (model edge) -> frontier
    grid = [round(e, 1) for e in
            [x * 0.5 for x in range(-4, 41)]]     # -2c .. +20c
    frontier = []
    for thr in grid:
        entered = [w for w in allw if w["edge_c"] >= thr]
        if not entered:
            continue
        m = metrics(entered, allw)
        m["edge_threshold_c"] = thr
        frontier.append(m)
    # operating point: lowest bad-entry within budget, then max recall
    feasible = [m for m in frontier
                if m["bad_entry_rate"] <= EPS_BAD_ENTRY
                and m["mean_ev_per_$1_c"] > 0]
    op = (max(feasible, key=lambda m: m["profitable_trade_recall"])
          if feasible else None)
    doc = {"generated_ts": int(time.time()),
           "objective": "max profitable-trade recall s.t. "
           f"P(neg EV|enter) <= {EPS_BAD_ENTRY} AND E[EV|enter] > 0",
           "arm": ARM, "note": "counterfactual entry at MARKET-IMPLIED "
           "price; observation-only; kb2 ~0.998 corr with market so a "
           "weak edge signal is the expected honest result",
           "n_windows": len(allw), "n_ex_post_profitable": len(prof),
           "base_rate_profitable": round(len(prof) / len(allw), 3),
           "risk_budget_epsilon": EPS_BAD_ENTRY,
           "frontier": frontier,
           "chosen_operating_point": op,
           "verdict": ("FEASIBLE_OPERATING_POINT" if op else
                       "NO_FEASIBLE_POINT — no edge threshold achieves "
                       "positive EV within the bad-entry budget "
                       "(consistent with: we do not beat the market)")}
    (RES / "decision_frontier.json").write_text(json.dumps(doc, indent=1))
    print(f"decision_frontier: {len(allw)} windows, "
          f"{len(prof)} ex-post profitable ({doc['base_rate_profitable']}); "
          f"verdict {doc['verdict'].split(' ')[0]}")
    if op:
        print(f"  op: edge>={op['edge_threshold_c']}c recall "
              f"{op['profitable_trade_recall']} bad_entry "
              f"{op['bad_entry_rate']} meanEV {op['mean_ev_per_$1_c']}c")


if __name__ == "__main__":
    main()
