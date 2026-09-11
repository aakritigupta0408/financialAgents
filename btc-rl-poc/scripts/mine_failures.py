"""FAILURE STORE (PM/Karpathy direction 09-10) — the data flywheel.

Every losing champion bet becomes an automatically-labelled example
so research is driven by "what are the top recurring failure modes?"
rather than only "what's the Brier score?". Observation-only
(Research/Teaching plane) — reads ledgers, mutates no research state,
touches no experiment.

Emits results/failure_store.json:
  * per-window failure mechanism labels (PRE-ENTRY attributable),
  * the dominant failure cluster over the last N bad windows,
  * the 20 worst windows by economic loss,
  * ONE proposed information improvement, derived from the top cluster.
"""
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_RECENT = 100


def classify(b, mkt_p):
    """Primary failure mechanism for a LOST bet, by priority.
    All labels are attributable to PRE-ENTRY state."""
    p = float(b.get("p_model") or 0.5)
    side_yes = b.get("side") == "yes"
    conf_for_side = p if side_yes else 1 - p     # our conviction
    price = float(b.get("price_c") or 0)
    mins = float(b.get("mins_left") or 99)
    tags = []
    if conf_for_side < 0.5:
        tags.append("BET_AGAINST_OWN_MODEL")     # anomaly
    if conf_for_side >= 0.65:
        tags.append("CONFIDENTLY_WRONG")
    if price >= 70:
        tags.append("EXPENSIVE_LOSER")           # selection should skip
    if mins < 3:
        tags.append("LATE_ENTRY")
    if 0.5 <= conf_for_side < 0.65:
        tags.append("THIN_EDGE_LOST")
    if mkt_p is not None:
        mkt_for_other = (1 - mkt_p) if side_yes else mkt_p
        if mkt_for_other >= 0.62:
            tags.append("MARKET_KNEW")           # market favored other side
    if not tags:
        tags.append("UNCLASSIFIED")
    priority = ["BET_AGAINST_OWN_MODEL", "MARKET_KNEW",
                "CONFIDENTLY_WRONG", "EXPENSIVE_LOSER", "LATE_ENTRY",
                "THIN_EDGE_LOST", "UNCLASSIFIED"]
    primary = next(t for t in priority if t in tags)
    return primary, tags


IMPROVEMENT = {
    "MARKET_KNEW": "market leads on these — need EARLIER or genuinely "
        "INDEPENDENT information (F1 microstructure / F-AV1), not a "
        "better BTC-price model",
    "CONFIDENTLY_WRONG": "confidence is miscalibrated in these states — "
        "add uncertainty / regime context so the model abstains when it "
        "shouldn't be sure",
    "EXPENSIVE_LOSER": "selection should abstain at high entry price — "
        "exactly the execution-selection lever (validated, insufficient "
        "alone); needs an information gate on top",
    "LATE_ENTRY": "late entries are uneconomic — the market has already "
        "priced it; weight early-window information (economic half-life)",
    "THIN_EDGE_LOST": "no real edge here — these are coin-flips; a "
        "residual model should learn delta~=0 and abstain",
    "BET_AGAINST_OWN_MODEL": "ledger anomaly — a bet placed against the "
        "model's own side; investigate the decision path",
    "UNCLASSIFIED": "add instrumentation — this loss is not explained by "
        "current pre-entry features",
}


def main():
    kb = [json.loads(l) for l in
          (RES / "kalshi_binary_log.jsonl").open() if l.strip()]
    mkt = {r.get("ticker"): r.get("mkt_p_up") for r in kb
           if r.get("mkt_p_up") is not None}
    bets = [json.loads(l) for l in (RES / "kb_bets.jsonl").open()
            if l.strip()]
    bets = [b for b in bets if b.get("actual") is not None
            and b.get("pnl_c") is not None]
    bets.sort(key=lambda b: b.get("made_ts", 0))

    bad = [b for b in bets if float(b["pnl_c"]) < 0]
    recent_bad = bad[-N_RECENT:]
    labelled = []
    for b in recent_bad:
        primary, tags = classify(b, mkt.get(b.get("ticker")))
        labelled.append({"ticker": b.get("ticker"),
                         "made_ts": b.get("made_ts"),
                         "pnl_c": round(float(b["pnl_c"]), 0),
                         "p_model": b.get("p_model"),
                         "side": b.get("side"),
                         "price_c": b.get("price_c"),
                         "mins_left": b.get("mins_left"),
                         "primary": primary, "tags": tags})
    clusters = Counter(x["primary"] for x in labelled)
    top_cluster = clusters.most_common(1)[0] if clusters else ("NONE", 0)
    worst20 = sorted(labelled, key=lambda x: x["pnl_c"])[:20]

    doc = {"generated_ts": int(time.time()),
           "plane": "D-Research (data flywheel); observation-only",
           "bets_total": len(bets), "bad_total": len(bad),
           "window_recent_bad": len(recent_bad),
           "failure_clusters": dict(clusters.most_common()),
           "top_cluster": {"mechanism": top_cluster[0],
                           "count": top_cluster[1],
                           "share": round(top_cluster[1]
                                          / max(1, len(labelled)), 3)},
           "proposed_information_improvement":
               IMPROVEMENT.get(top_cluster[0], "n/a"),
           "worst_20_windows": worst20}
    (RES / "failure_store.json").write_text(json.dumps(doc, indent=1))
    print(f"failure_store: {len(bad)} bad of {len(bets)} bets · "
          f"top cluster {top_cluster[0]} ({top_cluster[1]}/"
          f"{len(labelled)}) · clusters {dict(clusters.most_common())}")


if __name__ == "__main__":
    main()
