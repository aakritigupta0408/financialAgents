"""One-shot: register the TA decision-layer workstream on the project
board (idempotent)."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "results" / "board.json"
b = json.load(p.open())
ids = {w.get("id") for w in b["workstreams"]}
if "W-decision-layer" not in ids:
    b["workstreams"].append({
        "id": "W-decision-layer",
        "status": "active",
        "note": ("TA metrics program: decision layer live "
                 "(decision_board.json — CI/P(Δ>0)/MDE/power/veto/"
                 "slices/states, 10-min cron). Queued next: post-fill "
                 "markout + capacity metrics (need new capture), "
                 "offline→online retention tracker, Brier "
                 "decomposition + ECE, risk–coverage curves, "
                 "redundancy/kill board, factorial interaction for "
                 "combo treatments")})
    json.dump(b, p.open("w"), indent=1)
    print("workstream added")
else:
    print("already present")
