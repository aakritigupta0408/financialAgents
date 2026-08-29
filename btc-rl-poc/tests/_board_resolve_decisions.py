"""One-shot: mark D-gambler-sizing and D-m1-future resolved on the
project board (owner decisions, 2026-08-29). Idempotent; nothing is
deleted — resolved cards keep their history."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "results" / "board.json"
b = json.load(p.open())
for d in b.get("open_decisions", []):
    if d.get("id") == "D-gambler-sizing" and d.get("status") != "resolved":
        d["status"] = "resolved"
        d["resolution"] = ("owner 08-29: KEEP 33% (the aggressive "
                           "curriculum is the exhibit) + v3 profit "
                           "sweep — any settle lifting bankroll above "
                           "the $10k start is withdrawn on the spot "
                           "(wd_c ledger on the row), withdrawals "
                           "never re-staked, fresh $10k at cutover "
                           "PT4_RESET2_TS")
    if d.get("id") == "D-m1-future" and d.get("status") != "resolved":
        d["status"] = "resolved"
        d["resolution"] = ("owner 08-29: redesign with SHORTER memory "
                           "(150→50 windows, warm 20, refit 3) AND "
                           "keep shadow-only as a drift instrument — "
                           "p_m1 is never a decision input; "
                           "config-sovereign from_dict so the retune "
                           "actually takes effect")
json.dump(b, p.open("w"), indent=1)
print("resolved:", [d.get("id") for d in b.get("open_decisions", [])
                    if d.get("status") == "resolved"])
