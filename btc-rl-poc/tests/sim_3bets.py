"""Replay: exactly-3-bets-per-window vs the live exactly-1 policy.

Same rules generalized: strike whenever the called side shows >=3c edge
under 85c (one bet per minute max) until the quota is filled; any
remaining quota is force-filled in the final minutes (called side if
legal, else the only legal side). Prices = logged market mid -/+ 1c.
Emits JSON with per-window cumulative net P&L (after Kalshi fees) for
both policies, for the comparison page.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from btc_rl.metrics import kalshi_fee_c

rows = [json.loads(l) for l in
        (ROOT / "results" / "kalshi_binary_log.jsonl").read_text().splitlines()]
byw = defaultdict(list)
for r in rows:
    if (r.get("variant", "kb") == "kb" and r.get("mkt_p_up") is not None
            and r["actual"] is not None):
        byw[r["ticker"]].append(r)
for v in byw.values():
    v.sort(key=lambda r: r["made_ts"])
windows = sorted(byw, key=lambda t: byw[t][0]["made_ts"])


def leg(side, price, p, outcome):
    won = (side == "yes") == bool(outcome)
    return ((100 - price) if won else -price) - kalshi_fee_c(price), won


def run(quota):
    per_window = []
    wins = bets = 0
    for t in windows:
        calls = byw[t]
        outcome = calls[0]["actual"]
        placed = 0
        pnl = 0.0
        for i, r in enumerate(calls):
            if placed >= quota:
                break
            p = r["p_up"]
            mid = r["mkt_p_up"] * 100
            side = "yes" if p >= 0.5 else "no"
            price = (mid + 1) if side == "yes" else 100 - (mid - 1)
            edge = 100 * (p if side == "yes" else 1 - p) - price
            remaining = len(calls) - i
            must = remaining <= (quota - placed) + 2  # force-fill window end
            if price < 85 and (edge >= 3 or must):
                d, won = leg(side, price, p, outcome)
                pnl += d; wins += won; placed += 1; bets += 1
            elif must:
                other = "no" if side == "yes" else "yes"
                oprice = (mid + 1) if other == "yes" else 100 - (mid - 1)
                if oprice < 85:
                    d, won = leg(other, oprice, p, outcome)
                    pnl += d; wins += won; placed += 1; bets += 1
        per_window.append(round(pnl, 1))
    cum = []
    s = 0.0
    for x in per_window:
        s += x
        cum.append(round(s, 1))
    dd = peak = 0.0
    for v in cum:
        peak = max(peak, v)
        dd = max(dd, peak - v)
    fees = sum(0 for _ in ())  # fees already inside leg()
    return {"bets": bets, "wins": wins, "win_rate": round(wins / bets, 3),
            "net_c": round(cum[-1], 1), "per_bet": round(cum[-1] / bets, 2),
            "max_dd": round(dd, 1), "cum": cum}


out = {"windows": len(windows), "one": run(1), "three": run(3)}
print(json.dumps({k: (v if k == "windows" else {x: y for x, y in v.items()
                                               if x != "cum"})
                  for k, v in out.items()}, indent=1))
(ROOT / "results" / "bet_policy_sim.json").write_text(json.dumps(out))
