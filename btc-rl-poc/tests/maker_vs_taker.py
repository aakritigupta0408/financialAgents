"""Maker vs taker execution on identical sides/timing, offline, from the
true-quote corpus (results/kalshi_history.jsonl, per-minute bid/ask).

Policy under test: at the first mid-window biddable minute (<=10 left),
on the market-favorite side —
  TAKER: buy at the ask, fee in.
  MAKER: post at (side bid + 1c); filled iff a LATER minute's side ask
         <= limit (the market traded down through us); unfilled = no bet.
Reports per-execution EV, fill rate, and the paired delta.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in
        (ROOT / "results" / "kalshi_history.jsonl").open()]
wins = defaultdict(list)
for r in rows:
    wins[r["ticker"]].append(r)
for v in wins.values():
    v.sort(key=lambda r: r["ts"])

fee = lambda c: math.ceil(7 * (c / 100) * (1 - c / 100))
tak = []
mak = []
filled = 0
placed = 0
for rs in wins.values():
    entry_i = None
    for i, r in enumerate(rs):
        if r["mins_left"] <= 10:
            entry_i = i
            break
    if entry_i is None:
        continue
    r = rs[entry_i]
    outcome = r["outcome"]
    fav_yes = r["price_c"] >= 50
    ask = r["yes_ask_c"] if fav_yes else 100 - r["yes_bid_c"]
    bid = r["yes_bid_c"] if fav_yes else 100 - r["yes_ask_c"]
    if not 5 <= ask < 80 or bid < 1:
        continue
    won = int(fav_yes == bool(outcome))
    tak.append((100 - ask - fee(ask)) if won else -(ask + fee(ask)))
    limit = min(bid + 1, ask)   # never worse than the touch
    placed += 1
    hit = False
    for q in rs[entry_i + 1:]:
        qask = q["yes_ask_c"] if fav_yes else 100 - q["yes_bid_c"]
        if qask <= limit:
            hit = True
            break
    if hit:
        filled += 1
        mak.append((100 - limit - fee(limit)) if won
                   else -(limit + fee(limit)))

nt, nm = len(tak), len(mak)
evt = sum(tak) / nt
evm = sum(mak) / nm if nm else float("nan")
print(f"windows with entry: {nt}")
print(f"TAKER : n={nt}, EV {evt:+.2f}c/contract "
      f"({evt/ (sum(abs(x) for x in tak)/nt) if nt else 0:+.1%} rough)")
print(f"MAKER : placed {placed}, filled {filled} "
      f"({filled/placed:.0%} fill rate), EV {evm:+.2f}c/contract on fills")
print(f"per-contract execution delta on filled orders vs taking: "
      f"{evm - evt:+.2f}c")
avg_cost_t = sum(a for a in (abs(x) for x in tak)) / nt
print(f"\nnote: maker fills skew adverse (filled when price moves toward "
      f"you) — this rule prices that honestly.")
