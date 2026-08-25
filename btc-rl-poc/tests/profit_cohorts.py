"""Mine the true-quote corpus for honestly positive-EV betting cohorts.

Grid over (phase bucket x price bucket x favorite/underdog), train on the
first 70% of windows, keep cohorts with EV>0 and n>=80 on train, then
validate each on the last 30%. The survivors define the profit book's
entry rule. Taker fills at the true ask, fees in, no leakage.
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
tickers = sorted(wins, key=lambda t: wins[t][0]["ts"])
cut = tickers[int(len(tickers) * .7)]
cut_ts = wins[cut][0]["ts"]

fee = lambda c: math.ceil(7 * (c / 100) * (1 - c / 100))
PHASES = (("early", 10.5, 15.1), ("mid", 5.5, 10.5), ("late", 0, 5.5))
PRICES = ((5, 20), (20, 35), (35, 50), (50, 65), (65, 80))


def entries(ts_filter):
    out = defaultdict(list)
    for tk in tickers:
        rs = wins[tk]
        if not ts_filter(rs[0]["ts"]):
            continue
        seen = set()
        for r in rs:
            for ph, lo, hi in PHASES:
                if not lo <= r["mins_left"] < hi or (tk, ph) in seen:
                    continue
                for fav in (True, False):
                    yes = (r["price_c"] >= 50) == fav
                    ask = r["yes_ask_c"] if yes else 100 - r["yes_bid_c"]
                    for plo, phi2 in PRICES:
                        if not plo <= ask < phi2:
                            continue
                        won = int(yes == bool(r["outcome"]))
                        net = ((100 - ask - fee(ask)) if won
                               else -(ask + fee(ask)))
                        out[(ph, (plo, phi2), fav)].append(net / ask)
                seen.add((tk, ph))
    return out


tr = entries(lambda ts: ts < cut_ts)
te = entries(lambda ts: ts >= cut_ts)
print(f"{'cohort':34s} {'nTr':>4s} {'evTr':>7s} {'nTe':>4s} {'evTe':>7s}")
survivors = []
for k, v in sorted(tr.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    if len(v) < 80:
        continue
    evtr = sum(v) / len(v)
    if evtr <= 0.01:
        continue
    tv = te.get(k, [])
    evte = sum(tv) / len(tv) if tv else float("nan")
    ph, (plo, phi2), fav = k
    name = f"{ph:5s} {plo}-{phi2}c {'favorite' if fav else 'underdog'}"
    print(f"{name:34s} {len(v):4d} {evtr:+7.1%} {len(tv):4d} {evte:+7.1%}")
    if tv and evte > 0.01 and len(tv) >= 30:
        survivors.append((name, k, evtr, evte, len(tv)))
print("\nheld-out SURVIVORS (train+test both positive):")
for name, k, a, b, n in survivors:
    print(f"  {name}: train {a:+.1%}, test {b:+.1%} (n={n})")
