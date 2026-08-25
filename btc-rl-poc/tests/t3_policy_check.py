"""The T-3 favored-side policy on the full corpus, chronological split:
enter once per window at the last minute with mins_left <= 3.5, favored
side, true ask < 80c, taker fill, fees in."""
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
cut_ts = wins[tickers[int(len(tickers) * .7)]][0]["ts"]
fee = lambda c: math.ceil(7 * (c / 100) * (1 - c / 100))


def run(ts_filter):
    evs = []
    for tk in tickers:
        rs = wins[tk]
        if not ts_filter(rs[0]["ts"]):
            continue
        cand = [r for r in rs if r["mins_left"] <= 3.5]
        if not cand:
            continue
        r = cand[0]
        yes = r["price_c"] >= 50
        ask = r["yes_ask_c"] if yes else 100 - r["yes_bid_c"]
        if not 5 <= ask < 80:
            continue
        won = int(yes == bool(r["outcome"]))
        net = (100 - ask - fee(ask)) if won else -(ask + fee(ask))
        evs.append((net, ask, won))
    return evs


for name, f in (("train", lambda ts: ts < cut_ts),
                ("test ", lambda ts: ts >= cut_ts)):
    e = run(f)
    n = len(e)
    net = sum(x[0] for x in e)
    stake = sum(x[1] for x in e)
    wr = sum(x[2] for x in e) / n
    # win-rate CI -> EV CI at avg cost
    import math as m2
    z = 1.96
    den = 1 + z * z / n
    half = z * m2.sqrt(wr * (1 - wr) / n + z * z / (4 * n * n))
    plo = (wr + z * z / (2 * n) - half) / den
    cost = (stake + sum(fee(x[1]) for x in e)) / n
    evlo = (plo * 100 - cost) / cost
    print(f"{name}: n={n}, win {wr:.1%}, net {net:+.0f}c on {stake:.0f}c "
          f"staked = EV {net/stake:+.2%}/$1  (CI-low EV {evlo:+.2%})")
