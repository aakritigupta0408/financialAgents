"""Noise-floor estimate: how good can ANY predictor get, per horizon?

Persistence MAE = E|price(t+h) - price(t)| — the score of predicting "no
change". If prices were a pure martingale this is the exact optimum; any
real model's headroom below it is bounded by the (small) predictable
fraction of the move.
"""
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
prices = json.load(open(RES / "recent_prices.json"))
by_ts = {p["ts"]: p["c"] for p in prices}
ts_sorted = sorted(by_ts)

print(f"bars: {len(ts_sorted)}  span: {(ts_sorted[-1]-ts_sorted[0])/3600:.1f}h"
      f"  price ~${by_ts[ts_sorted[-1]]:,.0f}")
for h in (1, 5, 15, 30):
    moves = [by_ts[t + h * 60] - by_ts[t] for t in ts_sorted
             if t + h * 60 in by_ts]
    if len(moves) < 30:
        continue
    mabs = sum(abs(m) for m in moves) / len(moves)
    mu = sum(moves) / len(moves)
    sd = math.sqrt(sum((m - mu) ** 2 for m in moves) / len(moves))
    up = sum(1 for m in moves if m > 0) / len(moves)
    # Gaussian check: for a centered normal, E|x| = sd * sqrt(2/pi)
    print(f"  h{h:>2}: persistence-floor MAE ${mabs:6.0f}   sigma ${sd:6.0f} "
          f"(gauss E|x| ${sd * math.sqrt(2/math.pi):6.0f})   P(up) {up:.2f}   "
          f"n={len(moves)}")
