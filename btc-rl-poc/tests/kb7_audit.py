"""kb7 forensics — is it too good to be true?
(a) interval coverage: do its 80% bands contain ~80% of outcomes?
(b) calibration by confidence bucket (stated vs delivered)
(c) window-clustered significance vs market (refresh)
(d) the 13/14 biddable wins: time-clustered in one regime day?
"""
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb7"
        and r.get("actual") is not None]

# (a) interval coverage — needs settle price; use base at settle? we
# don't store settle price on kb7 rows, but hit + call + strike + bounds
# let us check: outcome close >= strike; interval covers strike-side?
# Proper check: was the SETTLE CLOSE inside [q80_lo, q80_hi]? We don't
# store the settle close on rows... derive from any kb row of the same
# window? kb rows don't store it either. Use recent_prices minute closes.
px = {}
try:
    for q in json.loads((ROOT / "results" / "recent_prices.json").read_text()):
        px[q["ts"]] = q["c"]
except Exception:
    pass
bounded = [r for r in rows if r.get("q80_lo") is not None]
cov = tot = 0
for r in bounded:
    close = px.get(r["close_ts"])
    if close is None:
        continue
    tot += 1
    cov += int(r["q80_lo"] <= close <= r["q80_hi"])
print(f"(a) interval coverage: {cov}/{tot} settled closes inside the 80% "
      f"band ({cov/max(1,tot):.0%}; target ~80%)")

# (b) calibration buckets
print("(b) stated confidence vs delivered:")
bucks = defaultdict(list)
for r in rows:
    c = max(r["p_up"], 1 - r["p_up"])
    bucks[round(c * 10) / 10].append(r["hit"])
for b in sorted(bucks):
    v = bucks[b]
    if len(v) >= 15:
        print(f"    conf ~{b:.1f}: delivered {sum(v)/len(v):.1%} (n={len(v)})")

# (c) clustered significance vs market
byw = defaultdict(list)
for r in rows:
    if r.get("mkt_p_up") is not None:
        byw[r["ticker"]].append(
            (r["p_up"] - r["actual"]) ** 2
            - (r["mkt_p_up"] - r["actual"]) ** 2)
means = [sum(v) / len(v) for v in byw.values()]
n = len(means)
if n >= 5:
    m = sum(means) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (n - 1))
    t = m / (sd / math.sqrt(n))
    print(f"(c) window-clustered Brier vs market: {n} windows, "
          f"mean diff {m:+.4f}, t = {t:.2f} "
          f"({'significant' if abs(t) > 2 else 'not yet'})")

# (d) the biddable-confident wins by day
ent = [r for r in rows if r.get("mkt_p_up") is not None
       and r["mins_left"] <= 10
       and max(r["p_up"], 1 - r["p_up"]) >= 0.70]
bid = []
for r in ent:
    side = "yes" if r["call"] else "no"
    ask = 100 * (r["mkt_p_up"] if side == "yes" else 1 - r["mkt_p_up"]) + 2.5
    if 5 <= ask < 80:
        bid.append(r)
byday = defaultdict(lambda: [0, 0])
for r in bid:
    d = datetime.fromtimestamp(r["made_ts"], PT).strftime("%m-%d")
    byday[d][0] += 1
    byday[d][1] += r["hit"]
print("(d) biddable confident entries by day:",
      {d: f"{h}/{n2}" for d, (n2, h) in sorted(byday.items())})
uniq = len({r["ticker"] for r in bid})
print(f"    across {uniq} distinct windows "
      f"({len(bid)} minute-entries) — effective n is the window count")
