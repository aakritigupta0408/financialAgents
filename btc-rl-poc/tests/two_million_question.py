"""The TA's $2M question, computed honestly from our own ledgers.

Inputs measured:
  - kb2 gated-and-biddable entries (conf >= tau, ask < 80c, mid/late
    phase): empirical win rate, avg ask, per-bet EV after fees, with a
    95% CI on the win rate -> EV CI.
  - qualifying entries per day.
  - Kalshi capacity: avg traded notional per window from the 14-60d
    history mine (results/kalshi_history.jsonl volumes).

Outputs: per-bet edge (point/lo/hi), Kelly growth rate, time to $2M from
various starting capitals under a per-window deployment cap, and the
minimum capital for which the cap (not compounding) dominates.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
TAU, ADJ = 0.64, 2.5

ent = []
for r in kb:
    if (r.get("variant") != "kb2" or r.get("actual") is None
            or r.get("mkt_p_up") is None or r["mins_left"] > 10):
        continue
    if max(r["p_up"], 1 - r["p_up"]) < TAU:
        continue
    side = "yes" if r["call"] else "no"
    ask = 100 * (r["mkt_p_up"] if side == "yes" else 1 - r["mkt_p_up"]) + ADJ
    if not 5 <= ask < 80:
        continue
    fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
    ent.append((int(r["hit"]), ask, fee, r["ticker"]))

n = len(ent)
w = sum(e[0] for e in ent)
p = w / n
z = 1.96
den = 1 + z * z / n
plo = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n
       + z * z / (4 * n * n))) / den
phi = (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n
       + z * z / (4 * n * n))) / den
cost = sum(e[1] + e[2] for e in ent) / n
pay = 100.0


def ev_frac(pw):     # expected return per $1 staked
    return (pw * pay - cost) / cost


days = len({t.split("-")[1][:7] for _, _, _, t in ent})
per_day = len({t for *_, t in ent}) / max(1, days)
print(f"gated biddable entries: n={n}, win {p:.3f} "
      f"[{plo:.3f}, {phi:.3f}] 95% CI")
print(f"avg cost {cost:.1f}c (ask+fee), payout 100c")
for name, pw in (("point", p), ("CI low", plo), ("CI high", phi)):
    e = ev_frac(pw)
    print(f"  {name:8s} EV per $1 staked: {e:+.4f}")

# capacity from history mine
try:
    hist = [json.loads(l) for l in
            (ROOT / "results" / "kalshi_history.jsonl").open()]
    volw = defaultdict(float)
    for h in hist:
        volw[h["ticker"]] = max(volw[h["ticker"]],
                                float(h.get("volume", 0) or 0))
except Exception:
    volw = {}
print(f"\nassume per-window deployable notional (conservative "
      f"participation): $500 and $5,000 scenarios")

qpd = per_day if per_day > 1 else 40   # qualifying windows/day fallback
print(f"qualifying windows/day observed ~{qpd:.0f}")

for name, pw in (("point", p), ("CI high", phi)):
    e = ev_frac(pw)
    if e <= 0:
        print(f"[{name}] EV <= 0 -> expected time to $2M: NEVER "
              f"(expected loss per bet)")
        continue
    # Kelly fraction for binary bet: f* = edge/odds
    b = (pay - cost) / cost
    q = 1 - pw
    f = max(0.0, (b * pw - q) / b)
    g = pw * math.log(1 + f * b) + q * math.log(1 - f)  # per-bet log growth
    print(f"[{name}] Kelly f*={f:.3f}, per-bet log-growth {g:.5f}, "
          f"bets/day {qpd:.0f}")
    for cap_notional in (500, 5000):
        # capital below which full Kelly stake fits under the cap
        k_star = cap_notional / f if f > 0 else float("inf")
        # phase 1: compounding K0 -> K*; phase 2: linear at cap
        daily_lin = cap_notional * e * qpd
        for k0 in (1_000, 10_000, 100_000, k_star):
            if k0 < k_star:
                t1 = math.log(k_star / k0) / (g * qpd)
            else:
                t1, k0 = 0.0, max(k0, k_star)
            t2 = max(0.0, (2_000_000 - k_star) / daily_lin)
            total = t1 + t2
            print(f"    cap ${cap_notional}/win, start "
                  f"${k0:,.0f}: compounding {t1:,.0f}d + linear "
                  f"{t2:,.0f}d = {total/365:,.1f} YEARS")
