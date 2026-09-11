"""LONG-RUN end of life per trader: analytic growth rates + 180-day MC.

Per-bet log-growth for fraction f at cost c (cents), win prob p:
    g(f) = p*ln(1 + f*b) + (1-p)*ln(1 - f)   with b = (100-c)/c
g > 0  -> survives and compounds until the depth cap, then linear
g < 0  -> median decays exponentially: practical ruin, just slower
The Kelly fraction f* = (p*(1+b) - 1)/b marks the growth optimum;
beyond ~2f* growth goes negative even WITH a real edge.
"""
import math
import random

COST = 75.0
B = (100 - COST) / COST
DAYS = 180
BETS_D = 20
DISC_D = 4
DEPTH_CAP = 450_000
PATHS = 5_000
START = 100_000
random.seed(11)


def g(f, p):
    return p * math.log(1 + f * B) + (1 - p) * math.log(1 - f)


print("analytic per-bet log growth g(f) [x1000] and Kelly f*:")
print(f"{'p':>5s} {'Kelly f*':>9s} {'g(10%)':>8s} {'g(33%)':>8s} "
      f"{'verdict 10% / 33%':>28s}")
for p in (0.72, 0.75, 0.79, 0.84):
    fstar = max(0.0, (p * (1 + B) - 1) / B)
    g10, g33 = g(0.10, p), g(0.33, p)
    v = ("dies" if g10 < 0 else "compounds") + " / " + \
        ("dies" if g33 < 0 else "compounds")
    print(f"{p:5.2f} {fstar:9.2f} {1000*g10:8.2f} {1000*g33:8.2f} {v:>28s}")
print()


def run(p, frac, bpd, ladder=False):
    finals, banked_all, ruined = [], [], 0
    n = DAYS * bpd
    for _ in range(PATHS):
        k, bank, lvl = START, 0, START
        for _ in range(n):
            stake = min(frac * k, DEPTH_CAP)
            ncon = int(stake // COST)
            if ncon < 1:
                ruined += 1
                break
            st = ncon * COST
            k -= st
            if random.random() < p:
                k += ncon * 100
            if ladder:
                while k >= 11 * lvl:
                    bank += lvl
                    k -= lvl
                    lvl *= 10
        finals.append(k + bank)
        banked_all.append(bank)
    finals.sort()
    m = PATHS // 2
    return (finals[m], finals[-PATHS // 20], ruined / PATHS,
            sorted(banked_all)[m])


print(f"180-day MC ({PATHS} paths): median / 95th pct / P(practical ruin) "
      f"/ median banked")
for sname, p in (("p=0.75 break-even", 0.75), ("p=0.79 live rate", 0.79)):
    print(f"== {sname} ==")
    for name, frac, bpd, lad, boost in (
            ("Follower 10%", .10, BETS_D, False, 0),
            ("Ladder 10%+bank", .10, BETS_D, True, 0),
            ("Disciplined sel.", .10, DISC_D, False, .05),
            ("Gambler 33%", .33, BETS_D, False, 0)):
        med, hi, ru, bk = run(min(.95, p + boost), frac, bpd, lad)
        print(f"  {name:>17s}: median ${med/100:>10,.0f} · 95th "
              f"${hi/100:>11,.0f} · ruin {100*ru:5.1f}% · banked "
              f"${bk/100:>9,.0f}")
