"""End-of-life (final-presentation horizon) fund projections, per trader.

Monte Carlo (20k paths) over a 14-day horizon with the desk's measured
parameters: avg entry cost ~75c incl fee (payout 100c), leader-gated
entry rate ~20 bets/day (observed day one), disciplined ~4/day, stake
capped by ~$4,500 live near-touch depth. Win-rate scenarios bracket the
evidence: 0.72 (the all-entries cohort EV of ~-10%/$1), 0.75 (the
break-even line), 0.79 (the desk's live day-one run rate). The
disciplined trader's tier is measured stronger than all-entries; he
gets a conservative +5 points (tier measured ~+20 pts, thin n).

Policies: follower 10% - ladder 10% + bank $1k at $11k x10 -
disciplined 10% on his subset - gambler 33% (all depth-capped).
Reported: median / mean / 5th / 95th percentile terminal funds and
P(ending under $500). Research simulation; not advice.
"""
import random

COST = 75.0          # cents per contract incl fee
PAYOUT = 100.0
DAYS = 14
BETS_D = 20          # leader-gated entries/day (observed)
DISC_D = 4           # disciplined entries/day (observed)
DEPTH_CAP = 450_000  # cents (~$4.5k near-touch, measured proxy)
PATHS = 20_000
START = 100_000

random.seed(7)


def run(policy, p, frac, bets_per_day, ladder=False):
    finals, banked_all, under = [], [], 0
    n = DAYS * bets_per_day
    for _ in range(PATHS):
        k, bank, lvl = START, 0, START
        for _ in range(n):
            stake = min(frac * k, DEPTH_CAP)
            ncon = int(stake // COST)
            if ncon < 1:
                break
            st = ncon * COST
            k -= st
            if random.random() < p:
                k += ncon * PAYOUT
            if ladder:
                while k >= 11 * lvl:
                    bank += lvl
                    k -= lvl
                    lvl *= 10
        total = k + bank
        finals.append(total)
        banked_all.append(bank)
        if total < 50_000:
            under += 1
    finals.sort()
    m = PATHS // 2
    return (finals[m], sum(finals) / PATHS,
            finals[PATHS // 20], finals[-PATHS // 20],
            under / PATHS, sorted(banked_all)[m])


SCEN = [("pessimistic (all-entries cohort)", 0.72),
        ("break-even line", 0.75),
        ("live day-one run rate", 0.79)]
TRADERS = [
    ("Follower 10%", 0.10, BETS_D, False, 0.00),
    ("Ladder 10%+bank", 0.10, BETS_D, True, 0.00),
    ("Disciplined 10% sel.", 0.10, DISC_D, False, 0.05),
    ("Gambler 33%", 0.33, BETS_D, False, 0.00),
]
print(f"horizon {DAYS} days · cost {COST:.0f}c · depth cap "
      f"${DEPTH_CAP/100:,.0f} · {PATHS} paths\n")
for sname, p in SCEN:
    print(f"== win rate {p:.2f} — {sname} ==")
    print(f"{'trader':>20s} {'median':>10s} {'mean':>10s} {'5th pct':>10s} "
          f"{'95th pct':>12s} {'P(<$500)':>9s} {'banked':>9s}")
    for name, frac, bpd, lad, boost in TRADERS:
        med, mean, lo, hi, pu, bk = run(name, min(0.95, p + boost),
                                        frac, bpd, lad)
        print(f"{name:>20s} ${med/100:>9,.0f} ${mean/100:>9,.0f} "
              f"${lo/100:>9,.0f} ${hi/100:>11,.0f} {100*pu:>8.1f}% "
              f"${bk/100:>8,.0f}")
    print()
