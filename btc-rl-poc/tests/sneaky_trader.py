"""The stealth trader — can execution discipline avoid being caught?

Kyle-(1985)-flavored detection model: market makers estimate flow
toxicity. The trader's HEAT rises with his take fraction of near-touch
depth and decays when he abstains; his effective edge decays linearly
in heat, and at full heat he is 'caught' — quoted out entirely.

    heat <- lam * heat + alpha * (stake / D)      each window
    e_eff = e * max(0, 1 - heat / hmax)

Policies compared, per arm EV scenario:
  loud     25% of bankroll, every qualifying window (the naive $2M math)
  capped   never exceed D/2 per window, still every window
  stealth  take fraction f of D, participate in fraction q of windows,
           (f, q) grid-searched — the sim discovers Kyle's answer
Closed form for the steady state: with x = f*q and
beta = alpha / ((1 - lam) * hmax), extraction/day ∝ x * (1 - beta*x),
maximized at x* = 1/(2*beta): harvest HALF of what detection tolerates.

Models sizing/timing discipline ONLY — no manipulation, no multi-
account games, nothing that breaks exchange rules. Research simulation
for a class; not advice; nothing is traded.
"""
import itertools

D = 500.0          # near-touch depth per window ($, measured proxy)
BETS_D = 30        # qualifying windows per day
LAM = 0.97         # heat decay per window (half-life ~23 windows)
ALPHA = 0.06       # heat added by taking the full touch once
HMAX = 1.0         # heat at which the trader is fully quoted out
BETA = ALPHA / ((1 - LAM) * HMAX)
YEARS_CAP = 200


def simulate(e, frac_of_D, q, start=1000.0, greedy_bankroll=False):
    """Deterministic EV sim. Returns (days_to_2M or None, total, caught_day).
    caught_day = first day e_eff fell below 5% of e (loud gets this)."""
    K, heat, caught = start, 0.0, None
    days = 0
    while K < 2e6 and days < 365 * YEARS_CAP:
        for _ in range(BETS_D):
            e_eff = e * max(0.0, 1.0 - heat / HMAX)
            stake = min(0.25 * K, D) if greedy_bankroll \
                else min(0.10 * K, frac_of_D * D)
            take = stake if greedy_bankroll else (stake * q)
            K += take * e_eff                      # expected value per window
            heat = LAM * heat + ALPHA * (take / D)
        days += 1
        if caught is None and e * 0.05 > e * max(0.0, 1.0 - heat / HMAX):
            caught = days
    return (days if K >= 2e6 else None), K, caught


def fdays(d):
    if d is None:
        return f">{YEARS_CAP}y"
    return f"{d}d" if d < 365 else f"{d/365:.1f}y"


SCEN = [("kb2 measured", -0.102), ("kb4 measured", -0.168),
        ("kb7 measured", 0.028), ("kb5 warm-start", 0.189),
        ("hypothetical +5%", 0.05), ("hypothetical +10%", 0.10)]

xstar = min(1.0, 1.0 / (2 * BETA))
print(f"detection: beta={BETA:.1f} -> optimal take x* = {xstar:.2f} of "
      f"depth (Kyle: harvest half of what detection tolerates)")
print(f"{'scenario':>18s} {'per $1':>7s} {'loud: caught':>13s} "
      f"{'loud total':>11s} {'stealth $/day':>13s} {'stealth $2M':>12s} "
      f"{'best (f,q)':>11s}")
for nm, e in SCEN:
    if e <= 0:
        print(f"{nm:>18s} {100*e:6.1f}%          –           –  "
              f"        never        never          –")
        continue
    _, k_loud, caught = simulate(e, 1.0, 1.0, greedy_bankroll=True)
    # grid-search stealth
    best = None
    for f, q in itertools.product((.1, .2, .3, .4, .5, .75, 1.0),
                                  (.2, .4, .6, .8, 1.0)):
        d2m, _, _ = simulate(e, f, q)
        if d2m is not None and (best is None or d2m < best[0]):
            best = (d2m, f, q)
    x = best[1] * best[2] if best else 0
    daily = BETS_D * x * D * e * (1 - BETA * x) if best else 0
    print(f"{nm:>18s} {100*e:6.1f}% {fdays(caught):>13s} "
          f"${k_loud - 1000:>9,.0f} {daily:>12,.0f} "
          f"{fdays(best[0]) if best else '>200y':>12s} "
          f"{f'({best[1]:.2f},{best[2]:.1f})' if best else '–':>11s}")
print("\nloud total = all he extracts before markets quote him out;"
      "\nstealth = grid-searched (f, q); closed-form optimum x* matches.")
