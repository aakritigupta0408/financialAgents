"""Two questions the re-triage raises, before we change the plan:

Q1. Was 08/27 detectable EX ANTE? If the market's own predictability
    (its Brier on settled windows) collapsed that day, a regime gate
    could have stood us down. Uses only information available before
    each window settles — trailing market Brier, not same-day hindsight.

Q2. Is the "toxic hour" real, or an artifact of the one bad day?
    An hour effect that only exists on 08/27 is the regime wearing a
    clock costume; shipping M7 on it would be superstition.
"""
import datetime
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = datetime.timezone(datetime.timedelta(hours=-7))


def load(n):
    return [json.loads(l) for l in (ROOT / "results" / n).open()
            if l.strip()]


def day(ts):
    return datetime.datetime.fromtimestamp(ts, PT).strftime("%m/%d")


def hour(ts):
    return datetime.datetime.fromtimestamp(ts, PT).hour


kb = load("kalshi_binary_log.jsonl")
# one decision per window from the CONTROL arm (kb) + the market's quote
win = {}
for r in kb:
    if (r.get("variant") or "kb") != "kb" or r.get("actual") is None \
            or r.get("mins_left") is None or r["mins_left"] > 12 \
            or r.get("mkt_p_up") is None:
        continue
    tk = r["ticker"]
    if tk not in win or r["mins_left"] > win[tk]["mins_left"]:
        win[tk] = r
seq = sorted(win.values(), key=lambda r: r["close_ts"])
print(f"windows with a market quote: {len(seq)}")

print("\n" + "=" * 68)
print("Q1 · was 08/27 detectable in advance?")
print("=" * 68)
print("MARKET Brier per day (the market's own skill — our regime proxy)")
byd = defaultdict(list)
for r in seq:
    byd[day(r["close_ts"])].append(r)
for d in sorted(byd):
    rs = byd[d]
    mb = sum((r["mkt_p_up"] - r["actual"]) ** 2 for r in rs) / len(rs)
    ob = sum((r["p_up"] - r["actual"]) ** 2 for r in rs) / len(rs)
    macc = sum(1 for r in rs
               if (r["mkt_p_up"] >= 0.5) == bool(r["actual"])) / len(rs)
    print(f"  {d}: n={len(rs):3d} · market Brier {mb:.3f} · market acc "
          f"{100*macc:5.1f}% · our Brier {ob:.3f}")

print("\nTRAILING (ex-ante) signal: market accuracy over the previous 20")
print("settled windows, as it stood at each window's entry —")
print("could a gate have seen 08/27 coming?")
hist = []
flagged = defaultdict(lambda: [0, 0])
for r in seq:
    prev = hist[-20:]
    if len(prev) >= 20:
        acc = sum(prev) / len(prev)
        d = day(r["close_ts"])
        flagged[d][0] += 1
        if acc < 0.62:                    # candidate stand-down trigger
            flagged[d][1] += 1
    hist.append(1 if (r["mkt_p_up"] >= 0.5) == bool(r["actual"]) else 0)
for d in sorted(flagged):
    n, f = flagged[d]
    print(f"  {d}: {f:3d}/{n:3d} windows ({100*f/max(1,n):5.1f}%) would "
          f"have been flagged 'low predictability' BEFORE trading")

print("\n" + "=" * 68)
print("Q2 · is the toxic hour real, or is it just 08/27?")
print("=" * 68)
pt = [t for t in load("pt_trades.jsonl") if t.get("actual") is not None]
grid = defaultdict(lambda: [0, 0])
for t in pt:
    g = grid[(hour(t["close_ts"]), day(t["close_ts"]))]
    g[0] += 1
    g[1] += 0 if t["win"] else 1
hours = sorted({h for h, _ in grid})
days = sorted({d for _, d in grid})
print("     " + "".join(f"{d:>12s}" for d in days) + "      TOTAL")
suspect = []
for h in hours:
    cells, tn, tl = [], 0, 0
    for d in days:
        n, l = grid.get((h, d), [0, 0])
        tn += n
        tl += l
        cells.append(f"{l}/{n}" if n else "·")
    if tn < 6:
        continue
    daysbad = sum(1 for d in days
                  if grid.get((h, d), [0, 0])[0] >= 3
                  and grid[(h, d)][1] / grid[(h, d)][0] > 0.5)
    daysseen = sum(1 for d in days if grid.get((h, d), [0, 0])[0] >= 3)
    mark = ""
    if tl / tn > 0.45:
        mark = ("  <- BAD on %d of %d days" % (daysbad, daysseen)
                + ("  [ARTIFACT: one day only]"
                   if daysbad <= 1 and daysseen >= 2 else "  [PERSISTENT]"))
        suspect.append((h, daysbad, daysseen))
    print(f"  {h:02d}h" + "".join(f"{c:>12s}" for c in cells)
          + f"   {tl}/{tn}{mark}")
print("\nverdict:")
if not suspect:
    print("  no hour clears the 45%-loss bar -> M7 has no target: DROP")
else:
    per = [h for h, bd, ds in suspect if bd >= 2]
    art = [h for h, bd, ds in suspect if bd <= 1]
    print(f"  persistent bad hours (bad on >=2 separate days): "
          f"{per or 'NONE'}")
    print(f"  single-day artifacts (regime wearing a clock costume): "
          f"{art or 'none'}")
    if not per:
        print("  -> M7 rests entirely on single-day artifacts: DROP IT")
    else:
        print(f"  -> hours {per} look persistent BEFORE multiplicity")

# --- multiplicity: how many "persistent bad hours" would pure luck give?
print("\n" + "-" * 68)
print("MULTIPLICITY CHECK — we tested 24 hours; how many would look")
print("'bad on >=2 of 3 days' by CHANCE at the desk's own loss rate?")
base = sum(1 for t in pt if not t["win"]) / len(pt)


def binom_at_least(k, n, p):
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


# a day-cell is "bad" if >half its bids lost; cells here run 3-4 bids
cell_bad_3 = binom_at_least(2, 3, base)      # 2+ of 3 lost
cell_bad_4 = binom_at_least(3, 4, base)      # 3+ of 4 lost
cell_bad = (cell_bad_3 + cell_bad_4) / 2     # typical cell size 3-4
p_persist = binom_at_least(2, 3, cell_bad)   # bad on 2+ of 3 days
exp_hours = 24 * p_persist
print(f"  desk baseline loss rate: {100*base:.1f}%")
print(f"  P(a 3-4 bid day-cell looks 'bad'): {100*cell_bad:.1f}%")
print(f"  P(an hour looks bad on >=2 of 3 days | pure chance): "
      f"{100*p_persist:.1f}%")
print(f"  => expected 'persistent bad hours' from luck alone: "
      f"{exp_hours:.2f} of 24")
print(f"  => actually observed: {len(per)}")
# a count vs a mean is not a test — use the binomial tail across the
# 24 hours we searched (this IS the multiple-comparisons correction)
pval = binom_at_least(len(per), 24, p_persist) if per else 1.0
print(f"  => P(>= {len(per)} such hours | pure chance, 24 searched) = "
      f"{pval:.2f}")
if pval > 0.05:
    print("\n  VERDICT: NOT SIGNIFICANT (p > 0.05). The hour pattern is")
    print("  indistinguishable from what searching 24 hours produces by")
    print("  chance -> M7 SHOULD BE DROPPED. Shipping it would be data")
    print("  dredging (Arnott, Harvey & Markowitz 2019, backtest")
    print("  protocol). Revisit only with a pre-registered hypothesis")
    print("  and >= 3x this sample.")
else:
    print(f"\n  VERDICT: significant (p = {pval:.3f}) — M7 may proceed")
    print(f"  for hours {per}, with a pre-registered re-check at 2x n.")
