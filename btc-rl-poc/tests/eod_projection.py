"""What EOD return does each treatment imply, at honest sizing?

EV per $1 staked is not a daily return — it must be multiplied by how
much of the bankroll is actually at risk per day. This converts each
treatment's measured per-window EV into an end-of-day return using
half-Kelly sizing (the project's standing rule) and the treatment's own
measured bet frequency, then reports the whole distribution rather than
a single flattering number.

Everything here is a PROJECTION from backfilled windows, not a live
result. The SPRT verdicts on the tracker are the real gate.
"""
import datetime
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = datetime.timezone(datetime.timedelta(hours=-7))
recs = [json.loads(l) for l in
        (ROOT / "results" / "treatments.jsonl").open() if l.strip()]
st = json.loads((ROOT / "results" / "online_status.json").read_text())
labels = {t["key"]: t["label"] for t in st.get("treatments", [])}

days = defaultdict(set)
for r in recs:
    d = datetime.datetime.fromtimestamp(r["close_ts"], PT).strftime("%m/%d")
    days[d].add(r["ticker"])
nday = max(1, len(days))
wpd = len(recs) / nday                      # windows per day observed

print(f"{len(recs)} windows over {nday} days ({wpd:.0f} windows/day)\n")
print(f"{'treatment':42s} {'bets/day':>9s} {'EV/$1':>8s} "
      f"{'½-Kelly f':>10s} {'EOD return':>11s}")
rows = []
for key in recs[0]["ev"]:
    evs = [r["ev"][key] for r in recs if r["ev"].get(key) is not None]
    if not evs:
        continue
    n = len(evs)
    ev = sum(evs) / n
    bets_day = n / nday
    # half-Kelly on the measured win rate at the measured price:
    # f* = (p(1+b) - 1)/b ; b = net odds. Recover p and b from EV and
    # the observed win fraction so sizing is not invented.
    wins = sum(1 for e in evs if e > 0)
    p = wins / n
    b = (sum(e for e in evs if e > 0) / wins) if wins else 0.0
    f = 0.0
    if b > 0:
        f = max(0.0, (p * (1 + b) - 1) / b)
    f = min(0.10, 0.5 * f)                  # half-Kelly, 10% cap
    # compounding across the day's bets
    g = 0.0
    if f > 0:
        g = bets_day * math.log(1 + f * ev) if 1 + f * ev > 0 else -1
    eod = math.exp(g) - 1 if g > -1 else -1
    rows.append((labels.get(key, key), bets_day, ev, f, eod))
REAL = {"champion_real", "t_exec", "t_exec_reg"}
key_of = {labels.get(k, k): k for k in recs[0]["ev"]}
for lab, bd, ev, f, eod in sorted(rows, key=lambda x: -x[4]):
    tag = "  REAL FILLS" if key_of.get(lab) in REAL else ""
    print(f"{lab[:42]:42s} {bd:9.1f} {100*ev:+7.2f}% {100*f:9.1f}% "
          f"{100*eod:+10.1f}%{tag}")

print("\n" + "!" * 68)
print("PRICING BASIS WARNING — read before believing any row above.")
print("!" * 68)
print("Rows without the REAL FILLS tag are priced from the modelled")
print("decision-time quote, which was measured to sit 2.1c BELOW the")
print("ask the desk actually pays. That understates cost on every")
print("entry, so those EOD figures are UPPER BOUNDS, not forecasts.")
print("The same artifact inflated a treatment to +23% paired before it")
print("was caught. Only the REAL FILLS rows reflect achievable fills.")
real_rows = [r for r in rows if key_of.get(r[0]) in REAL]
best = max(real_rows, key=lambda r: r[4]) if real_rows else None
if best:
    print(f"\nBest honest (real-fill) projection: {best[0]} at "
          f"{100*best[4]:+.1f}% EOD")
    if best[4] < 0.20:
        print(f"  -> BELOW a 20% EOD target by "
              f"{100*(0.20-best[4]):.1f} points. Claiming 20% from the")
        print("     model-priced rows would be repeating the exact error")
        print("     this audit exists to catch.")
    else:
        print("  -> clears a 20% EOD target on the honest basis.")

print("\nreading:")
print("  EV/$1 is per WINDOW ENTERED; EOD return compounds it over the")
print("  day's entries at half-Kelly, so a selective treatment with a")
print("  huge per-bet edge can still post a modest daily return, and a")
print("  frequent one with a thin edge can beat it.")
print("  These are PROJECTIONS from backfilled windows. The SPRT")
print("  verdicts on the tracker are the gate that actually promotes.")
