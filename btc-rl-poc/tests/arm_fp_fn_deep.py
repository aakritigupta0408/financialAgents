"""FP/FN deep dive: WHY (mechanism), WHERE (time/price context),
WHAT (regime), WHO (arms alone vs with the market, correlated errors).

Same decision definition as arm_fp_fn.py: one biddable decision per
(arm, window) — earliest row with mins_left <= 12 and conf >= 0.62.
"""
import datetime
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = datetime.timezone(datetime.timedelta(hours=-7))
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]

ARMS = ["kb", "kb2", "kb3", "kb4", "kb6", "kb7", "kb8", "kb9"]  # no kb5:
# its call is an EV price play, not a direction belief — separate species

dec = defaultdict(dict)
for r in kb:
    v = r.get("variant") or "kb"
    if (v not in ARMS or r.get("actual") is None
            or r.get("mins_left") is None or r["mins_left"] > 12
            or max(r["p_up"], 1 - r["p_up"]) < 0.62):
        continue
    tk = r["ticker"]
    if tk not in dec[v] or r["mins_left"] > dec[v][tk]["mins_left"]:
        dec[v][tk] = r

flat = [(v, r) for v in ARMS for r in dec[v].values()]


def kind(r):
    if r["call"] == 1 and not r["actual"]:
        return "FP"
    if r["call"] == 0 and r["actual"]:
        return "FN"
    return "OK"


# ---- WHAT: the regime — how often did windows actually settle UP? ----
win_out = {}
for v, r in flat:
    win_out[r["ticker"]] = r["actual"]
ups = sum(win_out.values())
print(f"=== WHAT · regime ===")
print(f"windows with any biddable decision: {len(win_out)}, "
      f"settled UP {ups} ({100*ups/len(win_out):.1f}%) — "
      f"arms' aggregate UP-call share: "
      f"{100*sum(1 for _, r in flat if r['call'] == 1)/len(flat):.1f}%")
byday = defaultdict(lambda: [0, 0])
for tk, a in win_out.items():
    d = tk.split("-")[1][:7]
    byday[d][0] += 1
    byday[d][1] += a
for d, (n, u) in sorted(byday.items()):
    print(f"  {d}: {u}/{n} up ({100*u/n:.0f}%)")

# ---- WHERE: hour of day (PT) ----
print("\n=== WHERE · error rate by hour (PT), all arms pooled ===")
byh = defaultdict(lambda: [0, 0, 0])   # n, fp, fn
for v, r in flat:
    h = datetime.datetime.fromtimestamp(r["close_ts"], PT).hour
    k = kind(r)
    byh[h][0] += 1
    byh[h][1] += k == "FP"
    byh[h][2] += k == "FN"
for h in sorted(byh):
    n, fp, fn = byh[h]
    bar = "#" * int(40 * (fp + fn) / n)
    print(f"  {h:02d}h  n={n:3d}  FP {100*fp/n:4.1f}%  FN {100*fn/n:4.1f}%"
          f"  wrong {100*(fp+fn)/n:4.1f}%  {bar}")

# ---- WHERE: market decidedness (proxy for distance to strike —
# decision rows carry no spot price; |mkt_p_up-0.5| ~ 0 means the
# window is a knife-edge coin flip at entry) ----
print("\n=== WHERE · error rate by market decidedness |mkt_p_up-0.5| ===")
BINS = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.51)]
byd = defaultdict(lambda: [0, 0, 0])
for v, r in flat:
    if r.get("mkt_p_up") is None:
        continue
    d = abs(r["mkt_p_up"] - 0.5)
    for lo, hi in BINS:
        if lo <= d < hi:
            k = kind(r)
            byd[(lo, hi)][0] += 1
            byd[(lo, hi)][1] += k == "FP"
            byd[(lo, hi)][2] += k == "FN"
            break
for (lo, hi), (n, fp, fn) in sorted(byd.items()):
    if not n:
        continue
    print(f"  |m-.5| {lo:.2f}-{hi:.2f}  n={n:4d}  FP {100*fp/n:4.1f}%  "
          f"FN {100*fn/n:4.1f}%  wrong {100*(fp+fn)/n:4.1f}%")

# ---- WHY: with or against the market? (adverse-selection anatomy) ----
print("\n=== WHY · wrong calls: WITH the market or AGAINST it? ===")
print("(with = arm and market on same side of 50; against = arm disputes)")
tab = defaultdict(lambda: [0, 0])
for v, r in flat:
    if r.get("mkt_p_up") is None:
        continue
    mside = r["mkt_p_up"] >= 0.5
    aside = r["call"] == 1
    k = kind(r)
    grp = "with" if mside == aside else "against"
    tab[grp][0] += 1
    tab[grp][1] += k != "OK"
for g, (n, w) in tab.items():
    print(f"  {g:7s} market: n={n:4d}, wrong {w:3d} ({100*w/n:.1f}%)")

# ---- WHY: does confidence help? error by confidence bucket ----
print("\n=== WHY · wrong% by arm confidence bucket ===")
CB = [(0.62, 0.70), (0.70, 0.77), (0.77, 0.85), (0.85, 1.01)]
byc = defaultdict(lambda: [0, 0, 0])
for v, r in flat:
    c = max(r["p_up"], 1 - r["p_up"])
    for lo, hi in CB:
        if lo <= c < hi:
            k = kind(r)
            byc[(lo, hi)][0] += 1
            byc[(lo, hi)][1] += k == "FP"
            byc[(lo, hi)][2] += k == "FN"
            break
for (lo, hi), (n, fp, fn) in sorted(byc.items()):
    print(f"  conf {lo:.2f}-{hi:.2f}  n={n:4d}  FP {100*fp/n:4.1f}%  "
          f"FN {100*fn/n:4.1f}%  wrong {100*(fp+fn)/n:4.1f}%")

# ---- WHO: correlated or idiosyncratic? wrong-arm count per window ----
print("\n=== WHO · when a window fools arms, how many at once? ===")
per_win = defaultdict(lambda: [0, 0])
for v, r in flat:
    per_win[r["ticker"]][0] += 1
    per_win[r["ticker"]][1] += kind(r) != "OK"
hist = defaultdict(int)
for tk, (n, w) in per_win.items():
    if n >= 4:                      # windows most arms judged
        hist[round(8 * w / n) / 8] += 1
groups = {0.0: "0 wrong", 0.25: "~quarter", 0.5: "~half",
          0.75: "~3/4", 1.0: "ALL wrong"}
tot = sum(hist.values())
acc = defaultdict(int)
for frac, cnt in hist.items():
    key = min(groups, key=lambda g: abs(g - frac))
    acc[key] += cnt
for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
    c = acc.get(g, 0)
    print(f"  {groups[g]:9s}: {c:4d} windows ({100*c/max(1,tot):.0f}%)"
          f"  {'#' * int(40*c/max(1,tot))}")

# ---- WHO: each arm's errors split solo vs herd ----
print("\n=== WHO · per arm: wrong bids that were SOLO vs shared herd ===")
wrong_by_win = defaultdict(set)
for v, r in flat:
    if kind(r) != "OK":
        wrong_by_win[r["ticker"]].add(v)
for v in ARMS:
    solo = herd = 0
    for r in dec[v].values():
        if kind(r) == "OK":
            continue
        others = wrong_by_win[r["ticker"]] - {v}
        if others:
            herd += 1
        else:
            solo += 1
    n = len(dec[v])
    if n:
        print(f"  {v:4s}  wrong {solo+herd:3d}  solo {solo:3d}  "
              f"herd {herd:3d}  (solo share "
              f"{100*solo/max(1,solo+herd):.0f}%)")
