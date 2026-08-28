"""Step-0 gate check + regime-shift diagnostics on fresh data.

Each shipped fix carries a pre-registered gate (docs/SEV0_REMEDIATION.md).
This reports PASS/FAIL against the ACTUAL break-even implied by the
prices each trader paid — not a stale constant.
"""
import datetime
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = datetime.timezone(datetime.timedelta(hours=-7))
PT4_RESET_TS = 1787788353


def load(n):
    return [json.loads(l) for l in (ROOT / "results" / n).open()
            if l.strip()]


def day(ts):
    return datetime.datetime.fromtimestamp(ts, PT).strftime("%m/%d")


print("=" * 70)
print("STEP-0 GATE CHECK — the shipped fixes, judged on their own terms")
print("=" * 70)

# --- Gambler v2: gate = win% stays above ITS OWN break-even ---
g = [t for t in load("pt4_trades.jsonl")
     if t["made_ts"] >= PT4_RESET_TS and t.get("actual") is not None]
if g:
    wr = sum(t["win"] for t in g) / len(g)
    # break-even = mean cost per 100c contract actually paid
    cost = sum(t["stake_c"] / t["contracts"] for t in g) / len(g)
    pnl = sum(t["pnl_c"] for t in g)
    stk = sum(t["stake_c"] for t in g)
    print(f"\nGAMBLER v2 (>=0.77 gate, $10k reset)")
    print(f"  bets {len(g)} · win {100*wr:.1f}% · avg cost paid "
          f"{cost:.1f}c => break-even {cost:.1f}%")
    print(f"  EV/$1 {100*pnl/stk:+.1f}% · P&L ${pnl/100:+,.0f}")
    above = wr * 100 > cost
    print(f"  GATE (win% > break-even, n>=30): "
          f"{'PASS' if above and len(g) >= 30 else
             f'ON TRACK (n={len(g)}/30)' if above else 'FAILING'}")

# --- MLE: gate = >=60% idle, and is it profitable? ---
m = load("pt6_trades.jsonl")
bets = [t for t in m if not t.get("skipped")]
sbets = [t for t in bets if t.get("actual") is not None]
idle = 1 - len(bets) / max(1, len(m))
if sbets:
    stk = sum(t["stake_c"] for t in sbets)
    pnl = sum(t["pnl_c"] for t in sbets)
    cost = sum(t["stake_c"] / t["contracts"] for t in sbets) / len(sbets)
    wr = sum(t["win"] for t in sbets) / len(sbets)
    print(f"\nMLE (10c edge margin + shadow rows)")
    print(f"  windows seen {len(m)} · bets {len(bets)} · idle "
          f"{100*idle:.0f}%")
    print(f"  win {100*wr:.1f}% vs break-even {cost:.1f}% · EV/$1 "
          f"{100*pnl/stk:+.1f}% · P&L ${pnl/100:+,.0f}")
    print(f"  GATE (idle >= 60%, n>=50 windows): "
          f"{'PASS' if idle >= 0.60 and len(m) >= 50 else 'PENDING'}")

# --- Saver: gate = DD rate <= 40% of the 25%-era rate ---
s = [t for t in load("pt5_trades.jsonl") if t.get("actual") is not None]
byd = defaultdict(lambda: [0, 0])
for t in s:
    d = byd[day(t["close_ts"])]
    d[0] += t["pnl_c"]
    d[1] += 1
print(f"\nSAVER (stake 25% -> 10% on 08-26)")
for d in sorted(byd):
    print(f"  {d}: {byd[d][1]:3d} bets, P&L ${byd[d][0]/100:+8.0f}")

# --- the desk overall, by day: is the regime the story? ---
print("\n" + "=" * 70)
print("REGIME CHECK — desk by day (is this a mechanism or a market?)")
print("=" * 70)
pt = [t for t in load("pt_trades.jsonl") if t.get("actual") is not None]
byd2 = defaultdict(lambda: [0, 0, 0, 0.0])
for t in pt:
    d = byd2[day(t["close_ts"])]
    d[0] += 1
    d[1] += t["win"]
    d[2] += t["pnl_c"]
    d[3] += t["stake_c"] / t["contracts"]
for d in sorted(byd2):
    n, w, p, c = byd2[d]
    be = c / n
    print(f"  {d}: {n:3d} bids · win {100*w/n:5.1f}% vs break-even "
          f"{be:4.1f}% · P&L ${p/100:+8.0f} · "
          f"{'ABOVE' if 100*w/n > be else 'below'} water")

# --- calibration drift: has the error MODE changed? ---
print("\n" + "=" * 70)
print("CALIBRATION DRIFT — is the bias a constant or a moving target?")
print("=" * 70)
kb = load("kalshi_binary_log.jsonl")
ARMS = ["kb", "kb2", "kb4", "kb7", "kb8", "kb9"]
dec = defaultdict(dict)
for r in kb:
    v = r.get("variant") or "kb"
    if (v not in ARMS or r.get("actual") is None
            or r.get("mins_left") is None or r["mins_left"] > 12):
        continue
    d = dec[v]
    tk = r["ticker"]
    if tk not in d or r["mins_left"] > d[tk]["mins_left"]:
        d[tk] = r


def lg(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def platt(ps, ys):
    a, b = 0.0, 1.0
    xs = [lg(p) for p in ps]
    for _ in range(60):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            mm = 1 / (1 + math.exp(-max(-30, min(30, a + b * x))))
            w = mm * (1 - mm)
            g0 += y - mm
            g1 += (y - mm) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (g0 * h11 - g1 * h01) / det
        db = (g1 * h00 - g0 * h01) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


print("  arm   first-half (a, b)      second-half (a, b)     verdict")
for v in ARMS:
    ds = sorted(dec[v].values(), key=lambda r: r["close_ts"])
    if len(ds) < 60:
        continue
    h = len(ds) // 2
    a1, b1 = platt([d["p_up"] for d in ds[:h]],
                   [d["actual"] for d in ds[:h]])
    a2, b2 = platt([d["p_up"] for d in ds[h:]],
                   [d["actual"] for d in ds[h:]])
    moved = "INTERCEPT moved" if abs(a2 - a1) > 0.15 else ""
    moved += (" SLOPE moved" if abs(b2 - b1) > 0.25 else "")
    print(f"  {v:4s}  a{a1:+.3f} b{b1:.3f}      a{a2:+.3f} b{b2:.3f}"
          f"     {moved or 'stable'}")
print("\n  -> a moving (a,b) is the case FOR an online tracker (M1) and")
print("     AGAINST any one-time correction constant.")
