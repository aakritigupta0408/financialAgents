"""Is kbf trivial (= market at T-3), and does a dynamic early-lock rule
preserve 80/80 while decoupling from the fixed decision time?

A) Baseline check on settled windows: kbf accuracy vs 'market favorite
   at the same T-3 minute' on identical windows.
B) Dynamic lock: walk each window's per-minute kb calls (calibrated p);
   LOCK the call the first minute |p-0.5| >= m (scan m on the first 70%
   of windows for the largest m holding all four class metrics >= 0.80
   with earliest average lock); evaluate frozen m on the last 30%.
   Report accuracy, per-class P/R, average lock time, and how often the
   lock beats the market's convergence (|p_mkt - 0.5| at lock < |p| ).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in (ROOT / "results" / "kalshi_binary_log.jsonl").open()]

wins = defaultdict(list)
for r in kb:
    if r.get("variant", "kb") == "kb" and r.get("actual") is not None:
        wins[r["ticker"]].append(r)
for v in wins.values():
    v.sort(key=lambda r: -r["mins_left"])

tickers = sorted(wins, key=lambda t: wins[t][0]["close_ts"])


def prf(calls):
    """calls: list of (call, actual) -> dict of per-class P/R + acc."""
    out = {}
    for cls, val in (("up", 1), ("down", 0)):
        tp = sum(1 for c, a in calls if c == val and a == val)
        fp = sum(1 for c, a in calls if c == val and a != val)
        fn = sum(1 for c, a in calls if c != val and a == val)
        out[cls] = (tp / (tp + fp) if tp + fp else 0.0,
                    tp / (tp + fn) if tp + fn else 0.0)
    out["acc"] = sum(1 for c, a in calls if c == a) / len(calls)
    return out


# A) fixed T-3: model vs market favorite at the same minute
fixed_model, fixed_mkt = [], []
for t in tickers:
    rows = [r for r in wins[t] if r["mins_left"] <= 3.4]
    if not rows:
        continue
    r = rows[0]
    a = r["actual"]
    fixed_model.append((r["call"], a))
    if r.get("mkt_p_up") is not None:
        fixed_mkt.append((int(r["mkt_p_up"] >= 0.5), a))
m, k = prf(fixed_model), prf(fixed_mkt)
print(f"A) fixed T-3 on {len(fixed_model)} windows:")
print(f"   model  acc {m['acc']:.1%}  UP {m['up'][0]:.2f}/{m['up'][1]:.2f} "
      f"DOWN {m['down'][0]:.2f}/{m['down'][1]:.2f}")
print(f"   market acc {k['acc']:.1%}  UP {k['up'][0]:.2f}/{k['up'][1]:.2f} "
      f"DOWN {k['down'][0]:.2f}/{k['down'][1]:.2f}   (n={len(fixed_mkt)})")

# B) dynamic lock
cut = int(len(tickers) * 0.7)
tr_t, te_t = tickers[:cut], tickers[cut:]


def run_lock(ts_list, margin):
    calls, locks, beat_mkt = [], [], 0
    for t in ts_list:
        rows = wins[t]
        a = rows[0]["actual"]
        pick = None
        for r in rows:
            if abs(r["p_up"] - 0.5) >= margin and r["mins_left"] >= 3.0:
                pick = r
                break
        if pick is None:
            late = [r for r in rows if r["mins_left"] <= 3.4]
            pick = late[0] if late else rows[-1]
        calls.append((pick["call"], a))
        locks.append(pick["mins_left"])
        if (pick.get("mkt_p_up") is not None
                and abs(pick["p_up"] - 0.5)
                > abs(pick["mkt_p_up"] - 0.5)):
            beat_mkt += 1
    return calls, locks, beat_mkt


best = None
for m100 in range(10, 45, 2):
    margin = m100 / 100
    calls, locks, _ = run_lock(tr_t, margin)
    p = prf(calls)
    if all(x >= 0.80 for x in (*p["up"], *p["down"])):
        avg = sum(locks) / len(locks)
        if best is None or avg > best[1]:
            best = (margin, avg, p)
if best is None:
    print("B) no margin holds 80/80 on train — dynamic lock not shippable")
    sys.exit(0)
margin = best[0]
calls, locks, beat = run_lock(te_t, margin)
p = prf(calls)
avg = sum(locks) / len(locks)
early = sum(1 for L in locks if L > 3.4) / len(locks)
print(f"B) dynamic lock margin {margin:.2f} (train avg lock "
      f"{best[1]:.1f}m left):")
print(f"   held-out {len(calls)} windows: acc {p['acc']:.1%}  "
      f"UP {p['up'][0]:.2f}/{p['up'][1]:.2f}  "
      f"DOWN {p['down'][0]:.2f}/{p['down'][1]:.2f}")
print(f"   avg lock {avg:.1f} min left | locked before T-3: {early:.0%} "
      f"| more confident than market at lock: {beat}/{len(calls)}")
