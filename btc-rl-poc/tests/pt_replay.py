"""Replay the $1K Desk policy over logged history (validation only).

Same rules as the live trader in btc_rl/online.py: start $1,000, follow
the current best bidder (last-10 settled gate-clearing decisions among
PT_ARMS), enter once per window at the first tau-clearing minute with
mins_left <= 12, risk at most 10% of funds, Kalshi fee included.

Caveat (stated on anything shown): logged rows carry the market MID,
not the book, so the replay ask is mid + 2.5c (the SEL_CF_ASK_ADJ
convention measured for this market). The live trader uses real asks.
Leadership uses only windows SETTLED before entry time — no leakage.
"""
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.online import PT_ARMS, PT_TAU, PT_LAST_N, PT_MIN_REC  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
ASK_ADJ = 2.5
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
kb = [r for r in kb if r.get("actual") is not None]

# decision per (arm, window): first tau-clearing minute, mins_left<=12
byw = defaultdict(lambda: defaultdict(list))
for r in kb:
    if r.get("variant") in PT_ARMS:
        byw[r["ticker"]][r["variant"]].append(r)
windows = []
for tk, arms in byw.items():
    any_rows = next(iter(arms.values()))
    windows.append((any_rows[0]["close_ts"], tk, arms))
windows.sort()

# per-arm decision history in settle order for leadership lookback
dec_hist = defaultdict(list)     # arm -> [(close_ts, hit, brier)]
for cts, tk, arms in windows:
    for arm, rows in arms.items():
        rows.sort(key=lambda r: -r["mins_left"])
        for r in rows:
            if max(r["p_up"], 1 - r["p_up"]) >= PT_TAU:
                dec_hist[arm].append(
                    (cts, r["hit"], (r["p_up"] - r["actual"]) ** 2))
                break

bank = 100_000
trades = []
for cts, tk, arms in windows:
    # leader from decisions SETTLED strictly before this window's entries
    entry_cut = cts - 900          # window open
    best = None
    for arm in PT_ARMS:
        past = [d for d in dec_hist[arm] if d[0] <= entry_cut]
        past = past[-PT_LAST_N:]
        if len(past) < PT_MIN_REC:
            continue
        wins = sum(h for _, h, _ in past)
        br = sum(b for _, _, b in past) / len(past)
        key = (wins / len(past), -br)
        if best is None or key > best[0]:
            best = (key, arm)
    if not best:
        continue
    arm = best[1]
    rows = sorted(arms.get(arm, []), key=lambda r: -r["mins_left"])
    pick = next((r for r in rows if r["mins_left"] <= 12
                 and max(r["p_up"], 1 - r["p_up"]) >= PT_TAU
                 and r.get("mkt_p_up") is not None), None)
    if not pick:
        continue
    sy = pick["p_up"] >= 0.5
    ask = 100 * (pick["mkt_p_up"] if sy else 1 - pick["mkt_p_up"]) + ASK_ADJ
    if not 5 <= ask < 80:
        continue
    fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
    ncon = int((0.10 * bank) // (ask + fee))
    if ncon < 1:
        continue
    stake = int(ncon * (ask + fee))
    win = int((pick["call"] == 1) == bool(pick["actual"]))
    payout = ncon * 100 if win else 0
    bank += payout - stake
    trades.append((cts, arm, win, payout - stake, bank))

n = len(trades)
w = sum(t[2] for t in trades)
print(f"replayed trades: {n}  wins {w}  ({w/max(1,n):.0%})")
print(f"final bankroll: ${(trades[-1][4] if trades else 100000)/100:,.2f} "
      f"(start $1,000.00)")
byday = defaultdict(lambda: [0, 0])
for cts, arm, win, pnl, bk in trades:
    d = datetime.fromtimestamp(cts, PT).strftime("%m-%d")
    byday[d][0] += 1
    byday[d][1] += pnl
print("by day (PT):")
for d, (cnt, pnl) in sorted(byday.items()):
    print(f"  {d}: {cnt:3d} trades  {pnl/100:+9.2f} $")
lead = defaultdict(int)
for _, arm, *_ in trades:
    lead[arm] += 1
print("leadership share:", dict(sorted(lead.items(), key=lambda x: -x[1])))
print("\ncaveat: replay asks = mid + 2.5c (logged mids, not book); live "
      "trader uses real asks.")
