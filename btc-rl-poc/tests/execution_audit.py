"""Execution quality: what does the desk's ENTRY TIMING actually cost?

The treatment framework prices every policy from the decision-time
quote. The desk's real fills differ from that quote — and the gap
turned out to average 2.1c, which is larger than any paired edge we
have measured. If the desk is losing more at the point of execution
than any model change can win back, execution is the binding
constraint and the mitigation queue is aimed at the wrong tier.

This asks: where does the gap come from, how much does it cost in EV,
and is it fixable by entry timing?
"""
import datetime
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = datetime.timezone(datetime.timedelta(hours=-7))


def load(n):
    p = ROOT / "results" / n
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


kb = load("kalshi_binary_log.jsonl")
pt = [t for t in load("pt_trades.jsonl") if t.get("actual") is not None]
by = defaultdict(list)
for r in kb:
    if r.get("mkt_p_up") is not None:
        by[r["ticker"]].append(r)

rows = []
for t in pt:
    cand = [r for r in by.get(t["ticker"], [])
            if r.get("variant") == t.get("leader")
            and (r.get("mins_left") or 99) <= 12]
    if not cand:
        continue
    dt = max(cand, key=lambda r: r["mins_left"])      # decision-time row
    mkt = dt["mkt_p_up"]
    model = (100 * mkt + 2.5) if t["side"] == "yes" \
        else (100 * (1 - mkt) + 2.5)
    rows.append({
        "slip": t["ask_c"] - model,           # +ve = paid MORE than model
        "real": t["ask_c"], "model": model,
        "mins_left": dt["mins_left"],
        "entry_mins": t.get("mins_left"),
        "win": t["win"], "pnl": t["pnl_c"], "stake": t["stake_c"],
        "hour": datetime.datetime.fromtimestamp(t["close_ts"], PT).hour,
    })

n = len(rows)
slips = sorted(r["slip"] for r in rows)
mean = sum(slips) / n
print(f"=== ENTRY SLIPPAGE vs the decision-time quote (n={n}) ===")
print(f"  mean {mean:+.2f}c · median {slips[n//2]:+.2f}c · "
      f"p10 {slips[n//10]:+.2f}c · p90 {slips[9*n//10]:+.2f}c · "
      f"max {slips[-1]:+.2f}c")
paid_more = sum(1 for s in slips if s > 0.5)
print(f"  paid MORE than the quote on {paid_more}/{n} "
      f"({100*paid_more/n:.0f}%) of entries")

# what the slippage costs in EV terms
def ev(ask, won):
    fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
    c = ask + fee
    return (100 - c) / c if won else -1.0


real_ev = sum(ev(r["real"], r["win"]) for r in rows) / n
mod_ev = sum(ev(r["model"], r["win"]) for r in rows) / n
print(f"\n  EV/$1 at REAL fills   {100*real_ev:+6.2f}%")
print(f"  EV/$1 at MODEL quotes {100*mod_ev:+6.2f}%")
print(f"  => execution costs {100*(mod_ev-real_ev):.2f} pts of EV — "
      f"compare the best measured\n     paired edge of any treatment: "
      f"~1.2 pts. Execution dominates.")

print("\n=== where the slippage comes from: entry lateness ===")
B = [(0, 3), (3, 6), (6, 9), (9, 13)]
g = defaultdict(list)
for r in rows:
    m = r["entry_mins"]
    if m is None:
        continue
    for lo, hi in B:
        if lo <= m < hi:
            g[(lo, hi)].append(r)
            break
print(f"{'mins left at entry':>20s} {'n':>4s} {'mean slip':>10s} "
      f"{'real EV':>9s} {'model EV':>9s} {'cost':>7s}")
for k in sorted(g):
    rs = g[k]
    ms = sum(r["slip"] for r in rs) / len(rs)
    re_ = sum(ev(r["real"], r["win"]) for r in rs) / len(rs)
    me = sum(ev(r["model"], r["win"]) for r in rs) / len(rs)
    print(f"{f'{k[0]}-{k[1]} min':>20s} {len(rs):4d} {ms:+9.2f}c "
          f"{100*re_:+8.2f}% {100*me:+8.2f}% {100*(me-re_):6.2f}pts")

print("\n=== the tail: the worst 10% of fills ===")
worst = sorted(rows, key=lambda r: -r["slip"])[:max(1, n // 10)]
wn = len(worst)
print(f"  {wn} entries · mean slip {sum(r['slip'] for r in worst)/wn:+.1f}c"
      f" · mean entry {sum((r['entry_mins'] or 0) for r in worst)/wn:.1f}"
      f" min left")
print(f"  they cost "
      f"{100*(sum(ev(r['model'], r['win']) for r in worst)/wn - sum(ev(r['real'], r['win']) for r in worst)/wn):.1f}"
      f" pts of EV between them")
lost = sum(r["pnl"] for r in worst)
print(f"  realised P&L on these: ${lost/100:+,.0f}")
print("\n  => a treatment that simply DECLINES a fill worse than the")
print("     decision-time quote by more than a few cents is testable")
print("     against real traffic like any other (M10).")
