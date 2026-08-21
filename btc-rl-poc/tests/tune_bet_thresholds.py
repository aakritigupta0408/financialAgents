"""Threshold sweep for the one-bet policy, replayed over the kb call log.

Simulates: strike at the first minute where the called side has
edge >= edge_min AND |p-0.5| >= conf_min; else forced entry at <=3 min
(called side if <85c, else other side). Executable prices approximated
from the logged market mid +/- 1c half-spread. Scores win rate and net
P&L after the Kalshi fee. Windows without quotes are skipped (matches
live behavior).
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from btc_rl.metrics import kalshi_fee_c

rows = [json.loads(l) for l in
        (ROOT / "results" / "kalshi_binary_log.jsonl").read_text().splitlines()]
byw = defaultdict(list)
for r in rows:
    if r.get("mkt_p_up") is not None and r["actual"] is not None:
        byw[r["ticker"]].append(r)
for v in byw.values():
    v.sort(key=lambda r: r["made_ts"])

def simulate(edge_min, conf_min):
    n = wins = 0
    pnl = 0.0
    for ticker, calls in byw.items():
        outcome = calls[0]["actual"]
        bet = None
        for r in calls:
            p = r["p_up"]
            mid = r["mkt_p_up"] * 100
            yes_ask, yes_bid = mid + 1, mid - 1
            side = "yes" if p >= 0.5 else "no"
            price = yes_ask if side == "yes" else 100 - yes_bid
            edge = (100 * p - price) if side == "yes" else (100 * (1 - p) - price)
            conf = abs(p - 0.5)
            forced = r["mins_left"] <= 3
            if price < 85 and ((edge >= edge_min and conf >= conf_min)
                               or forced):
                bet = (side, price, p)
                break
            if forced:  # called side priced out: only legal side
                other = "no" if side == "yes" else "yes"
                oprice = (100 - yes_bid) if other == "no" else yes_ask
                if oprice < 85:
                    bet = (other, oprice, p)
                break
        if bet is None:
            continue
        side, price, p = bet
        won = (side == "yes") == bool(outcome)
        n += 1
        wins += won
        pnl += ((100 - price) if won else -price) - kalshi_fee_c(price)
    return n, wins, pnl

def simulate_on(windows, edge_min, conf_min):
    saved = dict(byw)
    try:
        globals()["byw"] = {k: byw_all[k] for k in windows}
        return simulate(edge_min, conf_min)
    finally:
        globals()["byw"] = saved


# ── leakage-free protocol: chronological 60/40 split, tune early,
#    evaluate the single chosen policy once on the held-out tail ─────────
byw_all = dict(byw)
ordered = sorted(byw_all, key=lambda t: byw_all[t][0]["made_ts"])
cut = int(len(ordered) * 0.6)
train_w, test_w = ordered[:cut], ordered[cut:]
print(f"windows with quotes+outcome: {len(ordered)} "
      f"(tune on first {len(train_w)}, hold out last {len(test_w)})")

print("\nTUNE (training windows only):")
print(f"{'edge>=':>7} {'conf>=':>7} {'bets':>5} {'win%':>6} {'netP&L c':>9} {'c/bet':>7}")
best = None
for edge_min in (3, 5, 7, 10, 12, 15):
    for conf_min in (0.0, 0.05, 0.10, 0.15, 0.20):
        n, w, pnl = simulate_on(train_w, edge_min, conf_min)
        if n < 10:
            continue
        tag = "  <- current policy" if (edge_min, conf_min) == (5, 0.0) else ""
        print(f"{edge_min:>7} {conf_min:>7.2f} {n:>5} {100*w/n:>5.0f}% "
              f"{pnl:>+9.0f} {pnl/n:>+7.1f}{tag}")
        if best is None or pnl / n > best[3] / best[2]:
            best = (edge_min, conf_min, n, pnl, w)

e, c, *_ = best
print(f"\nchosen on train (best net c/bet): edge>={e}, conf>={c}")
print("\nHELD-OUT evaluation (single shot, later windows):")
for label, (em, cm) in (("chosen", (e, c)), ("current", (5, 0.0))):
    n, w, pnl = simulate_on(test_w, em, cm)
    print(f"  {label:>8} (edge>={em}, conf>={cm}): {n} bets, "
          f"win {100*w/n:.0f}%, net {pnl:+.0f}c ({pnl/n:+.1f} c/bet)")
