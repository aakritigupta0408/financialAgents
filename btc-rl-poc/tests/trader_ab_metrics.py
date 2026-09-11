"""Revenue A/B metrics for all five paper traders — the professional
scoreboard the TA asked for (net P&L, returns, win rate, avg win/bid,
max drawdown, idle rate, today's depletion)."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
STARTS = {"pt": 1000, "pt2": 1000, "pt3": 1000, "pt4": 1000, "pt5": 10000}
NAME = {"pt": "Follower", "pt2": "Ladder", "pt3": "Disciplined",
        "pt4": "Gambler", "pt5": "Saver"}


def load(f):
    try:
        return [json.loads(l) for l in open("results/" + f)]
    except Exception:
        return []


# total settled windows (for idle rate) — count distinct tickers with a
# kb7 biddable-phase row (the desk's opportunity set)
kb = [json.loads(l) for l in open("results/kalshi_binary_log.jsonl")]
opp = {r["ticker"] for r in kb if r.get("variant") == "kb7"
       and r.get("mins_left", 99) <= 12 and r.get("actual") is not None}
now = max((r["made_ts"] for r in kb), default=0)
today = datetime.fromtimestamp(now, PT).strftime("%Y-%m-%d")

print(f"opportunity windows (kb7 biddable, settled): {len(opp)}\n")
hdr = ("trader", "net$", "ret%", "trades", "win%", "avg$/trade",
       "avgWin/bid", "maxDD$", "idle%", "today$")
print(("{:>11}" + "{:>9}" * 9).format(*hdr))
for key, f in (("pt", "pt_trades.jsonl"), ("pt2", "pt2_trades.jsonl"),
               ("pt3", "pt3_trades.jsonl"), ("pt4", "pt4_trades.jsonl"),
               ("pt5", "pt5_trades.jsonl")):
    t = load(f)
    s = [x for x in t if x.get("actual") is not None]
    if not s:
        continue
    s.sort(key=lambda x: x["close_ts"])
    start = STARTS[key]
    net = sum(x["pnl_c"] for x in s) / 100
    # for saver, wealth includes banked savings
    banked = sum(x.get("skim_c", 0) for x in s) / 100 if key == "pt5" else 0
    wins = [x for x in s if x["win"]]
    wr = len(wins) / len(s)
    avg_trade = net / len(s)
    avg_win = (sum(x["pnl_c"] for x in wins) / 100 / len(wins)
               if wins else 0)
    # equity curve max drawdown (dollars)
    eq, peak, dd = start, start, 0
    for x in s:
        eq += x["pnl_c"] / 100
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    traded_windows = len({x["ticker"] for x in s})
    idle = 1 - traded_windows / max(1, len(opp))
    tp = sum(x["pnl_c"] for x in s
             if datetime.fromtimestamp(x["close_ts"], PT).strftime("%Y-%m-%d")
             == today) / 100
    ret = 100 * net / start
    print(("{:>11}" + "{:>9.2f}" * 2 + "{:>9}" + "{:>9.0f}"
           + "{:>9.2f}" * 3 + "{:>9.0f}" + "{:>9.2f}").format(
        NAME[key], net, ret, len(s), 100 * wr, avg_trade, avg_win,
        dd, 100 * idle, tp))
