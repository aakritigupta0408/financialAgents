"""Tier replay over the mined Kalshi history — pre-registered Method B.

Applies Sagemon's frozen tier rule to every mined window with REAL
bid/ask quotes (results/kalshi_history.jsonl): at each minute with
mins_left <= 12, recompute kb7 (Chronos) from historical closes
STRICTLY before that minute (no look-ahead); enter at the first minute
where confidence >= 0.77 and the side's real ask is in [5, 80); fees
in. One entry per window. Reports the pre-registered outputs: windows,
win rate + Wilson 95% LB, EV per $1 at real asks, and Method A's
market-paired excess score (win - implied q), t CLUSTERED BY DAY.

Labeled REPLAY evidence — corroboration for the live stream, never
merged with it. Bars cached locally (not committed).
"""
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.online import _chronos_p_up            # noqa: E402
from btc_rl.sources import fetch_coinbase_candles  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
TAU, LO, HI = 0.77, 5, 80
CACHE = ROOT / "results" / "hist_bars_cache.jsonl"

rows = [json.loads(l) for l in
        (ROOT / "results" / "kalshi_history.jsonl").open()]
byw = defaultdict(list)
for r in rows:
    if r.get("outcome") is not None:
        byw[r["ticker"]].append(r)
for tk in byw:
    byw[tk].sort(key=lambda r: r["ts"])
print(f"windows in mine: {len(byw)}")

t_lo = min(r["ts"] for r in rows) - 700 * 60
t_hi = max(r["ts"] for r in rows) + 120

bars = {}
if CACHE.exists():
    for l in CACHE.open():
        b = json.loads(l)
        bars[b["ts"]] = b["c"]
    print(f"bar cache: {len(bars)} minutes")
if not bars or min(bars) > t_lo or max(bars) < t_hi - 3600:
    print("fetching historical bars from Coinbase…")
    import time as _t
    cur = datetime.fromtimestamp(t_lo, tz=timezone.utc)
    end = datetime.fromtimestamp(t_hi, tz=timezone.utc)
    fetched = 0
    while cur < end:
        nxt = min(cur + timedelta(hours=4), end)
        try:
            for b in fetch_coinbase_candles(cur, nxt):
                bars[b["ts"]] = b["close"]
        except Exception as e:
            print("  chunk failed, continuing:", e)
        fetched += 1
        if fetched % 30 == 0:
            print(f"  …{fetched} chunks, {len(bars)} minutes")
        cur = nxt
        _t.sleep(0.13)
    with CACHE.open("w") as f:
        for ts in sorted(bars):
            f.write(json.dumps({"ts": ts, "c": bars[ts]}) + "\n")
    print(f"fetched {len(bars)} minutes; cached")

keys = sorted(bars)
closes = [bars[k] for k in keys]
import bisect

entries = []          # (day, win, cost_c, conf, ask)
evaluated = 0
for tk, rs in sorted(byw.items(), key=lambda kv: kv[1][0]["ts"]):
    outcome = rs[0]["outcome"]
    for r in rs:
        if r["mins_left"] > 12 or r.get("yes_bid_c") is None \
                or r.get("yes_ask_c") is None:
            continue
        i = bisect.bisect_left(keys, r["ts"])
        if i < 520:
            continue
        evaluated += 1
        out = _chronos_p_up(closes[:i], r["strike"],
                            int(max(1, round(r["mins_left"]))))
        if not out:
            continue
        p7 = out[0]
        conf = max(p7, 1 - p7)
        if conf < TAU:
            continue
        sy = p7 >= 0.5
        ask = r["yes_ask_c"] if sy else 100 - r["yes_bid_c"]
        if not LO <= ask < HI:
            continue
        fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
        win = int(sy == bool(outcome))
        day = datetime.fromtimestamp(r["ts"], PT).strftime("%m-%d")
        entries.append((day, win, ask + fee, conf, ask))
        break                                   # one entry per window

n = len(entries)
print(f"\nChronos evaluations: {evaluated} · tier entries: {n} windows")
if n:
    w = sum(e[1] for e in entries)
    cost = sum(e[2] for e in entries) / n
    p = w / n
    z = 1.645
    den = 1 + z * z / n
    lb = (p + z * z / (2 * n) - z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n))) / den
    ev = (p * 100 - cost) / cost
    print(f"win rate: {w}/{n} = {100*p:.1f}%  ·  Wilson 95% LB "
          f"{100*lb:.1f}%  ·  avg cost {cost:.1f}c  ·  "
          f"EV {100*ev:+.1f}%/$1 (real asks, fees in)")
    # Method A: market-paired excess score, clustered by day
    byd = defaultdict(list)
    for day, win, c, _, _ in entries:
        byd[day].append(win - c / 100.0)
    means = [sum(v) / len(v) for v in byd.values()]
    k = len(means)
    mu = sum(means) / k
    sd = math.sqrt(sum((x - mu) ** 2 for x in means) / max(1, k - 1))
    t = mu / (sd / math.sqrt(k)) if sd else float("nan")
    print(f"Method A excess score (win − implied q): mean {mu:+.4f} "
          f"over {k} days · clustered t = {t:.2f} "
          f"({'significant' if t > 1.7 else 'not significant'})")
    perday = defaultdict(int)
    for day, *_ in entries:
        perday[day] += 1
    print(f"entries/day: min {min(perday.values())}, "
          f"median {sorted(perday.values())[len(perday)//2]}, "
          f"max {max(perday.values())} across {len(perday)} days")
