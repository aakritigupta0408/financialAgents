"""kb7 context-length ablation: does Chronos-Bolt call windows better
with more price history? Replay recent windows offline: for each, feed
the last N minute-closes ending at a mid-window decision minute, read
P(close >= strike) from the quantiles, score vs the settled outcome.
Same readout code path as live (_chronos_p_up)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.online import _chronos_p_up            # noqa: E402
from btc_rl.sources import fetch_range             # noqa: E402
from datetime import datetime, timedelta           # noqa: E402

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
wins = {}
for r in kb:
    if r.get("variant") == "kb2" and r.get("actual") is not None \
            and 6 <= r["mins_left"] <= 9:
        wins.setdefault(r["ticker"], r)   # one mid-window decision each
rows = sorted(wins.values(), key=lambda r: r["made_ts"])[-260:]
print(f"windows: {len(rows)}")

now = datetime.now().astimezone()
bars = fetch_range(now - timedelta(hours=64), now)
closes = {b["ts"]: b["close"] for b in bars}
keys = sorted(closes)
print(f"bars: {len(keys)}")

res = defaultdict(lambda: [0, 0, 0.0])
for r in rows:
    upto = [closes[k] for k in keys if k < r["made_ts"]]
    if len(upto) < 2100:
        continue
    horizon = int(max(1, round(r["mins_left"])))
    for N in (256, 512, 1024, 2048):
        out = _chronos_p_up(upto[-N:], r["strike"], horizon)
        if not out:
            continue
        p = out[0]          # (p_up, q80_w, q80_lo, q80_hi) since the 4-tuple change
        s = res[N]
        s[0] += 1
        s[1] += int((p >= 0.5) == bool(r["actual"]))
        s[2] += (p - r["actual"]) ** 2
print(f"{'ctx':>5s} {'n':>4s} {'acc':>7s} {'brier':>7s}")
for N in (256, 512, 1024, 2048):
    n, h, b = res[N]
    if n:
        print(f"{N:5d} {n:4d} {h/n:7.1%} {b/n:7.3f}")
