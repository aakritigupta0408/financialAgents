"""Warm-start kb5 on the deployment distribution: every BIDDABLE side of
every settled minute (mid/late, ask<80 at mid+2.5 adj), label = that
side won. Chronological (settle-order) online SGD — prequential metrics
reported, including the pre-registered one: EV/$1 on confident entries
(p_hat*100 >= ask+fee+3) in the final quarter."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.agents import BinaryLogit                       # noqa: E402
from btc_rl.online import KB5_DIM, _kb5_features            # noqa: E402
from datetime import datetime                               # noqa: E402
from zoneinfo import ZoneInfo                               # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
by = defaultdict(dict)
for r in kb:
    v = r.get("variant", "kb")
    if v in ("kb2", "kb3", "kb4") and r.get("actual") is not None:
        by[(r["ticker"], r["made_ts"])][v] = r

samples = []
for (tk, ts), d in by.items():
    r2 = d.get("kb2")
    if not r2 or r2.get("mkt_p_up") is None or r2["mins_left"] > 10:
        continue
    r3, r4 = d.get("kb3"), d.get("kb4")
    bx = (r3 or {}).get("bx")
    if not bx or len(bx) < 4:
        continue
    pf = ((r3 or {}).get("pf") or [0, 0, 0, 0])
    pf = (list(pf) + [0.0] * 4)[:4]
    hot = datetime.fromtimestamp(ts, PT).hour in (18, 19, 20, 1)
    for sy in (True, False):
        ask = 100 * (r2["mkt_p_up"] if sy else 1 - r2["mkt_p_up"]) + 2.5
        if not 5 <= ask < 80:
            continue
        x = _kb5_features(sy, ask, r2["p_up"], (r3 or {}).get("p_up", .5),
                          (r4 or {}).get("p_up", .5), r2["mkt_p_up"],
                          bx, pf, r2["mins_left"], hot)
        won = int(sy == bool(r2["actual"]))
        fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
        samples.append((r2["close_ts"], x, won, ask, fee))

samples.sort(key=lambda s: s[0])
m = BinaryLogit(KB5_DIM)
q4 = len(samples) * 3 // 4
kept = []
for i, (_, x, y, ask, fee) in enumerate(samples):
    pw = m.predict(x)
    if i >= q4 and pw * 100 >= ask + fee + 3:
        kept.append((y, ask, fee))
    m.update(x, y)
print(f"replayed {len(samples)} biddable sides")
if kept:
    n = len(kept)
    w = sum(k[0] for k in kept)
    cost = sum(k[1] + k[2] for k in kept) / n
    ev = (w / n * 100 - cost) / cost
    print(f"final-quarter CONFIDENT entries: n={n}, win {w/n:.1%}, "
          f"avg cost {cost:.1f}c, EV/$1 {ev:+.1%}")
else:
    print("final quarter: no confident entries (model too conservative)")
out = ROOT / "results" / "kb5_logit.json"
out.write_text(json.dumps(m.to_dict()))
print(f"saved warm checkpoint: updates {m.updates}")
