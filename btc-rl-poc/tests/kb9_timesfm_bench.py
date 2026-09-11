"""kb9 candidate: TimesFM 2.5 (200M) zero-shot vs live kb7 recipe.

Same pre-registered gauntlet as tests/kb8_gauntlet.py: one mid-window
decision (6-9 mins left) per settled window, identical readout
(quantiles interpolated at the strike), window-clustered paired Brier
t vs kb7-replay. Gate to go live: t < -2. Offline only — touches
nothing live.
"""
import bisect
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.online import _chronos_p_up            # noqa: E402

QS = [i / 10 for i in range(1, 10)]


def quantile_p_up(vals, strike):
    if strike <= vals[0]:
        return 0.95
    if strike >= vals[-1]:
        return 0.05
    pr = 0.5
    for i in range(len(vals) - 1):
        if vals[i] <= strike <= vals[i + 1]:
            frac = ((strike - vals[i]) / (vals[i + 1] - vals[i])
                    if vals[i + 1] > vals[i] else 0.5)
            pr = 1.0 - (QS[i] + frac * (QS[i + 1] - QS[i]))
            break
    return min(.95, max(.05, pr))


kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
wins = {}
for r in kb:
    if r.get("variant") == "kb2" and r.get("actual") is not None \
            and r.get("mkt_p_up") is not None and 6 <= r["mins_left"] <= 9:
        wins.setdefault(r["ticker"], r)
rows = sorted(wins.values(), key=lambda r: r["made_ts"])[-260:]
print(f"windows: {len(rows)}")

bars = {}
for l in (ROOT / "results" / "hist_bars_cache.jsonl").open():
    b = json.loads(l)
    bars[b["ts"]] = b["c"]
keys = sorted(bars)
closes = [bars[k] for k in keys]
print(f"bars: {len(keys)}")

import numpy as np                                  # noqa: E402
import timesfm                                      # noqa: E402
print("loading TimesFM 2.5 200M…")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch")
tfm.compile(timesfm.ForecastConfig(
    max_context=1024, max_horizon=16, normalize_inputs=True,
    use_continuous_quantile_head=True, fix_quantile_crossing=True))

res = []           # (ticker, p_tfm, p_kb7, y, mkt, sec)
for r in rows:
    i = bisect.bisect_left(keys, r["made_ts"])
    if i < 1050:
        continue
    horizon = int(max(1, round(r["mins_left"])))
    out7 = _chronos_p_up(closes[:i], r["strike"], horizon)
    if not out7:
        continue
    t0 = time.time()
    try:
        pt, qt = tfm.forecast(horizon=horizon,
                              inputs=[np.array(closes[i - 1024:i],
                                               dtype=np.float32)])
        q = np.asarray(qt)[0, horizon - 1]
        vals = sorted(float(x) for x in (q[1:10] if q.shape[-1] >= 10
                                         else q))
        p_t = quantile_p_up(vals, r["strike"])
    except Exception as e:
        print("tfm failed:", str(e)[:120])
        continue
    res.append((r["ticker"], p_t, out7[0], r["actual"], r["mkt_p_up"],
                time.time() - t0))

n = len(res)
print(f"\npaired windows: {n}")
if n >= 10:
    acc_t = sum((p >= .5) == bool(y) for _, p, _, y, _, _ in res) / n
    acc_7 = sum((p7 >= .5) == bool(y) for _, _, p7, y, _, _ in res) / n
    br_t = sum((p - y) ** 2 for _, p, _, y, _, _ in res) / n
    br_7 = sum((p7 - y) ** 2 for _, _, p7, y, _, _ in res) / n
    d = [(p - y) ** 2 - (p7 - y) ** 2 for _, p, p7, y, _, _ in res]
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    t = mu / (sd / math.sqrt(n))
    lat = sorted(s for *_, s in res)[n // 2]
    print(f"TimesFM: acc {acc_t:.1%} brier {br_t:.3f} · kb7: acc "
          f"{acc_7:.1%} brier {br_7:.3f}")
    print(f"paired Brier diff {mu:+.4f}, t = {t:.2f} "
          f"(gate: t < -2 to go live) · median latency {lat:.2f}s")
