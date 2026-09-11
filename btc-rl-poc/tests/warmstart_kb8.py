"""Warm-start kb8 (calibrated decorrelation stack over kb7 + market).

kb7's own log is too thin to train on (~15 settled windows), so we
REPLAY the kb7 signal over history: for every settled kb2 row in the
bar-fetchable range, call the live _chronos_p_up on the bar-close
prefix strictly before made_ts (decision-time data only, no leakage),
build b8x from the row's LOGGED decision-time values, and train
prequentially in settle order.

Live-faithful ordering: rows are grouped by close_ts; every row in a
settle group is PREDICTED before any row in it updates the weights —
same-window rows share one outcome and must not teach each other.

Reports full + final-quarter prequential accuracy and window-clustered
paired Brier t vs the kb7 replay and vs the market, then saves
results/kb8_logit.json for the daemon to pick up.
"""
import bisect
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.online import _chronos_p_up, _kb8_features, KB8_DIM  # noqa: E402
from btc_rl.agents import BinaryLogit                            # noqa: E402
from btc_rl.sources import fetch_range                           # noqa: E402

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb3"
        and r.get("actual") is not None and r.get("bx")
        and len(r["bx"]) >= 4 and r.get("pf")]
print(f"settled kb3 rows (bx+pf carriers): {len(rows)}")

now = datetime.now().astimezone()
bars = fetch_range(now - timedelta(hours=64), now)
keys = [b["ts"] for b in bars]
closes = [b["close"] for b in bars]
print(f"bars: {len(bars)}")

# Chronos replay is the slow step (~0.05s/call) — cache it as a durable
# artifact so diagnostics can iterate without re-running the model.
CACHE = ROOT / "results" / "kb8_replay.jsonl"
cache = {}
if CACHE.exists():
    for l in CACHE.open():
        c = json.loads(l)
        cache[(c["made_ts"], c["ticker"])] = (c["p7"], c["w80"])
new_cache_rows = []

samples = []          # (close_ts, ticker, b8x, outcome, p7, mkt)
for r in rows:
    ck = (r["made_ts"], r["ticker"])
    if ck in cache:
        p7, w80 = cache[ck]
    else:
        i = bisect.bisect_left(keys, r["made_ts"])
        if i < 520:
            continue
        horizon = int(max(1, round(r["mins_left"])))
        out = _chronos_p_up(closes[:i], r["strike"], horizon)
        if not out:
            continue
        p7, w80, _, _ = out
        new_cache_rows.append({"made_ts": r["made_ts"],
                               "ticker": r["ticker"],
                               "p7": p7, "w80": w80})
    pf = (r.get("pf") or []) + [0.0] * 4
    b8x = _kb8_features(p7, w80, r.get("mkt_p_up"), r["bx"], pf[:4],
                        r["mins_left"])
    if b8x is None or len(b8x) != KB8_DIM:
        continue
    samples.append((r["close_ts"], r["ticker"], b8x, r["actual"],
                    p7, r.get("mkt_p_up")))
samples.sort(key=lambda s: s[0])
if new_cache_rows:
    with CACHE.open("a") as f:
        for c in new_cache_rows:
            f.write(json.dumps(c) + "\n")
print(f"replayed samples: {len(samples)} across "
      f"{len({s[1] for s in samples})} windows "
      f"({len(new_cache_rows)} fresh, rest cached)")

# prequential, settle-grouped: predict a whole close_ts group, then update
m = BinaryLogit(KB8_DIM)
groups = defaultdict(list)
for s in samples:
    groups[s[0]].append(s)
preds = []            # (ticker, p8, outcome, p7, mkt)
for cts in sorted(groups):
    g = groups[cts]
    for _, tk, x, y, p7, mkt in g:
        preds.append((tk, m.predict(x), y, p7, mkt))
    for _, _, x, y, _, _ in g:
        m.update(x, y)

n = len(preds)
acc = sum((p >= .5) == bool(y) for _, p, y, _, _ in preds) / n
q = preds[-n // 4:]
acc_q = sum((p >= .5) == bool(y) for _, p, y, _, _ in q) / len(q)
print(f"prequential: acc {acc:.1%} (n={n}), final quarter {acc_q:.1%} "
      f"(n={len(q)}), trained updates {m.updates}")

# is the gap cold-start drag or design? quarter-by-quarter, side by side
print("quarters (kb8 / kb7-replay / market acc):")
qs = n // 4
for qi in range(4):
    seg = preds[qi * qs: (qi + 1) * qs if qi < 3 else n]
    a8 = sum((p >= .5) == bool(y) for _, p, y, _, _ in seg) / len(seg)
    a7 = sum((p7 >= .5) == bool(y) for _, _, y, p7, _ in seg) / len(seg)
    wm = [(mkt, y) for _, _, y, _, mkt in seg if mkt is not None]
    am = sum((mk >= .5) == bool(y) for mk, y in wm) / max(1, len(wm))
    print(f"  Q{qi+1}: {a8:5.1%} / {a7:5.1%} / {am:5.1%}  (n={len(seg)})")


def clustered_t(diff_rows):
    """diff_rows: (ticker, diff). One mean per window, t over windows."""
    byw = defaultdict(list)
    for tk, d in diff_rows:
        byw[tk].append(d)
    means = [sum(v) / len(v) for v in byw.values()]
    k = len(means)
    mu = sum(means) / k
    sd = math.sqrt(sum((x - mu) ** 2 for x in means) / (k - 1))
    return k, mu, mu / (sd / math.sqrt(k)) if sd else float("nan")


k, mu, t = clustered_t([(tk, (p - y) ** 2 - (p7 - y) ** 2)
                        for tk, p, y, p7, _ in preds])
print(f"vs kb7-replay: {k} windows, mean Brier diff {mu:+.4f}, "
      f"clustered t = {t:.2f}  (negative = kb8 better)")
mk = [(tk, (p - y) ** 2 - (mkt - y) ** 2)
      for tk, p, y, _, mkt in preds if mkt is not None]
k, mu, t = clustered_t(mk)
print(f"vs market:     {k} windows, mean Brier diff {mu:+.4f}, "
      f"clustered t = {t:.2f}")

(ROOT / "results" / "kb8_logit.json").write_text(json.dumps(m.to_dict()))
print("saved results/kb8_logit.json")
