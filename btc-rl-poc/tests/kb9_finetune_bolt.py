"""kb9 candidate: fine-tune Chronos-Bolt-small on OUR BTC minute bars.

Train on ~60 days of Coinbase minute closes STRICTLY BEFORE the
evaluation windows (cutoff = earliest gauntlet decision − 1h; no
leakage). Native quantile loss via the model's own training forward.
Then the pre-registered gauntlet: one decision per settled window,
paired Brier vs the STOCK kb7 recipe, window-clustered t; gate to go
live t < -2. Saves to results/chronos_bolt_ft/ — the live daemon never
reads that path, so nothing running is touched.
"""
import bisect
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch                                        # noqa: E402
from chronos import BaseChronosPipeline             # noqa: E402

QS = [i / 10 for i in range(1, 10)]
CTX, HOR = 512, 64
STEPS, BATCH, LR = 500, 8, 1e-5
OUT = ROOT / "results" / "chronos_bolt_ft"

bars = {}
for l in (ROOT / "results" / "hist_bars_cache.jsonl").open():
    b = json.loads(l)
    bars[b["ts"]] = b["c"]
keys = sorted(bars)
closes = [bars[k] for k in keys]
print(f"bars: {len(keys)} ({(keys[-1]-keys[0])/86400:.1f} days)")

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
wins = {}
for r in kb:
    if r.get("variant") == "kb2" and r.get("actual") is not None \
            and r.get("mkt_p_up") is not None and 6 <= r["mins_left"] <= 9:
        wins.setdefault(r["ticker"], r)
rows = sorted(wins.values(), key=lambda r: r["made_ts"])[-260:]
cutoff = min(r["made_ts"] for r in rows) - 3600
cut_i = bisect.bisect_left(keys, cutoff)
train = closes[:cut_i]
print(f"eval windows: {len(rows)} · train minutes: {len(train)} "
      f"(strictly before cutoff)")

pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small", device_map="cpu",
    torch_dtype=torch.float32)
model = pipe.model
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR)

starts = list(range(0, len(train) - CTX - HOR - 1, 16))
random.seed(13)
random.shuffle(starts)
print(f"training samples available: {len(starts)} · steps: {STEPS}")
t0 = time.time()
for step in range(STEPS):
    bs = [starts[(step * BATCH + j) % len(starts)] for j in range(BATCH)]
    ctx = torch.tensor([train[s:s + CTX] for s in bs],
                       dtype=torch.float32)
    tgt = torch.tensor([train[s + CTX:s + CTX + HOR] for s in bs],
                       dtype=torch.float32)
    out = model(context=ctx, target=tgt)
    loss = out.loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"  step {step:4d} loss {float(loss):.4f} "
              f"({(time.time()-t0)/(step+1):.2f}s/step)")
model.eval()
OUT.mkdir(exist_ok=True)
model.save_pretrained(OUT)
print(f"saved fine-tuned model -> {OUT}")

ft = BaseChronosPipeline.from_pretrained(str(OUT), device_map="cpu",
                                         torch_dtype=torch.float32)


def p_up(pipeline, hist, strike, horizon):
    ctx = torch.tensor(hist[-512:], dtype=torch.float32).unsqueeze(0)
    q, _ = pipeline.predict_quantiles(ctx, prediction_length=max(1, horizon),
                                      quantile_levels=QS)
    vals = [float(x) for x in q[0, -1]]
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


stock = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small", device_map="cpu",
    torch_dtype=torch.float32)
res = []
for r in rows:
    i = bisect.bisect_left(keys, r["made_ts"])
    if i < 520:
        continue
    horizon = int(max(1, round(r["mins_left"])))
    with torch.no_grad():
        pf = p_up(ft, closes[:i], r["strike"], horizon)
        ps = p_up(stock, closes[:i], r["strike"], horizon)
    res.append((pf, ps, r["actual"]))
n = len(res)
acc_f = sum((p >= .5) == bool(y) for p, _, y in res) / n
acc_s = sum((p >= .5) == bool(y) for _, p, y in res) / n
br_f = sum((p - y) ** 2 for p, _, y in res) / n
br_s = sum((p - y) ** 2 for _, p, y in res) / n
d = [(pf - y) ** 2 - (ps - y) ** 2 for pf, ps, y in res]
mu = sum(d) / n
sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
t = mu / (sd / math.sqrt(n))
print(f"\nheld-out gauntlet ({n} windows, one decision each):")
print(f"  fine-tuned: acc {acc_f:.1%} brier {br_f:.3f}")
print(f"  stock kb7 : acc {acc_s:.1%} brier {br_s:.3f}")
print(f"  paired Brier diff {mu:+.4f}, t = {t:.2f} "
      f"(gate: t < -2 to go live)")
