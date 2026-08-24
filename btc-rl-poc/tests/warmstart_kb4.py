"""Warm-start kb4 by replaying settled history through its own online
SGD, in settlement order (same update rule, no leakage: features are the
logged per-minute values, outcome from the window close)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.agents import BinaryLogit                     # noqa: E402
from btc_rl.online import KB4_DIM, _kb4_features          # noqa: E402

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
by = defaultdict(dict)
for r in kb:
    v = r.get("variant", "kb")
    if v in ("kb2", "kb3") and r.get("actual") is not None:
        by[(r["ticker"], r["made_ts"])][v] = r

samples = []
for (tk, ts), d in by.items():
    if "kb2" not in d or "kb3" not in d:
        continue
    r2, r3 = d["kb2"], d["kb3"]
    bx = r3.get("bx")
    if not bx or len(bx) < 4:
        continue
    pf = r3.get("pf") or [0.0, 0.0, 0.0, 0.0]
    pf = (pf + [0.0] * 4)[:4]
    x = _kb4_features(r2["p_up"], r3["p_up"], r2.get("mkt_p_up"),
                      bx, pf, r2["mins_left"])
    samples.append((r2["close_ts"], x, r2["actual"]))

samples.sort(key=lambda s: s[0])   # settle order — as live would have
m = BinaryLogit(KB4_DIM)
correct = 0
for i, (_, x, y) in enumerate(samples):
    if (m.predict(x) >= 0.5) == bool(y):
        correct += 1
    m.update(x, y)
print(f"replayed {len(samples)} settled minutes; "
      f"prequential accuracy {correct/len(samples):.1%}")
# prequential accuracy over the final quarter (post-warmup skill)
m2 = BinaryLogit(KB4_DIM)
seen = 0; ok = 0
for i, (_, x, y) in enumerate(samples):
    if i >= len(samples) * 3 // 4:
        seen += 1
        ok += int((m2.predict(x) >= 0.5) == bool(y))
    m2.update(x, y)
print(f"final-quarter prequential accuracy: {ok/max(1,seen):.1%} (n={seen})")
out = ROOT / "results" / "kb4_logit.json"
out.write_text(json.dumps(m.to_dict()))
print(f"saved warm checkpoint: dim {m.dim}, updates {m.updates}")
