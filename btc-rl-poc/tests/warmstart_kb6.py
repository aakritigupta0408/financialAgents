"""Warm-start kb6 by joining settled kb rows with the historical
snapshots that carry perp/tape/whale/OI fields (logging began 2026-08-25
evening). Chronological settle-order replay through its own SGD."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.agents import BinaryLogit                     # noqa: E402
from btc_rl.online import KB6_DIM, _kb6_features          # noqa: E402

snaps = []
for l in (ROOT / "results" / "live_snapshots.jsonl").open():
    try:
        s = json.loads(l)
        if s.get("perp_gap_bp") is not None or s.get("tape_imb_1m") is not None:
            snaps.append(s)
    except Exception:
        continue
snaps.sort(key=lambda s: s["ts"])
print(f"snapshots with fast-info fields: {len(snaps)}")

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb3"
        and r.get("actual") is not None and r.get("bx")
        and r["made_ts"] >= (snaps[0]["ts"] if snaps else 1e18)]

def nearest(ts):
    lo, hi = 0, len(snaps) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if snaps[mid]["ts"] < ts:
            lo = mid + 1
        else:
            hi = mid
    for c in (lo - 1, lo):
        if 0 <= c < len(snaps) and abs(snaps[c]["ts"] - ts) <= 90:
            return snaps[c]
    return None

samples = []
for r in rows:
    s = nearest(r["made_ts"])
    if not s:
        continue
    pf = (r.get("pf") or [0, 0, 0, 0])
    pf = (list(pf) + [0.0] * 4)[:4]
    x = _kb6_features(s, r.get("mkt_p_up"), r["bx"], pf, r["mins_left"])
    samples.append((r["close_ts"], x, r["actual"]))
samples.sort(key=lambda t: t[0])
m = BinaryLogit(KB6_DIM)
q4 = len(samples) * 3 // 4
ok = seen = 0
for i, (_, x, y) in enumerate(samples):
    if i >= q4:
        seen += 1
        ok += int((m.predict(x) >= 0.5) == bool(y))
    m.update(x, y)
print(f"replayed {len(samples)} joined minutes; "
      f"final-quarter prequential accuracy "
      f"{ok/max(1, seen):.1%} (n={seen})")
(ROOT / "results" / "kb6_logit.json").write_text(json.dumps(m.to_dict()))
print(f"saved warm checkpoint: updates {m.updates}")
