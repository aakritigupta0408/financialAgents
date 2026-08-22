"""Where does direction skill actually live? Compare, on settled rows:
(a) sign of level predictions (what the chart currently plots) per arm,
(b) t8's distributional P(up) with confidence gates (selective calls),
(c) the binary stack (kb2/kb3) for reference.
All read-only."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
preds = [json.loads(l) for l in
         (ROOT / "results" / "prediction_log.jsonl").open()]

print("== (a) sign of LEVEL predictions (current chart source) ==")
acc = defaultdict(lambda: [0, 0])
for p in preds:
    if p.get("actual") is None or p["pred"] == p["price_now"] \
            or p["actual"] == p["price_now"]:
        continue
    k = (p["variant"], p["horizon"])
    acc[k][1] += 1
    acc[k][0] += int((p["pred"] > p["price_now"])
                     == (p["actual"] > p["price_now"]))
for (v, h), (w, n) in sorted(acc.items()):
    if n >= 80 and v in ("consensus", "consensus-h1", "consensus-h15",
                         "consensus-h30", "t8", "t2", "h5", "t8-h15"):
        print(f"  {v:15s} h{h:<3d} {w/n:.1%}  (n={n})")

print("== (b) t8 P(up) with confidence gate (|p-0.5| >= m) ==")
# t8 rows carry lo/hi native band; direction prob approx via delta/sigma?
# use rows with 'sigma' and delta: p_up ~ Phi(delta/sigma) is monotone in
# delta/sigma, so gate on |delta|/sigma directly.
for h in (5, 15, 30):
    rows = [p for p in preds
            if p["variant"] == "t8" and p.get("horizon") == h
            and p.get("actual") is not None and p.get("sigma")
            and p["actual"] != p["price_now"] and p["pred"] != p["price_now"]]
    n_all = len(rows)
    if n_all < 60:
        continue
    for m in (0.0, 0.3, 0.6, 1.0):
        sel = [p for p in rows
               if abs(p["pred"] - p["price_now"]) / p["sigma"] >= m]
        if len(sel) < 25:
            continue
        w = sum(int((p["pred"] > p["price_now"])
                    == (p["actual"] > p["price_now"])) for p in sel)
        print(f"  t8 h{h:<3d} gate {m:.1f}sigma: {w/len(sel):.1%} "
              f"coverage {len(sel)/n_all:.0%} (n={len(sel)})")

print("== (c) binary stack reference (15-min windows) ==")
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
for v in ("kb", "kb2", "kb3"):
    rows = [r for r in kb if r.get("variant", "kb") == v
            and r.get("hit") is not None]
    if rows:
        w = sum(r["hit"] for r in rows)
        print(f"  {v:4s} per-minute calls: {w/len(rows):.1%} (n={len(rows)})")
