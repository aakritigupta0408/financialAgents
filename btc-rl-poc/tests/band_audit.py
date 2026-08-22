"""80%-band audit: per arm x horizon, median band width and EMPIRICAL
coverage on the last 150 settled rows with bands. Widths legitimately
differ per arm (conformal bands track each arm's own residuals); coverage
is the consistency contract — every arm should sit near 80%."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").open()]

st = defaultdict(list)
for r in rows:
    if r.get("actual") is None or r.get("lo") is None:
        continue
    st[(r["variant"], r["horizon"])].append(r)

print(f"{'arm':14s} {'h':>3s} {'n':>4s} {'med width':>9s} {'coverage':>8s} "
      f"{'src':>7s}")
for (v, h), rs in sorted(st.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    rs = rs[-150:]
    if len(rs) < 30:
        continue
    ws = sorted(r["hi"] - r["lo"] for r in rs)
    cov = sum(1 for r in rs if r["lo"] <= r["actual"] <= r["hi"]) / len(rs)
    src = rs[-1].get("band_src", "conf")
    flag = "" if 0.68 <= cov <= 0.92 else "  <-- MISCALIBRATED"
    print(f"{v:14s} {h:3d} {len(rs):4d} {ws[len(ws)//2]:8.0f}$ "
          f"{cov:7.0%} {src:>7s}{flag}")
