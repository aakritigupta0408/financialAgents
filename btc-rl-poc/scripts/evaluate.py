"""Multi-angle evaluation of all live predictors from the prediction ledger.

Usage: python scripts/evaluate.py
Angles: accuracy, bias, error distribution, betting economics, volatility
regimes, paired head-to-head on identical slots, consensus vs voters.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").read_text().splitlines()]
scored = [r for r in rows if r["actual"] is not None]

ARMS = ["h5", "h15", "h30", "t2-h5", "t2-h15", "t2-h30",
        "t3-h5", "t3-h15", "t3-h30", "t4-h5", "t4-h15", "t4-h30",
        "t5-h5", "t5-h15", "t5-h30", "t6-h5", "t6-h15", "t6-h30",
        "consensus"]


def q(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


print("=" * 100)
print("ANGLE 1 — Accuracy, bias, distribution, betting (all scored rows)")
print(f'{"arm":12s} {"n":>4s} {"MAE":>6s} {"MSE":>7s} {"med|e|":>7s} '
      f'{"bias":>7s} {"<=$10":>6s} {"dir":>5s} {"replay":>7s} {"hits":>5s}')
for arm in ARMS:
    a = [r for r in scored if r["variant"] == arm]
    if not a:
        continue
    errs = [r["err"] if r.get("err") is not None else 0 for r in a]
    abse = [abs(e) for e in errs]
    moved = [r for r in a if r.get("delta")]
    dir_hit = sum(1 for r in moved
                  if (r["pred"] - r["price_now"]) * (r["actual"] - r["price_now"]) > 0)
    mean_dev = sum(abs(r["pred"] - r["price_now"]) for r in a) / len(a)
    mean_move = sum(abs(r["actual"] - r["price_now"]) for r in a) / len(a)
    replay = max(0.0, 1 - mean_dev / mean_move) if mean_move else 0.0
    print(f'{arm:12s} {len(a):4d} {sum(abse)/len(a):6.1f} '
          f'{sum(e*e for e in errs)/len(a)/1000:6.1f}k '
          f'{q(abse, .5):7.1f} {sum(errs)/len(a):+7.1f} '
          f'{sum(e <= 10 for e in abse)/len(a):6.1%} '
          f'{(dir_hit/len(moved) if moved else 0):5.0%} {replay:7.0%} '
          f'{sum(bool(r["hit"]) for r in a):5d}')

print()
print("=" * 100)
print("ANGLE 2 — Paired head-to-head on IDENTICAL slots (who was closer?)")
by_slot: dict[tuple, dict] = defaultdict(dict)
for r in scored:
    by_slot[(r["made_ts"], r["horizon"])][r["variant"]] = abs(r.get("err") or 0)
for h in (5, 15, 30):
    pairs = [("h" + str(h), f"t2-h{h}"), ("h" + str(h), f"t3-h{h}"),
             (f"t2-h{h}", f"t3-h{h}")]
    for a, b in pairs:
        common = [(v[a], v[b]) for (ts, hh), v in by_slot.items()
                  if hh == h and a in v and b in v]
        if not common:
            continue
        aw = sum(1 for ea, eb in common if ea < eb)
        bw = sum(1 for ea, eb in common if eb < ea)
        print(f'h{h:<3d} {a:8s} vs {b:8s}: n={len(common):3d}  '
              f'{a} closer {aw/len(common):5.1%} | {b} closer {bw/len(common):5.1%} '
              f'| tied {(len(common)-aw-bw)/len(common):5.1%}')

print()
print("=" * 100)
print("ANGLE 3 — Volatility regime (state vol bucket: 0 calm / 1 normal / 2 wild)")
for arm in ("h15", "t2-h15", "t3-h15"):
    a = [r for r in scored if r["variant"] == arm and r.get("state")]
    line = f"{arm:8s}"
    for vb in (0, 1, 2):
        g = [abs(r["err"]) for r in a if r["state"][2] == vb]
        line += f"  vol{vb}: " + (f"MAE ${sum(g)/len(g):5.1f} (n={len(g):3d})"
                                  if g else "     —        ")
    print(line)

print()
print("=" * 100)
print("ANGLE 4 — Consensus vs its own voters (same slots, horizon 5)")
cons_slots = {r["made_ts"]: abs(r.get("err") or 0)
              for r in scored if r["variant"] == "consensus"}
for voter in ("h5", "t2-h5", "t3-h5"):
    common = [(cons_slots[r["made_ts"]], abs(r.get("err") or 0))
              for r in scored
              if r["variant"] == voter and r["made_ts"] in cons_slots]
    if not common:
        continue
    cw = sum(1 for c, v in common if c < v)
    print(f'consensus vs {voter:6s}: n={len(common):3d}  consensus closer '
          f'{cw/len(common):5.1%}  voter closer '
          f'{sum(1 for c, v in common if v < c)/len(common):5.1%}')

print()
print("=" * 100)
print("ANGLE 5 — Policy behavior (chosen deltas per arm; convergence signal)")
from collections import Counter
for arm in ARMS[:-1]:
    a = [r for r in rows if r["variant"] == arm]
    if not a:
        continue
    c = Counter(r["delta"] for r in a)
    top = ", ".join(f"{d:+d}×{n}" for d, n in c.most_common(5))
    print(f"{arm:8s} distinct deltas={len(c):2d}  top: {top}")
