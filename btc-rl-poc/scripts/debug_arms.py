"""Per-arm debug: era-split online metrics + action pathology flags.

Usage: python scripts/debug_arms.py [--era-min 90]
Era cut = rows committed in the last N minutes (post-upgrade behavior)
vs everything before.
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--era-min", type=int, default=90)
args = ap.parse_args()

rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").read_text().splitlines()]
cut = time.time() - args.era_min * 60

ARMS = ["h", "t2", "t3", "t4", "t5", "t6", "t7"]


def era(rs):
    sc = [r for r in rs if r["actual"] is not None]
    if not sc:
        return None
    mae = sum(r["abs_err"] for r in sc) / len(sc)
    bias = sum(r["err"] for r in sc) / len(sc)
    mdev = sum(abs(r["delta"]) for r in sc) / len(sc)
    return f"n={len(sc):3d} MAE=${mae:6.1f} bias={bias:+7.1f} |δ|=${mdev:5.1f}"


print(f"{'arm':6s} {'h':>3s}  OLD ERA{'':38s}NEW ERA (last {args.era_min}m)")
for a in ARMS:
    for h in (5, 15, 30):
        vn = f"{a}{h}" if a == "h" else f"{a}-h{h}"
        rs = [r for r in rows if r["variant"] == vn]
        old = era([r for r in rs if r["made_ts"] < cut])
        new = era([r for r in rs if r["made_ts"] >= cut])
        pend = sum(1 for r in rs if r["made_ts"] >= cut and r["actual"] is None)
        print(f"{a:6s} {h:3d}  {old or '—':45s}{new or '—'}"
              + (f"  (+{pend} pending)" if pend else ""))
    print()

# pathology flags on the newest rows
print("NEW-ERA action behavior (last committed rows per arm):")
for a in ARMS[1:]:
    for h in (5, 15, 30):
        vn = f"{a}-h{h}"
        rs = [r for r in rows if r["variant"] == vn and r["made_ts"] >= cut]
        if not rs:
            continue
        deltas = [r["delta"] for r in rs]
        adjs = [r.get("bias_adj", 0) for r in rs]
        arms_used = len({r.get("arm") for r in rs})
        print(f"{vn:8s} deltas={deltas[-6:]} bias_adj={adjs[-1]:+d} "
              f"distinct_arms={arms_used}/{len(rs)}")
