"""Diagnose whether the arms are actually learning. Usage: python scripts/diagnose_learning.py"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").read_text().splitlines()]
sc = [r for r in rows if r["actual"] is not None and r["variant"] != "consensus"]

print("HOURLY SKILL vs persistence (positive = model better), h5 arms:")
for v in ("h5", "t2-h5", "t3-h5", "t4-h5", "t5-h5"):
    a = [r for r in sc if r["variant"] == v]
    byh = defaultdict(list)
    for r in a:
        byh[r["target_ts"] // 3600].append(
            abs(r["actual"] - r["price_now"]) - abs(r["err"]))
    hours = sorted(byh)
    line = f"{v:7s}"
    for h in hours[-8:]:
        line += f" {sum(byh[h])/len(byh[h]):+6.1f}"
    print(line, f"  (n/hr≈{len(a)//max(1,len(hours))})")

print()
print("CHOSEN DELTA (are models even deviating from persistence?):")
for v in ("t2-h5", "t3-h5", "t4-h5", "t5-h5"):
    a = [r for r in rows if r["variant"] == v]
    recent = a[-24:]
    zeros = sum(1 for r in recent if r["delta"] == 0)
    mean_abs = sum(abs(r["delta"]) for r in recent) / max(1, len(recent))
    print(f"{v:7s} last24: mean|delta|=${mean_abs:5.1f}  delta==0: {zeros}/24")

s = json.loads((ROOT / "results" / "online_status.json").read_text())
print()
print("updates this session:", s["online_updates_session"],
      "| retrains:", s["retrains_this_session"])
print("bandit pulls:", {v: st["states_known"] for v, st in s["variants"].items()
                        if v.startswith(("t2", "t5"))})
errs = [abs(r["err"]) for r in sc if r["horizon"] == 5][-200:]
m = sum(errs) / len(errs)
sd = math.sqrt(sum((e - m) ** 2 for e in errs) / len(errs))
print(f"h5 reward scale: mean shaped r = {-m/100:.3f}, per-sample sd = {sd/100:.3f} "
      f"(LinUCB exploration bonus starts at ~1.0)")
