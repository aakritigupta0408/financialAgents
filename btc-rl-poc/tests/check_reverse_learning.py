"""Is the consensus 'learning staircase' getting worse over time, or just
flat-negative? Reverse learning = per-slot advantage DECLINING across
quarters; noise-floor tax = roughly constant small negative."""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
rows = [json.loads(l) for l in (RES / "prediction_log.jsonl").read_text().splitlines()]

def adv_series(variant, h):
    sc = sorted((r for r in rows if r["variant"] == variant and r["horizon"] == h
                 and r["actual"] is not None), key=lambda r: r["target_ts"])
    return [abs(r["actual"] - r["price_now"]) - r["abs_err"] for r in sc]

for name, v, h in (("consensus +5m", "consensus", 5),
                   ("t10 +5m", "t10-h5", 5),
                   ("t10 +30m", "t10-h30", 30),
                   ("control +5m", "h5", 5)):
    a = adv_series(v, h)
    if len(a) < 40:
        continue
    q = len(a) // 4
    quarters = [sum(a[i*q:(i+1)*q]) / q for i in range(4)]
    wins = sum(1 for x in a if x > 0)
    print(f"{name:14s} n={len(a):4d}  total ${sum(a):+7.0f}  "
          f"win {100*wins/len(a):3.0f}%  $/slot by quarter: "
          + "  ".join(f"{x:+6.2f}" for x in quarters))
