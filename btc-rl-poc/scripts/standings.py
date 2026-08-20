"""Current A/B standings: MAE ranking + DM + paired wins, per horizon/era.
Usage: python scripts/standings.py"""
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").read_text().splitlines()]
sc = [r for r in rows if r["actual"] is not None]
ARMS = {"ctl": "h", "rp": "rp-h", "t2": "t2-h", "t6": "t6-h",
        "t7": "t7-h", "t8": "t8-h", "t9": "t9-h", "t10": "t10-h",
        "t11": "t11-h", "cal": "cal-h", "agg": None}


def variant(a, h):
    if a == "agg":
        return "consensus" if h == 5 else f"consensus-h{h}"
    return ARMS[a] + str(h)


def dm(d, q):
    n = len(d)
    if n < 20:
        return None
    m = sum(d) / n
    dev = [x - m for x in d]
    v = sum(x * x for x in dev) / n
    for lag in range(1, q + 1):
        w = 1 - lag / (q + 1)
        v += 2 * w * sum(dev[i] * dev[i - lag] for i in range(lag, n)) / n
    return m / math.sqrt(v / n) if v > 0 else None


cut = time.time() - 6 * 3600
for era, rows_e in (("ALL-TIME", sc),
                    ("LAST 6H (clean era)", [r for r in sc if r["made_ts"] >= cut])):
    print("=" * 76)
    print(era)
    for h in (1, 5, 15, 30):
        base = {r["made_ts"]: r["abs_err"] for r in rows_e
                if r["variant"] == variant("ctl", h)}
        stats = []
        for a in ARMS:
            g = [r for r in rows_e if r["variant"] == variant(a, h)]
            if len(g) < 15:
                continue
            mae = sum(r["abs_err"] for r in g) / len(g)
            d = [r["abs_err"] - base[r["made_ts"]] for r in g
                 if r["made_ts"] in base]
            t = dm(d, max(0, math.ceil(h / 5) - 1)) if a != "ctl" else None
            wins = sum(1 for r in g if r["made_ts"] in base
                       and r["abs_err"] < base[r["made_ts"]])
            common = sum(1 for r in g if r["made_ts"] in base)
            stats.append((mae, a, len(g), t, wins, common))
        if not stats:
            continue
        stats.sort()
        top = " | ".join(
            f"{a} ${mae:.0f}" + (f" (DM {t:+.1f})" if t is not None
                                 and abs(t) > 1.6 else "")
            for mae, a, n, t, w, c in stats[:4])
        mae, a, n, t, w, c = stats[0]
        extra = f" — beats ctl on {w}/{c} slots" if c and a != "ctl" else ""
        print(f"  +{h:2d}m  {top}   >> leader: {a} (n={n}){extra}")
