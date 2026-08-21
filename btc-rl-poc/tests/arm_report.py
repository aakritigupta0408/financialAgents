"""Per-arm learning report: MASE eras, direction+PT, coverage, gate record,
direction P&L — the evidence base for a strengths/shortcomings review."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from btc_rl.history import load_history
from btc_rl.metrics import pt_test

rows = [json.loads(l) for l in
        (ROOT / "results" / "prediction_log.jsonl").read_text().splitlines()]
sc = [r for r in rows if r["actual"] is not None]

FAMS = ["h", "t2-h", "t6-h", "t7-h", "t8-h", "t9-h", "t10-h", "t11-h"]

print(f"{'arm':>8} {'n':>5} {'MASE all':>8} {'1st→2nd half':>13} "
      f"{'dir%':>5} {'PTz':>5} {'cov%':>5} {'P&L bps':>8}")
for h in (1, 5, 15, 30):
    for fam in FAMS:
        v = f"{fam}{h}"
        g = [r for r in sc if r["variant"] == v]
        if len(g) < 30:
            continue
        naive = [abs(r["actual"] - r["price_now"]) for r in g]

        def mase_of(rs):
            nv = sum(abs(r["actual"] - r["price_now"]) for r in rs) / len(rs)
            return (sum(r["abs_err"] for r in rs) / len(rs)) / nv if nv else 0
        half = len(g) // 2
        moved = [r for r in g if r["delta"]]
        z = pt_test([r["pred"] > r["price_now"] for r in moved],
                    [r["actual"] > r["price_now"] for r in moved]) \
            if len(moved) >= 20 else None
        dirp = (100 * sum(1 for r in moved
                          if (r["pred"] > r["price_now"])
                          == (r["actual"] > r["price_now"])) / len(moved)
                if moved else 0)
        banded = [r for r in g if r.get("in_band") is not None]
        cov = (100 * sum(r["in_band"] for r in banded) / len(banded)
               if banded else 0)
        pnl = sum((1 if r["delta"] > 0 else -1)
                  * (r["actual"] - r["price_now"]) / r["price_now"] * 1e4
                  for r in moved)
        print(f"{v:>8} {len(g):>5} {mase_of(g):>8.3f} "
              f"{mase_of(g[:half]):>6.2f}→{mase_of(g[half:]):<5.2f} "
              f"{dirp:>4.0f}% {z if z is None else round(z, 1)!s:>5} "
              f"{cov:>4.0f}% {pnl:>+7.0f}")
    print()

print("GATE RECORD (hourly replay kept vs reverted, from metrics_history):")
gates = defaultdict(lambda: [0, 0])
for r in load_history("retrain"):
    for arm, hs in (r.get("gate") or {}).items():
        fam = arm.split("-")[0] if "-" in arm else "ctl/rp"
        for hk, d in hs.items():
            gates[arm][0 if not d["reverted"] else 1] += 1
for arm in sorted(gates):
    k, rev = gates[arm]
    print(f"  {arm:>10}: kept {k}, reverted {rev}")

print()
print("t11 vs t2 twin check (paired slots, should be ~equal with no votes):")
for h in (1, 5, 15, 30):
    a = {r["made_ts"]: r["abs_err"] for r in sc if r["variant"] == f"t11-h{h}"}
    b = {r["made_ts"]: r["abs_err"] for r in sc if r["variant"] == f"t2-h{h}"}
    common = sorted(set(a) & set(b))
    if len(common) >= 30:
        print(f"  +{h:>2}m n={len(common)}: t11 "
              f"${sum(a[t] for t in common)/len(common):.0f} vs t2 "
              f"${sum(b[t] for t in common)/len(common):.0f}")
