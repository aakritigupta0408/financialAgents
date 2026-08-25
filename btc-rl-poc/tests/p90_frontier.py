"""Precision-first frontiers for kb2 and kb4: smallest tau where BOTH
class precisions clear a target, with coverage and recall at that point."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]


def prf(sel):
    out = {}
    for cls, val in (("up", 1), ("down", 0)):
        tp = sum(1 for r in sel if r["call"] == val and r["actual"] == val)
        fp = sum(1 for r in sel if r["call"] == val and r["actual"] != val)
        fn = sum(1 for r in sel if r["call"] != val and r["actual"] == val)
        out[cls] = (tp / (tp + fp) if tp + fp else None,
                    tp / (tp + fn) if tp + fn else None)
    return out


for v in ("kb2", "kb4"):
    rows = [r for r in kb if r.get("variant") == v
            and r.get("actual") is not None]
    print(f"\n== {v} ({len(rows)} settled calls) ==")
    for target in (0.85, 0.90, 0.95):
        found = None
        for t100 in range(55, 100):
            tau = t100 / 100
            sel = [r for r in rows
                   if max(r["p_up"], 1 - r["p_up"]) >= tau]
            if len(sel) < 40:
                break
            s = prf(sel)
            if ((s["up"][0] or 0) >= target
                    and (s["down"][0] or 0) >= target):
                found = (tau, s, len(sel))
                break
        if found:
            tau, s, n = found
            # recall vs ALL actuals in the full row set
            full = prf(rows)
            rec_up = sum(1 for r in rows if r["call"] == 1
                         and r["actual"] == 1
                         and max(r["p_up"], 1 - r["p_up"]) >= tau) / \
                max(1, sum(1 for r in rows if r["actual"] == 1))
            rec_dn = sum(1 for r in rows if r["call"] == 0
                         and r["actual"] == 0
                         and max(r["p_up"], 1 - r["p_up"]) >= tau) / \
                max(1, sum(1 for r in rows if r["actual"] == 0))
            print(f"  P{int(target*100)}: tau {tau:.2f} -> "
                  f"UP prec {s['up'][0]:.3f} DOWN prec {s['down'][0]:.3f} "
                  f"| coverage {n/len(rows):.0%} "
                  f"| global recall UP {rec_up:.0%} DOWN {rec_dn:.0%}")
        else:
            print(f"  P{int(target*100)}: not reachable (n floor)")
