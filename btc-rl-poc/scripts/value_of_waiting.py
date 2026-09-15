"""VALUE OF WAITING — accuracy & coverage vs DECISION TIME (mins-left).

The capstone baseline is 89% hit @ 90% coverage. Our 11-min-entry ceiling was ~0.66-0.70;
the biggest proven lever is entry TIMING (accuracy rises as the window fills). This measures
the full curve: for each decision time T (mins-left), the barrier first-passage probability
P(final >= target) and its confidence, then a coverage/hit sweep — to find the (T, threshold)
where 89%@90% appears. The barrier is a PARAMETER-FREE physics formula (no training, no
leakage): z = (price - target) / (sigma * sqrt(time_remaining)), P = Phi(z), with the
60s-averaged (Asian) settlement variance reduction.

Data: results/brti_decision_dataset.jsonl — many intra-window rows per window with
time_remaining_s, current_brti, official_target, cb_rvol_30s, exact_yes.
"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "brti_decision_dataset.jsonl"
OUT = ROOT / "research" / "value_of_waiting.json"
AVG_VAR = 1.0 / 3     # variance of trailing-60s average vs point (~1/3)


def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load():
    byw = defaultdict(list)
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1) and d.get("time_remaining_s") is not None and d.get("current_brti"):
            byw[d.get("market_window_id")].append(d)
    return byw


def barrier_p(cur, tgt, tr_s, rvol30, cur_for_vol):
    """P(60s-avg final >= tgt) from first-passage. rvol30 = $ vol over 30s (cb_rvol_30s)."""
    if rvol30 and cur_for_vol:
        sig_per_s = (rvol30) / math.sqrt(30.0)              # $/sqrt(s)
    else:
        sig_per_s = cur * 1e-4 / math.sqrt(30.0)            # fallback ~1bp/30s
    sigma_T = sig_per_s * math.sqrt(max(1.0, tr_s)) * math.sqrt(AVG_VAR + (1 - AVG_VAR) * 0)
    z = (cur - tgt) / max(sigma_T, 1e-6)
    return _phi(z)


def run():
    byw = load()
    targets_min = [12, 10, 8, 6, 5, 4, 3, 2, 1, 0.5]
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rep = {"n_windows": len(byw), "by_decision_time": {}}
    hit90_at = None
    for T in targets_min:
        tr_target = T * 60.0
        preds = []   # (confidence, correct, p_up, label)
        for wid, rows in byw.items():
            rows2 = [r for r in rows if (r.get("time_remaining_s") or 0) >= tr_target - 20]
            if not rows2:
                continue
            d = min(rows2, key=lambda r: abs((r.get("time_remaining_s") or 0) - tr_target))
            cur = d["current_brti"]; tgt = d.get("official_target") or d.get("official_target_kalshi")
            trs = d.get("time_remaining_s"); rvol = d.get("cb_rvol_30s")
            if not tgt or not trs:
                continue
            p = barrier_p(cur, tgt, trs, rvol, cur)
            conf = abs(p - 0.5) * 2
            correct = int((p >= 0.5) == bool(d["exact_yes"]))
            preds.append((conf, correct))
        if not preds:
            continue
        arr = np.array(preds, float)
        conf, corr = arr[:, 0], arr[:, 1]
        sweep = []
        best_at_cov90 = None
        for c in thresholds:
            m = conf >= c
            cov = float(m.mean())
            hit = float(corr[m].mean()) if m.any() else None
            sweep.append({"thr": c, "coverage": round(cov, 4), "hit": round(hit, 4) if hit is not None else None})
            if hit is not None and cov >= 0.90:
                if best_at_cov90 is None or hit > best_at_cov90[1]:
                    best_at_cov90 = (c, hit, cov)
            if hit is not None and hit >= 0.89 and cov >= 0.90 and hit90_at is None:
                hit90_at = {"mins_left": T, "thr": c, "hit": round(hit, 4), "coverage": round(cov, 4)}
        rep["by_decision_time"][str(T)] = {
            "n": len(preds), "overall_hit": round(float(corr.mean()), 4),
            "best_hit_at_cov>=0.90": ({"thr": best_at_cov90[0], "hit": round(best_at_cov90[1], 4),
                                       "coverage": round(best_at_cov90[2], 4)} if best_at_cov90 else None),
            "sweep": sweep}
    rep["first_89hit_90cov"] = hit90_at
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"n_windows={rep['n_windows']}")
    print(f"{'minsLeft':>8} {'overallHit':>10} {'bestHit@cov>=.90':>18}")
    for T in targets_min:
        r = rep["by_decision_time"].get(str(T))
        if not r:
            continue
        b = r["best_hit_at_cov>=0.90"]
        print(f"{T:>8} {r['overall_hit']:>10} "
              f"{(str(b['hit'])+' @cov '+str(b['coverage'])+' thr '+str(b['thr'])) if b else 'na':>18}")
    print("FIRST 89%@90%:", rep["first_89hit_90cov"])
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    run()
