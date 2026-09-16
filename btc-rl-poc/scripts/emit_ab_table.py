"""EMIT the A/B comparison table — every treatment vs the T0 control, with significance.

For each treatment arm, matches its settled trades to T0's by window (ticker), computes the
PAIRED per-contract P&L difference (treatment - T0) on the overlapping windows, and reports:
  n (matched), hit-rate, net $, EV/trade (c), Δ EV/trade vs T0 (paired mean), a bootstrap 95%
  CI on that Δ, a paired t-test p-value, and a significance verdict. Per-contract normalization
  makes different stake sizes comparable; pairing on the same windows removes market-regime
  noise (a real A/B, not two separate series). Writes results/ab_table.json (rendered on home).
Read-only research. Metric = EV, the north-star.
"""
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "ab_table.json"

TREATMENTS = [("tv", "Value Gate"), ("ob", "Open+6 Barrier"),
              ("cg33", "Gated·33%"), ("fm", "Chronos-Bolt")]
CONTROL = ("pt", "T0 Baseline Follower")


def _settled(name):
    p = RES / f"{name}_trades.jsonl"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("actual") is not None and r.get("contracts"):
            out.append(r)
    return out


def _pc(rows):
    """ticker -> per-contract net pnl (cents), for pairing."""
    return {r["ticker"]: (r.get("pnl_c") or 0) / r["contracts"] for r in rows if r.get("ticker")}


def _summary(rows):
    if not rows:
        return {"n": 0, "hit_rate": None, "net_usd": 0.0, "ev_per_trade_c": None}
    wins = sum(1 for r in rows if r.get("win"))
    pnl = sum(r.get("pnl_c") or 0 for r in rows)
    return {"n": len(rows), "hit_rate": round(wins / len(rows), 4),
            "net_usd": round(pnl / 100, 2),
            "ev_per_trade_c": round(pnl / len(rows), 1)}


def _boot_ci(d, iters=5000):
    if len(d) < 2:
        return [None, None]
    idx = np.random.default_rng(0).integers(0, len(d), size=(iters, len(d)))
    means = d[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 1), round(float(np.percentile(means, 97.5)), 1)]


def _paired_p(d):
    """Paired t-test p-value (two-sided) via normal/t approx; None if too few."""
    n = len(d)
    if n < 3:
        return None
    m = float(d.mean()); sd = float(d.std(ddof=1))
    if sd == 0:
        return 0.0 if m != 0 else 1.0
    t = m / (sd / math.sqrt(n))
    try:
        from scipy import stats
        return round(float(2 * stats.t.sf(abs(t), n - 1)), 4)
    except Exception:
        return round(float(math.erfc(abs(t) / math.sqrt(2))), 4)   # normal approx


def run():
    t0 = _settled(CONTROL[0]); t0_pc = _pc(t0)
    rows = [{"arm": "pt", "name": CONTROL[1], "role": "CONTROL", **_summary(t0),
             "matched_n": None, "delta_ev_c": None, "ci95": [None, None],
             "p_value": None, "sig": "reference"}]
    for aid, name in TREATMENTS:
        rr = _settled(aid); summ = _summary(rr)
        pc = _pc(rr)
        common = [t for t in pc if t in t0_pc]
        d = np.array([pc[t] - t0_pc[t] for t in common], float)   # paired Δ per-contract
        p = _paired_p(d)
        mean_d = round(float(d.mean()), 1) if len(d) else None
        ci = _boot_ci(d) if len(d) >= 2 else [None, None]
        if len(d) < 10:
            sig = "collecting (need ~20+ paired)"
        elif p is not None and p < 0.05:
            sig = ("SIGNIFICANT + " if mean_d > 0 else "SIGNIFICANT - ")
        else:
            sig = "not significant"
        rows.append({"arm": aid, "name": name, "role": "TREATMENT", **summ,
                     "matched_n": len(d), "delta_ev_c": mean_d, "ci95": ci,
                     "p_value": p, "sig": sig})
    doc = {"schema": "ab-table-1", "metric": "EV per trade (per-contract net, cents)",
           "note": ("Paired vs T0 on matched windows (per-contract-normalized). Δ EV/trade > 0 "
                    "means the treatment beats the control on the same windows. 95% CI = bootstrap; "
                    "p = paired t-test. Small n -> wide CI / 'collecting'."),
           "rows": rows}
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"ab_table -> {OUT}")
    print(f"{'arm':6}{'n':>5}{'hit':>7}{'net$':>9}{'EV/tr':>8}{'matched':>8}{'ΔEV':>7}{'p':>8}  sig")
    for r in rows:
        print(f"{r['arm']:6}{str(r['n']):>5}{str(r['hit_rate']):>7}{r['net_usd']:>9}"
              f"{str(r['ev_per_trade_c']):>8}{str(r['matched_n']):>8}{str(r['delta_ev_c']):>7}"
              f"{str(r['p_value']):>8}  {r['sig']}")


if __name__ == "__main__":
    run()
