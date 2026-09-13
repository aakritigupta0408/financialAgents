"""§7/§9/§32 follow-up — IS THE RESIDUAL GAP CALIBRATION OR INFORMATION?

The residual Oracle (Family A) added nothing OOS, but a tree given only
logit(p_mech) gained ~5% BSS — suggesting the surviving mech-vs-Kalshi gap is a
model-FORM (calibration) problem, not missing information. Test that directly:
fit a monotone recalibration of MECH_FAIR_BRTI (isotonic, train windows only, NO
features) and measure OOS Brier + how much of the mech-vs-Kalshi gap it closes,
by time-to-expiry. If recalibration alone closes most of the gap, the next model
work is heavy-tail/better-calibrated mechanics — not more features.

Writes research/oracle/mech_recalibration_result.json.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "results" / "brti_decision_dataset.jsonl"
MFB = ROOT / "research" / "oracle" / "mech_fair_brti_result.json"
OUT = ROOT / "research" / "oracle" / "mech_recalibration_result.json"


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def brier(p, y, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def p_mech_row(r, sig):
    lvl, tte = r["current_brti"], max(1.0, r["time_remaining_s"])
    p = Phi(r["brti_distance_to_target"] / (sig * lvl * math.sqrt(tte)))
    rem = r.get("settle_remaining") or 0
    req = r.get("required_remaining_average")
    if r["time_remaining_s"] <= 60 and rem > 0 and req is not None:
        s_rem = sig * lvl * math.sqrt(tte) / math.sqrt(rem)
        p = Phi((lvl - req) / max(1e-6, s_rem))
    return min(1 - 1e-4, max(1e-4, p))


def main():
    from sklearn.isotonic import IsotonicRegression
    sig = json.load(MFB.open())["sigma_rel_per_sqrt_s"]
    rows = [json.loads(l) for l in DEC.open() if l.strip()]
    rows = [r for r in rows if r.get("current_brti") is not None
            and r["time_remaining_s"] >= 1]
    for r in rows:
        r["p_mech"] = p_mech_row(r, sig)

    first, perwin = {}, {}
    for r in rows:
        tk = r["market_window_id"]
        first[tk] = min(first.get(tk, 1e18), r["decision_time"])
        perwin[tk] = perwin.get(tk, 0) + 1
    order = sorted(first, key=lambda t: first[t]); n = len(order)
    trw = set(order[:int(.7 * n)])
    tr = [r for r in rows if r["market_window_id"] in trw]
    te = [r for r in rows if r["market_window_id"] not in trw]

    def col(rs, k):
        return np.array([r[k] for r in rs], float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(col(tr, "p_mech"), col(tr, "exact_yes"),
            sample_weight=1.0 / np.array([perwin[r["market_window_id"]] for r in tr]))

    yte = col(te, "exact_yes")
    wte = np.array([1.0 / perwin[r["market_window_id"]] for r in te])
    pm = col(te, "p_mech")
    pr = np.clip(iso.predict(pm), 1e-4, 1 - 1e-4)
    pk = col(te, "k_prob")
    tte = col(te, "time_remaining_s") / 60.0

    def B(p, m=None):
        if m is None:
            return round(brier(p, yte, wte), 4)
        return round(brier(p[m], yte[m], wte[m]), 4)

    bm, br, bk = B(pm), B(pr), B(pk)
    by_time = {}
    for lo, hi, nm in [(0, 2, "T-2..0"), (2, 5, "T-5..2"), (5, 9, "T-9..5"), (9, 20, "T-15..9")]:
        m = (tte >= lo) & (tte < hi)
        if m.sum() < 20:
            continue
        by_time[nm] = {"n": int(m.sum()), "mech": B(pm, m), "mech_recal": B(pr, m),
                       "kalshi": B(pk, m),
                       "recal_closed_gap": round((B(pm, m) - B(pr, m))
                                                 / max(1e-9, B(pm, m) - B(pk, m)), 3)
                       if B(pm, m) > B(pk, m) else None}
    gap_before = bm - bk
    gap_after = br - bk
    doc = {"n_test": len(te), "windows": n,
           "brier": {"MECH_BRTI": bm, "MECH_BRTI_recalibrated": br, "KALSHI": bk},
           "mech_vs_kalshi_gap_before_recal": round(gap_before, 4),
           "mech_vs_kalshi_gap_after_recal": round(gap_after, 4),
           "fraction_of_gap_closed_by_recalibration":
               round((gap_before - gap_after) / gap_before, 3) if gap_before > 1e-9 else None,
           "by_time_to_expiry": by_time,
           "conclusion": None}
    fc = doc["fraction_of_gap_closed_by_recalibration"]
    if fc is not None:
        doc["conclusion"] = (
            "GAP_IS_MOSTLY_CALIBRATION — a monotone recalibration of mechanics "
            "(no features) closes {:.0f}% of the mech-vs-Kalshi gap; the residual "
            "is model-FORM (heavy-tail σ), not missing information.".format(fc * 100)
            if fc >= 0.5 else
            "GAP_IS_MIXED — recalibration closes {:.0f}%; a real information "
            "component remains (needs Families B-E capture).".format(fc * 100))
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"MECH recalibration — n_test {len(te)}")
    print(f"  Brier: MECH_BRTI {bm}  recalibrated {br}  KALSHI {bk}")
    print(f"  mech-vs-kalshi gap: {gap_before:+.4f} -> {gap_after:+.4f} "
          f"(closed {fc*100 if fc else 0:.0f}%)")
    for k, v in by_time.items():
        print(f"  {k:9s} mech {v['mech']} recal {v['mech_recal']} kalshi {v['kalshi']} "
              f"closed {v['recal_closed_gap']}")
    print("CONCLUSION:", doc["conclusion"])


if __name__ == "__main__":
    main()
