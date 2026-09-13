"""§6-8, §32, §47 — MECH_FAIR_BRTI and the GAP DECOMPOSITION.

Rebuild the mechanics control on EXACT BRTI contract state (no Coinbase proxy),
then answer the decisive question: does the previous large mid-window Kalshi gap
survive once MECH_FAIR uses exact current BRTI?

Fair value (Gaussian diffusion control, §7):
  z = (current_brti - official_target) / (sigma * current_brti * sqrt(tte))
  p_mech = Phi(z)
Settlement-aware refinement (§8) for rows inside the final 60s: the settlement
value is the 60s mean; already-observed samples are deterministic, so only the
unknown remainder is simulated. p reduces to Phi over required_remaining_average.

Reports Brier for 50/50, MECH_COINBASE_PROXY (historical), MECH_FAIR_BRTI, Kalshi,
overall + by time-to-expiry + by exact difficulty, and the gap decomposition.
Window-split, window-weighted, official labels. Writes
research/oracle/mech_fair_brti_result.json.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "results" / "brti_decision_dataset.jsonl"
OLD = ROOT / "research" / "oracle" / "mech_fair_result.json"   # Coinbase-proxy Brier
OUT = ROOT / "research" / "oracle" / "mech_fair_brti_result.json"


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
Phiv = np.vectorize(Phi)


def brier(p, y, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def main():
    rows = [json.loads(l) for l in DEC.open() if l.strip()]
    rows = [r for r in rows if r.get("current_brti") is not None
            and r.get("time_remaining_s", 0) >= 1]
    if len(rows) < 400:
        print("insufficient backfilled rows:", len(rows)); return

    # window split by first decision_time
    first = {}
    for r in rows:
        tk = r["market_window_id"]
        first[tk] = min(first.get(tk, 1e18), r["decision_time"])
    order = sorted(first, key=lambda t: first[t])
    n = len(order)
    trw = set(order[:int(.7 * n)])
    perwin = {}
    for r in rows:
        perwin[r["market_window_id"]] = perwin.get(r["market_window_id"], 0) + 1

    def arr(rs, k):
        return np.array([rs_[k] for rs_ in rs], float)

    tr = [r for r in rows if r["market_window_id"] in trw]
    te = [r for r in rows if r["market_window_id"] not in trw]

    dist_tr = arr(tr, "brti_distance_to_target")
    lvl_tr = arr(tr, "current_brti")
    tte_tr = arr(tr, "time_remaining_s")
    ytr = arr(tr, "exact_yes")

    # calibrate a single relative sigma-per-sqrt-second on train Brier
    best_sig, best_b = None, 9.0
    for sig in [4e-5, 6e-5, 8e-5, 1.1e-4, 1.5e-4, 2.0e-4, 2.8e-4, 4e-4, 6e-4]:
        denom = sig * lvl_tr * np.sqrt(tte_tr)
        p = Phiv(dist_tr / denom)
        b = brier(p, ytr, np.ones_like(ytr))
        if b < best_b:
            best_b, best_sig = b, sig

    def mech_brti(rs):
        d = arr(rs, "brti_distance_to_target")
        lvl = arr(rs, "current_brti")
        tte = arr(rs, "time_remaining_s")
        p = Phiv(d / (best_sig * lvl * np.sqrt(tte)))
        # §8 settlement-aware override in the final 60s
        for i, r in enumerate(rs):
            rem = r.get("settle_remaining") or 0
            seen = r.get("settle_seen") or 0
            req = r.get("required_remaining_average")
            if r["time_remaining_s"] <= 60 and rem > 0 and req is not None and seen >= 0:
                # unknown remainder mean ~ Normal(current_brti, s_rem); YES iff mean>=req
                s_rem = best_sig * r["current_brti"] * math.sqrt(
                    max(1.0, r["time_remaining_s"])) / math.sqrt(rem)
                z = (r["current_brti"] - req) / max(1e-6, s_rem)
                p[i] = Phi(z)
        return np.clip(p, 1e-4, 1 - 1e-4)

    yte = arr(te, "exact_yes")
    wte = np.array([1.0 / perwin[r["market_window_id"]] for r in te])
    p_mech = mech_brti(te)
    p_mkt = arr(te, "k_prob")
    p_half = np.full(len(te), 0.5)
    tte = arr(te, "time_remaining_s") / 60.0

    # exact difficulty (§5): |z| on the causal scale; small |z| = HARD (near boundary)
    z_te = np.abs(arr(te, "brti_distance_to_target")) / (
        best_sig * arr(te, "current_brti") * np.sqrt(arr(te, "time_remaining_s")))
    q33, q66 = np.quantile(z_te, [0.33, 0.66])
    diff = np.where(z_te <= q33, "HARD", np.where(z_te <= q66, "MED", "EASY"))

    def B(p):
        return round(brier(p, yte, wte), 4)

    brs = {"50_50": B(p_half), "MECH_FAIR_BRTI": B(p_mech), "KALSHI": B(p_mkt)}
    old = json.load(OLD.open()) if OLD.exists() else {}
    mech_coin = old.get("brier", {}).get("MECH_FAIR")
    brs["MECH_COINBASE_PROXY_historical"] = mech_coin

    def prof(mask):
        m = mask & (wte > 0)
        if m.sum() < 20:
            return None
        return {"n": int(m.sum()),
                "mech_brti": round(brier(p_mech[m], yte[m], wte[m]), 4),
                "kalshi": round(brier(p_mkt[m], yte[m], wte[m]), 4),
                "mech_minus_kalshi": round(brier(p_mech[m], yte[m], wte[m])
                                           - brier(p_mkt[m], yte[m], wte[m]), 4)}
    by_time = {}
    for lo, hi, nm in [(0, 2, "T-2..0"), (2, 5, "T-5..2"), (5, 9, "T-9..5"), (9, 20, "T-15..9")]:
        v = prof((tte >= lo) & (tte < hi))
        if v:
            by_time[nm] = v
    by_diff = {k: prof(diff == k) for k in ("HARD", "MED", "EASY")}
    by_diff = {k: v for k, v in by_diff.items() if v}

    # decomposition
    b5, bb, bk = brs["50_50"], brs["MECH_FAIR_BRTI"], brs["KALSHI"]
    mech_gain = b5 - bb
    mkt_gain = b5 - bk
    frac = (mech_gain / mkt_gain) if mkt_gain > 1e-9 else None
    # coinbase-proxy fraction, same statistic
    frac_coin = ((b5 - mech_coin) / mkt_gain) if (mech_coin and mkt_gain > 1e-9) else None

    doc = {
        "n_test": len(te), "windows": n, "sigma_rel_per_sqrt_s": best_sig,
        "base_rate": round(float(yte.mean()), 3),
        "brier": brs,
        "decomposition": {
            "mech_brti_gain_over_50_50": round(mech_gain, 4),
            "kalshi_gain_over_50_50": round(mkt_gain, 4),
            "kalshi_advantage_vs_mech_brti": round(bb - bk, 4),
            "frac_of_kalshi_edge_explained_by_MECH_BRTI":
                (round(frac, 3) if frac is not None else None),
            "frac_explained_by_MECH_COINBASE_proxy":
                (round(frac_coin, 3) if frac_coin is not None else None),
            "improvement_from_exact_brti":
                (round(frac - frac_coin, 3) if (frac is not None and frac_coin is not None) else None)},
        "by_time_to_expiry": by_time,
        "by_difficulty": by_diff,
        "gap_verdict": None,
        "note": "MECH_FAIR_BRTI uses exact current BRTI as contract state; Gaussian "
                "diffusion + settlement-aware final-60s override; no Kalshi input. "
                "Difficulty = |distance|/(sigma*brti*sqrt(tte)); HARD=near boundary.",
    }
    # classify the mid-window mystery (§32)
    mid = [by_time.get("T-9..5"), by_time.get("T-5..2")]
    mid = [m for m in mid if m]
    mid_gap = np.mean([m["mech_minus_kalshi"] for m in mid]) if mid else None
    if mid_gap is not None:
        doc["mid_window_mech_minus_kalshi"] = round(float(mid_gap), 4)
        doc["gap_verdict"] = ("GAP_MOSTLY_DISAPPEARS" if mid_gap < 0.005
                              else "GAP_SHRINKS_BUT_REMAINS" if mid_gap < 0.02
                              else "GAP_REMAINS_LARGE")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))

    print(f"MECH_FAIR_BRTI — n_test {len(te)}, base rate {yte.mean():.3f}, sigma {best_sig}")
    print(f"  Brier: 50/50 {brs['50_50']}  MECH_COINBASE {mech_coin}"
          f"  MECH_BRTI {brs['MECH_FAIR_BRTI']}  KALSHI {brs['KALSHI']}")
    print(f"  frac of Kalshi edge explained: COINBASE {frac_coin}  ->  BRTI {frac}")
    print("  by time (mech_brti / kalshi / mech-kalshi):")
    for k, v in by_time.items():
        print(f"    {k:9s} n={v['n']:4d} mech {v['mech_brti']} kalshi {v['kalshi']} gap {v['mech_minus_kalshi']:+.4f}")
    print("  by difficulty:", {k: (v["mech_brti"], v["kalshi"]) for k, v in by_diff.items()})
    print("  MID-WINDOW gap (mech-kalshi):", doc.get("mid_window_mech_minus_kalshi"),
          "->", doc["gap_verdict"])


if __name__ == "__main__":
    main()
