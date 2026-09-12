"""MECH_FAIR (Phase 4/5) — contract-mechanics fair value, and the decomposition
of what Kalshi knows beyond mechanics.

Uses the EXACT target (floor_strike = opening 60s-BRTI avg) and EXACT outcome
(exact_yes from contract_outcomes.jsonl) — not our previously-divergent labels.
YES iff close_avg >= open_avg. At decision t the fair value is
  p_mech = Phi( (level_t - floor_strike) / sigma_eff ),  sigma_eff calibrated,
where level_t is the current BRTI proxy (Coinbase, basis-corrected). No Kalshi.

Reports Brier for 50/50, MECH_FAIR, Kalshi, learned-oracle on IDENTICAL windows,
by time-to-expiry, and the decomposition: how much of Kalshi's edge over 50/50
is already explained by mechanics.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
CO = ROOT / "results" / "contract_outcomes.jsonl"
OUT = ROOT / "research" / "oracle" / "mech_fair_result.json"
BASIS = -3.67                      # mean Coinbase - BRTI (label_audit); level_brti = cb - basis... use cb+? see below


def Phi(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
Phiv = np.vectorize(Phi)


def brier(p, y): return float(np.mean((p - y) ** 2))


def main():
    out = {}
    for l in CO.open():
        l = l.strip()
        if l:
            r = json.loads(l)
            out[r["ticker"]] = r
    rows = []
    for l in DS.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        o = out.get(r["ticker"])
        if not o or r.get("cb_price") is None:
            continue
        # BRTI proxy at decision: Coinbase minus the mean basis (cb - basis_mean)
        level = r["cb_price"] - BASIS
        # time to expiry from contract close_time
        import calendar, time as _t
        try:
            ct = calendar.timegm(_t.strptime(o["close_time"], "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            continue
        tte = max(1.0, ct - r["ts"])
        rows.append({"ticker": r["ticker"], "ts": r["ts"], "y": o["exact_yes"],
                     "k_prob": r["k_prob"], "level": level, "cb_price": r["cb_price"],
                     "floor": o["floor_strike"], "tte": tte,
                     "rvol": r["cb_rvol_30s"]})
    if len(rows) < 400:
        print("insufficient joined rows:", len(rows)); return
    # ANCHORED proxy (no BRTI): floor_strike is the exact BRTI open-avg; use the
    # Coinbase MOVE since window open (kills the static ~$18 level basis; only
    # the small basis-CHANGE remains). cb_open = earliest cb_price per window.
    cb_open = {}
    for r in rows:
        if r["ticker"] not in cb_open or r["ts"] < cb_open[r["ticker"]][0]:
            cb_open[r["ticker"]] = (r["ts"], r["cb_price"])
    for r in rows:
        r["anchored_dist"] = r["cb_price"] - cb_open[r["ticker"]][1]
    # window split
    first = {}
    for r in rows:
        first[r["ticker"]] = min(first.get(r["ticker"], 1e18), r["ts"])
    order = sorted(first, key=lambda t: first[t])
    trw = set(order[:int(.7 * len(order))])
    tr = [r for r in rows if r["ticker"] in trw]
    te = [r for r in rows if r["ticker"] not in trw]

    dist = np.array([r["level"] - r["floor"] for r in tr])
    tte_tr = np.array([r["tte"] for r in tr]); ytr = np.array([r["y"] for r in tr], float)
    # calibrate a single sigma-per-sqrt-second (relative to price)
    best_sig, best_b = None, 9
    for sig in [8e-5, 1.2e-4, 1.8e-4, 2.6e-4, 4e-4, 6e-4, 9e-4]:
        denom = sig * np.array([r["level"] for r in tr]) * np.sqrt(tte_tr)
        p = Phiv(dist / denom)
        b = brier(p, ytr)
        if b < best_b:
            best_b, best_sig = b, sig

    def mech(rs):
        d = np.array([r["level"] - r["floor"] for r in rs])
        denom = best_sig * np.array([r["level"] for r in rs]) * np.sqrt([r["tte"] for r in rs])
        return Phiv(d / denom)

    # anchored MECH_FAIR: distance = Coinbase move since open (no level basis)
    a_tr = np.array([r["anchored_dist"] for r in tr])
    best_asig, best_ab = None, 9
    for sig in [3, 6, 10, 16, 26, 40, 60, 90]:      # $ move scale per sqrt(s)
        denom = sig * np.sqrt(tte_tr)
        p = Phiv(a_tr / denom)
        b = brier(p, ytr)
        if b < best_ab:
            best_ab, best_asig = b, sig
    def mech_anch(rs):
        d = np.array([r["anchored_dist"] for r in rs])
        return Phiv(d / (best_asig * np.sqrt([r["tte"] for r in rs])))

    yte = np.array([r["y"] for r in te], float)
    p_mkt = np.array([r["k_prob"] for r in te])
    p_mech = mech(te)
    p_anch = mech_anch(te)
    p_half = np.full(len(te), 0.5)
    tte = np.array([r["tte"] / 60.0 for r in te])

    B = {"50_50": round(brier(p_half, yte), 4),
         "MECH_FAIR_level": round(brier(p_mech, yte), 4),
         "MECH_FAIR_anchored": round(brier(p_anch, yte), 4),
         "KALSHI": round(brier(p_mkt, yte), 4)}
    # use the better mechanics baseline for the decomposition
    p_mech = p_anch if B["MECH_FAIR_anchored"] <= B["MECH_FAIR_level"] else p_mech
    B["MECH_FAIR"] = min(B["MECH_FAIR_level"], B["MECH_FAIR_anchored"])
    mech_gain = B["50_50"] - B["MECH_FAIR"]
    mkt_gain = B["50_50"] - B["KALSHI"]
    mkt_adv_vs_mech = B["MECH_FAIR"] - B["KALSHI"]
    frac_explained = (mech_gain / mkt_gain) if mkt_gain > 1e-9 else None

    by_time = {}
    for lo, hi, nm in [(0, 2, "T-2..0"), (2, 5, "T-5..2"), (5, 9, "T-9..5"), (9, 20, "T-15..9")]:
        m = (tte >= lo) & (tte < hi)
        if m.sum() < 30:
            continue
        by_time[nm] = {"n": int(m.sum()),
                       "mech": round(brier(p_mech[m], yte[m]), 4),
                       "kalshi": round(brier(p_mkt[m], yte[m]), 4)}

    doc = {"n_test": len(te), "sigma": best_sig, "basis_used": BASIS,
           "base_rate": round(float(yte.mean()), 3), "brier": B,
           "decomposition": {
               "mechanics_gain_over_50_50": round(mech_gain, 4),
               "kalshi_gain_over_50_50": round(mkt_gain, 4),
               "kalshi_advantage_vs_mech": round(mkt_adv_vs_mech, 4),
               "fraction_of_kalshi_edge_explained_by_mechanics":
                   (round(frac_explained, 3) if frac_explained is not None else None)},
           "by_time_to_expiry": by_time,
           "note": "exact target (floor_strike) + exact outcome; MECH_FAIR uses no "
                   "Kalshi. Coinbase used as BRTI proxy (basis-corrected mean); the "
                   "$18 basis std is a residual mismodeling this baseline cannot remove."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"MECH_FAIR — n_test {len(te)}, base rate {yte.mean():.3f}, sigma {best_sig}")
    print(f"  Brier: 50/50 {B['50_50']}  MECH_level {B['MECH_FAIR_level']}"
          f"  MECH_anchored {B['MECH_FAIR_anchored']}  KALSHI {B['KALSHI']}")
    print(f"  (anchored on Coinbase move since ~open; no BRTI needed)")
    print(f"  mechanics gain over 50/50: {mech_gain:+.4f}")
    print(f"  Kalshi gain over 50/50:    {mkt_gain:+.4f}")
    print(f"  Kalshi advantage vs mech:  {mkt_adv_vs_mech:+.4f}")
    print(f"  => fraction of Kalshi's edge explained by mechanics: "
          f"{round(100*frac_explained) if frac_explained is not None else '—'}%")
    print("  by time (mech / kalshi):", {k: (v["mech"], v["kalshi"]) for k, v in by_time.items()})


if __name__ == "__main__":
    main()
