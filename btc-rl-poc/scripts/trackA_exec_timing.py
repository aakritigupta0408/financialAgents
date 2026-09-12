"""TRACK A — can the confirmed T0.5 lead improve ENTRY ECONOMICS for an
already-decided settlement thesis? (handoff Track A). NOT settlement alpha.

Frozen side: we intend to BUY the contract at the current executable price.
The T0.5 model predicts the near-term reprice; a timing policy may DEFER the
entry ~15s when it predicts the price will fall (cheaper for a buyer), else
ENTER NOW. We measure the entry-price improvement in cents/$1 — and, the
make-or-break test, whether it survives realistic decision->fill LATENCY
(we observe features at t but can only act at t+L, by which point Kalshi
has already partly repriced).

Control = enter at t+L (frozen behavior). Treatment differs only on DEFER.
Improvement (YES buyer) on a deferred window = (price@t+L) - (price@+15s)
= dprob_L - dprob_15  (>0 when the price fell as predicted).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
OUT = ROOT / "research" / "t05_repricing" / "trackA_result.json"
FEATS = ["cb_ofi_30s", "cb_ret_30s", "cb_rvol_30s", "cb_l1_imb",
         "cb_micro_dev", "trade_n_30s", "k_spread"]
LATENCIES = [0, 1, 2, 5]          # seconds; dprob_0 == 0
FEE_FLOOR_C = 1.75                 # ~ Kalshi fee at p=0.5, cents/$1


def ridge(X, y, lam=1.0):
    n, d = X.shape
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def main():
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    rows = [r for r in rows if r.get("dprob_15") is not None]
    # window split (disjoint, chronological)
    first = {}
    for r in rows:
        first[r["ticker"]] = min(first.get(r["ticker"], 1e18), r["ts"])
    order = sorted(first, key=lambda t: first[t])
    cut = set(order[:int(.7 * len(order))])
    tr = [r for r in rows if r["ticker"] in cut]
    te = [r for r in rows if r["ticker"] not in cut]

    Xtr = np.array([[r[f] for f in FEATS] for r in tr], float)
    ytr = np.array([r["dprob_15"] for r in tr], float)      # target: 15s move
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd
    beta = ridge(Xtr, ytr, lam=5.0)

    Xte = (np.array([[r[f] for f in FEATS] for r in te], float) - mu) / sd
    pred = Xte @ beta                                       # predicted 15s move
    d15 = np.array([r["dprob_15"] for r in te], float)
    dL = {L: np.array([(r.get("dprob_%d" % L) or 0.0) if L else 0.0
                       for r in te], float) for L in LATENCIES}
    kspread = np.array([r.get("k_spread", 0.0) for r in te], float)

    # correlation of predicted vs actual near-term move (sanity)
    ic = float(np.corrcoef(pred, d15)[0, 1])

    # policy: DEFER to +15s when we predict the price will FALL by > eps
    results = {}
    best = None
    for eps in [0.001, 0.002, 0.005]:
        defer = pred <= -eps                                 # predict drop
        for L in LATENCIES:
            # improvement per entry (cents/$1): only deferred windows differ
            impr = np.where(defer, (dL[L] - d15) * 100.0, 0.0)
            cov = float(defer.mean())
            mean_impr = float(impr.mean())                   # over ALL entries
            mean_impr_def = float(impr[defer].mean()) if defer.any() else 0.0
            # net of half-spread cost incurred to act (round-trip proxy)
            net = mean_impr - 0.0                             # entry-only; fee same both arms
            results.setdefault("eps_%.3f" % eps, {})["L%d" % L] = {
                "defer_coverage": round(cov, 3),
                "mean_improvement_c_per_$1": round(mean_impr, 4),
                "mean_improvement_on_deferred_c": round(mean_impr_def, 4),
                "adverse_rate": round(float((impr < 0).mean()), 3)}
            key = (eps, L)
            if L == 2 and (best is None or mean_impr > best[1]):
                best = (key, mean_impr, cov)

    # verdict: does it survive latency (L>=2s) and beat the fee floor?
    l2 = results.get("eps_0.002", {}).get("L2", {})
    l0 = results.get("eps_0.002", {}).get("L0", {})
    impr_l0 = l0.get("mean_improvement_c_per_$1", 0.0)
    impr_l2 = l2.get("mean_improvement_c_per_$1", 0.0)
    if impr_l2 <= 0.0:
        verdict = "NO_EXECUTION_VALUE"
    elif impr_l2 < 0.3 or impr_l2 < 0.15 * FEE_FLOOR_C:
        verdict = "REPRICE_SIGNAL_NOT_EXECUTABLE"
    else:
        verdict = "EXECUTION_TIMING_PROMISING"

    out = {"n_test_entries": len(te), "n_train": len(tr),
           "predicted_move_IC": round(ic, 4),
           "median_k_spread_c": round(float(np.median(kspread) * 100), 3),
           "fee_floor_c": FEE_FLOOR_C,
           "improvement_vs_latency_eps0.002": {
               "L0": impr_l0, "L1": results["eps_0.002"]["L1"]["mean_improvement_c_per_$1"],
               "L2": impr_l2, "L5": results["eps_0.002"]["L5"]["mean_improvement_c_per_$1"]},
           "policies": results, "verdict": verdict,
           "note": "entry-only timing for a FROZEN side; improvement is the "
                   "price a buyer saves by deferring on a predicted drop; the "
                   "signal must survive decision->fill latency."}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"Track A — {len(te)} test entries, predicted-move IC {ic:+.3f}")
    print(f"median Kalshi spread {out['median_k_spread_c']:.2f}c · fee floor {FEE_FLOOR_C}c")
    print("entry improvement (c/$1) vs latency @eps0.002:")
    for L in LATENCIES:
        r = results["eps_0.002"]["L%d" % L]
        print(f"  L={L}s  improve {r['mean_improvement_c_per_$1']:+.4f}c/$1"
              f"  defer_cov {r['defer_coverage']:.2f}  adverse {r['adverse_rate']:.2f}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
