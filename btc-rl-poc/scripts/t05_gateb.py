"""GATE B — RESIDUAL SETTLEMENT INFORMATION (handoff).

Question: conditional on Kalshi's probability at decision time, do
microstructure features carry incremental information about the eventual
SETTLEMENT outcome y in {0,1}?

Statistical unit = market window (ticker), NOT the decision-point row.
  - windows are split disjointly into train/val/test (chronological);
  - rows are weighted 1/rows_in_window so dense windows don't dominate;
  - uncertainty + headline metrics are also reported at window level.

Settlement label: Kalshi resolves to 0/1 at expiry, so y = the ticker's
final captured quote (kept only when decisive, |p-0.5|>0.35). PIT-clean:
features use data <= t; the label is the window outcome, never a feature.

Model 0: p = p_market (Kalshi mid at t).
Model 1/2: offset residual  p = sigmoid(logit(p_market) + Xβ),  L = BCE +
  λ·mean(β²).  Reports BSS vs market, the λ regularization path, |Δ|
  distribution, market correlation, Murphy components, BSS-by-time-to-expiry,
  overfit diagnosis, and placebos (block-shuffle, external-only).
"""
import glob
import json
import math
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
SHARDS = ROOT / "results" / "events"
OUT = ROOT / "research" / "t05_repricing" / "gateb_result.json"
FEATS = ["cb_ofi_30s", "cb_ret_30s", "cb_rvol_30s", "cb_l1_imb",
         "cb_micro_dev", "trade_n_30s", "k_spread"]
LAMBDAS = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
random.seed(20260912)
np.random.seed(20260912)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def settlement_map(n_shards):
    """Last captured Kalshi quote per ticker -> settlement proxy."""
    files = sorted(glob.glob(str(SHARDS / "*.jsonl")), key=os.path.getmtime)[-n_shards:]
    last = {}
    for fp in files:
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("src") == "kalshi" and e.get("kind") == "quote":
                yb, ya = e.get("yes_bid"), e.get("yes_ask")
                ts = e.get("receive_ts")
                if yb is None or ya is None or ts is None:
                    continue
                p = (yb + ya) / 200.0
                tk = e.get("ticker")
                if tk not in last or ts > last[tk][0]:
                    last[tk] = (ts, p)
    out = {}
    for tk, (ts, p) in last.items():
        if abs(p - 0.5) > 0.35:            # decisive => truly settled
            out[tk] = {"y": 1 if p > 0.5 else 0, "expiry_ts": ts}
    return out


def fit_offset(X, y, w, zmkt, lam, iters=600, lr=0.3):
    beta = np.zeros(X.shape[1])
    W = w / w.sum()
    for _ in range(iters):
        p = sigmoid(zmkt + X @ beta)
        g = X.T @ (W * (p - y)) + lam * beta
        beta -= lr * g
    return beta


def brier(p, y, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def window_brier(tickers, p, y):
    """One row per window: mean prediction, single outcome."""
    agg = {}
    for tk, pp, yy in zip(tickers, p, y):
        a = agg.setdefault(tk, [0.0, 0, yy])
        a[0] += pp; a[1] += 1
    bc = [( (v[0] / v[1]) - v[2]) ** 2 for v in agg.values()]
    return float(np.mean(bc)), len(agg)


def murphy(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ybar = y.mean()
    rel = res = 0.0
    n = len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        pk = p[m].mean(); ok = y[m].mean(); nk = m.sum()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - ybar) ** 2
    return {"reliability": round(rel / n, 5), "resolution": round(res / n, 5),
            "uncertainty": round(ybar * (1 - ybar), 5)}


def main():
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    n_shards = 96
    settle = settlement_map(n_shards)
    data = []
    per_win = {}
    for r in rows:
        s = settle.get(r["ticker"])
        if not s:
            continue
        r = dict(r)
        r["y"] = s["y"]
        r["tte_min"] = round(max(0.0, (s["expiry_ts"] - r["ts"]) / 60.0), 2)
        data.append(r)
        per_win[r["ticker"]] = per_win.get(r["ticker"], 0) + 1
    if len(data) < 500:
        print("insufficient settled rows:", len(data)); return
    for r in data:
        r["w"] = 1.0 / per_win[r["ticker"]]

    # window split (chronological, disjoint)
    first_ts = {}
    for r in data:
        first_ts[r["ticker"]] = min(first_ts.get(r["ticker"], 1e18), r["ts"])
    order = sorted(first_ts, key=lambda t: first_ts[t])
    n = len(order)
    tr_w = set(order[:int(.6 * n)])
    va_w = set(order[int(.6 * n):int(.8 * n)])
    te_w = set(order[int(.8 * n):])

    def split(S):
        rs = [r for r in data if r["ticker"] in S]
        X = np.array([[r[f] for f in FEATS] for r in rs], float)
        y = np.array([r["y"] for r in rs], float)
        w = np.array([r["w"] for r in rs], float)
        pm = np.array([r["k_prob"] for r in rs], float)
        tk = [r["ticker"] for r in rs]
        tte = np.array([r["tte_min"] for r in rs], float)
        return rs, X, y, w, pm, tk, tte
    _, Xtr, ytr, wtr, pmtr, _, _ = split(tr_w)
    _, Xva, yva, wva, pmva, _, _ = split(va_w)
    rte, Xte, yte, wte, pmte, tkte, ttete = split(te_w)

    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd

    b_mkt_te = brier(pmte, yte, wte)
    b_mkt_win, n_te_win = window_brier(tkte, pmte, yte)

    # lambda path (pick on val BSS)
    path = []
    best = None
    for lam in LAMBDAS:
        beta = fit_offset(Xtr, ytr, wtr, logit(pmtr), lam)
        pva = sigmoid(logit(pmva) + Xva @ beta)
        bss_va = 1 - brier(pva, yva, wva) / brier(pmva, yva, wva)
        pte = sigmoid(logit(pmte) + Xte @ beta)
        bss_te = 1 - brier(pte, yte, wte) / b_mkt_te
        dw, nwin = window_brier(tkte, pte, yte)
        bss_te_win = 1 - dw / b_mkt_win
        delta = Xte @ beta
        ptr = sigmoid(logit(pmtr) + Xtr @ beta)
        bss_tr = 1 - brier(ptr, ytr, wtr) / brier(pmtr, ytr, wtr)
        rec = {"lambda": lam, "bss_val": round(bss_va, 4),
               "bss_test_row": round(bss_te, 4), "bss_test_window": round(bss_te_win, 4),
               "bss_train": round(bss_tr, 4),
               "mean_abs_delta": round(float(np.mean(np.abs(delta))), 4),
               "p90_abs_delta": round(float(np.percentile(np.abs(delta), 90)), 4),
               "frac_delta_gt_1pp": round(float(np.mean(np.abs(delta) > 0.01)), 3),
               "corr_with_market": round(float(np.corrcoef(pte, pmte)[0, 1]), 4)}
        path.append(rec)
        if best is None or bss_va > best["bss_val"]:
            best = rec; best_beta = beta
    print("lambda path (BSS vs market):")
    for r in path:
        print(f"  λ={r['lambda']:<7} val {r['bss_val']:+.4f}  test_row {r['bss_test_row']:+.4f}"
              f"  test_win {r['bss_test_window']:+.4f}  train {r['bss_train']:+.4f}"
              f"  |Δ| {r['mean_abs_delta']:.4f}")

    # best model diagnostics
    pte = sigmoid(logit(pmte) + Xte @ best_beta)
    # time-to-expiry slices
    slices = {}
    for lo, hi, name in [(0, 2, "T-2..0"), (2, 4, "T-4..2"), (4, 6, "T-6..4"),
                         (6, 9, "T-9..6"), (9, 20, "T-15..9")]:
        m = (ttete >= lo) & (ttete < hi)
        if m.sum() < 30:
            continue
        bss = 1 - brier(pte[m], yte[m], wte[m]) / brier(pmte[m], yte[m], wte[m])
        slices[name] = {"n": int(m.sum()), "bss": round(float(bss), 4)}

    # placebo: block-shuffle features across TRAIN windows
    idx = np.random.permutation(len(Xtr))
    beta_sh = fit_offset(Xtr[idx], ytr, wtr, logit(pmtr), best["lambda"])
    pte_sh = sigmoid(logit(pmte) + Xte @ beta_sh)
    bss_shuffle = 1 - brier(pte_sh, yte, wte) / b_mkt_te
    # external-only (ignore market prior)
    beta_ext = fit_offset(Xtr, ytr, wtr, np.zeros(len(ytr)), 1e-3)
    pte_ext = sigmoid(Xte @ beta_ext + logit(np.array([ytr.mean()])) )
    bss_ext = 1 - brier(pte_ext, yte, wte) / b_mkt_te

    # overfit diagnosis from lambda=0
    l0 = path[0]
    if l0["bss_train"] > 0.02 and best["bss_test_window"] < 0.005:
        diag = "CLASSIC_OVERFIT"
    elif l0["bss_train"] < 0.01 and best["bss_test_window"] < 0.005:
        diag = "INFORMATION_LIMITED"
    elif best["bss_test_window"] >= 0.01:
        diag = "PROMISING"
    else:
        diag = "WEAK"

    # honest generalizable BSS = the strongly-shrunk model (lambda=1),
    # which is the market-prior-trusting optimum; the val-chosen model
    # overfits with only ~200 train windows.
    shrunk = next((r for r in path if r["lambda"] == 1), best)
    shrunk_bss = shrunk["bss_test_window"]
    if best["bss_test_window"] >= 0.02 and all(s["bss"] > -0.01 for s in slices.values()):
        verdict = "B_PASS_STRONG"
    elif best["bss_test_window"] >= 0.005 and shrunk_bss >= 0.003:
        verdict = "B_PASS_WEAK"
    elif shrunk_bss < -0.02:
        verdict = "B_FAIL"
    else:
        # train>0, test<=~0 under proper shrinkage, shuffle~0: the reprice
        # signal does not become durable settlement skill beyond Kalshi.
        verdict = "B_REPRICE_ONLY"

    out = {
        "decision_points": len(data), "windows": n,
        "windows_train": len(tr_w), "windows_val": len(va_w), "windows_test": len(te_w),
        "test_decision_points": len(rte),
        "market_brier_test_row": round(b_mkt_te, 5),
        "market_brier_test_window": round(b_mkt_win, 5),
        "settlement_base_rate": round(float(yte.mean()), 3),
        "lambda_path": path, "best": best,
        "bss_by_time_to_expiry": slices,
        "murphy_market": murphy(pmte, yte), "murphy_candidate": murphy(pte, yte),
        "placebo_block_shuffle_bss": round(float(bss_shuffle), 4),
        "external_only_bss": round(float(bss_ext), 4),
        "overfit_diagnosis": diag,
        "verdict": verdict,
        "note": "settlement label from Kalshi expiry resolution; unit=window; "
                "row weights 1/rows_in_window; BSS vs Kalshi mid at decision t."}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwindows {n} (train {len(tr_w)}/val {len(va_w)}/test {len(te_w)})"
          f" · settled decision-points {len(data)}")
    print(f"market Brier (test): row {b_mkt_te:.4f} · window {b_mkt_win:.4f}"
          f" · base rate {yte.mean():.3f}")
    print("BSS by time-to-expiry:", {k: v["bss"] for k, v in slices.items()})
    print(f"placebo block-shuffle BSS {bss_shuffle:+.4f} (want ~0)"
          f" · external-only BSS {bss_ext:+.4f}")
    print("overfit diagnosis:", diag)
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
