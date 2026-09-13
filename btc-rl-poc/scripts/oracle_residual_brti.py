"""§15-23 — INDEPENDENT RESIDUAL ORACLE on EXACT-BRTI mechanics.

  p_mech   = MECH_FAIR_BRTI (exact contract state; settlement-aware)
  delta(X) = f(independent info)          # X = single-venue Coinbase microstructure
  p_oracle = sigmoid( logit(p_mech) + delta(X) )

The question (§15): what independent information should move us away from the
mechanically justified probability — especially MID-WINDOW (T-9..2), where the
exact-BRTI gap to Kalshi survives?

Rigor: window split, window-weighted, official labels, difficulty = frozen causal
|z|. Baselines (§22): logistic residual (mandatory) with a shrinkage path, and a
histogram gradient-boosting tree (LightGBM-equivalent; lightgbm not installed) via
the ablation tree(logit_pmech) vs tree(logit_pmech + X). Deliberate-overfit check
(§23): train vs test BSS. Scored vs MECH_FAIR_BRTI (not Kalshi).

FAMILY A only (single-venue Coinbase microstructure). Multi-venue / derivatives /
options / news are prospective-capture-pending — reported as NOT-YET-TESTABLE.

Writes research/oracle/oracle_residual_brti_result.json.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "results" / "brti_decision_dataset.jsonl"
T05 = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
MFB = ROOT / "research" / "oracle" / "mech_fair_brti_result.json"
OUT = ROOT / "research" / "oracle" / "oracle_residual_brti_result.json"
FEATS = ["cb_ofi_30s", "cb_l1_imb", "cb_micro_dev", "cb_ret_30s",
         "cb_rvol_30s", "trade_n_30s"]


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
Phiv = np.vectorize(Phi)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def brier(p, y, w): return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def p_mech_row(r, sig):
    """MECH_FAIR_BRTI for one decision row (mirror of mech_fair_brti.py)."""
    lvl, tte = r["current_brti"], max(1.0, r["time_remaining_s"])
    p = Phi(r["brti_distance_to_target"] / (sig * lvl * math.sqrt(tte)))
    rem = r.get("settle_remaining") or 0
    req = r.get("required_remaining_average")
    if r["time_remaining_s"] <= 60 and rem > 0 and req is not None:
        s_rem = sig * lvl * math.sqrt(tte) / math.sqrt(rem)
        p = Phi((lvl - req) / max(1e-6, s_rem))
    return min(1 - 1e-4, max(1e-4, p))


def main():
    sig = json.load(MFB.open())["sigma_rel_per_sqrt_s"]
    # join microstructure features from t05 on (ticker, ts)
    feats = {}
    for l in T05.open():
        if not l.strip():
            continue
        r = json.loads(l)
        feats[(r["ticker"], r["ts"])] = r
    rows = []
    for l in DEC.open():
        if not l.strip():
            continue
        d = json.loads(l)
        f = feats.get((d["market_window_id"], d["decision_time"]))
        if not f or any(f.get(k) is None for k in FEATS):
            continue
        if d.get("current_brti") is None or d["time_remaining_s"] < 1:
            continue
        d["p_mech"] = p_mech_row(d, sig)
        d.update({k: f[k] for k in FEATS})
        rows.append(d)
    if len(rows) < 400:
        print("insufficient joined rows:", len(rows)); return

    first, perwin = {}, {}
    for r in rows:
        tk = r["market_window_id"]
        first[tk] = min(first.get(tk, 1e18), r["decision_time"])
        perwin[tk] = perwin.get(tk, 0) + 1
    order = sorted(first, key=lambda t: first[t]); n = len(order)
    trw = set(order[:int(.6 * n)]); vaw = set(order[int(.6 * n):int(.8 * n)])

    def split(S):
        rs = [r for r in rows if r["market_window_id"] in S]
        X = np.array([[r[f] for f in FEATS] for r in rs], float)
        y = np.array([r["exact_yes"] for r in rs], float)
        w = np.array([1.0 / perwin[r["market_window_id"]] for r in rs])
        pm = np.array([r["p_mech"] for r in rs])
        pk = np.array([r["k_prob"] for r in rs])
        tte = np.array([r["time_remaining_s"] / 60.0 for r in rs])
        z = np.abs(np.array([r["brti_distance_to_target"] for r in rs])) / (
            sig * np.array([r["current_brti"] for r in rs])
            * np.sqrt([r["time_remaining_s"] for r in rs]))
        return rs, X, y, w, pm, pk, tte, z

    _, Xtr, ytr, wtr, pmtr, _, _, _ = split(trw)
    _, Xva, yva, wva, pmva, _, _, _ = split(vaw)
    rte, Xte, yte, wte, pmte, pkte, ttete, zte = split(set(order[int(.8 * n):]))
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd

    # frozen difficulty on test |z|
    q33, q66 = np.quantile(zte, [0.33, 0.66])
    diff = np.where(zte <= q33, "HARD", np.where(zte <= q66, "MED", "EASY"))

    # ---- logistic residual with offset + shrinkage path (§22 linear, §27 path) ----
    def fit(lam, it=800, lr=0.3):
        b = np.zeros(Xtr.shape[1]); W = wtr / wtr.sum()
        for _ in range(it):
            p = sigmoid(logit(pmtr) + Xtr @ b)
            b -= lr * (Xtr.T @ (W * (p - ytr)) + lam * b)
        return b
    bmech_te = brier(pmte, yte, wte)
    path, best, bb = [], None, None
    for lam in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        b = fit(lam)
        pv = sigmoid(logit(pmva) + Xva @ b)
        pt = sigmoid(logit(pmte) + Xte @ b)
        ptr = sigmoid(logit(pmtr) + Xtr @ b)
        rec = {"lambda": lam,
               "bss_val": round(1 - brier(pv, yva, wva) / brier(pmva, yva, wva), 4),
               "bss_test": round(1 - brier(pt, yte, wte) / bmech_te, 4),
               "bss_train": round(1 - brier(ptr, ytr, wtr) / brier(pmtr, ytr, wtr), 4),
               "mean_abs_delta": round(float(np.mean(np.abs(Xte @ b))), 4)}
        path.append(rec)
        if best is None or rec["bss_val"] > best["bss_val"]:
            best, bb = rec, b
    p_lin = sigmoid(logit(pmte) + Xte @ bb)

    # ---- histogram GBT ablation: tree(logit_pmech) vs tree(logit_pmech + X) (§22) ----
    tree_res = {"available": False}
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        base_tr = logit(pmtr).reshape(-1, 1); base_te = logit(pmte).reshape(-1, 1)
        def tree_bss(Xtr_, Xte_):
            m = HGB(max_depth=3, max_iter=200, learning_rate=0.05,
                    l2_regularization=1.0, min_samples_leaf=60)
            m.fit(Xtr_, ytr, sample_weight=wtr)
            pt = np.clip(m.predict_proba(Xte_)[:, 1], 1e-4, 1 - 1e-4)
            return 1 - brier(pt, yte, wte) / bmech_te, pt
        bss_mech_only, _ = tree_bss(base_tr, base_te)
        bss_mech_x, p_tree = tree_bss(np.hstack([base_tr, Xtr]),
                                      np.hstack([base_te, Xte]))
        tree_res = {"available": True, "impl": "sklearn HistGradientBoosting (lightgbm N/A)",
                    "bss_test_mech_only": round(float(bss_mech_only), 4),
                    "bss_test_mech_plus_X": round(float(bss_mech_x), 4),
                    "incremental_bss_from_features": round(float(bss_mech_x - bss_mech_only), 4)}
    except Exception as e:
        tree_res = {"available": False, "error": str(e)[:120]}

    # ---- stratified incremental value of the linear residual (§21/§31) ----
    def strat(mask):
        m = mask & (wte > 0)
        if m.sum() < 25:
            return None
        bm = brier(pmte[m], yte[m], wte[m])
        return {"n": int(m.sum()),
                "mech_brier": round(bm, 4),
                "oracle_bss_vs_mech": round(1 - brier(p_lin[m], yte[m], wte[m]) / bm, 4),
                "kalshi_brier": round(brier(pkte[m], yte[m], wte[m]), 4)}
    by_time = {}
    for lo, hi, nm in [(0, 2, "T-2..0"), (2, 5, "T-5..2"), (5, 9, "T-9..5"), (9, 20, "T-15..9")]:
        v = strat((ttete >= lo) & (ttete < hi))
        if v:
            by_time[nm] = v
    mid = strat((ttete >= 2) & (ttete < 9))
    by_diff = {k: strat(diff == k) for k in ("HARD", "MED", "EASY")}
    by_diff = {k: v for k, v in by_diff.items() if v}

    # diagnosis (§23)
    tr0 = path[0]["bss_train"]; bt = best["bss_test"]
    diag = ("PROMISING" if bt >= 0.005 and tr0 > 0
            else "CLASSIC_OVERFIT" if tr0 > 0.02 and bt < 0.003
            else "INFORMATION_LIMITED" if tr0 <= 0.02
            else "OPTIMIZATION_FAILURE")

    doc = {
        "family": "A_single_venue_coinbase_microstructure",
        "features": FEATS,
        "n_test": len(rte), "windows": n,
        "mech_brier_test": round(bmech_te, 4),
        "kalshi_brier_test": round(brier(pkte, yte, wte), 4),
        "linear_residual": {"lambda_path": path, "best": best},
        "tree_baseline": tree_res,
        "by_time_to_expiry": by_time,
        "mid_window_T9to2": mid,
        "by_difficulty": by_diff,
        "diagnosis": diag,
        "not_yet_testable_families": {
            "B_cross_venue_lead_lag": "needs multi-venue capture (Binance/OKX/Kraken)",
            "C_derivatives": "adapter ready (OKX); prospective capture pending",
            "D_options": "AV adapter ready; prospective capture pending",
            "E_news": "AV adapter ready; prospective capture pending"},
        "note": "delta scored vs MECH_FAIR_BRTI; official labels; no Kalshi input. "
                "Family A is the only historically-testable independent source.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"Residual Oracle (Family A) — n_test {len(rte)}, mech Brier {bmech_te:.4f}, "
          f"kalshi {doc['kalshi_brier_test']}")
    print("λ path (BSS vs MECH_BRTI):")
    for r in path:
        print(f"  λ={r['lambda']:<6} val {r['bss_val']:+.4f} test {r['bss_test']:+.4f} "
              f"train {r['bss_train']:+.4f} |Δ|={r['mean_abs_delta']}")
    if tree_res.get("available"):
        print(f"tree: mech-only BSS {tree_res['bss_test_mech_only']:+.4f} -> "
              f"+features {tree_res['bss_test_mech_plus_X']:+.4f} "
              f"(incremental {tree_res['incremental_bss_from_features']:+.4f})")
    print("mid-window T-9..2 oracle BSS vs mech:", mid["oracle_bss_vs_mech"] if mid else None)
    print("by difficulty BSS:", {k: v["oracle_bss_vs_mech"] for k, v in by_diff.items()})
    print("DIAGNOSIS:", diag)


if __name__ == "__main__":
    main()
