"""P0.6 — MECHANICS-RESIDUAL ORACLE. The core question (override §33):
after fixing the target (exact floor_strike) and the label (official outcome),
does independent information improve the mechanics fair value MECH_FAIR,
especially MID-WINDOW where Kalshi's residual advantage lives?

p_oracle = sigmoid( logit(p_mech) + Δ(X_independent) ),  no Kalshi input.
Scored by BSS vs MECH_FAIR (not vs Kalshi), window-split, window-weighted,
shrinkage path, stratified by time-to-expiry AND by mechanics difficulty
(frozen causal |z|). Official labels from contract_outcomes.jsonl.
"""
import calendar
import json
import math
import time as _t
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
CO = ROOT / "results" / "contract_outcomes.jsonl"
OUT = ROOT / "research" / "oracle" / "oracle_residual_result.json"
SIG = 8e-5                      # MECH_FAIR calibrated sigma (mech_fair.py)
BASIS = -3.67
FEATS = ["cb_ofi_30s", "cb_l1_imb", "cb_micro_dev", "cb_ret_30s",
         "cb_rvol_30s", "trade_n_30s"]


def Phi(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
Phiv = np.vectorize(Phi)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def brier(p, y, w): return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def main():
    out = {json.loads(l)["ticker"]: json.loads(l) for l in CO.open() if l.strip()}
    rows = []
    for l in DS.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        o = out.get(r["ticker"])
        if not o or r.get("cb_price") is None:
            continue
        try:
            ct = calendar.timegm(_t.strptime(o["close_time"], "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            continue
        tte = max(1.0, ct - r["ts"])
        level = r["cb_price"] - BASIS
        z = (level - o["floor_strike"]) / (SIG * level * math.sqrt(tte))
        rows.append({"ticker": r["ticker"], "ts": r["ts"], "y": float(o["exact_yes"]),
                     "k_prob": r["k_prob"], "p_mech": Phi(z), "z": z,
                     "tte_min": tte / 60.0,
                     **{f: r[f] for f in FEATS}})
    if len(rows) < 400:
        print("insufficient:", len(rows)); return
    # frozen difficulty bins by |z| (small |z| = near boundary = HARD)
    az = np.array([abs(r["z"]) for r in rows])
    q33, q66 = np.quantile(az, [0.33, 0.66])
    for r in rows:
        a = abs(r["z"]); r["diff"] = "HARD" if a <= q33 else "MED" if a <= q66 else "EASY"

    first = {}
    for r in rows:
        first[r["ticker"]] = min(first.get(r["ticker"], 1e18), r["ts"])
    order = sorted(first, key=lambda t: first[t]); n = len(order)
    trw = set(order[:int(.6 * n)]); vaw = set(order[int(.6 * n):int(.8 * n)])
    tew = set(order[int(.8 * n):])
    perwin = {}
    for r in rows:
        perwin[r["ticker"]] = perwin.get(r["ticker"], 0) + 1

    def sp(S):
        rs = [r for r in rows if r["ticker"] in S]
        X = np.array([[r[f] for f in FEATS] for r in rs], float)
        y = np.array([r["y"] for r in rs]); w = np.array([1.0 / perwin[r["ticker"]] for r in rs])
        pm = np.array([r["p_mech"] for r in rs]); pk = np.array([r["k_prob"] for r in rs])
        tte = np.array([r["tte_min"] for r in rs]); diff = [r["diff"] for r in rs]
        return rs, X, y, w, pm, pk, tte, diff
    _, Xtr, ytr, wtr, pmtr, _, _, _ = sp(trw)
    _, Xva, yva, wva, pmva, _, _, _ = sp(vaw)
    rte, Xte, yte, wte, pmte, pkte, ttete, diffte = sp(tew)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd

    def fit(lam, it=700, lr=0.3):
        b = np.zeros(Xtr.shape[1]); W = wtr / wtr.sum()
        for _ in range(it):
            p = sigmoid(logit(pmtr) + Xtr @ b)
            b -= lr * (Xtr.T @ (W * (p - ytr)) + lam * b)
        return b

    bmech_te = brier(pmte, yte, wte)
    path = []; best = None
    for lam in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        b = fit(lam)
        pv = sigmoid(logit(pmva) + Xva @ b)
        bss_v = 1 - brier(pv, yva, wva) / brier(pmva, yva, wva)
        pt = sigmoid(logit(pmte) + Xte @ b)
        bss_t = 1 - brier(pt, yte, wte) / bmech_te
        ptr = sigmoid(logit(pmtr) + Xtr @ b)
        bss_tr = 1 - brier(ptr, ytr, wtr) / brier(pmtr, ytr, wtr)
        rec = {"lambda": lam, "bss_val_vs_mech": round(bss_v, 4),
               "bss_test_vs_mech": round(bss_t, 4), "bss_train_vs_mech": round(bss_tr, 4)}
        path.append(rec)
        if best is None or bss_v > best["bss_val_vs_mech"]:
            best = rec; bb = b
    pte = sigmoid(logit(pmte) + Xte @ bb)

    # stratify improvement over mechanics
    def strat(mask_fn, keys):
        d = {}
        for k in keys:
            m = np.array([mask_fn(i, k) for i in range(len(yte))])
            if m.sum() < 25:
                continue
            bm = brier(pmte[m], yte[m], wte[m])
            d[k] = {"n": int(m.sum()),
                    "mech_brier": round(bm, 4),
                    "oracle_bss_vs_mech": round(1 - brier(pte[m], yte[m], wte[m]) / bm, 4),
                    "kalshi_brier": round(brier(pkte[m], yte[m], wte[m]), 4)}
        return d
    tbin = [(0, 2, "T-2..0"), (2, 5, "T-5..2"), (5, 9, "T-9..5"), (9, 20, "T-15..9")]
    by_time = strat(lambda i, k: k[0] <= ttete[i] < k[1] if isinstance(k, tuple) else False,
                    [(a, b) for a, b, _ in tbin])
    by_time = {nm: strat(lambda i, kk=(a, b): kk[0] <= ttete[i] < kk[1], [(a, b)]).get((a, b))
               for a, b, nm in tbin}
    by_time = {k: v for k, v in by_time.items() if v}
    by_diff = strat(lambda i, k: diffte[i] == k, ["HARD", "MED", "EASY"])

    diag = ("PROMISING" if best["bss_test_vs_mech"] >= 0.005 and path[0]["bss_train_vs_mech"] > 0
            else "CLASSIC_OVERFIT" if path[0]["bss_train_vs_mech"] > 0.02 and best["bss_test_vs_mech"] < 0.003
            else "INFORMATION_LIMITED")
    doc = {"n_test": len(rte), "windows": n, "mech_brier_test": round(bmech_te, 4),
           "kalshi_brier_test": round(brier(pkte, yte, wte), 4),
           "lambda_path": path, "best": best,
           "by_time_to_expiry": by_time, "by_difficulty": by_diff,
           "diagnosis": diag,
           "note": "Δ_independent scored vs MECH_FAIR (not Kalshi); official labels; "
                   "no Kalshi input; difficulty = frozen |z| (HARD=near boundary)."}
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"Mechanics-residual Oracle — n_test {len(rte)}, mech Brier {bmech_te:.4f}, kalshi {doc['kalshi_brier_test']}")
    print("λ path (BSS of oracle vs MECH_FAIR):")
    for r in path:
        print(f"  λ={r['lambda']:<6} val {r['bss_val_vs_mech']:+.4f}  test {r['bss_test_vs_mech']:+.4f}  train {r['bss_train_vs_mech']:+.4f}")
    print("by time (oracle BSS vs mech · mech Brier · kalshi Brier):")
    for k, v in by_time.items():
        print(f"  {k:9s} n={v['n']:4d}  bss {v['oracle_bss_vs_mech']:+.4f}  mech {v['mech_brier']}  kalshi {v['kalshi_brier']}")
    print("by difficulty:", {k: v["oracle_bss_vs_mech"] for k, v in by_diff.items()})
    print("DIAGNOSIS:", diag)


if __name__ == "__main__":
    main()
