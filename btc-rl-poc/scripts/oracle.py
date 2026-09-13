"""HISTORICAL / SUPERSEDED (see architecture/dangling_threads.json DT-03).
Kept as evidence. The CURRENT canonical Oracle uses EXACT BRTI as contract truth
(scripts/mech_fair_brti.py, scripts/oracle_residual_brti.py, scripts/freeze_oracle.py).
This module models P(final Coinbase settlement > strike) — a Gen-4/5 assumption:
Coinbase spot is NOT the settlement benchmark (BRTI is), so its contract state is
a proxy. Do not treat its Coinbase-as-mechanic semantics as current.

INDEPENDENT ORACLE (handoff O1/O2/O5) — an independent estimate of
P(final Coinbase settlement > strike) using BTC-only information (never
Kalshi price), evaluated by EARLINESS, with fixed-confidence lock curves.

Strike / expiry / settlement come from contract metadata + Kalshi's own
expiry resolution (allowed: the strike defines the event; the settlement
is the label). Kalshi's probability (k_prob) is used ONLY as the market
benchmark for comparison — never as an Oracle input.

Models:
  MARKET      : p = k_prob (Kalshi mid at t)                [benchmark]
  ORACLE_phys : p = Phi( (price-strike)/(sigma*sqrt(tte)) ), sigma
                calibrated on train — pure BTC distance/vol, independent
  ORACLE_learn: logistic on [z, rvol, ret, ofi, l1_imb, tte] -> y, no Kalshi
Reports Brier/accuracy/calibration by time-to-expiry, Oracle vs market,
and fixed-confidence lock curves (coverage x accuracy x earliness).
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
KLOG = ROOT / "results" / "kalshi_binary_log.jsonl"
OUT = ROOT / "research" / "oracle" / "oracle_result.json"


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


Phiv = np.vectorize(Phi)


def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def strike_map():
    m = {}
    for l in KLOG.open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        tk = r.get("ticker")
        if tk and r.get("strike") is not None and r.get("close_ts") is not None:
            m[tk] = (r["strike"], r["close_ts"], r.get("actual"))
    return m


def fit_logistic(X, y, lam=1e-2, it=800, lr=0.3):
    b = np.zeros(X.shape[1]); bias = 0.0
    for _ in range(it):
        p = sigmoid(bias + X @ b)
        g = X.T @ (p - y) / len(y) + lam * b
        b -= lr * g; bias -= lr * (p - y).mean()
    return b, bias


def brier(p, y): return float(np.mean((p - y) ** 2))


def calib(p, y, bins=((0.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01))):
    # fold to confidence of the *called* side
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = np.where(p >= 0.5, y, 1 - y)
    out = []
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 20:
            out.append({"bin": f"{int(lo*100)}-{int(hi*100)}", "n": int(m.sum()),
                        "expected": round(float(conf[m].mean()), 3),
                        "observed": round(float(correct[m].mean()), 3)})
    return out


def main():
    sm = strike_map()
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    data = []
    for r in rows:
        s = sm.get(r["ticker"])
        if not s or s[2] is None or r.get("cb_price") is None:
            continue
        strike, close_ts, actual = s
        tte = max(1.0, close_ts - r["ts"])
        dist = (r["cb_price"] - strike) / r["cb_price"]
        data.append({"ticker": r["ticker"], "ts": r["ts"], "y": float(actual),
                     "k_prob": r["k_prob"], "dist": dist, "tte_s": tte,
                     "tte_min": tte / 60.0, "rvol": r["cb_rvol_30s"],
                     "ret": r["cb_ret_30s"], "ofi": r["cb_ofi_30s"],
                     "l1": r["cb_l1_imb"]})
    if len(data) < 500:
        print("insufficient oracle rows:", len(data)); return
    # window split
    first = {}
    for d in data:
        first[d["ticker"]] = min(first.get(d["ticker"], 1e18), d["ts"])
    order = sorted(first, key=lambda t: first[t]); n = len(order)
    trw = set(order[:int(.7 * n)]); tew = set(order[int(.7 * n):])
    tr = [d for d in data if d["ticker"] in trw]
    te = [d for d in data if d["ticker"] in tew]

    # calibrate physics sigma on train
    def phys(d, sig):
        return Phi(d["dist"] / (sig * math.sqrt(d["tte_s"])))
    ytr = np.array([d["y"] for d in tr])
    best_sig, best_b = None, 1e9
    for sig in [1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1.2e-4, 2e-4, 4e-4]:
        p = np.array([phys(d, sig) for d in tr])
        bb = brier(p, ytr)
        if bb < best_b:
            best_b, best_sig = bb, sig

    # learned oracle (BTC-only)
    FEATS = ["z", "rvol", "ret", "ofi", "l1", "tte_min"]
    def vec(d):
        z = d["dist"] / (best_sig * math.sqrt(d["tte_s"]))
        return [z, d["rvol"], d["ret"], d["ofi"], d["l1"], d["tte_min"]]
    Xtr = np.array([vec(d) for d in tr]); Xte = np.array([vec(d) for d in te])
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    b, bias = fit_logistic((Xtr - mu) / sd, ytr)

    yte = np.array([d["y"] for d in te])
    p_mkt = np.array([d["k_prob"] for d in te])
    p_phys = np.array([phys(d, best_sig) for d in te])
    p_learn = sigmoid(bias + ((Xte - mu) / sd) @ b)
    tte = np.array([d["tte_min"] for d in te])

    def acc(p): return float(np.mean((p >= 0.5) == (yte >= 0.5)))
    summary = {}
    for name, p in [("MARKET", p_mkt), ("ORACLE_phys", p_phys), ("ORACLE_learn", p_learn)]:
        summary[name] = {"brier": round(brier(p, yte), 4), "accuracy": round(acc(p), 4)}

    # by time-to-expiry
    bins = [(0, 2, "T-2..0"), (2, 4, "T-4..2"), (4, 6, "T-6..4"),
            (6, 9, "T-9..6"), (9, 20, "T-15..9")]
    by_time = {}
    for lo, hi, nm in bins:
        m = (tte >= lo) & (tte < hi)
        if m.sum() < 30:
            continue
        by_time[nm] = {"n": int(m.sum()),
                       "market_brier": round(brier(p_mkt[m], yte[m]), 4),
                       "oracle_phys_brier": round(brier(p_phys[m], yte[m]), 4),
                       "oracle_learn_brier": round(brier(p_learn[m], yte[m]), 4),
                       "oracle_ahead_of_market": bool(brier(p_learn[m], yte[m]) < brier(p_mkt[m], yte[m]))}

    # LOCK CURVES (fixed confidence) on ORACLE_learn, per test window
    win = {}
    for d, p in zip(te, p_learn):
        win.setdefault(d["ticker"], []).append((d["tte_min"], p, d["y"]))
    locks = {}
    for c in [0.60, 0.70, 0.80, 0.90]:
        locked = 0; correct = 0; lock_times = []
        for tk, pts in win.items():
            pts.sort(key=lambda x: -x[0])            # earliest (largest tte) first
            call = None
            for tt, p, y in pts:
                if p >= c or p <= 1 - c:
                    call = (1 if p >= 0.5 else 0, tt, y); break
            if call:
                locked += 1
                if call[0] == call[2]:
                    correct += 1
                lock_times.append(call[1])
        nw = len(win)
        locks["c=%.2f" % c] = {"coverage": round(locked / nw, 3),
                               "accuracy": round(correct / locked, 3) if locked else None,
                               "mean_lock_tte_min": round(float(np.mean(lock_times)), 2) if lock_times else None,
                               "no_call_rate": round(1 - locked / nw, 3),
                               "locked_windows": locked, "total_windows": nw}

    verdict = ("ORACLE_MATCHES_MARKET" if abs(summary["ORACLE_learn"]["brier"] - summary["MARKET"]["brier"]) < 0.003
               else "ORACLE_BEATS_MARKET" if summary["ORACLE_learn"]["brier"] < summary["MARKET"]["brier"]
               else "ORACLE_BEHIND_MARKET")
    out = {"n_windows": n, "windows_train": len(trw), "windows_test": len(tew),
           "test_points": len(te), "phys_sigma_per_sqrt_s": best_sig,
           "base_rate": round(float(yte.mean()), 3),
           "summary": summary, "by_time_to_expiry": by_time,
           "lock_curves": locks, "calibration_oracle_learn": calib(p_learn, yte),
           "verdict": verdict,
           "note": "Independent Oracle uses BTC-only features + strike/tte; "
                   "Kalshi used only as benchmark. Settlement label from "
                   "contract resolution. Development sample."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"Oracle — {n} windows (test {len(tew)}), {len(te)} test points, base rate {yte.mean():.3f}")
    print(f"phys sigma {best_sig}")
    for k, v in summary.items():
        print(f"  {k:14s} Brier {v['brier']}  acc {v['accuracy']}")
    print("by time-to-expiry (Brier: market / oracle_learn):")
    for nm, v in by_time.items():
        print(f"  {nm:9s} n={v['n']:5d}  mkt {v['market_brier']:.4f}  oracle {v['oracle_learn_brier']:.4f}"
              f"  {'ORACLE AHEAD' if v['oracle_ahead_of_market'] else ''}")
    print("lock curves (fixed confidence):")
    for k, v in locks.items():
        print(f"  {k}  coverage {v['coverage']}  accuracy {v['accuracy']}"
              f"  mean_lock_tte {v['mean_lock_tte_min']}min  no_call {v['no_call_rate']}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
