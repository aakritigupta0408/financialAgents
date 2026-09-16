"""TUNE FOR EV — find the value gate that maximizes net EV on T0's actual trades.

T0 loses at 69% hit because it buys favorites: it needs P(win)*100 > price + fee, which it
doesn't check. This: (1) calibrates T0's confidence p_arm -> true P(win) (isotonic, fit on the
first 60%), (2) computes per-trade EV_c = P(win)*100 - ask - fee, (3) sweeps an edge threshold
e and, on the held-out 40%, only keeps trades with EV_c >= e, reporting realized net $, EV/
trade, hit, coverage, and a trade-level Sharpe. Finds the EV-maximizing gate. All walk-forward:
the calibration + threshold generalize; no look-ahead. Uses ACTUAL prices T0 paid.
"""
import json, math
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "ev_optimize.json"


def _fee_c(price_c):
    # Kalshi trading fee ~ ceil(0.07 * C * p * (1-p)); per-contract at p=price/100
    p = price_c / 100.0
    return math.ceil(7 * p * (1 - p))


def load():
    rows = [json.loads(l) for l in (ROOT / "results" / "pt_trades.jsonl").open() if l.strip()]
    s = [r for r in rows if r.get("actual") is not None and r.get("ask_c") is not None
         and r.get("contracts") and r.get("p_arm") is not None]
    s.sort(key=lambda r: r.get("close_ts") or 0)
    return s


def run():
    s = load()
    n = len(s); mid = int(n * 0.6)
    p_arm = np.array([r["p_arm"] for r in s], float)          # T0 confidence for its chosen side
    win = np.array([1 if r.get("win") else 0 for r in s], int)
    ask = np.array([r["ask_c"] for r in s], float)
    pnl_c = np.array([r.get("pnl_c") or 0 for r in s], float)
    contracts = np.array([r["contracts"] for r in s], float)
    realized_pc = pnl_c / contracts                            # actual net cents PER CONTRACT

    # calibrate p_arm -> P(win) on TRAIN only
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_arm[:mid], win[:mid])
    p_win = iso.predict(p_arm)                                 # calibrated
    fee = np.array([_fee_c(a) for a in ask])
    ev_c = p_win * 100 - ask - fee                             # expected cents per contract

    tr = slice(mid, n)                                         # held-out test
    evt, rt = ev_c[tr], realized_pc[tr]
    test_pnl = pnl_c[tr]                                        # actual net cents per trade
    rep = {"n": n, "n_test": n - mid,
           "calibration": {"train_hit": round(float(win[:mid].mean()), 4),
                           "mean_p_arm": round(float(p_arm[:mid].mean()), 4),
                           "mean_calibrated_p_win": round(float(p_win[:mid].mean()), 4)},
           "always_trade_test": {"n": int((n - mid)), "hit": round(float(win[tr].mean()), 4),
                                 "net_usd": round(float(test_pnl.sum() / 100), 2),
                                 "ev_per_trade_c": round(float(test_pnl.sum() / (n - mid)), 1)},
           "gate_sweep": []}
    best = None
    for e in (-20, -10, -5, 0, 2, 5, 8, 12, 16, 20):
        m = evt >= e
        if m.sum() < 5:
            continue
        net = float(test_pnl[m].sum() / 100)
        k = int(m.sum())
        hit = float(win[tr][m].mean())
        ev_trade = float(test_pnl[m].sum() / k)
        sharpe = float(rt[m].mean() / (rt[m].std() + 1e-9))
        row = {"edge_thr_c": e, "coverage": round(k / (n - mid), 3), "n": k,
               "hit": round(hit, 4), "net_usd": round(net, 2),
               "ev_per_trade_c": round(ev_trade, 1), "trade_sharpe": round(sharpe, 3)}
        rep["gate_sweep"].append(row)
        if best is None or net > best["net_usd"]:
            best = row
    rep["ev_optimal_gate"] = best
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"n={n} test={n-mid}")
    print(f"calibration: mean p_arm {rep['calibration']['mean_p_arm']} -> calibrated p_win {rep['calibration']['mean_calibrated_p_win']} (train hit {rep['calibration']['train_hit']})")
    a = rep["always_trade_test"]
    print(f"ALWAYS-TRADE (T0, test): n={a['n']} hit {a['hit']} net ${a['net_usd']} EV {a['ev_per_trade_c']}c/trade")
    print("VALUE-GATE sweep (test):")
    for r in rep["gate_sweep"]:
        star = "  <== EV-max" if r is best else ""
        print(f"  edge>={r['edge_thr_c']:>3}c  cov {r['coverage']:.2f} (n={r['n']:3})  hit {r['hit']}  net ${r['net_usd']:>8.2f}  EV {r['ev_per_trade_c']:>6.1f}c  sharpe {r['trade_sharpe']}{star}")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    run()
