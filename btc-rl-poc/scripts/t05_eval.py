"""T0.5 evaluation — does microstructure predict Kalshi REPRICING?

For each feature x and horizon H, the information coefficient IC = Spearman
rank-correlation(x_at_t, dprob_{t->t+H}), with a moving-block bootstrap 95%
CI (blocks preserve the strong within-window autocorrelation, so the CI is
honest about effective sample size). Also a walk-forward logistic AUC for
the sign of the +60s reprice from all features.

Interpretation guardrail (handoff §30, §61-A): cb_ret_30s is MECHANICALLY
coupled to Kalshi repricing (Kalshi prob is ~a function of BTC price), so a
nonzero IC there is expected and is NOT tradeable information. The
informative test is whether ORDER FLOW (cb_ofi_30s, cb_l1_imb, cb_micro_dev)
predicts the future reprice — i.e. whether Kalshi LAGS order flow.
"""
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
OUT = ROOT / "research" / "t05_repricing" / "t05_result.json"
FEATS = ["cb_ofi_30s", "cb_l1_imb", "cb_micro_dev", "cb_ret_30s",
         "cb_rvol_30s", "trade_n_30s", "k_spread"]
MECHANICAL = {"cb_ret_30s"}     # coupled to price, not information
HORIZONS = [5, 15, 30, 60]
BLOCK = 40                       # ~ one window's worth of 15s samples
B = 500
random.seed(20260912)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    saa = sum((a[i] - ma) ** 2 for i in range(n))
    sbb = sum((b[i] - mb) ** 2 for i in range(n))
    return sab / math.sqrt(saa * sbb) if saa > 0 and sbb > 0 else 0.0


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def block_ci(x, y):
    """Moving-block bootstrap CI on Spearman IC."""
    n = len(x)
    nb = max(1, n // BLOCK)
    starts = list(range(0, n - BLOCK + 1)) or [0]
    ics = []
    for _ in range(B):
        idx = []
        for _ in range(nb):
            s = random.choice(starts)
            idx.extend(range(s, min(n, s + BLOCK)))
        ics.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    ics.sort()
    return ics[int(0.025 * B)], ics[int(0.975 * B)]


def logistic_auc_walkforward(rows, H):
    """Sign(dprob_H) from standardized features; chronological 70/30 split."""
    key = "dprob_%d" % H
    data = [r for r in rows if r.get(key) is not None and abs(r[key]) > 1e-9]
    if len(data) < 200:
        return None
    cut = int(len(data) * 0.7)
    tr, te = data[:cut], data[cut:]
    mu = {f: sum(r[f] for r in tr) / len(tr) for f in FEATS}
    sd = {f: (sum((r[f] - mu[f]) ** 2 for r in tr) / len(tr)) ** 0.5 or 1.0 for f in FEATS}
    def vec(r):
        return [ (r[f] - mu[f]) / sd[f] for f in FEATS ]
    w = [0.0] * len(FEATS); b = 0.0; lr = 0.1
    for _ in range(300):
        for r in tr:
            x = vec(r); y = 1.0 if r[key] > 0 else 0.0
            z = b + sum(w[i] * x[i] for i in range(len(w)))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            g = p - y
            for i in range(len(w)):
                w[i] -= lr * (g * x[i] + 1e-4 * w[i])
            b -= lr * g
    # AUC on test
    pos = [ ]; neg = [ ]
    for r in te:
        x = vec(r); z = b + sum(w[i] * x[i] for i in range(len(w)))
        p = 1 / (1 + math.exp(-max(-30, min(30, z))))
        (pos if r[key] > 0 else neg).append(p)
    if not pos or not neg:
        return None
    wins = 0
    for pp in pos:
        for nn in neg:
            wins += 1 if pp > nn else 0.5 if pp == nn else 0
    return round(wins / (len(pos) * len(neg)), 4), len(te)


def main():
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    n = len(rows)
    out = {"n_points": n, "block_size": BLOCK, "horizons": {}}
    print(f"T0.5 evaluation — {n} points\n")
    any_flow_signal = False
    for H in HORIZONS:
        key = "dprob_%d" % H
        y = [r[key] for r in rows if r.get(key) is not None]
        rr = [r for r in rows if r.get(key) is not None]
        hres = {}
        print(f"H=+{H}s (n={len(y)})")
        for f in FEATS:
            x = [r[f] for r in rr]
            ic = spearman(x, y)
            lo, hi = block_ci(x, y)
            sig = (lo > 0) or (hi < 0)
            tag = "MECHANICAL" if f in MECHANICAL else ("SIGNAL" if sig else "flat")
            if sig and f not in MECHANICAL and abs(ic) >= 0.03:
                any_flow_signal = True
            hres[f] = {"IC": round(ic, 4), "ci95": [round(lo, 4), round(hi, 4)],
                       "significant": sig, "kind": tag}
            print(f"  {f:14s} IC {ic:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  {tag}")
        auc = logistic_auc_walkforward(rows, H)
        hres["logistic_auc_walkforward"] = auc[0] if auc else None
        hres["auc_test_n"] = auc[1] if auc else None
        if auc:
            print(f"  -> walk-forward logistic AUC (sign) = {auc[0]}  (n_test {auc[1]})")
        out["horizons"]["+%ds" % H] = hres
        print()
    out["verdict"] = ("ORDER_FLOW_PREDICTS_REPRICING" if any_flow_signal
                      else "NO_REPRICING_PREDICTABILITY_FROM_FLOW")
    out["note"] = ("cb_ret_30s coupling is mechanical (price->prob) and not "
                   "tradeable; the test is whether ORDER FLOW leads the reprice.")
    OUT.write_text(json.dumps(out, indent=1))
    print("VERDICT:", out["verdict"])
    print("written", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
