"""PURE CANDLE-HISTORY model — the clean, leak-free core question:
given the full history of completed 15-min BTC candles (their opens/closes) AND the current
candle's known OPEN, predict the current candle's CLOSE direction (close >= open).

No intra-window microstructure, no cross-venue (Coinbase-leads-BRTI) latency — ONLY past
completed candles + the current open. This isolates "how predictable is the next 15-min
return sign from candle history alone." 6,337 candles from results/contract_outcomes.jsonl
(open=floor_strike, close=expiration_value, label=exact_yes). Walk-forward 60/40 + label-
shuffle leakage canary + hit-rate at coverage>=0.90 (target 89%@90%).
"""
import json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "contract_outcomes.jsonl"
OUT = ROOT / "research" / "candle_history_model.json"


def _ep(t):
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def candles():
    C = []
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        o = d.get("floor_strike"); c = d.get("expiration_value"); y = d.get("exact_yes")
        ot = _ep(d.get("open_time"))
        if o and c and y in (0, 1) and ot:
            C.append({"open": float(o), "close": float(c), "y": int(y), "ot": ot})
    C.sort(key=lambda r: r["ot"])
    return C


def build(C, K=32):
    """Features at candle N from candles[..N-1] + open_N. Strictly past-only."""
    closes = np.array([c["close"] for c in C], float)
    opens = np.array([c["open"] for c in C], float)
    rets = np.diff(np.log(closes))                      # close-to-close log returns
    X, y, ts = [], [], []
    for i in range(K, len(C)):
        openN = opens[i]
        past_c = closes[i - K:i]                         # last K closes (all < N)
        past_r = np.diff(np.log(past_c))
        cl = past_c[-1]                                  # last completed close
        def mom(k): return math.log(cl / past_c[-k]) * 1e4 if k < len(past_c) and past_c[-k] > 0 else 0.0
        up = past_r[past_r > 0].sum(); dn = -past_r[past_r < 0].sum()
        rsi = up / (up + dn) if (up + dn) else 0.5
        # streak of up/down candles
        signs = np.sign([c2["close"] - c2["open"] for c2 in C[i - K:i]])
        streak = 0
        for s in signs[::-1]:
            if s == signs[-1] and s != 0: streak += 1
            else: break
        streak *= (signs[-1] if len(signs) else 0)
        gap = math.log(openN / cl) * 1e4 if cl > 0 else 0.0      # open_N vs last close
        vol = past_r.std() * 1e4 if len(past_r) > 1 else 0.0
        e = datetime.fromtimestamp(C[i]["ot"], tz=timezone.utc)
        f = [gap, mom(1), mom(3), mom(6), mom(12), mom(24), rsi, float(streak), vol,
             past_r[-1] * 1e4, past_r[-3:].mean() * 1e4, (openN - past_c.mean()) / past_c.mean() * 1e4,
             math.sin(2 * math.pi * e.hour / 24), math.cos(2 * math.pi * e.hour / 24), e.weekday()]
        X.append(f); y.append(C[i]["y"]); ts.append(C[i]["ot"])
    return np.nan_to_num(np.array(X, float)), np.array(y, int)


def _sel(p, y):
    conf = np.abs(p - 0.5) * 2; pred = (p >= 0.5).astype(int)
    best90 = None
    for c in np.linspace(0, 0.95, 20):
        m = conf >= c
        if m.mean() >= 0.90 and m.any():
            hit = (pred[m] == y[m]).mean()
            if best90 is None or hit > best90[1]:
                best90 = (round(float(c), 3), round(float(hit), 4), round(float(m.mean()), 4))
    return best90


def run():
    C = candles()
    X, y = build(C)
    n = len(y); mid = int(n * 0.6); yte = y[mid:]
    rep = {"n_candles": len(C), "n_samples": n, "base_up": round(float(y.mean()), 4)}
    # GBM
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=500,
                                        l2_regularization=2.0, min_samples_leaf=30, random_state=17)
    gb.fit(X[:mid], y[:mid]); pg = gb.predict_proba(X[mid:])[:, 1]
    rep["gbm"] = {"oos_acc": round(float(((pg >= 0.5).astype(int) == yte).mean()), 4),
                  "in_sample_acc": round(float((gb.predict(X[:mid]) == y[:mid]).mean()), 4),
                  "best_hit@cov>=.90": _sel(pg, yte)}
    # logistic
    sc = StandardScaler().fit(X[:mid])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[:mid]), y[:mid])
    pl = lr.predict_proba(sc.transform(X[mid:]))[:, 1]
    rep["logistic"] = {"oos_acc": round(float(((pl >= 0.5).astype(int) == yte).mean()), 4),
                       "best_hit@cov>=.90": _sel(pl, yte)}
    # leakage canary
    rng = np.random.default_rng(0); ysh = y.copy(); ysh[:mid] = rng.permutation(ysh[:mid])
    gbc = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=500,
                                         l2_regularization=2.0, min_samples_leaf=30, random_state=17)
    gbc.fit(X[:mid], ysh[:mid]); pc = gbc.predict_proba(X[mid:])[:, 1]
    rep["leakage_canary_oos"] = round(float(((pc >= 0.5).astype(int) == yte).mean()), 4)
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    run()
