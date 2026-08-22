"""Entry-timing policy from hindsight hard positives (optimal stopping).

For every historical minute, the side a live bidder would lean toward is
proxied by the market favorite. Label = "this minute was a near-optimal
entry": the favored side's TRUE ask is within EPS_C of the best ask still
available from now to close (labels use the future — that's allowed for
training targets; features are strictly current-state).

Train a logistic scorer, evaluate chronologically: simulated policy =
enter the favored side the first minute score >= theta (forced at T-3 if
never), vs baselines (enter immediately / always T-3 / hindsight oracle).

Usage: python3 tests/entry_policy.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EPS_C = 3.0
MAX_C = 80.0


def fee_c(price_c: float) -> float:
    p = price_c / 100.0
    return float(math.ceil(7.0 * p * (1.0 - p)))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def feats(r: dict, prev3: dict | None) -> list[float]:
    fav_yes = r["price_c"] >= 50
    ask = r["yes_ask_c"] if fav_yes else 100 - r["yes_bid_c"]
    spread = max(0.0, r["yes_ask_c"] - r["yes_bid_c"])
    qd = (r["price_c"] - prev3["price_c"]) / 100 if prev3 else 0.0
    s = 1.0 if fav_yes else -1.0
    pf = r.get("pf") or [0.0, 0.0, 0.0]
    return [1.0, r["mins_left"] / 15.0, ask / 100.0, spread / 100.0,
            abs(r["price_c"] - 50) / 50.0, s * qd * 5.0,
            s * pf[0], pf[1], s * pf[2] / 200.0]


def side_ask(r: dict, yes: bool) -> float:
    return r["yes_ask_c"] if yes else 100 - r["yes_bid_c"]


def main() -> None:
    rows = [json.loads(l) for l in
            (ROOT / "results" / "kalshi_history.jsonl").open()]
    wins = defaultdict(list)
    for r in rows:
        wins[r["ticker"]].append(r)
    for v in wins.values():
        v.sort(key=lambda r: r["ts"])

    samples = []   # (x, label, ticker, ts)
    for tk, rs in wins.items():
        for i, r in enumerate(rs):
            fav_yes = r["price_c"] >= 50
            ask_now = side_ask(r, fav_yes)
            if not 1.0 <= ask_now < MAX_C:
                continue
            best_left = min(side_ask(q, fav_yes) for q in rs[i:])
            label = int(ask_now - best_left <= EPS_C)
            prev3 = rs[i - 3] if i >= 3 else None
            samples.append((feats(r, prev3), label, tk, r["ts"],
                            r["mins_left"]))

    samples.sort(key=lambda s: s[3])
    cut = samples[int(len(samples) * 0.7)][3]
    tr = [s for s in samples if s[3] <= cut]
    te_tk = {s[2] for s in samples if s[3] > cut}
    print(f"{len(samples)} minute-samples, {len(wins)} windows, "
          f"train {len(tr)}, test windows {len(te_tk)}")

    d = len(tr[0][0])
    w = [0.0] * d
    for ep in range(30):
        lr = 0.3 / (1 + 0.2 * ep)
        for x, y, *_ in tr:
            g = (sigmoid(sum(a * b for a, b in zip(w, x))) - y) * lr
            for j in range(d):
                w[j] -= g * x[j]

    def replay(policy) -> tuple[float, float, int]:
        """Return (avg net_c/window, avg entry price, n) on held-out."""
        tot = tot_px = n = 0
        for tk in te_tk:
            rs = wins[tk]
            pick = None
            for i, r in enumerate(rs):
                fav_yes = r["price_c"] >= 50
                ask = side_ask(r, fav_yes)
                if not 1.0 <= ask < MAX_C:
                    continue
                if policy(r, rs, i):
                    pick = (r, fav_yes, ask)
                    break
            if pick is None:   # forced late entry
                for r in rs:
                    if r["mins_left"] <= 3.2:
                        fav_yes = r["price_c"] >= 50
                        ask = side_ask(r, fav_yes)
                        if 1.0 <= ask < MAX_C:
                            pick = (r, fav_yes, ask)
                            break
            if pick is None:
                continue
            r, fav_yes, ask = pick
            won = fav_yes == bool(r["outcome"])
            net = (100 - ask - fee_c(ask)) if won else -(ask + fee_c(ask))
            tot += net
            tot_px += ask
            n += 1
        return (tot / n if n else 0, tot_px / n if n else 0, n)

    def model_pol(theta):
        def p(r, rs, i):
            prev3 = rs[i - 3] if i >= 3 else None
            return sigmoid(sum(a * b for a, b in
                               zip(w, feats(r, prev3)))) >= theta
        return p

    print(f"{'policy':22s} {'net/window':>10s} {'avg entry':>9s} {'n':>4s}")
    for name, pol in (
            ("enter immediately", lambda r, rs, i: True),
            ("always T-3", lambda r, rs, i: r["mins_left"] <= 3.2),
            ("timing model .5", model_pol(0.5)),
            ("timing model .6", model_pol(0.6)),
            ("timing model .7", model_pol(0.7))):
        net, px, n = replay(pol)
        print(f"{name:22s} {net:+9.1f}c {px:8.1f}c {n:4d}")
    # hindsight oracle (upper bound): winning side at its cheapest ask
    tot = n = 0
    for tk in te_tk:
        rs = wins[tk]
        yes = bool(rs[0]["outcome"])
        asks = [side_ask(r, yes) for r in rs
                if 1.0 <= side_ask(r, yes) < MAX_C]
        if asks:
            a = min(asks)
            tot += 100 - a - fee_c(a)
            n += 1
    print(f"{'hindsight oracle':22s} {tot/max(n,1):+9.1f}c {'':>9s} {n:4d}")
    print("weights:", [round(v, 3) for v in w])


if __name__ == "__main__":
    main()
