"""Selector retarget backtest: can we reach 93% bet-profit precision
(the RA benchmark) and at what coverage — evaluated chronologically.

Training = the daemon's own _sel_training_set (real bets + counterfactual
per-minute entries) built ONLY from the first 70% of windows, with
profit-weighted positives (hindsight signal: winning entries weighted by
realized net profit, so cheap winners teach more). Threshold scanned on
train for precision targets {0.80, 0.93}; evaluated on the LAST 30%:
real bets, plus bidder-like counterfactual entries (edge>=3c or late)
to widen the tiny eval set. Selector still only selects from the
bidder's bets in production — this is training/eval signal only.

Usage: python3 tests/selector93_backtest.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from btc_rl.online import (_sel_features, _sel_predict,        # noqa: E402
                           _sel_training_set, _train_sel_model)


def fee_c(price_c: float) -> float:
    p = price_c / 100.0
    return float(math.ceil(7.0 * p * (1.0 - p)))


def profit_weight(samples):
    """Hindsight augmentation: reweight winning samples by realized net
    profit (cheap winners carry the most signal); losers keep weight."""
    out = []
    for x, y, wt, price_c, is_real in samples:
        if y:
            wt = wt * (0.5 + (100.0 - price_c - fee_c(price_c)) / 50.0)
        out.append((x, y, wt, price_c, is_real))
    return out


def scan_theta(w, samples, target):
    scored = [(_sel_predict(w, x), y, pc) for x, y, _, pc, _ in samples]
    for t in range(50, 96):
        th = t / 100.0
        kept = [(y, pc) for p, y, pc in scored
                if p >= max(th, (pc + fee_c(pc)) / 100.0)]
        if len(kept) < 12:
            return None
        if sum(y for y, _ in kept) / len(kept) >= target:
            return th
    return None


def evaluate(w, th, samples):
    scored = [(_sel_predict(w, x), y, pc) for x, y, _, pc, _ in samples]
    kept = [(y, pc) for p, y, pc in scored
            if p >= max(th, (pc + fee_c(pc)) / 100.0)]
    if not kept:
        return {"n": len(samples), "kept": 0}
    prec = sum(y for y, _ in kept) / len(kept)
    pnl = sum((100 - pc - fee_c(pc)) if y else -(pc + fee_c(pc))
              for y, pc in kept)
    return {"n": len(samples), "kept": len(kept),
            "coverage": round(len(kept) / len(samples), 3),
            "precision": round(prec, 3), "profit_c": round(pnl, 1)}


def main() -> None:
    kb = [json.loads(l) for l in
          (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
    bets = [json.loads(l) for l in
            (ROOT / "results" / "kb_bets.jsonl").open()]
    closes = sorted({r["close_ts"] for r in kb
                     if r.get("actual") is not None})
    cut = closes[int(len(closes) * 0.7)]

    kb_tr = [r for r in kb if r["close_ts"] <= cut]
    kb_te = [r for r in kb if r["close_ts"] > cut]
    bets_tr = [b for b in bets if b["close_ts"] <= cut]
    bets_te = [b for b in bets if b["close_ts"] > cut
               and b.get("win") is not None]

    base = _sel_training_set(kb_tr, bets_tr)
    test_real = _sel_training_set([], bets_te)
    # bidder-like counterfactual entries in the held-out span widen eval
    test_cf = [(x, y, wt, pc, ir) for x, y, wt, pc, ir
               in _sel_training_set(kb_te, [])
               if x[3] >= 0.03 or x[4] <= 0.35]  # edge>=3c or <=5.2min left

    # rich variant: 9 features — the 7 bet features plus kb3's (online
    # logit, 12 engineered inputs) prob of the taken side at that same
    # minute and its confidence. Same timestamp, no future info.
    k3 = {(r["ticker"], round(r["mins_left"])): r["p_up"]
          for r in kb if r.get("variant") == "kb3"}

    def rich_set(kb_rows, bet_rows):
        out = []
        for b in bet_rows:
            if b.get("win") is None:
                continue
            x = _sel_features(b["side"], b["price_c"], b["p_model"],
                              b["mins_left"], bool(b.get("forced")))
            p3 = k3.get((b["ticker"], round(b["mins_left"])))
            p3s = (p3 if b["side"] == "yes" else 1 - p3) \
                if p3 is not None else 0.5
            out.append((x + [p3s, abs(p3s - 0.5) * 2], b["win"], 3.0,
                        b["price_c"], True))
        for r in kb_rows:
            if (r.get("variant", "kb") != "kb" or r.get("hit") is None
                    or r.get("mkt_p_up") is None):
                continue
            side = "yes" if r["call"] else "no"
            pc = 100.0 * (r["mkt_p_up"] if side == "yes"
                          else 1.0 - r["mkt_p_up"])
            if not 1.0 <= pc <= 99.0:
                continue
            x = _sel_features(side, pc, r["p_up"], r["mins_left"], False)
            p3 = k3.get((r["ticker"], round(r["mins_left"])))
            p3s = (p3 if side == "yes" else 1 - p3) \
                if p3 is not None else 0.5
            out.append((x + [p3s, abs(p3s - 0.5) * 2], int(r["hit"]), 1.0,
                        pc, False))
        return out

    def train_nd(samples, epochs=40):
        d = len(samples[0][0])
        w = [0.0] * d
        for ep in range(epochs):
            lr = 0.2 / (1.0 + 0.15 * ep)
            for x, y, wt, _, _ in samples:
                g = (_sel_predict(w, x) - y) * wt * lr
                for i in range(d):
                    w[i] -= g * x[i]
        return w

    rich_tr = rich_set(kb_tr, bets_tr)
    rich_te_real = rich_set([], bets_te)
    rich_te_cf = [s for s in rich_set(kb_te, [])
                  if s[0][3] >= 0.03 or s[0][4] <= 0.35]
    wr = train_nd(rich_tr)
    for target in (0.80, 0.93):
        th = scan_theta(wr, rich_tr, target)
        if th is None:
            print(f"[rich9] target {target:.0%}: unreachable on train")
            continue
        r1 = evaluate(wr, th, rich_te_real)
        r2 = evaluate(wr, th, rich_te_cf)
        print(f"[rich9] target {target:.0%} -> theta {th:.2f} | "
              f"held-out REAL bets: {r1} | bidder-like cf: {r2}")

    print(f"train samples {len(base)} (cut at 70% of "
          f"{len(closes)} settled windows)")
    for name, train in (("plain", base), ("profit-weighted",
                                          profit_weight(base))):
        w = _train_sel_model(train)
        for target in (0.80, 0.93):
            th = scan_theta(w, train, target)
            if th is None:
                print(f"[{name}] target {target:.0%}: unreachable on train")
                continue
            r1 = evaluate(w, th, test_real)
            r2 = evaluate(w, th, test_cf)
            print(f"[{name}] target {target:.0%} -> theta {th:.2f} | "
                  f"held-out REAL bets: {r1} | bidder-like cf: {r2}")


if __name__ == "__main__":
    main()
