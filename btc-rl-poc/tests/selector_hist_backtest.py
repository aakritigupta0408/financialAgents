"""Does 14 days of TRUE-quote history improve the selector? Chronological
backtest: train on (real bets + live counterfactuals + historical
simulated mandatory bets), evaluate on the SAME held-out sets as
tests/selector93_backtest.py so results are comparable.

Historical sim bets: favored side at entry minutes ~{12, 8, 5, 3} when its
true ask < 80c — spans the live bidder's entry distribution. p_model is
proxied by the market price (flagged: the live feature uses our model's
prob). Historical pf is mapped to the live pf shape with a $150~sigma
drift proxy and 3-min quote drift from the contract price.

Usage: python3 tests/selector_hist_backtest.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from btc_rl.online import (_sel_features, _sel_kb3_feats,      # noqa: E402
                           _sel_path_feats, _sel_predict,
                           _sel_training_set, _train_sel_model)


def fee_c(pc: float) -> float:
    p = pc / 100.0
    return float(math.ceil(7.0 * p * (1.0 - p)))


def hist_sim_bets(rows: list[dict]) -> list[tuple]:
    wins = defaultdict(list)
    for r in rows:
        wins[r["ticker"]].append(r)
    out = []
    for rs in wins.values():
        rs.sort(key=lambda r: r["ts"])
        picked = set()
        for target in (12.0, 8.0, 5.0, 3.0):
            r = min(rs, key=lambda q: abs(q["mins_left"] - target))
            if abs(r["mins_left"] - target) > 1.5 or id(r) in picked:
                continue
            picked.add(id(r))
            i = rs.index(r)
            prev3 = rs[i - 3] if i >= 3 else None
            qd = (r["price_c"] - prev3["price_c"]) / 100 if prev3 else 0.0
            pf_raw = r.get("pf") or [0.0, 0.0, 0.0]
            pf_live = [pf_raw[0], pf_raw[1],
                       max(-4.0, min(4.0, pf_raw[2] / 150.0)), qd]
            # BOTH sides, mirroring the live bidder which bets underdogs
            # on model edge as often as favorites
            for side, ask in (("yes", r["yes_ask_c"]),
                              ("no", 100 - r["yes_bid_c"])):
                if not 1.0 <= ask < 80.0:
                    continue
                win = int((side == "yes") == bool(r["outcome"]))
                x = _sel_features(side, ask, r["price_c"] / 100,
                                  r["mins_left"], False)
                x += _sel_kb3_feats({}, r["ticker"], r["mins_left"], side)
                x += _sel_path_feats(pf_live, side)
                out.append((x, win, 1.0, ask, False, r["ts"]))
    return out


def evaluate(w, th, samples):
    kept = []
    for x, y, _, pc, *_ in samples:
        if _sel_predict(w, x) >= max(th, (pc + fee_c(pc)) / 100.0):
            kept.append((y, pc))
    if not kept:
        return {"kept": 0}
    prec = sum(y for y, _ in kept) / len(kept)
    pnl = sum((100 - pc - fee_c(pc)) if y else -(pc + fee_c(pc))
              for y, pc in kept)
    return {"n": len(samples), "kept": len(kept),
            "coverage": round(len(kept) / len(samples), 3),
            "precision": round(prec, 3), "profit_c": round(pnl, 1)}


def scan(w, samples, target):
    scored = [(_sel_predict(w, x), y, pc) for x, y, _, pc, *_ in samples]
    for t in range(50, 96):
        th = t / 100.0
        kept = [(y, pc) for p, y, pc in scored
                if p >= max(th, (pc + fee_c(pc)) / 100.0)]
        if len(kept) < 15:
            return None
        if sum(y for y, _ in kept) / len(kept) >= target:
            return th
    return None


def main() -> None:
    kb = [json.loads(l) for l in
          (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
    bets = [json.loads(l) for l in
            (ROOT / "results" / "kb_bets.jsonl").open()]
    hist = [json.loads(l) for l in
            (ROOT / "results" / "kalshi_history.jsonl").open()]

    closes = sorted({r["close_ts"] for r in kb
                     if r.get("actual") is not None})
    cut = closes[int(len(closes) * 0.7)]
    live_tr = _sel_training_set([r for r in kb if r["close_ts"] <= cut],
                                [b for b in bets if b["close_ts"] <= cut])
    te_real = _sel_training_set([], [b for b in bets if b["close_ts"] > cut
                                     and b.get("win") is not None])
    te_cf = [s for s in _sel_training_set(
        [r for r in kb if r["close_ts"] > cut], [])
        if s[0][3] >= 0.03 or s[0][4] <= 0.35]

    sim = hist_sim_bets(hist)
    sim_tr = [s for s in sim if s[5] <= cut]      # no look-ahead into eval
    print(f"live train {len(live_tr)} | hist sim bets {len(sim_tr)} "
          f"(of {len(sim)}) | eval: {len(te_real)} real, {len(te_cf)} cf")

    for name, train in (("live-only", live_tr),
                        ("live + history", live_tr
                         + [s[:5] for s in sim_tr])):
        w = _train_sel_model(train)
        for target in (0.90, 0.93):
            th = scan(w, train, target)
            if th is None:
                print(f"[{name}] target {target:.0%}: unreachable on train")
                continue
            print(f"[{name}] target {target:.0%} theta {th:.2f} | real "
                  f"{evaluate(w, th, te_real)} | cf {evaluate(w, th, te_cf)}")


if __name__ == "__main__":
    main()
