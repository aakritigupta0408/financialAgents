"""Hindsight-optimal entry mining (offline, read-only).

For every RESOLVED 15-min window, replay the logged per-minute quotes and
find when the profit-maximizing valid entry existed: the winning side is
known after settlement, so the optimal entry is the minute that side was
cheapest while still valid (<85c, market accepts at the ask). These
hindsight optima become POSITIVE training samples for the selector (and
timing evidence for the bidder). The selector still only ever chooses
among the bidder's actual bets — hindsight rows are used purely as
training signal, never as decisions.

Prices: per-minute rows log mkt_p_up (mid). Ask ~ mid + half-spread; we
add a conservative +2c slippage so "optimal" is not flattered.

Usage: python3 tests/hindsight_entries.py
Writes results/hindsight_entries.json and prints a summary.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SLIP_C = 2.0          # mid -> ask conservatism, cents
MAX_PRICE_C = 85.0    # broker validity: entries only under 85c


def fee_c(price_c: float) -> float:
    p = price_c / 100.0
    return float(math.ceil(7.0 * p * (1.0 - p)))


def main() -> None:
    rows = [json.loads(l) for l in
            (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
    by_win = defaultdict(list)
    for r in rows:
        if (r.get("variant", "kb") == "kb" and r.get("mkt_p_up") is not None
                and r.get("actual") is not None):
            by_win[r["ticker"]].append(r)

    out, timing, mkt_agreed = [], [], 0
    for ticker, rs in by_win.items():
        rs.sort(key=lambda r: -r["mins_left"])
        actual = rs[0]["actual"]
        win_side = "yes" if actual else "no"
        best = None
        for r in rs:
            p_mid = r["mkt_p_up"] if win_side == "yes" else 1 - r["mkt_p_up"]
            price = 100.0 * p_mid + SLIP_C
            if not 1.0 <= price < MAX_PRICE_C:
                continue
            net = 100.0 - price - fee_c(price)
            if best is None or net > best["net_c"]:
                best = {"ticker": ticker, "side": win_side,
                        "price_c": round(price, 1),
                        "net_c": round(net, 1),
                        "mins_left": r["mins_left"],
                        "p_model": r["p_up"],
                        "model_agreed": int((r["p_up"] >= 0.5) == bool(actual)),
                        "close_ts": r["close_ts"]}
        if best:
            out.append(best)
            timing.append(best["mins_left"])
            mkt_agreed += int(best["price_c"] - SLIP_C > 50)

    (ROOT / "results" / "hindsight_entries.json").write_text(
        json.dumps({"n_windows": len(out), "slip_c": SLIP_C,
                    "entries": out}))

    n = len(out)
    timing.sort()
    med = timing[n // 2] if n else 0
    early = sum(1 for t in timing if t > 10) / n * 100 if n else 0
    late = sum(1 for t in timing if t <= 5) / n * 100 if n else 0
    agreed = sum(e["model_agreed"] for e in out) / n * 100 if n else 0
    avg_net = sum(e["net_c"] for e in out) / n if n else 0
    print(f"windows with a valid hindsight-profitable entry: {n} "
          f"of {len(by_win)}")
    print(f"optimal entry timing: median {med:.1f} min left, "
          f"{early:.0f}% strike >10 min out, {late:.0f}% in final 5 min")
    print(f"model already agreed with winning side at the optimal "
          f"minute: {agreed:.0f}%")
    print(f"market already favored winner (price>50c) at optimum: "
          f"{mkt_agreed / n * 100:.0f}%")
    print(f"avg hindsight net profit: {avg_net:.1f}c per contract "
          f"(+{SLIP_C:.0f}c slippage, fees in)")


if __name__ == "__main__":
    main()
