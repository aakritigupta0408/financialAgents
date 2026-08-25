"""Why is kb3's rolling accuracy declining? Decompose: model vs market on
identical calls, by recency and by window phase, plus market regime
(whipsaw) over the same span."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]

k3 = [r for r in kb if r.get("variant") == "kb3" and r.get("hit") is not None
      and r.get("trained") is not None]
k3.sort(key=lambda r: r["trained"])
n = len(k3)
half = n // 2


def acc(rows):
    return sum(r["hit"] for r in rows) / len(rows) if rows else float("nan")


def mkt_acc(rows):
    rs = [r for r in rows if r.get("mkt_p_up") is not None]
    return (sum(((r["mkt_p_up"] >= 0.5) == bool(r["actual"])) for r in rs)
            / len(rs) if rs else float("nan"))


for name, rows in (("first half", k3[:half]), ("second half", k3[half:]),
                   ("last 30", k3[-30:])):
    early = [r for r in rows if r["mins_left"] > 10]
    mid = [r for r in rows if 5 <= r["mins_left"] <= 10]
    late = [r for r in rows if r["mins_left"] < 5]
    print(f"{name:11s} n={len(rows):3d}  kb3 {acc(rows):.2f}  "
          f"mkt {mkt_acc(rows):.2f}  | kb3 by phase "
          f"E {acc(early):.2f}({len(early)}) M {acc(mid):.2f}({len(mid)}) "
          f"L {acc(late):.2f}({len(late)})")

# regime: outcome flip rate between consecutive windows (whipsaw proxy)
wins = {}
for r in kb:
    if r.get("variant", "kb") == "kb" and r.get("actual") is not None:
        wins[r["close_ts"]] = r["actual"]
ordered = [wins[t] for t in sorted(wins)]
flips_all = sum(1 for a, b in zip(ordered, ordered[1:]) if a != b)
recent = ordered[-12:]
flips_recent = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
print(f"window outcome flip rate: all {flips_all}/{len(ordered)-1} "
      f"({flips_all/(len(ordered)-1):.0%}) | last 12 windows "
      f"{flips_recent}/11 ({flips_recent/11:.0%})")

# confidence calibration drift: mean |p-0.5| first vs second half
for name, rows in (("first half", k3[:half]), ("second half", k3[half:])):
    conf = sum(abs(r["p_up"] - 0.5) for r in rows) / len(rows)
    bri = sum(r.get("brier", 0) for r in rows) / len(rows)
    print(f"{name}: mean confidence {conf:.3f}, mean brier {bri:.3f}")
