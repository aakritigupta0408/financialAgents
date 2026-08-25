"""Adverse-selection EV on biddable confident entries, per variant."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
TAUS = {"kb2": .64, "kb3": .83, "kb4": .78}
print(f"{'variant':8s} {'n':>4s} {'win':>6s} {'95% CI':>15s} "
      f"{'cost':>6s} {'EV/$1':>7s} {'EV hi':>7s}")
for v, tau in TAUS.items():
    ent = []
    for r in kb:
        if (r.get("variant") != v or r.get("actual") is None
                or r.get("mkt_p_up") is None or r["mins_left"] > 10):
            continue
        if max(r["p_up"], 1 - r["p_up"]) < tau:
            continue
        side = "yes" if r["call"] else "no"
        ask = 100 * (r["mkt_p_up"] if side == "yes"
                     else 1 - r["mkt_p_up"]) + 2.5
        if not 5 <= ask < 80:
            continue
        fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
        ent.append((int(r["hit"]), ask + fee))
    n = len(ent)
    if n < 40:
        print(f"{v:8s} {n:4d}  (too few)")
        continue
    w = sum(e[0] for e in ent)
    p = w / n
    cost = sum(e[1] for e in ent) / n
    z = 1.96
    den = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    plo = (p + z * z / (2 * n) - half) / den
    phi = (p + z * z / (2 * n) + half) / den
    ev = (p * 100 - cost) / cost
    evhi = (phi * 100 - cost) / cost
    print(f"{v:8s} {n:4d} {p:6.1%} [{plo:5.1%},{phi:5.1%}] "
          f"{cost:6.1f} {ev:+7.1%} {evhi:+7.1%}")
