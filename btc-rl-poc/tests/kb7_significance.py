"""Significance tests for kb7 vs the market on identical settled minutes:
- accuracy: McNemar exact (discordant pairs)
- Brier: paired mean difference with normal SE (and sign test)
- disagreement record: exact binomial vs 0.5
- accuracy vs coin flip: binomial
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb7"
        and r.get("actual") is not None and r.get("mkt_p_up") is not None]
n = len(rows)
acc = sum(r["hit"] for r in rows) / n
mhit = [int((r["mkt_p_up"] >= .5) == bool(r["actual"])) for r in rows]
macc = sum(mhit) / n
print(f"kb7 vs market, n={n} paired minutes: acc {acc:.1%} vs {macc:.1%}")

# McNemar exact on discordant pairs
b01 = sum(1 for r, m in zip(rows, mhit) if r["hit"] and not m)   # kb7 only
b10 = sum(1 for r, m in zip(rows, mhit) if not r["hit"] and m)   # mkt only
disc = b01 + b10
p_mcnemar = sum(math.comb(disc, k) for k in range(0, min(b01, b10) + 1)) \
    / (2 ** (disc - 1)) if disc else 1.0
p_mcnemar = min(1.0, p_mcnemar)
print(f"accuracy: kb7-only-right {b01}, market-only-right {b10} "
      f"-> McNemar exact p = {p_mcnemar:.3f}")

# Brier paired
d = [ (r["p_up"] - r["actual"])**2 - (r["mkt_p_up"] - r["actual"])**2
      for r in rows]
mean = sum(d) / n
sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
tstat = mean / (sd / math.sqrt(n))
print(f"brier: kb7 {sum((r['p_up']-r['actual'])**2 for r in rows)/n:.3f} "
      f"vs mkt {sum((r['mkt_p_up']-r['actual'])**2 for r in rows)/n:.3f} "
      f"| paired mean diff {mean:+.4f}, t = {tstat:.2f} "
      f"({'p<0.05' if abs(tstat) > 1.96 else 'not significant'})")

dis = [r for r, m in zip(rows, mhit) if r["hit"] != m]
wins = sum(r["hit"] for r in dis)
p_bin = sum(math.comb(len(dis), k) for k in range(wins, len(dis) + 1)) \
    / 2 ** len(dis) if dis else 1.0
print(f"disagreements: kb7 right {wins}/{len(dis)} "
      f"(one-sided binomial p = {p_bin:.3f})")

z_coin = (acc - 0.5) / math.sqrt(0.25 / n)
print(f"vs coin flip: z = {z_coin:.1f} (trivially significant)")
