"""Why is kb2 uncertain? Decompose its sub-gate (conf < 0.62) calls:
- phase (early/mid/late), strike distance, and — the key question —
  whether the MARKET is also uncertain on those same minutes.
- accuracy of sub-gate calls (is discarded info recoverable?)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
rows = [r for r in kb if r.get("variant") == "kb2"
        and r.get("actual") is not None and r.get("mkt_p_up") is not None]
TAU = 0.62
sub = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) < TAU]
gat = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= TAU]
print(f"kb2 settled quoted calls {len(rows)} | sub-gate {len(sub)} "
      f"({len(sub)/len(rows):.0%}) | gated {len(gat)}")

acc = lambda rs: sum(r["hit"] for r in rs) / len(rs) if rs else float("nan")
mconf = lambda r: abs(r["mkt_p_up"] - 0.5)
print(f"sub-gate accuracy {acc(sub):.1%} vs gated {acc(gat):.1%}")

mk_unc = [r for r in sub if mconf(r) < TAU - 0.5]
print(f"market ALSO uncertain (|mkt-.5| < .12) on sub-gate calls: "
      f"{len(mk_unc)}/{len(sub)} ({len(mk_unc)/len(sub):.0%})")
mk_conf = [r for r in sub if mconf(r) >= TAU - 0.5]
print(f"market CONFIDENT while kb2 wasn't: {len(mk_conf)} calls, "
      f"market acc there {sum(((r['mkt_p_up']>=.5)==bool(r['actual'])) for r in mk_conf)/max(1,len(mk_conf)):.1%}, "
      f"kb2 acc {acc(mk_conf):.1%}")

for name, f in (("early >10m", lambda m: m > 10),
                ("mid 5-10m", lambda m: 5 <= m <= 10),
                ("late <5m", lambda m: m < 5)):
    s = [r for r in sub if f(r["mins_left"])]
    a = [r for r in rows if f(r["mins_left"])]
    print(f"  {name:11s} sub-gate share {len(s)/max(1,len(a)):.0%} "
          f"of {len(a)} calls, sub-gate acc {acc(s):.1%}")

# strike distance among sub-gate calls (z from base/sigma proxy: use
# |base-strike| in dollars bucketed)
near = [r for r in sub if abs(r["base"] - r["strike"]) < 50]
print(f"sub-gate calls with price within $50 of strike: "
      f"{len(near)}/{len(sub)} ({len(near)/len(sub):.0%})")
