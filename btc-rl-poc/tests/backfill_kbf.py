"""Backfill kbf (the T-3 definitive window call) from already-logged kb
calls — deterministic derivation from live commits, so it is the
backtest materialized, not synthetic data. Rows are flagged
backfilled=true for full transparency. Idempotent."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "kalshi_binary_log.jsonl"
rows = [json.loads(l) for l in LOG.read_text().splitlines()]

have = {r["ticker"] for r in rows if r.get("variant") == "kbf"}
byw = defaultdict(list)
for r in rows:
    if r.get("variant", "kb") == "kb" and r["actual"] is not None:
        byw[r["ticker"]].append(r)

added = 0
out = list(rows)
for ticker, calls in byw.items():
    if ticker in have:
        continue
    calls.sort(key=lambda r: r["made_ts"])
    ok = [r for r in calls if r["mins_left"] >= 3]
    if not ok:
        continue
    src = ok[-1]
    out.append({
        "variant": "kbf", "ticker": ticker, "made_ts": src["made_ts"],
        "close_ts": src["close_ts"], "strike": src["strike"],
        "base": src["base"], "mins_left": src["mins_left"],
        "p_up": src["p_up"], "call": src["call"],
        "mkt_p_up": src.get("mkt_p_up"), "actual": src["actual"],
        "hit": src["hit"], "brier": src["brier"],
        "decide_at": src["mins_left"], "backfilled": True,
        **({"mkt_brier": src["mkt_brier"]} if "mkt_brier" in src else {}),
    })
    added += 1

out.sort(key=lambda r: r["made_ts"])
tmp = LOG.with_suffix(".tmp")
tmp.write_text("".join(json.dumps(r) + "\n" for r in out))
tmp.replace(LOG)
kbf = [r for r in out if r.get("variant") == "kbf" and r["actual"] is not None]
hits = sum(r["hit"] for r in kbf)
print(f"backfilled {added} kbf windows; total kbf settled {len(kbf)}, "
      f"accuracy {100*hits/len(kbf):.1f}%")
