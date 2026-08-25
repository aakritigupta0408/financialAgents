"""kb5 + Conviction Book status: activity, learning trajectory, EV test."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
k5 = [r for r in kb if r.get("variant") == "kb5"]
print("kb5 rows total:", len(k5))
if k5:
    r = k5[-1]
    print("latest age min:", round((time.time() - r["made_ts"]) / 60, 1),
          "| sample:", {k: r.get(k) for k in
                        ("call", "ask_c", "ev_c", "conf_entry", "trained")})
    settled = [x for x in k5 if x.get("actual") is not None]
    conf = [x for x in settled if x.get("conf_entry")]
    print("settled:", len(settled), "| confident entries:", len(conf),
          "| conf wins:", sum(x["hit"] for x in conf) if conf else 0)
    # learning over time: rolling accuracy + trained counter growth
    if len(settled) >= 40:
        h = len(settled) // 2
        a1 = sum(x["hit"] for x in settled[:h]) / h
        a2 = sum(x["hit"] for x in settled[h:]) / (len(settled) - h)
        print(f"accuracy first half {a1:.1%} -> second half {a2:.1%}")
try:
    pb = [json.loads(l) for l in (ROOT / "results" / "pb_bets.jsonl").open()]
    done = [b for b in pb if b.get("win") is not None]
    print("Conviction Book:", len(pb), "bets,",
          len(done), "settled,",
          sum(b["win"] for b in done), "wins,",
          f"net {sum(b['pnl_c'] for b in done):+.0f}c" if done else "")
    for b in pb[-5:]:
        print("  ", b["side"], f"@{b['price_c']}c",
              f"p={b['p_win']}", "->",
              "open" if b.get("win") is None
              else ("WON" if b["win"] else "lost"))
except FileNotFoundError:
    print("Conviction Book: no bets yet")
lg = json.loads((ROOT / "results" / "kb5_logit.json").read_text())
print("kb5 model updates:", lg["updates"])
