"""M14 — "The Oracle Call" (owner spec 2026-08-29): predict the
window's settlement 2-9 minutes IN, speaking ONLY when the claim can
honestly carry 80-90% certainty; otherwise say NO CALL.

Pre-registered rules (never fitted after the fact):
  * envelope: first decision row with 6 <= mins_left <= 13
    (= 2-9 minutes into the 15-minute window);
  * caller: kb2 (the pre-registered deliverable arm) — its claimed
    0.8-0.9 band verified 91% on the exploratory sample; kb5 logged
    as a shadow second caller for comparison, never merged;
  * call iff claimed confidence >= 0.80; else NO CALL (logged too —
    coverage is half the product);
  * prequential: rows are stamped pre-settle by the daemon; this
    emitter (10-min cron) only READS them, so every call is
    reconstructible and time-safe (leakage canaries guard the tape);
  * promise metric: selective accuracy of calls must be >= 0.80;
    report with Wilson 95% CI and coverage. The promise failing is a
    result, not a formatting problem.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CONF = 0.80
LO_MIN, HI_MIN = 6.0, 13.0
CALLERS = ("kb2", "kb5")


def wilson(k, n):
    if not n:
        return None
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    now = int(time.time())
    rows = [json.loads(l) for l in
            (RES / "kalshi_binary_log.jsonl").open() if l.strip()]
    # first in-envelope decision row per (ticker, variant)
    first = {}
    for r in sorted(rows, key=lambda r: -(r.get("mins_left") or 0)):
        if r.get("mins_left") is None or r.get("p_up") is None:
            continue
        if not (LO_MIN <= r["mins_left"] <= HI_MIN):
            continue
        v = r.get("variant") or "kb"
        if v in CALLERS:
            first.setdefault((r["ticker"], v), r)

    out = {"generated_ts": now, "config": {
        "conf_threshold": CONF, "envelope_mins_left": [LO_MIN, HI_MIN],
        "primary_caller": "kb2", "shadow_caller": "kb5",
        "promise": "selective accuracy of calls >= 0.80"}}
    for v in CALLERS:
        wins = [r for (tk, vv), r in first.items() if vv == v]
        settled = [r for r in wins if r.get("actual") is not None]
        calls = [r for r in settled
                 if max(r["p_up"], 1 - r["p_up"]) >= CONF]
        hits = sum(1 for r in calls
                   if (r["p_up"] >= 0.5) == bool(r["actual"]))
        sel_acc = hits / len(calls) if calls else None
        out[v] = {
            "windows_seen": len(wins), "settled": len(settled),
            "calls": len(calls),
            "coverage": round(len(calls) / len(settled), 3)
            if settled else None,
            "hits": hits,
            "selective_accuracy": round(sel_acc, 3)
            if sel_acc is not None else None,
            "wilson_ci95": wilson(hits, len(calls)),
            "promise_met": (sel_acc >= 0.80) if sel_acc is not None
            else None,
            "no_calls": len(settled) - len(calls),
        }
    # live pending call, if any (most recent unsettled window)
    pend = [r for (tk, vv), r in first.items()
            if vv == "kb2" and r.get("actual") is None]
    if pend:
        r = max(pend, key=lambda r: r.get("close_ts") or 0)
        c = max(r["p_up"], 1 - r["p_up"])
        out["live"] = {"ticker": r["ticker"],
                       "call": ("UP" if r["p_up"] >= 0.5 else "DOWN")
                       if c >= CONF else "NO CALL",
                       "claimed_conf": round(c, 3),
                       "mins_left_at_read": r["mins_left"]}
    (RES / "oracle_calls.json").write_text(json.dumps(out, indent=1))
    k2 = out.get("kb2", {})
    print(f"oracle_calls: kb2 {k2.get('calls')} calls / "
          f"{k2.get('settled')} settled · sel_acc "
          f"{k2.get('selective_accuracy')} CI {k2.get('wilson_ci95')} "
          f"· coverage {k2.get('coverage')} · promise_met "
          f"{k2.get('promise_met')}")


if __name__ == "__main__":
    main()
