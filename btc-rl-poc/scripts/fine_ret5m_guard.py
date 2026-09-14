"""Guard (NOT a semantic change): prove brti_fine.ret_5m's full-slice return spans
genuinely ~300s and is not a variable-length window depending on first-sample
availability. Reports the elapsed-seconds distribution (t0 - first_in_slice_ts)
across the 365 fine windows. If tightly centered on ~300s, done.
"""
import json
import statistics
import sys
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

INV = ROOT / "research" / "true15m" / "contract_inventory.jsonl"
SAMP = ROOT / "results" / "brti_decision_dataset.jsonl"
FEAT = ROOT / "research" / "true15m" / "fine_features.jsonl"
OUT = ROOT / "research" / "true15m" / "fine_ret5m_elapsed_guard.json"
LOOKBACK = 300


def build():
    ts = sorted({float(r["brti_sample_ts"]) for r in
                 (json.loads(l) for l in SAMP.open() if l.strip())
                 if r.get("brti_sample_ts")})
    fine_ids = {json.loads(l)["market_window_id"] for l in FEAT.open() if l.strip()}
    t0_by = {w["market_window_id"]: w["T0"]
             for w in (json.loads(l) for l in INV.open() if l.strip())}
    elapsed = []
    for wid in fine_ids:
        t0 = t0_by[wid]
        lo = bisect_left(ts, t0 - LOOKBACK); hi = bisect_right(ts, t0)
        if hi > lo:
            elapsed.append(t0 - ts[lo])          # t0 - first in-slice sample
    elapsed.sort()
    def q(p):
        return round(elapsed[min(len(elapsed) - 1, int(p * len(elapsed)))], 1)
    dist = {"n": len(elapsed), "min": round(elapsed[0], 1), "p05": q(0.05),
            "median": round(statistics.median(elapsed), 1), "p95": q(0.95),
            "max": round(elapsed[-1], 1)}
    tight = elapsed[0] >= 260 and elapsed[-1] <= 300.5      # within ~16s cadence of 300
    doc = {"schema_version": "fine-ret5m-guard-1", "generated_at": time.time(),
           "target_seconds": LOOKBACK, "elapsed_seconds": dist,
           "tight_around_300": bool(tight),
           "note": "ret_5m spans t0 to the first sample within [t0-300s, t0]; at ~16s "
                   "cadence the earliest sample sits ~284-300s before t0."}
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("INTEGRITY_CHECK", "brti_fine.ret_5m elapsed-time guard", lane="L7",
            fact=f"elapsed s: min {dist['min']}, p05 {dist['p05']}, median {dist['median']}, "
                 f"p95 {dist['p95']}, max {dist['max']} (n={dist['n']}).",
            interpretation="Full-slice return spans ~300s tightly; not a variable-length "
                           "window." if tight else "Elapsed spread wider than expected — inspect.",
            files=[str(OUT.relative_to(ROOT))])
    print(f"fine_ret5m_guard: elapsed s min={dist['min']} median={dist['median']} "
          f"max={dist['max']} tight_around_300={tight}")


if __name__ == "__main__":
    build()
