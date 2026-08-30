"""Emit results/f1_capture_qualification.json — Gate F1 (PM 08-30).

Machine-computed daily from the raw tapes; only PASS unlocks feature
evaluation, and "eligible" means allowed-to-evaluate, never
ready-for-model-inclusion. Verdict is EXTEND_CAPTURE until BOTH
duration (>=7d) and per-venue quality thresholds hold.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

CAPTURE_START = 1_788_090_000        # 2026-08-30 F-XVENUE go-live
MIN_DAYS = 7
MIN_COVERAGE = 0.95
# expected inter-event gaps (s) used to estimate coverage: a venue
# gap far beyond this on an active market counts as a capture gap
EXPECTED_GAP_S = {"binance": 30, "okx": 30, "kraken": 120}


def rows_from(dirname):
    out = []
    for sh in sorted(glob.glob(str(RES / dirname / "*.jsonl"))):
        for l in open(sh):
            if l.strip():
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out


def main():
    now = time.time()
    days = (now - CAPTURE_START) / 86400
    xr = rows_from("events_xvenue")
    venues = {}
    for v, exp_gap in EXPECTED_GAP_S.items():
        ts = sorted(r["ts_recv"] for r in xr if r.get("src") == v)
        if len(ts) < 2:
            venues[v] = {"events": len(ts), "state": "INSUFFICIENT"}
            continue
        span = ts[-1] - ts[0]
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b - a > exp_gap]
        gap_time = sum(gaps)
        coverage = 1 - gap_time / span if span > 0 else 0
        bad_order = sum(1 for a, b in zip(ts, ts[1:]) if b < a)
        venues[v] = {"events": len(ts),
                     "span_h": round(span / 3600, 1),
                     "coverage": round(coverage, 4),
                     "gaps_over_expected": len(gaps),
                     "gap_time_s": round(gap_time),
                     "out_of_order": bad_order,
                     "state": "OK" if coverage >= MIN_COVERAGE
                     and bad_order == 0 else "BELOW_GATE"}
    # synchronized tape: fraction of minute-states with all venues valid
    sync_rows = []
    p = RES / "xvenue_state.jsonl"
    if p.exists():
        for l in p.read_text().splitlines():
            try:
                sync_rows.append(json.loads(l))
            except Exception:
                pass
    all_valid = sum(1 for r in sync_rows
                    if all(v.get("valid") for v in
                           (r.get("venues") or {}).values()))
    combined = {"decision_states": len(sync_rows),
                "states_all_venues_valid": all_valid,
                "valid_fraction": round(all_valid / len(sync_rows), 4)
                if sync_rows else None}
    duration_ok = days >= MIN_DAYS
    quality_ok = venues and all(v.get("state") == "OK"
                                for v in venues.values())
    verdict = "PASS" if duration_ok and quality_ok else \
        "EXTEND_CAPTURE"
    doc = {"generated_ts": int(now),
           "capture_start": CAPTURE_START,
           "days_captured": round(days, 2),
           "gate": {"min_days": MIN_DAYS,
                    "min_coverage": MIN_COVERAGE,
                    "earliest_eligible": "2026-09-06"},
           "venues": venues, "combined_tape": combined,
           "verdict": verdict,
           "meaning": "PASS = allowed to EVALUATE (never = ready for "
                      "model inclusion)"}
    (RES / "f1_capture_qualification.json").write_text(
        json.dumps(doc, indent=1))
    print(f"f1_gate: {verdict} · {days:.2f}d · " +
          " ".join(f"{v}:{d.get('coverage', '—')}"
                   for v, d in venues.items()))


if __name__ == "__main__":
    main()
