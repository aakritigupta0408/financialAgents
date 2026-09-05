"""Emit results/f1_capture_qualification.json — GATE F1 v2.

PROTOCOL AMENDMENT (PM ratified 2026-09-03, governance-logged):
v1 ("7 calendar days, coverage over the whole span") measured the
laptop, not the market — a single 19.4h sleep outage inside the span
permanently poisoned the coverage ratio. v2 is principled:

  F1 = a CONTIGUOUS CLEAN CAPTURE WINDOW satisfying pre-registered
  statistical-sufficiency + integrity criteria. Documented machine
  outages TRUNCATE the eligible window (it restarts after the last
  all-venue gap >15 min); they never poison the whole span.

Pre-registered thresholds (frozen BEFORE any T1.1 model result
exists — the anti-peeking law: model outcomes never feed back into
gate criteria; sufficiency is about identifiability, not whether
the model wins):
  1. PIT violations = 0 (whole capture, ts_event vs ts_recv)
  2. schema bad rows = 0; out-of-order rate < 0.1% per venue
  3. per-venue 5-min presence >= 95% inside the eligible window
  4. continuity: no in-window venue gap >= 60 min
  5. ESS >= 150 per horizon (h5/h15/h30, 15-min grid, ACF-adjusted)
  6. MDE80 at rho=0.95 <= 8% of baseline MAE per horizon
  7. >= 5 chronological folds of >= 48 usable windows each
  8. non-degenerate outcomes (up-fraction in [0.30, 0.70])
  9. temporal diversity: >= 12 six-hour blocks, both trend signs;
     no vol tercile > 60% of usable windows
 10. no unresolved capture data-quality incident; all hourly shards
     present inside the window (provenance)
NOTE: all-venue 5s feature availability is a T1.1 FEATURE-SPEC
concern (kraken legitimately trades sparsely; registry expects gaps
to 120s) — deliberately NOT a capture-gate criterion.

Computation is shared with scripts/audit_f1_sufficiency.py
(scan/analyze) — one implementation, no private formulas.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from audit_f1_sufficiency import (HORIZONS, SHARDS, VENUES, analyze,
                                  machine_outages, scan)

CAPTURE_START = 1_788_090_000        # 2026-08-30 F-XVENUE go-live
GATE_VERSION = "2.0"
MIN_COVERAGE = 0.95
MIN_ESS = 150
MAX_MDE80_PCT = 8.0
OUTAGE_TRUNCATES_MIN = 15


def open_capture_incidents():
    p = RES / "incidents.jsonl"
    if not p.exists():
        return []
    resolved_markers = ("resolved", "closed")
    rows = []
    for l in p.read_text().splitlines():
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    out = []
    for r in rows:
        title = str(r.get("title", "")).lower()
        status = str(r.get("status", "")).lower()
        if ("capture" in title or "xvenue" in title) and not any(
                m in status for m in resolved_markers):
            out.append(r.get("title"))
    return out


def shards_complete(m_lo, m_hi):
    """Every UTC hour inside the window must have its shard file."""
    missing = []
    h = (m_lo * 60) // 3600
    h_end = (m_hi * 60) // 3600
    while h <= h_end:
        name = time.strftime("xvenue-%Y%m%d-%H.jsonl",
                             time.gmtime(h * 3600))
        if not (SHARDS / name).exists():
            missing.append(name)
        h += 1
    return missing


def main():
    now = time.time()
    days_total = round((now - CAPTURE_START) / 86400, 2)
    per_min, sec_present, integ = scan()

    outages = machine_outages(per_min, OUTAGE_TRUNCATES_MIN)
    all_minutes = set()
    for v in VENUES:
        all_minutes |= set(per_min[v])
    m_hi = max(all_minutes)
    m_lo = min(all_minutes)
    if outages:
        m_lo = max(m_lo, outages[-1][1])   # restart after last outage
    window_days = round((m_hi - m_lo + 1) / 1440, 2)

    body = analyze(per_min, sec_present, integ, m_lo, m_hi)
    checks = body["checks"]
    power = body["power"]

    ess_min = min((power[f"h{h}"].get("ess") or 0) for h in HORIZONS)
    mde_max = max((power[f"h{h}"]
                   .get("min_detectable_improvement", {})
                   .get("rho_0.95", {})
                   .get("mde80_pct_of_mae") or 999)
                  for h in HORIZONS)
    open_inc = open_capture_incidents()
    missing_shards = shards_complete(m_lo, m_hi)

    criteria = {
        "pit_zero": checks["pit_zero"],
        "schema_clean": all(g["bad_rows"] == 0
                            for g in integ.values()),
        "timestamps_ordered": checks["timestamps_ordered"],
        "coverage_95_in_window": checks["coverage_95"],
        "continuity_in_window": checks["continuity"],
        "ess_min_150": ess_min >= MIN_ESS,
        "mde80_max_8pct": mde_max <= MAX_MDE80_PCT,
        "fold_viability": checks["fold_viability"],
        "outcome_nondegenerate": checks["outcome_nondegenerate"],
        "temporal_diversity": checks["temporal_diversity"],
        "regime_no_domination": checks["regime_no_domination"],
        "no_open_capture_incident": not open_inc,
        "provenance_complete": not missing_shards,
    }
    verdict = "PASS" if all(criteria.values()) else "EXTEND_CAPTURE"

    # rough projection: ESS grows ~linearly with clean window length
    projection = None
    if ess_min > 0 and ess_min < MIN_ESS:
        need = window_days * (MIN_ESS / ess_min) - window_days
        projection = (f"~{round(need, 1)} more CLEAN days for "
                      f"ESS>={MIN_ESS} (linear estimate; any new "
                      "machine outage re-truncates)")

    # ---- PROPOSED v2.1 shadow (stitched clean segments) -----------
    # The 24-outage discovery: this laptop dozes several times a
    # day, so a contiguous >=3d clean window is fragile-to-
    # impossible. v2.1 stitches clean segments: outage minutes
    # leave the denominators, and an observation is usable only if
    # its whole lookback+label path avoids outages. NOT GOVERNING
    # until the PM formally ratifies; emitted for that decision.
    sb = analyze(per_min, sec_present, integ, outages=outages)
    sp = sb["power"]
    s_ess = min((sp[f"h{h}"].get("ess") or 0) for h in HORIZONS)
    s_mde = max((sp[f"h{h}"].get("min_detectable_improvement", {})
                 .get("rho_0.95", {}).get("mde80_pct_of_mae") or 999)
                for h in HORIZONS)
    s_checks = sb["checks"]
    s_criteria = {
        "pit_zero": s_checks["pit_zero"],
        "schema_clean": criteria["schema_clean"],
        "timestamps_ordered": s_checks["timestamps_ordered"],
        "coverage_95_ex_outages": s_checks["coverage_95"],
        "continuity_ex_outages": s_checks["continuity"],
        "ess_min_150": s_ess >= MIN_ESS,
        "mde80_max_8pct": s_mde <= MAX_MDE80_PCT,
        "fold_viability": s_checks["fold_viability"],
        "outcome_nondegenerate": s_checks["outcome_nondegenerate"],
        "temporal_diversity": s_checks["temporal_diversity"],
        "regime_no_domination": s_checks["regime_no_domination"],
        "no_open_capture_incident": not open_inc,
        "provenance_complete": not missing_shards,
    }
    # RATIFICATION (PM 09-04, pre-authorized conditional): v2.1
    # becomes GOVERNING iff the v2.1.1 missingness audit ran after
    # its registered prospective time and PASSed. Verified from the
    # artifact, never assumed.
    v21_governing = False
    try:
        _miss = json.loads((RES / "f1_missingness_audit.json")
                           .read_text())
        from audit_missingness import RERUN_NOT_BEFORE_TS
        v21_governing = (_miss.get("verdict") == "PASS"
                         and _miss.get("generated_ts", 0)
                         >= RERUN_NOT_BEFORE_TS)
    except Exception:
        pass
    shadow_v2_1 = {
        "status": ("GOVERNING — PM pre-ratified 09-04 conditional "
                   "on v2.1.1 missingness PASS; condition verified "
                   "from f1_missingness_audit.json"
                   if v21_governing else
                   "PROPOSED_NOT_GOVERNING — awaiting v2.1.1 "
                   "prospective missingness PASS"),
        "rule": "stitched clean segments — documented outages leave "
                "denominators; usable obs = clean [m-60, m+30] path",
        "would_pass": all(s_criteria.values()),
        "criteria": s_criteria,
        "failing": [k for k, v in s_criteria.items() if not v],
        "ess_min_across_horizons": round(s_ess, 1),
        "mde80_worst_pct_of_mae": (round(s_mde, 1)
                                   if s_mde < 999 else None),
        "usable_days_ex_outages": sb["span_days"],
        "power": sp}

    if v21_governing:
        verdict = ("PASS" if all(s_criteria.values())
                   else "EXTEND_CAPTURE")
    doc = {"generated_ts": int(now),
           "gate": "F1",
           "gate_version": "2.1" if v21_governing else GATE_VERSION,
           "governing_rule": ("v2.1 stitched clean segments"
                              if v21_governing
                              else "v2 contiguous clean window"),
           "amendment": "PM 2026-09-03 — clean-window truncation + "
                        "pre-registered statistical sufficiency; "
                        "v1 span-coverage rule retired (rationale: "
                        "f1_sufficiency_audit sha in commit 1caa9a9)",
           "anti_peeking_law": "criteria frozen before any T1.1 "
                               "model result; model outcomes never "
                               "amend this gate",
           "days": days_total,
           "eligible_window": {"start_utc": body["window_utc"][0],
                               "end_utc": body["window_utc"][1],
                               "days": window_days,
                               "machine_outages_all_capture":
                                   [{"start_utc": time.strftime(
                                       "%m-%d %H:%M",
                                       time.gmtime(a * 60)),
                                     "gap_min": b - a}
                                    for a, b in outages]},
           "criteria": s_criteria if v21_governing else criteria,
           "verdict": verdict,
           "failing": [k for k, v in
                       (s_criteria if v21_governing
                        else criteria).items() if not v],
           "projection": projection,
           "ess_min_across_horizons": round(ess_min, 1),
           "mde80_worst_pct_of_mae": (round(mde_max, 1)
                                      if mde_max < 999 else None),
           "open_capture_incidents": open_inc,
           "missing_shards": missing_shards[:10],
           "window_analysis": {k: body[k] for k in
                               ("coverage", "material_gaps",
                                "outcome_diversity", "power",
                                "temporal_blocks_6h",
                                "regime_concentration_of_usable")},
           "proposed_v2_1_stitched": shadow_v2_1,
           "integrity_whole_capture": integ}
    (RES / "f1_capture_qualification.json").write_text(
        json.dumps(doc, indent=1))
    print(f"F1 v2: {verdict} — window {window_days}d "
          f"(capture {days_total}d, {len(outages)} outage(s) "
          f"truncated), ess_min {round(ess_min, 1)}, "
          f"failing {doc['failing'] or 'none'}")


if __name__ == "__main__":
    main()
