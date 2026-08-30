"""Live ↔ replay parity (M2.5, PM 08-30) — the hard promotion gate.

For every PIT feature snapshot: rebuild the prediction OFFLINE from
the stored inputs and compare to the logged live output.

kb2 (blend-v1): exact deterministic recompute of
    clamp(bw*p_cal + (1-bw)*k_pup) — 100% of snapshots checked every
    run, tolerance 1e-4 (rounding of the logged value).
kb9 (TimesFM 2.5, frozen zero-shot): replaying a 200M-parameter
    foundation model inside a 10-minute cron is not proportionate, so
    rows are marked PARITY_DEFERRED_HEAVY and a SAMPLED replay
    (--sample N) can be run explicitly; deferred is a declared state,
    never treated as PASS.

Output: results/parity.json —
    {checked, pass, fail, deferred, by_model, failures[...],
     parity_state}  · parity_state FAIL blocks promotion (consumed by
    model_qualification) and is a red invariant.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TOL = 1e-4          # logged predictions are rounded to 4 decimals


def replay_kb2(feat):
    p_cal, k_pup, bw = feat["p_cal"], feat["k_pup"], feat["bw"]
    if k_pup is None:
        return p_cal
    return round(min(0.99, max(0.01, bw * p_cal + (1 - bw) * k_pup)), 4)


def replay_kb9(feat):
    """Explicit sampled replay only — heavy."""
    try:
        sys.path.insert(0, str(ROOT))
        from btc_rl import online as O
        fm = O._timesfm_p_up(feat["closes"], feat["strike"],
                             feat["horizon_min"])
        return fm[0] if fm else None
    except Exception:
        return None


def main(sample_kb9=0):
    p = RES / "feature_snapshots.jsonl"
    rows = []
    if p.exists():
        for l in p.open():
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
    checked = passed = failed = deferred = 0
    by_model, failures = {}, []
    kb9_rows = [r for r in rows if r["variant"] == "kb9"]
    kb9_sampled = set(id(r) for r in kb9_rows[-sample_kb9:]) \
        if sample_kb9 else set()
    for r in rows:
        v = r["variant"]
        st = by_model.setdefault(v, {"checked": 0, "pass": 0,
                                     "fail": 0, "deferred": 0})
        if v == "kb2":
            rep = replay_kb2(r["features"])
        elif v == "kb9" and id(r) in kb9_sampled:
            rep = replay_kb9(r["features"])
        else:
            deferred += 1
            st["deferred"] += 1
            continue
        checked += 1
        st["checked"] += 1
        diff = abs(rep - r["prediction"]) if rep is not None else None
        ok = diff is not None and diff <= TOL
        if ok:
            passed += 1
            st["pass"] += 1
        else:
            failed += 1
            st["fail"] += 1
            failures.append({
                "prediction_id": r["prediction_id"],
                "prediction_logged": r["prediction"],
                "prediction_replayed": rep,
                "abs_diff": diff,
                "feature_hash": r.get("feature_hash")})
    state = ("NO_SNAPSHOTS" if not rows else
             "FAIL" if failed else
             "PASS" if checked else "ALL_DEFERRED")
    doc = {"generated_ts": int(time.time()),
           "tolerance": TOL,
           "snapshots": len(rows), "checked": checked,
           "pass": passed, "fail": failed, "deferred": deferred,
           "by_model": by_model,
           "failures": failures[:20],
           "parity_state": state,
           "policy": "FAIL blocks model promotion and reds the "
                     "invariant wall; DEFERRED is declared, never "
                     "treated as PASS; kb9 replay via "
                     "check_parity.py --sample N"}
    (RES / "parity.json").write_text(json.dumps(doc, indent=1))
    print(f"parity: {state} · {passed}/{checked} checked pass · "
          f"{deferred} deferred · {len(rows)} snapshots")


if __name__ == "__main__":
    n = 0
    if "--sample" in sys.argv:
        n = int(sys.argv[sys.argv.index("--sample") + 1])
    main(sample_kb9=n)
