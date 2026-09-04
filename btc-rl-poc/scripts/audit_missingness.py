"""MISSINGNESS / OUTAGE-SELECTION AUDIT (PM 2026-09-03).

The condition on ratifying GATE_F1 v2.1: prove that stitching out
machine outages cannot bias the research. "The missing data came
from the laptop" is supported; "removing those periods cannot bias
our research" needs evidence. This audit compares RETAINED vs
EXCLUDED observations using the desk's INDEPENDENT Coinbase 1-min
candle source (REST, retrospective — it covers outage minutes our
own tape lacks).

PRE-DECLARED PASS CONDITIONS (written before results were seen):
  A. outcome parity  — |up_frac(ret) - up_frac(exc)| <= 0.10 per
     horizon (evaluated only if excluded n >= 30, else reported)
  B. volatility parity — median vol30 ratio ret/exc in [0.7, 1.4]
  C. trend parity   — median |ret60| ratio ret/exc in [0.5, 2.0]
  D. time-of-day    — every 3h UTC bucket retains >= 30% of its
     grid points (no session wiped out)
  E. outage trigger — median pre-outage vol30 percentile in
     [25, 75] (outages are not market-condition-triggered)
  F. gate re-check  — v2.1 shadow still shows ESS >= 150 per
     horizon and temporal/regime diversity PASS
Verdict PASS iff all evaluable conditions hold.
"""
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_f1_sufficiency import (HORIZONS, machine_outages, scan,
                                  sd)
from btc_rl.sources import fetch_range

GRID_STEP_MIN = 15
PATH_LO, PATH_HI = 60, 30            # v2.1 usable-path rule


def pct_rank(x, pool):
    return 100.0 * sum(1 for p in pool if p <= x) / max(1, len(pool))


def main():
    t0 = time.time()
    per_min, _sec, _integ = scan()
    outages = machine_outages(per_min)
    all_minutes = set()
    for v in per_min:
        all_minutes |= set(per_min[v])
    m_lo, m_hi = min(all_minutes), max(all_minutes)
    out_min = set()
    for a, b in outages:
        out_min.update(range(a + 1, b))

    # independent complete reference series (Coinbase candles)
    bars = fetch_range(
        datetime.fromtimestamp((m_lo - 90) * 60, tz=timezone.utc),
        datetime.fromtimestamp((m_hi + 60) * 60, tz=timezone.utc))
    close = {int(b["ts"] // 60): b["close"] for b in bars}

    def vol30(m):
        rs = [1e4 * math.log(close[x] / close[x - 1])
              for x in range(m - 29, m + 1)
              if x in close and x - 1 in close]
        return sd(rs) * math.sqrt(30) if len(rs) >= 20 else None

    def ret(m, h):
        if m in close and m + h in close:
            return 1e4 * math.log(close[m + h] / close[m])
        return None

    retained, excluded = [], []
    for m in range(m_lo + 60, m_hi - PATH_HI, GRID_STEP_MIN):
        row = {"m": m, "hour_utc": (m // 60) % 24,
               "vol30": vol30(m), "ret60": ret(m - 60, 60)}
        for h in HORIZONS:
            row[f"r{h}"] = ret(m, h)
        if row["vol30"] is None or any(row[f"r{h}"] is None
                                       for h in HORIZONS):
            continue                      # reference hole: drop both
        clean = not any(x in out_min
                        for x in range(m - PATH_LO, m + PATH_HI + 1))
        (retained if clean else excluded).append(row)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    def dist(rows):
        d = {"n": len(rows)}
        if rows:
            d["median_vol30"] = round(med([r["vol30"]
                                           for r in rows]), 1)
            d["median_abs_ret60"] = round(
                med([abs(r["ret60"]) for r in rows
                     if r["ret60"] is not None]), 1)
            for h in HORIZONS:
                rs = [r[f"r{h}"] for r in rows]
                d[f"up_frac_h{h}"] = round(
                    sum(1 for x in rs if x > 0) / len(rs), 3)
                d[f"median_abs_h{h}"] = round(med(
                    [abs(x) for x in rs]), 1)
        return d

    dr, de = dist(retained), dist(excluded)

    # D. time-of-day retention per 3h bucket
    buckets = {}
    for rows, key in ((retained, "ret"), (excluded, "exc")):
        for r in rows:
            b = r["hour_utc"] // 3
            buckets.setdefault(b, {"ret": 0, "exc": 0})[key] += 1
    tod = {f"utc_{3*b:02d}-{3*b+3:02d}": {
        **c, "retention": round(c["ret"] / (c["ret"] + c["exc"]), 3)}
        for b, c in sorted(buckets.items())}

    # E. were outages triggered by market conditions?
    vol_pool = [r["vol30"] for r in retained + excluded]
    pre_vol_pcts = []
    for a, _b in outages:
        v = vol30(a)
        if v is not None:
            pre_vol_pcts.append(round(pct_rank(v, vol_pool), 1))
    med_pre_pct = med(pre_vol_pcts)

    # F. gate v2.1 shadow re-check
    gate = json.loads((RES / "f1_capture_qualification.json")
                      .read_text())
    shadow = gate.get("proposed_v2_1_stitched", {})
    ess_ok = shadow.get("criteria", {}).get("ess_min_150", False)
    div_ok = (shadow.get("criteria", {}).get("temporal_diversity",
                                             False)
              and shadow.get("criteria", {}).get(
                  "regime_no_domination", False))

    exc_n_ok = de["n"] >= 30
    checks = {
        "A_outcome_parity": (all(
            abs(dr[f"up_frac_h{h}"] - de[f"up_frac_h{h}"]) <= 0.10
            for h in HORIZONS) if exc_n_ok else None),
        "B_volatility_parity": (
            0.7 <= dr["median_vol30"] / de["median_vol30"] <= 1.4
            if exc_n_ok and de.get("median_vol30") else None),
        "C_trend_parity": (
            0.5 <= dr["median_abs_ret60"] / de["median_abs_ret60"]
            <= 2.0 if exc_n_ok and de.get("median_abs_ret60")
            else None),
        "D_time_of_day": all(v["retention"] >= 0.30
                             for v in tod.values()),
        "E_outage_not_condition_triggered": (
            25 <= med_pre_pct <= 75 if med_pre_pct is not None
            else None),
        "F_gate_ess_and_diversity_hold": bool(ess_ok and div_ok),
    }
    evaluable = {k: v for k, v in checks.items() if v is not None}
    verdict = "PASS" if all(evaluable.values()) else "FAIL"
    doc = {
        "generated_ts": int(time.time()),
        "purpose": "condition on ratifying GATE_F1 v2.1 (PM 09-03)",
        "reference_series": "Coinbase 1m candles (independent, "
                            "covers outage minutes)",
        "grid_step_min": GRID_STEP_MIN,
        "usable_path_rule": f"[m-{PATH_LO}, m+{PATH_HI}] clean",
        "retained": dr, "excluded": de,
        "time_of_day_retention_3h_utc": tod,
        "pre_outage_vol_percentiles": pre_vol_pcts,
        "median_pre_outage_vol_pct": med_pre_pct,
        "gate_shadow_recheck": {"ess_min_150": ess_ok,
                                "diversity": div_ok},
        "checks": checks,
        "not_evaluable": [k for k, v in checks.items() if v is None],
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 1)}
    (RES / "f1_missingness_audit.json").write_text(
        json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in
                      ("retained", "excluded", "checks", "verdict",
                       "median_pre_outage_vol_pct")}, indent=1))


if __name__ == "__main__":
    main()
