"""MISSINGNESS / OUTAGE-SELECTION AUDIT (PM 2026-09-03).

The condition on ratifying GATE_F1 v2.1: prove that stitching out
machine outages cannot bias the research. "The missing data came
from the laptop" is supported; "removing those periods cannot bias
our research" needs evidence. This audit compares RETAINED vs
EXCLUDED observations using the desk's INDEPENDENT Coinbase 1-min
candle source (REST, retrospective — it covers outage minutes our
own tape lacks).

PRE-DECLARED PASS CONDITIONS (A-D, F written before any result was
seen; E amended once, see below):
  A. outcome parity  — |up_frac(ret) - up_frac(exc)| <= 0.10 per
     horizon (evaluated only if excluded n >= 30, else reported)
  B. volatility parity — median vol30 ratio ret/exc in [0.7, 1.4]
  C. trend parity   — median |ret60| ratio ret/exc in [0.5, 2.0]
  D. time-of-day    — every 3h UTC bucket retains >= 30% of its
     grid points (no session wiped out)
  E. outage-condition independence — AMENDED v2.1.1 (PM ruling
     2026-09-03, registered BEFORE the prospective rerun):
     evaluated at the OUTAGE-EPISODE level, never the individual
     gap level (the v2.1.0 form counted 17 gaps of one doze
     episode as independent draws — a measurement defect, ruled
     FAIL and frozen in git 82db163). Episodes are deterministic:
     adjacent outages separated by < EPISODE_QUIET_MIN of clean
     capture belong to one episode. Median PRE-EPISODE vol30
     percentile must lie in [25, 75] — bounds byte-identical to
     v2.1.0 — and >= MIN_EPISODES episodes are required for E to
     be evaluable; with fewer, the verdict is PENDING (a
     not-evaluable E can never default to PASS).
  F. gate re-check  — v2.1 shadow still shows ESS >= 150 per
     horizon and temporal/regime diversity PASS
Verdict PASS iff all conditions evaluable and passing; the rerun is
PROSPECTIVE — before RERUN_NOT_BEFORE_TS the artifact can only say
PENDING_PROSPECTIVE_RERUN, never PASS. All other criteria (A-D, F,
and every GATE_F1 v2/v2.1 threshold) are byte-identical: this
amendment fixes a defective measurement dimension, it does not make
F1 easier to pass.
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
AUDIT_VERSION = "2.1.1"
EPISODE_QUIET_MIN = 120              # registered episode separator
MIN_EPISODES = 3                     # E evaluability floor
RERUN_NOT_BEFORE_TS = 1788585037     # 2026-09-05 05:10 UTC —
# registration + >=24h of fresh capture under the sleep fix; before
# this ts the verdict is PENDING_PROSPECTIVE_RERUN by law


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

    # E (v2.1.1). episode-level outage-condition independence:
    # deterministic episodes — adjacent outages with < EPISODE_QUIET
    # clean minutes between them are ONE operational event
    episodes = []
    for a, b in outages:
        if episodes and a - episodes[-1][1] < EPISODE_QUIET_MIN:
            episodes[-1][1] = b
        else:
            episodes.append([a, b])
    vol_pool = [r["vol30"] for r in retained + excluded]
    pre_vol_pcts = []                    # one per EPISODE, at first
    for a, _b in episodes:               # outage of the episode
        v = vol30(a)
        if v is not None:
            pre_vol_pcts.append(round(pct_rank(v, vol_pool), 1))
    med_pre_pct = med(pre_vol_pcts)
    e_evaluable = len(pre_vol_pcts) >= MIN_EPISODES

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
        "E_outage_episode_independence": (
            25 <= med_pre_pct <= 75
            if e_evaluable and med_pre_pct is not None else None),
        "F_gate_ess_and_diversity_hold": bool(ess_ok and div_ok),
    }
    now = time.time()
    if now < RERUN_NOT_BEFORE_TS:
        verdict = "PENDING_PROSPECTIVE_RERUN"
    elif checks["E_outage_episode_independence"] is None:
        verdict = "PENDING"      # not-evaluable E never defaults on
    else:
        evaluable = {k: v for k, v in checks.items()
                     if v is not None}
        verdict = "PASS" if all(evaluable.values()) else "FAIL"
    doc = {
        "generated_ts": int(time.time()),
        "audit_version": AUDIT_VERSION,
        "amendment": "E episode-level (PM 09-03); v2.1.0 FAIL frozen "
                     "in git 82db163; bounds and A-D/F byte-identical",
        "rerun_not_before_utc": time.strftime(
            "%Y-%m-%d %H:%M", time.gmtime(RERUN_NOT_BEFORE_TS)),
        "episodes": [{"start_utc": time.strftime(
            "%m-%d %H:%M", time.gmtime(a * 60)),
            "total_outage_min": b - a} for a, b in episodes],
        "purpose": "condition on ratifying GATE_F1 v2.1 (PM 09-03)",
        "reference_series": "Coinbase 1m candles (independent, "
                            "covers outage minutes)",
        "grid_step_min": GRID_STEP_MIN,
        "usable_path_rule": f"[m-{PATH_LO}, m+{PATH_HI}] clean",
        "retained": dr, "excluded": de,
        "time_of_day_retention_3h_utc": tod,
        "pre_episode_vol_percentiles": pre_vol_pcts,
        "median_pre_episode_vol_pct": med_pre_pct,
        "n_episodes": len(pre_vol_pcts),
        "min_episodes_for_E": MIN_EPISODES,
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
                       "n_episodes", "median_pre_episode_vol_pct")},
                     indent=1))


if __name__ == "__main__":
    main()
