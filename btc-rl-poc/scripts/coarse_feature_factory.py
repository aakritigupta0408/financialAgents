"""L1 — COARSE feature factory (directive P1). Green provenance gate
(COARSE_MARK_AVAILABILITY_SEMANTICS PASS) authorizes building features from the
15-minute BRTI settlement-mark chain on the large ~6,189-window cohort.

Feature IDs are EXPLICITLY namespaced `brti_coarse.*` and say "N marks", never
`rsi_60m`, so no one can later mistake these for minute-candle indicators. Every
feature uses only marks with close_time <= T0 (proven available at T0). Label =
exact official BRTI outcome.

Writes research/true15m/coarse_features.jsonl + .meta.json + a feature-registry
fragment. This is FAMILY A's BTC-state backbone; it is NOT merged with the FINE
cohort (that stays separate, per directive).
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

INV = ROOT / "research" / "true15m" / "contract_inventory.jsonl"
OUT = ROOT / "research" / "true15m" / "coarse_features.jsonl"
META = ROOT / "research" / "true15m" / "coarse_features.meta.json"
REG = ROOT / "research" / "true15m" / "FEATURE_REGISTRY_coarse.json"
WINDOW_S = 900
NEED = 9                     # marks required (>=2h + headroom) -> defines the cohort


def _chain(hist, target_t0):
    """Contiguous prior settlement marks S ending at or before target_t0 (chronological)."""
    marks = []
    for w in reversed(hist):
        if not marks:
            if w["T1"] <= target_t0:
                marks.append(w)
        elif w["T1"] == marks[-1]["T0"]:
            marks.append(w)
        else:
            break
        if len(marks) >= 20:
            break
    marks.reverse()
    return marks


def _ema(xs, span):
    a = 2 / (span + 1)
    e = xs[0]
    for x in xs[1:]:
        e = a * x + (1 - a) * e
    return e


def _rsi_marks(S, n):
    if len(S) < n + 1:
        return None
    g = l = 0.0
    for i in range(-n, 0):
        d = S[i] - S[i - 1]
        g += max(d, 0); l += max(-d, 0)
    if g + l == 0:
        return 50.0
    rs = (g / n) / ((l / n) or 1e-9)
    return round(100 - 100 / (1 + rs), 3)


def _features(marks, ups):
    S = [m["settlement_brti_avg"] for m in marks]   # settlement-mark level path; S[-1] ends at T0
    n = len(S)
    r = [math.log(S[i] / S[i - 1]) for i in range(1, n) if S[i - 1] > 0]   # per-mark returns

    def ret(k):
        return round(math.log(S[-1] / S[-1 - k]), 6) if n > k and S[-1 - k] > 0 else None

    def rstd(w):
        seg = r[-w:]
        return round(statistics.pstdev(seg), 6) if len(seg) >= 2 else None

    # direction streak on outcomes
    streak = 1
    for i in range(len(ups) - 2, -1, -1):
        if ups[i] == ups[-1]:
            streak += 1
        else:
            break
    streak = streak if ups[-1] == 1 else -streak
    # trend slope + t-stat over last 8 marks (OLS on index)
    seg = S[-8:] if n >= 8 else S
    m = len(seg)
    xs = list(range(m)); xbar = sum(xs) / m; ybar = sum(seg) / m
    sxx = sum((x - xbar) ** 2 for x in xs) or 1e-9
    slope = sum((xs[i] - xbar) * (seg[i] - ybar) for i in range(m)) / sxx
    resid = [seg[i] - (ybar + slope * (xs[i] - xbar)) for i in range(m)]
    sse = sum(e * e for e in resid)
    se = math.sqrt((sse / max(1, m - 2)) / sxx) if m > 2 else None
    tstat = round(slope / se, 3) if se and se > 1e-12 else None
    # drawdown / runup / range position over last 8
    hi, lo, last = max(seg), min(seg), seg[-1]
    rng = (hi - lo) or 1e-9
    v_s, v_l = rstd(4), rstd(12)
    ret1 = ret(1)
    return {
        "brti_coarse.ret_1mark_15m": ret1,
        "brti_coarse.ret_2marks_30m": ret(2),
        "brti_coarse.ret_4marks_60m": ret(4),
        "brti_coarse.ret_8marks_120m": ret(8),
        "brti_coarse.mom_4_minus_8": round((ret(4) or 0) - (ret(8) or 0), 6),
        "brti_coarse.accel_1_minus_mean3": round((ret1 or 0) - (statistics.fmean(r[-4:-1]) if len(r) >= 4 else 0), 6),
        "brti_coarse.rstd_4marks": rstd(4),
        "brti_coarse.rstd_12marks": rstd(12),
        "brti_coarse.volratio_short_long": round((v_s / v_l), 4) if (v_s and v_l and v_l > 1e-12) else None,
        "brti_coarse.drawdown_8marks": round((last - hi) / hi, 6) if hi else None,
        "brti_coarse.runup_8marks": round((last - lo) / lo, 6) if lo else None,
        "brti_coarse.range_pos_8marks": round((last - lo) / rng, 4),
        "brti_coarse.dir_streak": streak,
        "brti_coarse.trend_slope_8marks": round(slope, 6),
        "brti_coarse.trend_tstat_8marks": tstat,
        "brti_coarse.autocorr_lag1": round(statistics.correlation(r[:-1], r[1:]), 4) if len(r) >= 6 else None,
        "brti_coarse.rsi_4marks": _rsi_marks(S, 4),
        "brti_coarse.rsi_8marks": _rsi_marks(S, 8),
        "brti_coarse.ema_fast4_vs_slow12": round((_ema(S[-8:], 4) - _ema(S[-12:], 12)) / S[-1], 6) if n >= 12 else None,
        "brti_coarse.macd_2_4_marks": round(_ema(S[-8:], 2) - _ema(S[-8:], 4), 4) if n >= 8 else None,
    }


def build():
    t0 = time.time()
    EV.emit("JOB_STARTED", "COARSE feature factory", lane="L1",
            narrative="Building brti_coarse.* features on the ~6,189-window large cohort "
                      "from the 15-min mark chain (provenance gate is green).")
    inv = [json.loads(l) for l in INV.open() if l.strip()]
    inv.sort(key=lambda r: r["T0"])
    rows, kept = [], 0
    for i, w in enumerate(inv):
        marks = _chain(inv[:i], w["T0"])
        if len(marks) < NEED:
            continue
        max_close = max(m["T1"] for m in marks)
        if max_close > w["T0"]:                      # PIT belt-and-suspenders
            continue
        feats = _features(marks, [m["official_outcome"] for m in marks])
        rows.append({"market_window_id": w["market_window_id"], "T0": w["T0"],
                     "official_outcome": w["official_outcome"],
                     "opening_target": w["opening_brti_avg"],
                     "cohort": "COARSE", "features": feats,
                     "max_source_close_ts": max_close})
        kept += 1
    body = "".join(json.dumps(x) + "\n" for x in rows)
    OUT.write_text(body)
    import hashlib
    sha = hashlib.sha256(body.encode()).hexdigest()[:16]
    fids = list(rows[0]["features"].keys()) if rows else []
    META.write_text(json.dumps({
        "schema_version": "coarse-features-1", "generated_at": time.time(),
        "cohort": "COARSE_BTC_STATE", "n_windows": kept, "n_features": len(fids),
        "feature_ids": fids, "resolution": "15m settlement marks",
        "provenance_gate": "COARSE_MARK_AVAILABILITY_SEMANTICS PASS",
        "content_sha256_16": sha,
        "note": "IDs namespaced brti_coarse.* ('N marks'); NOT minute-candle indicators. "
                "Do NOT merge with the FINE cohort.",
    }, indent=1))
    REG.write_text(json.dumps({
        "family": "brti_coarse", "resolution": "15m marks",
        "features": [{"feature_id": f, "source": "BRTI settlement-mark chain",
                      "available_for_decision": "<= T0 (gate PASS)"} for f in fids]}, indent=1))
    EV.emit("FEATURE_VALIDATED", f"COARSE features: {len(fids)} on {kept} windows", lane="L1",
            fact=f"{kept} windows, {len(fids)} brti_coarse.* features, sha {sha}. "
                 "All marks close_time <= T0.",
            interpretation="Family A backbone ready; low-frequency BTC state, legit pre-T0.",
            next_action="AV cross-asset as CORE-context; then coverage matrix + integrity.",
            files=[str(OUT.relative_to(ROOT))], metrics={"windows": kept, "features": len(fids)},
            duration_ms=int((time.time() - t0) * 1000))
    print(f"coarse_feature_factory: {kept} windows, {len(fids)} features, sha={sha}")


if __name__ == "__main__":
    build()
