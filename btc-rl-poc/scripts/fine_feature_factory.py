"""L7 — FINE feature factory (directive P1, Family-B challenger).

Builds brti_fine.* features ONLY from the ~363 genuinely high-resolution windows
(the ~16s BRTI sample block, Sep 8–12). Pre-open fine state for window n comes
from the in-block samples in [T0-H, T0] (the tail of the contiguous prior
windows). Every sample used has ts <= T0. IDs are namespaced brti_fine.* and are
NEVER merged into brti_coarse.* — the cohorts are scientifically distinct.

Source: results/brti_decision_dataset.jsonl (brti_sample_ts, current_brti).
Writes research/true15m/fine_features.jsonl + .meta.json + registry fragment.
This is the Family-B (~363-window) dataset; it must not become the main universe.
"""
import hashlib
import json
import math
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
OUT = ROOT / "research" / "true15m" / "fine_features.jsonl"
META = ROOT / "research" / "true15m" / "fine_features.meta.json"
REG = ROOT / "research" / "true15m" / "FEATURE_REGISTRY_fine.json"
LOOKBACK_S = 300           # 5m pre-open window for fine features
MIN_SAMPLES = 8            # ~16s cadence over 5m -> ~18 expected; require >= 8 real


def _series():
    ts, px = [], []
    seen = set()
    rows = []
    for l in SAMP.open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        t = r.get("brti_sample_ts"); b = r.get("current_brti")
        if t and b:
            rows.append((float(t), float(b)))
    rows.sort()
    for t, b in rows:
        if t in seen:
            continue
        seen.add(t); ts.append(t); px.append(b)
    return ts, px


def _fine_features(ts, px, lo, hi):
    """Features from samples in [T0-LOOKBACK, T0] = ts[lo:hi] (all <= T0)."""
    T = ts[lo:hi]; P = px[lo:hi]
    m = len(P)
    if m < MIN_SAMPLES:
        return None
    t0 = T[-1]
    last = P[-1]

    def px_at(dt):
        # latest price at or before (t0 - dt)
        tgt = t0 - dt
        i = bisect_right(T, tgt) - 1
        return P[i] if i >= 0 else None

    def ret(dt):
        p = px_at(dt)
        return round(math.log(last / p), 6) if (p and p > 0) else None

    rets = [math.log(P[i] / P[i - 1]) for i in range(1, m) if P[i - 1] > 0]
    rvol = round((sum(x * x for x in rets) / len(rets)) ** 0.5, 6) if rets else None
    # slope over the window (per second), + acceleration (2nd half vs 1st half return)
    span = (T[-1] - T[0]) or 1.0
    slope = round((P[-1] - P[0]) / span, 8)
    mid = m // 2
    r_first = math.log(P[mid] / P[0]) if P[0] > 0 else 0.0
    r_second = math.log(P[-1] / P[mid]) if P[mid] > 0 else 0.0
    hi_p, lo_p = max(P), min(P)
    rng = (hi_p - lo_p) or 1e-9
    signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in rets]
    persist = round(sum(1 for a, b in zip(signs, signs[1:]) if a == b and a != 0)
                    / max(1, len(signs) - 1), 4)
    return {
        "brti_fine.ret_30s": ret(30), "brti_fine.ret_1m": ret(60),
        "brti_fine.ret_2m": ret(120), "brti_fine.ret_3m": ret(180),
        "brti_fine.ret_5m": ret(300),
        "brti_fine.rvol_sample": rvol,
        "brti_fine.slope_per_s": slope,
        "brti_fine.accel_2nd_minus_1st": round(r_second - r_first, 6),
        "brti_fine.path_range_rel": round(rng / (statistics.fmean(P) or 1e-9), 6),
        "brti_fine.range_pos": round((last - lo_p) / rng, 4),
        "brti_fine.autocorr_lag1": round(statistics.correlation(rets[:-1], rets[1:]), 4)
            if len(rets) >= 6 else None,
        "brti_fine.sign_persistence": persist,
        "brti_fine.open_vs_mean": round((last / (statistics.fmean(P) or last)) - 1, 6),
        "brti_fine.n_samples": m,
    }


def build():
    t_start = time.time()
    EV.emit("JOB_STARTED", "FINE feature factory (Family-B challenger)", lane="L7",
            narrative="Building brti_fine.* features on the ~363 sub-minute windows only; "
                      "kept completely separate from brti_coarse.*.")
    ts, px = _series()
    inv = [json.loads(l) for l in INV.open() if l.strip()]
    inv.sort(key=lambda r: r["T0"])
    rows, kept = [], 0
    for w in inv:
        t0 = w["T0"]
        lo = bisect_left(ts, t0 - LOOKBACK_S)
        hi = bisect_right(ts, t0)                 # <= T0 strictly enforced
        if hi <= lo:
            continue
        feats = _fine_features(ts, px, lo, hi)
        if feats is None:
            continue
        rows.append({"market_window_id": w["market_window_id"], "T0": t0,
                     "official_outcome": w["official_outcome"], "cohort": "FINE",
                     "features": feats})
        kept += 1
    body = "".join(json.dumps(x) + "\n" for x in rows)
    OUT.write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()[:16]
    fids = [k for k in rows[0]["features"].keys()] if rows else []
    META.write_text(json.dumps({
        "schema_version": "fine-features-1", "generated_at": time.time(),
        "cohort": "FINE_BTC_STATE", "n_windows": kept, "n_features": len(fids),
        "feature_ids": fids, "resolution": "~16s samples, 5m pre-open lookback",
        "content_sha256_16": sha,
        "note": "Family-B challenger dataset. brti_fine.* only; NOT merged with coarse. "
                "Small N (~363) — must not be the main training universe.",
    }, indent=1))
    REG.write_text(json.dumps({"family": "brti_fine", "resolution": "~16s",
        "features": [{"feature_id": f, "source": "BRTI ~16s sample block",
                      "available_for_decision": "<= T0"} for f in fids]}, indent=1))
    EV.emit("FEATURE_VALIDATED", f"FINE features: {len(fids)} on {kept} windows", lane="L7",
            fact=f"{kept} windows, {len(fids)} brti_fine.* features, sha {sha}. "
                 "All samples ts <= T0; 5m pre-open lookback.",
            interpretation="Family-B high-resolution cohort ready (small N ~363). Later we "
                           "test whether fine info beats coarse despite the N loss — on "
                           "shared eligible windows only.",
            next_action="AV full backfill + derivatives/options/news resolution in parallel.",
            files=[str(OUT.relative_to(ROOT))], metrics={"windows": kept, "features": len(fids)},
            duration_ms=int((time.time() - t_start) * 1000))
    print(f"fine_feature_factory: {kept} windows, {len(fids)} features, sha={sha}")


if __name__ == "__main__":
    build()
