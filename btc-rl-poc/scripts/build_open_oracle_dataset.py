"""OPEN_ORACLE_15M — canonical T0 dataset (directive §4-5).

ONE market window = ONE example. At each KXBTC15M open T0 we freeze features
using only information available at or before T0, and label with the exact
official BRTI outcome 15 minutes later.

Source of truth: results/contract_outcomes.jsonl (6,337 settled windows, one row
each). Windows are contiguous: floor_strike[n] == expiration_value[n-1], so the
sequence of settlement marks S = expiration_value IS the 15-minute-sampled BRTI
path. The opening target of window n equals S[n-1] (the just-closed settlement,
known AT T0). Label:  Y = 1 iff expiration_value[n] >= floor_strike[n]  — i.e.
"did the 15-min-ahead settlement print at or above the level 15 min earlier?".

Every feature is a causal function of S[..n-1] (windows that closed at or before
T0). Diffusion mechanics at the open are driftless with distance 0, so
p_mech_15m == 0.5 by construction — recorded honestly as the mechanics control.
The open question is purely: does any pre-open information beat 0.5?

Emits results/open_oracle_15m_dataset.jsonl (+ .meta.json with the PIT audit and
a content hash). No external data — works for the full universe. Rich
independent-feature windows (F-XVENUE/F-DERIVATIVES) are a later extension.
"""
import calendar
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "contract_outcomes.jsonl"
OUT = ROOT / "results" / "open_oracle_15m_dataset.jsonl"
META = ROOT / "results" / "open_oracle_15m_dataset.meta.json"

FEATURE_VERSION = "open15m-feat-v1"
DATA_VERSION = "open15m-data-v1"
TARGET_VERSION = "open15m-Y=settle>=target-exactBRTI-v1"
CONTRACT_SPEC = "KXBTC15M_2026-09"
WINDOW_S = 900
MAX_LAG = 16                 # longest lookback (windows) used by any feature


def _epoch(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


def _load():
    rows = []
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        r["_t0"] = _epoch(r["open_time"])
        r["_t1"] = _epoch(r["close_time"])
        rows.append(r)
    rows.sort(key=lambda r: r["_t0"])
    # de-dupe by market window id (ticker), keep last
    seen = {}
    for r in rows:
        seen[r["ticker"]] = r
    return sorted(seen.values(), key=lambda r: r["_t0"])


def _features(hist):
    """Causal T0 features from the settlement-mark path of prior windows.
    hist = list of prior windows (chronological), all closed <= T0 of the
    target window. Returns (features dict, max_source_close_ts) or None if the
    immediately-preceding chain is too short/broken to form core features."""
    # require the immediately-preceding `MAX_LAG+1` windows to be contiguous so
    # returns/vols describe a real uninterrupted BTC path (no stitching a gap)
    chain = [hist[-1]]
    for i in range(len(hist) - 2, -1, -1):
        if hist[i]["_t1"] == chain[-1]["_t0"] and (chain[-1]["_t0"] - hist[i]["_t0"]) == WINDOW_S:
            chain.append(hist[i])
        else:
            break
        if len(chain) >= MAX_LAG + 2:
            break
    chain.reverse()                       # chronological
    if len(chain) < 6:                    # need a few marks for returns/vol
        return None
    S = [w["expiration_value"] for w in chain]   # settlement marks; S[-1] == target
    up = [w["exact_yes"] for w in chain]
    n = len(S)

    def logret(a, b):
        return math.log(a / b) if (a > 0 and b > 0) else 0.0

    r1 = [logret(S[i], S[i - 1]) for i in range(1, n)]     # 1-step (15m) returns
    def cum(k):
        return logret(S[-1], S[-1 - k]) if n > k else 0.0
    def rvol(w):
        seg = r1[-w:] if len(r1) >= 2 else r1
        return statistics.pstdev(seg) if len(seg) >= 2 else 0.0
    # signed run length of consecutive same-direction outcomes ending at last
    run = 1
    for i in range(len(up) - 2, -1, -1):
        if up[i] == up[-1]:
            run += 1
        else:
            break
    run_signed = run if up[-1] == 1 else -run
    v8 = rvol(8)
    feats = {
        "ret_1": round(r1[-1], 6),
        "ret_2": round(cum(2), 6),
        "ret_4": round(cum(4), 6),
        "ret_8": round(cum(8), 6),
        "rvol_4": round(rvol(4), 6),
        "rvol_8": round(v8, 6),
        "rvol_16": round(rvol(16), 6),
        "range_8": round((max(S[-8:]) - min(S[-8:])) / statistics.fmean(S[-8:]), 6) if n >= 8 else 0.0,
        "prior_up": up[-1],
        "run_signed": run_signed,
        "accel": round(r1[-1] - (statistics.fmean(r1[-4:-1]) if len(r1) >= 4 else 0.0), 6),
        "z_last": round(r1[-1] / v8, 4) if v8 > 1e-9 else 0.0,
        "level_k": round(math.log(S[-1]), 4),
    }
    return feats, chain[-1]["_t1"]


def build():
    rows = _load()
    out, violations, gaps = [], 0, 0
    base_up = 0
    for i in range(len(rows)):
        w = rows[i]
        if (w["_t1"] - w["_t0"]) != WINDOW_S:
            continue                                   # not a clean 15m window
        hist = rows[max(0, i - (MAX_LAG + 4)):i]        # strictly prior windows
        if not hist:
            continue
        fr = _features(hist)
        if fr is None:
            gaps += 1
            continue
        feats, max_src_close = fr
        # PIT audit: every source window closed at or before this window's open
        if max_src_close > w["_t0"]:
            violations += 1
            continue
        base_up += w["exact_yes"]
        out.append({
            "market_window_id": w["ticker"],
            "T0": w["_t0"], "T1": w["_t1"],
            "opening_target": w["floor_strike"],
            "official_outcome": w["exact_yes"],
            "settlement_value": w["expiration_value"],
            "p_mech_15m": 0.5,          # driftless diffusion, distance 0 at open
            "features": feats,
            "max_source_close_ts": max_src_close,
            "feature_version": FEATURE_VERSION,
            "contract_spec_version": CONTRACT_SPEC,
            "data_version": DATA_VERSION,
            "target_version": TARGET_VERSION,
        })
    body = "".join(json.dumps(r) + "\n" for r in out)
    OUT.write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()[:16]
    n = len(out)
    meta = {
        "schema_version": "open-oracle-15m-dataset-1",
        "generated_at": time.time(),
        "source": "results/contract_outcomes.jsonl",
        "market_window_n": n,               # == independent examples
        "raw_observation_n": n,             # 1 row per window (no tick inflation)
        "windows_skipped_gap_or_short": gaps,
        "post_open_information_violations": violations,   # MUST be 0
        "class_balance_up": round(base_up / n, 4) if n else None,
        "feature_families": {"F-CONTRACT/price-path": list(out[0]["features"].keys()) if out else []},
        "feature_version": FEATURE_VERSION, "data_version": DATA_VERSION,
        "target_version": TARGET_VERSION, "contract_spec_version": CONTRACT_SPEC,
        "content_sha256_16": sha,
        "note": "One KXBTC15M window = one example. Features causal <= T0 from the "
                "settlement-mark path; label = exact official BRTI outcome 15m later. "
                "p_mech_15m == 0.5 (driftless mechanics at the open) is the honest control.",
    }
    META.write_text(json.dumps(meta, indent=1))
    print(f"open_oracle_15m dataset: {n} windows (1 row each); "
          f"skipped_gap={gaps}; PIT violations={violations}; "
          f"base_rate_up={meta['class_balance_up']}; sha={sha}")


if __name__ == "__main__":
    build()
