"""The ONE cross-venue synchronization layer (PM 08-30).

Feature scripts never align timestamps independently — they call
states_at(). Rule: latest event AT OR BEFORE decision_ts per venue
(never a nearest-in-future quote), with explicit age and validity
against a max-age limit.

Emitter mode (audit chain): appends one synchronized state per
minute to results/xvenue_state.jsonl (idempotent by minute), so the
lead/lag research has a canonical prospective record from day one.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
XDIR = RES / "events_xvenue"
OUT = RES / "xvenue_state.jsonl"

VENUES = ("coinbase", "binance", "okx", "kraken")
MAX_AGE_S = 30.0


def _recent_rows(lookback_s=1200):
    """Rows from the newest shards covering the lookback window."""
    rows = []
    for sh in sorted(glob.glob(str(XDIR / "*.jsonl")))[-2:]:
        for l in open(sh):
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
    cut = time.time() - lookback_s
    return [r for r in rows if r.get("ts_recv", 0) >= cut]


def states_at(decision_ts, rows=None, max_age_s=MAX_AGE_S):
    """Per venue: latest trade at-or-before decision_ts with age and
    validity. Coinbase rides the primary tape (recent_prices), so
    xvenue covers binance/okx/kraken here; callers merge coinbase
    from the canonical price file."""
    rows = rows if rows is not None else _recent_rows()
    out = {}
    for v in ("binance", "okx", "kraken"):
        cand = [r for r in rows if r.get("src") == v
                and r.get("ts_recv", 9e18) <= decision_ts]
        if not cand:
            out[v] = {"valid": False, "reason": "no event at-or-"
                      "before decision_ts"}
            continue
        last = max(cand, key=lambda r: r["ts_recv"])
        age = decision_ts - last["ts_recv"]
        out[v] = {"px": last["px"],
                  "source_event_ts": last.get("ts_event"),
                  "receive_ts": last["ts_recv"],
                  "age_ms": round(age * 1000),
                  "valid": age <= max_age_s}
    return out


def shard_manifest():
    """Provenance: {shard_name: sha256[:16]} for every raw shard.
    Lives here so consumers never touch the shard directory."""
    import hashlib
    out = {}
    for p in sorted(XDIR.glob("*.jsonl")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def load_second_series():
    """One pass over ALL shards -> per-venue {second: last_px}.
    Lives HERE because this module is the one sanctioned reader of
    the raw shards (no-private-time-alignment law). Used by the
    T1.1 dataset freezer via batch_features()."""
    series = {v: {} for v in ("binance", "okx", "kraken")}
    for sh in sorted(glob.glob(str(XDIR / "*.jsonl"))):
        for l in open(sh):
            try:
                r = json.loads(l)
                series[r["src"]][int(r["ts_recv"])] = r["px"]
            except Exception:
                pass
    return {v: (sorted(d), d) for v, d in series.items()}


def _px_at(keys_map, v, ts):
    """Latest px at-or-before second ts (never future)."""
    import bisect
    keys, d = keys_map[v]
    i = bisect.bisect_right(keys, ts) - 1
    if i < 0:
        return None, None
    return d[keys[i]], ts - keys[i]


# F-XVENUE feature block for T1.1 (FEATURE_REGISTRY family; venue
# returns + consensus/dispersion/lead — coinbase/kalshi members of
# the family ride the primary tape, not these shards). ORDER IS
# FROZEN; the dataset manifest pins this file's sha256.
XV_FEATURES = ("binance_ret_5s_bps", "okx_ret_5s_bps",
               "kraken_ret_30s_bps", "consensus_30s_bps",
               "dispersion_bps", "binance_lead_bps",
               "okx_lead_bps", "n_venues_fresh_30s")


def features_at_batch(ts_list, keys_map=None, max_age_s=MAX_AGE_S):
    """PIT-safe F-XVENUE features at each decision second. Returns
    {ts: [8 floats] or None (insufficient venue data)}."""
    import math
    if keys_map is None:
        keys_map = load_second_series()
    out = {}
    for ts in ts_list:
        px, age, r5, r30 = {}, {}, {}, {}
        for v in ("binance", "okx", "kraken"):
            p, a = _px_at(keys_map, v, ts)
            px[v], age[v] = p, a
            for lb, dst in ((5, r5), (30, r30)):
                q, qa = _px_at(keys_map, v, ts - lb)
                if p and q and a is not None and a <= max_age_s \
                        and qa is not None and qa <= max_age_s + lb:
                    dst[v] = 1e4 * math.log(p / q)
        fresh = [v for v in ("binance", "okx", "kraken")
                 if age[v] is not None and age[v] <= 30]
        if "binance" not in fresh or "okx" not in fresh:
            out[ts] = None
            continue
        med30 = sorted(r30.values())[len(r30) // 2] if r30 else 0.0
        disp = 0.0
        pxs = [px[v] for v in fresh if px[v]]
        if len(pxs) >= 2:
            m = sorted(pxs)[len(pxs) // 2]
            disp = 1e4 * max(abs(p - m) / m for p in pxs)
        out[ts] = [r5.get("binance", 0.0), r5.get("okx", 0.0),
                   r30.get("kraken", 0.0),
                   (sum(r30.values()) / len(r30)) if r30 else 0.0,
                   disp,
                   r30.get("binance", 0.0) - med30,
                   r30.get("okx", 0.0) - med30,
                   float(len(fresh))]
    return out


def emit():
    """Append per-minute synchronized states (idempotent)."""
    seen = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines()[-100:]:
            try:
                seen.add(json.loads(l)["decision_ts"])
            except Exception:
                pass
    rows = _recent_rows()
    now_min = int(time.time()) // 60 * 60
    wrote = 0
    with OUT.open("a") as f:
        for k in range(10, 0, -1):
            ts = now_min - k * 60
            if ts in seen:
                continue
            st = states_at(ts, rows=rows)
            if not any(v.get("valid") for v in st.values()):
                continue        # capture not yet covering this minute
            f.write(json.dumps({"decision_ts": ts,
                                "schema": "xvenue-sync-v1",
                                "venues": st}) + "\n")
            wrote += 1
    print(f"xvenue_sync: +{wrote} synchronized minute-states")


if __name__ == "__main__":
    emit()
