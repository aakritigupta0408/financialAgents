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
