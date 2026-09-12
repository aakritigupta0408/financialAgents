"""R2A — DERIVATIVES capture adapter (prospective, PIT-safe).

Source: OKX public REST (free, no auth; Binance is geo-blocked here). These
are ORTHOGONAL-to-spot signals — funding, open interest, basis, positioning
(long/short account ratio) — i.e. leverage/forced-flow state, not another
BTC price copy. Missingness is explicit; an outage here must never affect
the core BTC/Kalshi capture (the caller treats this as an optional source).

Runtime: call snapshot() (or capture_once()) from the existing scheduler at a
modest cadence (e.g. every 15-30s). Records append to
results/deriv_capture.jsonl. Health via status().
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "deriv_capture.jsonl"
HEALTH = ROOT / "results" / "deriv_health.json"
BASE = "https://www.okx.com"
SWAP = "BTC-USD-SWAP"
IDX = "BTC-USD"
SCHEMA = "deriv-v1"
TIMEOUT = 8


_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": _UA,
                                                       "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        doc = json.loads(r.read().decode())
    if doc.get("code") not in ("0", 0):
        raise RuntimeError(doc.get("msg", "okx error"))
    return doc.get("data") or []


def snapshot():
    """One PIT-safe derivatives record. Fields absent on error are null with
    is_missing set; available_for_decision_ts = our receive time."""
    now = time.time()
    rec = {"source": "okx", "schema_version": SCHEMA,
           "received_ts": now, "available_for_decision_ts": now,
           "instrument": SWAP, "errors": []}
    def field(name, fn):
        try:
            fn()
        except Exception as e:            # optional source: never raise
            rec[name + "_missing"] = True
            rec["errors"].append(f"{name}:{str(e)[:60]}")

    def funding():
        d = _get(f"/api/v5/public/funding-rate?instId={SWAP}")[0]
        rec["funding_rate"] = float(d["fundingRate"])
        rec["next_funding_ts"] = int(d["fundingTime"]) / 1000.0
        rec["source_ts_funding"] = int(d.get("ts", 0)) / 1000.0 or None
    field("funding_rate", funding)

    def oi():
        d = _get(f"/api/v5/public/open-interest?instId={SWAP}")[0]
        rec["open_interest_ccy"] = float(d["oiCcy"])
        rec["open_interest_usd"] = float(d["oiUsd"])
        rec["source_ts_oi"] = int(d["ts"]) / 1000.0
    field("open_interest", oi)

    def basis():
        mk = _get(f"/api/v5/public/mark-price?instId={SWAP}")[0]
        ix = _get(f"/api/v5/market/index-tickers?instId={IDX}")[0]
        m, i = float(mk["markPx"]), float(ix["idxPx"])
        rec["mark_px"] = m
        rec["index_px"] = i
        rec["basis"] = round(m - i, 2)
        rec["basis_bps"] = round((m - i) / i * 1e4, 3) if i else None
    field("basis", basis)

    def posn():
        d = _get("/api/v5/rubik/stat/contracts/long-short-account-ratio"
                 "?ccy=BTC&period=5m")
        if d:
            rec["long_short_ratio"] = float(d[0][1])
            rec["source_ts_ls"] = int(d[0][0]) / 1000.0
    field("long_short_ratio", posn)

    rec["ok"] = not rec["errors"]
    return rec


def capture_once():
    rec = snapshot()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    state = "CONNECTED" if rec["ok"] else (
        "DEGRADED" if any(k in rec for k in ("funding_rate", "open_interest_ccy"))
        else "UNAVAILABLE")
    HEALTH.write_text(json.dumps({
        "source": "okx-derivatives", "state": state,
        "last_observation_ts": rec["received_ts"],
        "schema_version": SCHEMA, "errors": rec["errors"],
        "fields_present": [k for k in ("funding_rate", "open_interest_ccy",
                           "basis", "long_short_ratio") if k in rec]}, indent=1))
    return rec, state


if __name__ == "__main__":                    # live smoke test
    rec, state = capture_once()
    print("state:", state)
    for k in ("funding_rate", "open_interest_ccy", "mark_px", "index_px",
              "basis_bps", "long_short_ratio"):
        print(f"  {k}: {rec.get(k)}")
    if rec["errors"]:
        print("  errors:", rec["errors"])
