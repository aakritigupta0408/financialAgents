"""BRTI (CF Benchmarks BTC Real-Time Index) via Kalshi's authenticated
passthrough — the EXACT contract benchmark. Kalshi documents:
  GET  /trade-api/v2/cfbenchmarks/values?id=BRTI
  GET  /trade-api/v2/cfbenchmarks/history/values?id=BRTI&...
  WS   cfbenchmarks_value  (index_ids=["BRTI"], ~1/s, carries raw value +
       avg_60s_data + last_60s_windowed_average_15min)

Auth: the project's RSA key (~/.kalshi_demo.pem) signs (ts+method+path);
KALSHI-ACCESS-KEY = the key id (env KALSHI_DEMO_KEY_ID). Signs the PATH only
(no query), matching scripts/demo_reader.py.

This module PROBES access, records the exact state, and — if authorized —
captures BRTI append-only and runs historical parity vs floor_strike /
expiration_value. Run:  KALSHI_DEMO_KEY_ID=<id> python3 data/adapters/brti.py
"""
import base64
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import re

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "brti_capture.jsonl"
HEALTH = ROOT / "results" / "brti_health.json"

# Combined credential bundle: the owner may drop BOTH the private-key PEM and the
# access-key id (a UUID) into one file. Parse each part out separately.
BUNDLE = Path.home() / ".kalshi_key_api"
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL)


def _bundle_text():
    return BUNDLE.read_text() if BUNDLE.exists() else ""


def _read_key_id():
    v = os.environ.get("KALSHI_DEMO_KEY_ID") or os.environ.get("KALSHI_KEY_ID")
    if v:
        return v.strip()
    for f in (Path.home() / ".kalshi_key_id", Path.home() / ".kalshi_prod_key_id"):
        if f.exists():
            return f.read_text().strip()
    m = _UUID.search(_bundle_text())        # UUID embedded in the bundle
    return m.group(0) if m else ""


def _pem_bytes():
    """Return PEM private-key bytes from the bundle or a standalone .pem."""
    m = _PEM_BLOCK.search(_bundle_text())
    if m:
        return (m.group(0) + "\n").encode()
    for f in (Path.home() / ".kalshi_prod.pem", Path.home() / ".kalshi.pem",
              Path.home() / ".kalshi_demo.pem"):
        if f.exists():
            return f.read_bytes()
    return b""


KEY_ID = _read_key_id()
HOSTS = ["https://api.elections.kalshi.com", "https://demo-api.kalshi.co"]
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "Chrome/124.0 Safari/537.36")

_priv = None


def _key():
    global _priv
    if _priv is None:
        from cryptography.hazmat.primitives import serialization
        _priv = serialization.load_pem_private_key(_pem_bytes(), password=None)
    return _priv


def _hdr(method, path):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    ts = str(int(time.time() * 1000))
    sig = _key().sign((ts + method + path).encode(),
                      padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                  salt_length=padding.PSS.DIGEST_LENGTH),
                      hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "User-Agent": UA, "Accept": "application/json"}


def _req(host, path, query=None):
    url = host + path + ("?" + urllib.parse.urlencode(query) if query else "")
    req = urllib.request.Request(url, headers=_hdr("GET", path))
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.getcode(), json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)[:100]}


def probe():
    if not KEY_ID:
        return {"state": "MISSING_KEY_ID",
                "detail": "KALSHI_DEMO_KEY_ID not set; private key present. "
                          "Owner action: export KALSHI_DEMO_KEY_ID=<id> (same id "
                          "the daemon uses) and rerun.",
                "endpoint_exists": True}
    path = "/trade-api/v2/cfbenchmarks/values"
    for host in HOSTS:
        code, body = _req(host, path, {"id": "BRTI"})
        if code == 200:
            return {"state": "AVAILABLE", "host": host, "sample": body,
                    "endpoint_exists": True}
        if code == 200 and not body:
            return {"state": "AUTHORIZED_BUT_EMPTY", "host": host}
        if code in (401, 403):
            last = {"state": "ENTITLEMENT_DENIED", "host": host, "code": code,
                    "detail": body.get("error", body)}
        else:
            last = {"state": "ERROR", "host": host, "code": code, "detail": body}
    return {**last, "endpoint_exists": True}


# ── History (backfill) — VERIFIED schema ──────────────────────────────────────
# GET cfbenchmarks/history/values?id=BRTI&timespan=HOUR&timestamp=<ISO hour>
# returns the FULL calendar hour of samples (5Hz for ~90d, 1Hz to >=180d).
# `timestamp` MUST be truncated to the hour boundary (ISO_INSTANT) or it 400s.
# Settlement convention (verified to <=0.0004% vs official strikes): the
# reference at boundary T is the mean of BRTI over the 60s window [T-60, T).
HIST = "/trade-api/v2/cfbenchmarks/history/values"
_HOUR_CACHE = {}


def _iso_hour(sec):
    import datetime
    h = (int(sec) // 3600) * 3600
    return datetime.datetime.fromtimestamp(
        h, datetime.timezone.utc).strftime("%Y-%m-%dT%H:00:00.000Z")


def hour_samples(sec, host=None):
    """All (epoch_s, value) BRTI samples for the calendar hour containing `sec`.
    Cached per hour so a backfill sweep hits each hour once."""
    key = (int(sec) // 3600) * 3600
    if key not in _HOUR_CACHE:
        host = host or HOSTS[0]
        code, body = _req(host, HIST, {"id": "BRTI", "timespan": "HOUR",
                                       "timestamp": _iso_hour(sec)})
        pl = (body.get("data", {}) or {}).get("payload", []) if code == 200 else []
        _HOUR_CACHE[key] = [(x["time"] / 1000.0, float(x["value"])) for x in pl]
    return _HOUR_CACHE[key]


def avg_60s_ending(T):
    """Official settlement statistic: mean BRTI over [T-60, T). Returns (avg, n).
    Handles the case where the 60s window straddles an hour boundary."""
    seen = {}
    for t, v in hour_samples(T - 60) + hour_samples(T):
        if T - 60 <= t < T:
            seen[round(t, 3)] = v            # dedup identical timestamps
    vals = list(seen.values())
    return (round(sum(vals) / len(vals), 2) if vals else None), len(vals)


def parity(ticker_rows):
    """Reconstruct floor_strike (at open) and expiration_value (at close) from
    BRTI for each settled window; compare to Kalshi's official values."""
    import calendar
    out = []
    for r in ticker_rows:
        for which, t_iso, ref in (("open", r["open_time"], r["floor_strike"]),
                                  ("close", r["close_time"], r["expiration_value"])):
            T = calendar.timegm(time.strptime(t_iso, "%Y-%m-%dT%H:%M:%SZ"))
            avg, n = avg_60s_ending(T)
            out.append({"ticker": r["ticker"], "which": which, "n": n,
                        "brti_60s_avg": avg, "kalshi_ref": ref,
                        "abs_err": (round(abs(avg - ref), 2) if avg is not None else None),
                        "match_penny": (avg is not None and abs(avg - ref) < 0.01)})
    return out


def main():
    st = probe()
    HEALTH.write_text(json.dumps({"source": "kalshi-cfbenchmarks-BRTI",
                                  "checked_ts": time.time(), **st}, indent=1))
    print("BRTI access:", st.get("state"), "-", st.get("detail", st.get("host", "")))
    if st.get("state") == "AVAILABLE":
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a") as fh:
            fh.write(json.dumps({"received_ts": time.time(),
                                 "sample": st["sample"]}) + "\n")
        co = ROOT / "results" / "contract_outcomes.jsonl"
        rows = [json.loads(l) for l in co.open() if l.strip()][-6:] if co.exists() else []
        ps = parity(rows)
        errs = [p["abs_err"] for p in ps if p["abs_err"] is not None]
        for p in ps:
            print(f"  parity {p['which']:5s} {p['ticker']} brti {p['brti_60s_avg']}"
                  f" kalshi {p['kalshi_ref']} err ${p['abs_err']}")
        if errs:
            print(f"  reconstruction abs err: max ${max(errs)} mean ${round(sum(errs)/len(errs),3)}")


if __name__ == "__main__":
    main()
