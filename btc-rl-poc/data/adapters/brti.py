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

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "brti_capture.jsonl"
HEALTH = ROOT / "results" / "brti_health.json"
KEY_ID = os.environ.get("KALSHI_DEMO_KEY_ID", "")
PEM = Path.home() / ".kalshi_demo.pem"
HOSTS = ["https://api.elections.kalshi.com", "https://demo-api.kalshi.co"]
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "Chrome/124.0 Safari/537.36")

_priv = None


def _key():
    global _priv
    if _priv is None:
        from cryptography.hazmat.primitives import serialization
        _priv = serialization.load_pem_private_key(PEM.read_bytes(), password=None)
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


def parity(ticker_rows):
    """For a few settled windows, pull BRTI history around open-1min and
    close-1min, average 60 values, compare to floor_strike / expiration_value."""
    out = []
    for r in ticker_rows:
        # history endpoint params are documented as id + time range; exact param
        # names verified live once authorized.
        for label, t_iso, ref in (("open", r["open_time"], r["floor_strike"]),
                                   ("close", r["close_time"], r["expiration_value"])):
            code, body = _req(HOSTS[0], "/trade-api/v2/cfbenchmarks/history/values",
                              {"id": "BRTI", "end_ts": t_iso, "limit": 60})
            vals = [float(v.get("value")) for v in (body.get("values") or body.get("history") or [])
                    if v.get("value") is not None]
            avg = round(sum(vals) / len(vals), 2) if vals else None
            out.append({"ticker": r["ticker"], "which": label, "n": len(vals),
                        "brti_60s_avg": avg, "kalshi_ref": ref,
                        "match": (avg is not None and abs(avg - ref) < 0.01)})
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
        # parity on a few known windows
        co = ROOT / "results" / "contract_outcomes.jsonl"
        rows = [json.loads(l) for l in co.open() if l.strip()][-3:] if co.exists() else []
        for p in parity(rows):
            print("  parity", p["which"], p["ticker"], "brti", p["brti_60s_avg"],
                  "kalshi", p["kalshi_ref"], "match" if p["match"] else "MISMATCH")


if __name__ == "__main__":
    main()
