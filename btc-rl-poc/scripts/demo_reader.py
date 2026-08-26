"""Read-only Kalshi DEMO account reader — logs real fills/positions.

Polls the authenticated demo account for fills and positions and
appends new fills to results/demo_fills.jsonl. Read-only: places no
orders, moves no money. Proves the account-integration round trip when
a trade is placed in the demo UI.

Run:  KALSHI_DEMO_KEY_ID=<id> python3 scripts/demo_reader.py
"""
import base64
import datetime
import json
import os
import time
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

HOST = "https://demo-api.kalshi.co"          # demo only
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "demo_fills.jsonl"
KEY_ID = os.environ.get("KALSHI_DEMO_KEY_ID", "")
priv = serialization.load_pem_private_key(
    (Path.home() / ".kalshi_demo.pem").read_bytes(), password=None)


def hdr(method, path):
    ts = str(int(time.time() * 1000))
    sig = priv.sign((ts + method + path).encode(), padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


def main():
    print(f"demo reader up · read-only · host {HOST}")
    seen = set()
    if OUT.exists():
        for l in OUT.open():
            try:
                seen.add(json.loads(l).get("trade_id"))
            except Exception:
                pass
    while True:
        try:
            p = "/trade-api/v2/portfolio/fills"
            r = requests.get(HOST + p, headers=hdr("GET", p),
                             params={"limit": 50}, timeout=10)
            fills = r.json().get("fills", []) if r.status_code == 200 else []
            fresh = [f for f in fills if f.get("trade_id") not in seen]
            for f in reversed(fresh):
                seen.add(f.get("trade_id"))
                with OUT.open("a") as fh:
                    fh.write(json.dumps({"read_ts": int(time.time()),
                                         **f}) + "\n")
                print(datetime.datetime.now().strftime("%H:%M:%S"),
                      "FILL", f.get("ticker"), f.get("side"),
                      f.get("count"), "@", f.get("yes_price"))
            bp = "/trade-api/v2/portfolio/balance"
            b = requests.get(HOST + bp, headers=hdr("GET", bp), timeout=8)
            if fresh:
                print("  balance now:", b.json().get("balance"))
        except Exception as e:
            print("poll error:", str(e)[:120])
        time.sleep(8)


if __name__ == "__main__":
    main()
