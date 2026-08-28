"""Probe V2 order create on demo at $0 balance (1 lot @ $0.01, then cancel)."""
import base64
import os
import time
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

HOST = "https://demo-api.kalshi.co"
KEY_ID = os.environ.get("KALSHI_DEMO_KEY_ID", "")
priv = serialization.load_pem_private_key(
    (Path.home() / ".kalshi_demo.pem").read_bytes(), password=None)


def hdr(method, path):
    ts = str(int(time.time() * 1000))
    sig = priv.sign((ts + method + path).encode(), padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


m = requests.get(HOST + "/trade-api/v2/markets",
                 params={"limit": 1, "status": "open",
                         "series_ticker": "KXBTC15M"}, timeout=8).json()
tk = m["markets"][0]["ticker"]
print("market:", tk)
path = "/trade-api/v2/portfolio/events/orders"
body = {"ticker": tk,
        "client_order_id": f"probe2-{int(time.time())}",
        "side": "bid", "count": "1.00", "price": "0.0100",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": True, "subaccount": 0, "exchange_index": 0}
r = requests.post(HOST + path, json=body, headers=hdr("POST", path),
                  timeout=10)
print("order:", r.status_code, r.text[:400])
