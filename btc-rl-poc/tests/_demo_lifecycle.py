"""Prove the full order lifecycle on a demo-supported market: pick a
liquid baseball total, place a small marketable bid, read it back,
then cancel any remainder. Real sandbox money only."""
import base64
import json
import time
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

HOST = "https://external-api.demo.kalshi.co"
KEY_ID = "5548302b-00ab-4f99-ba61-aa4f1169b2e9"
priv = serialization.load_pem_private_key(
    (Path.home() / ".kalshi_demo.pem").read_bytes(), password=None)


def hdr(m, p):
    ts = str(int(time.time() * 1000))
    sig = priv.sign((ts + m + p).encode(), padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


# find a baseball market with a real ask to cross
ms = requests.get(HOST + "/trade-api/v2/markets",
                  params={"limit": 60, "status": "open",
                          "series_ticker": "KXMLBTOTAL"},
                  timeout=8).json().get("markets", [])
pick = None
for m in ms:
    if m.get("yes_ask") and 5 <= m["yes_ask"] <= 95:
        pick = m
        break
if not pick:
    print("no liquid baseball market; series list:",
          [m["ticker"] for m in ms[:3]])
    raise SystemExit
tk, ask = pick["ticker"], pick["yes_ask"]
print(f"market {tk} · yes_ask {ask}c")

op = "/trade-api/v2/portfolio/events/orders"
px = min(0.99, ask / 100 + 0.02)              # cross to fill 1 lot
body = {"ticker": tk, "client_order_id": f"life-{int(time.time()*1000) % 1000000}",
        "side": "bid", "count": "1.00", "price": f"{px:.4f}",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": False, "subaccount": 0, "exchange_index": 0}
o = requests.post(HOST + op, json=body, headers=hdr("POST", op),
                  timeout=10)
print("PLACE:", o.status_code, o.text[:220])
oid = None
try:
    oid = o.json().get("order", {}).get("order_id")
except Exception:
    pass
time.sleep(1.5)
pp = "/trade-api/v2/portfolio/positions"
pos = requests.get(HOST + pp, headers=hdr("GET", pp), timeout=8)
print("POSITIONS:", pos.status_code, pos.text[:220])
bp = "/trade-api/v2/portfolio/balance"
b = requests.get(HOST + bp, headers=hdr("GET", bp), timeout=8)
print("BALANCE:", b.json().get("balance"))
