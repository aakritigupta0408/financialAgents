"""Kalshi DEMO-exchange mirror for Sagemon (pt3) — zero real money.

Watches results/pt3_trades.jsonl and re-places each fresh paper entry
as an order on Kalshi's DEMO environment (fake balance, real order
lifecycle). Separate process: the live daemon is never touched.

SAFETY, by construction:
  - the host is HARD-CODED to the demo API; no production URL exists
    in this file, so production cannot be reached even by mistake
  - demo credentials only: env KALSHI_DEMO_KEY_ID + a PEM at
    ~/.kalshi_demo.pem (never in the repo)
  - without credentials it runs in DRY-RUN and only logs intentions
  - one order per window, capped at DEMO_MAX_CONTRACTS

Setup (one-time, human): create an account at https://demo.kalshi.co,
Profile -> API keys -> create, save the private key to
~/.kalshi_demo.pem (chmod 600), export KALSHI_DEMO_KEY_ID=<key id>.
Run:  python3 scripts/demo_trader.py
"""
import base64
import datetime
import json
import os
import time
from pathlib import Path

import requests

DEMO_HOST = "https://external-api.demo.kalshi.co"  # demo ONLY — by design
# (new unified engine host; the old demo-api host now 410s orders)
ROOT = Path(__file__).resolve().parent.parent
PT3 = ROOT / "results" / "pt3_trades.jsonl"
OUT = ROOT / "results" / "demo_orders.jsonl"
PEM = Path.home() / ".kalshi_demo.pem"
KEY_ID = os.environ.get("KALSHI_DEMO_KEY_ID", "")
FRESH_S = 300
DEMO_MAX_CONTRACTS = 500
POLL_S = 20

LIVE = bool(KEY_ID) and PEM.exists()
_priv = None
if LIVE:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    _priv = serialization.load_pem_private_key(
        PEM.read_bytes(), password=None)


def _sign(method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    msg = (ts + method + path).encode()
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    sig = _priv.sign(msg, padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


def place(trade: dict) -> dict:
    """V2 unified book: buying YES at q == bid at q; buying NO at q ==
    ask (sell yes) at 1-q. Prices are 4-decimal dollar strings."""
    path = "/trade-api/v2/portfolio/events/orders"
    yes_side = trade["side"] == "yes"
    px = trade["ask_c"] / 100.0 if yes_side \
        else (100.0 - trade["ask_c"]) / 100.0
    body = {
        "ticker": trade["ticker"],
        "client_order_id":
            f"sagemon-{trade['ticker']}-{trade['made_ts']}",
        "side": "bid" if yes_side else "ask",
        "count": f"{min(int(trade['contracts']), DEMO_MAX_CONTRACTS)}.00",
        "price": f"{px:.4f}",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": False,
        "subaccount": 0,
        "exchange_index": 0,
    }
    if not LIVE:
        return {"dry_run": True, "would_send": body}
    r = requests.post(DEMO_HOST + path, json=body,
                      headers=_sign("POST", path), timeout=10)
    return {"status": r.status_code, "resp": r.json()
            if r.headers.get("content-type", "").startswith(
                "application/json") else r.text[:200]}


def main() -> None:
    print(f"demo mirror up · {'LIVE on DEMO exchange' if LIVE else 'DRY-RUN (no credentials)'}"
          f" · host {DEMO_HOST}")
    seen = set()
    if OUT.exists():
        for l in OUT.open():
            try:
                seen.add(json.loads(l)["client_order_id"])
            except Exception:
                pass
    while True:
        try:
            trades = [json.loads(l) for l in PT3.open()] \
                if PT3.exists() else []
            now = time.time()
            for t in trades:
                coid = f"sagemon-{t['ticker']}-{t['made_ts']}"
                if coid in seen or t.get("actual") is not None \
                        or now - t["made_ts"] > FRESH_S:
                    continue
                res = place(t)
                seen.add(coid)
                rec = {"ts": int(now), "client_order_id": coid,
                       "ticker": t["ticker"], "side": t["side"],
                       "count": min(int(t["contracts"]),
                                    DEMO_MAX_CONTRACTS),
                       "price_c": t["ask_c"], "result": res}
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(datetime.datetime.now().strftime("%H:%M:%S"),
                      "mirrored", coid, "->",
                      res.get("status", "dry_run"))
        except Exception as e:
            print("poll error:", str(e)[:120])
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
