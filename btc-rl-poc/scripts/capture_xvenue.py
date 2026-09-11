"""F-XVENUE capture — CAPTURE_ONLY (PM 08-30).

Prospective raw recording of free cross-venue BTC trade tape:
Binance (btcusdt@trade) and OKX (BTC-USDT trades). This feed touches
NOTHING live — no model, no policy, no experiment reads it. It exists
so that if the Model Researcher's diagnosis ever implicates missing
cross-venue information, the historical tape already exists.
(F-MICRO — Coinbase trades + L1 + Kalshi quotes — is already captured
by scripts/event_capture.py.)

Shards: results/events_xvenue/xvenue-YYYYMMDD-HH.jsonl
Row:    {src, ts_recv, ts_event, px, qty, side}
Heartbeat: results/xvenue_capture.json (capture_watchdog restarts on
stale, same discipline as the primary tape).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "events_xvenue"
HB = ROOT / "results" / "xvenue_capture.json"
OUT.mkdir(exist_ok=True)

_lock = threading.Lock()
_counts = {"binance": 0, "okx": 0, "kraken": 0, "errors": 0}


def _write(row):
    shard = OUT / time.strftime("xvenue-%Y%m%d-%H.jsonl",
                                time.gmtime())
    with _lock:
        with shard.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _heartbeat():
    while True:
        tmp = HB.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"alive_at": time.time(), **_counts}))
        tmp.replace(HB)
        time.sleep(10)


def _binance():
    def on_msg(ws, msg):
        try:
            d = json.loads(msg)
            if d.get("e") == "trade":
                _counts["binance"] += 1
                _write({"src": "binance",
                        "ts_recv": round(time.time(), 3),
                        "ts_event": d["T"] / 1000.0,
                        "px": float(d["p"]), "qty": float(d["q"]),
                        "side": "sell" if d.get("m") else "buy"})
        except Exception:
            _counts["errors"] += 1
    while True:
        try:
            # data-stream.binance.vision = official market-data-only
            # mirror; stream.binance.com returns 451 (geo-restricted)
            # from this network
            ws = websocket.WebSocketApp(
                "wss://data-stream.binance.vision:9443/ws/"
                "btcusdt@trade",
                on_message=on_msg)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception:
            _counts["errors"] += 1
        time.sleep(5)


def _okx():
    sub = json.dumps({"op": "subscribe",
                      "args": [{"channel": "trades",
                                "instId": "BTC-USDT"}]})

    def on_open(ws):
        ws.send(sub)

    def on_msg(ws, msg):
        try:
            d = json.loads(msg)
            for t in d.get("data") or []:
                _counts["okx"] += 1
                _write({"src": "okx",
                        "ts_recv": round(time.time(), 3),
                        "ts_event": int(t["ts"]) / 1000.0,
                        "px": float(t["px"]), "qty": float(t["sz"]),
                        "side": t.get("side")})
        except Exception:
            _counts["errors"] += 1
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://ws.okx.com:8443/ws/v5/public",
                on_open=on_open, on_message=on_msg)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception:
            _counts["errors"] += 1
        time.sleep(5)


def _kraken():
    sub = json.dumps({"event": "subscribe",
                      "pair": ["XBT/USD"],
                      "subscription": {"name": "trade"}})

    def on_open(ws):
        ws.send(sub)

    def on_msg(ws, msg):
        try:
            d = json.loads(msg)
            if isinstance(d, list) and len(d) >= 4 \
                    and d[-2] == "trade":
                for t in d[1]:
                    _counts["kraken"] += 1
                    _write({"src": "kraken",
                            "ts_recv": round(time.time(), 3),
                            "ts_event": float(t[2]),
                            "px": float(t[0]), "qty": float(t[1]),
                            "side": "buy" if t[3] == "b" else "sell"})
        except Exception:
            _counts["errors"] += 1
    while True:
        try:
            ws = websocket.WebSocketApp("wss://ws.kraken.com",
                                        on_open=on_open,
                                        on_message=on_msg)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception:
            _counts["errors"] += 1
        time.sleep(5)


if __name__ == "__main__":
    threading.Thread(target=_heartbeat, daemon=True).start()
    threading.Thread(target=_binance, daemon=True).start()
    threading.Thread(target=_kraken, daemon=True).start()
    _okx()
