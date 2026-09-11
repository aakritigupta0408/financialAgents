"""F-MICRO deep capture — CAPTURE_ONLY (feature program Phase 1).

Fills the two P0 gaps the registry declared:
  * Coinbase L2: public level2_batch feed (50ms batched deltas +
    snapshot per connect) for BTC-USD.
  * Kalshi microstructure: full orderbook (public REST, polled 1s,
    stored only on change) + trades (public REST, min_ts cursor) for
    the ACTIVE KXBTC15M window — this is the "how and when did
    Kalshi absorb the move" side of the lead/lag research target.

Nothing live reads this. Shards: results/events_micro/
micro-YYYYMMDD-HH.jsonl · heartbeat results/micro_capture.json
(capture_watchdog restarts on stale).

Coinbase trades already ride the primary tape (event_capture.py);
not duplicated here.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request

from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "events_micro"
HB = ROOT / "results" / "micro_capture.json"
OUT.mkdir(exist_ok=True)

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"

_lock = threading.Lock()
_counts = {"cb_l2": 0, "k_book": 0, "k_trade": 0, "errors": 0}


def _write(row):
    shard = OUT / time.strftime("micro-%Y%m%d-%H.jsonl", time.gmtime())
    with _lock:
        with shard.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _heartbeat():
    while True:
        tmp = HB.with_suffix(".tmp")
        tmp.write_text(json.dumps({"alive_at": time.time(),
                                   **_counts}))
        tmp.replace(HB)
        time.sleep(10)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-rl"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read())


def _cb_l2():
    sub = json.dumps({"type": "subscribe",
                      "product_ids": ["BTC-USD"],
                      "channels": ["level2_batch"]})

    def on_open(ws):
        ws.send(sub)

    def on_msg(ws, msg):
        try:
            d = json.loads(msg)
            t = d.get("type")
            if t == "snapshot":
                _counts["cb_l2"] += 1
                _write({"src": "cb_l2", "type": "snapshot",
                        "ts_recv": round(time.time(), 3),
                        "bids": d["bids"][:25], "asks": d["asks"][:25]})
            elif t == "l2update":
                _counts["cb_l2"] += 1
                _write({"src": "cb_l2", "type": "update",
                        "ts_recv": round(time.time(), 3),
                        "ts_event": d.get("time"),
                        "changes": d.get("changes")})
        except Exception:
            _counts["errors"] += 1
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://ws-feed.exchange.coinbase.com",
                on_open=on_open, on_message=on_msg)
            ws.run_forever(ping_interval=25, ping_timeout=10)
        except Exception:
            _counts["errors"] += 1
        time.sleep(5)


def _active_ticker():
    try:
        d = _get(f"{KALSHI}/markets?limit=5&series_ticker={SERIES}"
                 "&status=open")
        ms = d.get("markets") or []
        ms.sort(key=lambda m: m.get("close_time") or "")
        return ms[0]["ticker"] if ms else None
    except Exception:
        _counts["errors"] += 1
        return None


def _kalshi():
    tk, tk_at = None, 0.0
    last_book_hash = None
    trade_cursor = int(time.time())
    while True:
        now = time.time()
        if now - tk_at > 60 or tk is None:
            new = _active_ticker()
            if new and new != tk:
                tk, last_book_hash = new, None
            tk_at = now
        if tk is None:
            time.sleep(5)
            continue
        try:
            ob = _get(f"{KALSHI}/markets/{tk}/orderbook")
            book = ob.get("orderbook") or {}
            h = hashlib.sha256(json.dumps(
                book, sort_keys=True).encode()).hexdigest()[:12]
            if h != last_book_hash:      # store only on change
                last_book_hash = h
                _counts["k_book"] += 1
                _write({"src": "k_book", "ticker": tk,
                        "ts_recv": round(time.time(), 3),
                        "book": book, "hash": h})
        except Exception:
            _counts["errors"] += 1
        try:
            tr = _get(f"{KALSHI}/markets/trades?ticker={tk}"
                      f"&min_ts={trade_cursor}&limit=100")
            trades = tr.get("trades") or []
            if trades:
                for t in trades:
                    _counts["k_trade"] += 1
                    _write({"src": "k_trade", "ticker": tk,
                            "ts_recv": round(time.time(), 3),
                            "trade": t})
                trade_cursor = max(
                    trade_cursor,
                    max(int(time.mktime(time.strptime(
                        t["created_time"][:19],
                        "%Y-%m-%dT%H:%M:%S")))
                        for t in trades if t.get("created_time")))
        except Exception:
            _counts["errors"] += 1
        time.sleep(1.0)


if __name__ == "__main__":
    threading.Thread(target=_heartbeat, daemon=True).start()
    threading.Thread(target=_cb_l2, daemon=True).start()
    _kalshi()
