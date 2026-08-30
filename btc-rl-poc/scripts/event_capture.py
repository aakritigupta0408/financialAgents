"""Layer A of the Alpha Capture Engine (TA architecture, adopted
2026-08-29): EVENT-LEVEL point-in-time capture. P0 — every other
layer is blocked on this data existing.

Streams:
  * Coinbase Exchange WebSocket — matches (trades) + ticker (L1
    best bid/ask + last) for BTC-USD; exchange_ts from the venue,
    receive_ts stamped at socket read, persist_ts at write.
  * Kalshi active-window orderbook — REST poll every POLL_S seconds
    (the venue exposes no public stream to us): yes/no bid/ask +
    top depth for the active KXBTC15M ticker, discovered every
    cycle from the markets endpoint.

Timestamps (master schema §4): exchange_ts, receive_ts, persist_ts.
(feature_asof_ts / decision_ts belong to the consumer layers.)

Output: results/events/YYYYMMDD_HH.jsonl (hour-sharded append-only;
one line per event: {src, kind, exchange_ts, receive_ts, persist_ts,
...payload}). A status heartbeat goes to results/event_capture.json
every 10s so the meta-monitor can watch this process like any other.

Honest limits, recorded: Kalshi granularity = poll cadence (~1s),
not true event time; Coinbase L2 depth deltas are NOT subscribed in
v1 (bandwidth/disk discipline — L1 + trades first; L2 is the first
upgrade once lead-lag half-life justifies it).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "events"
OUT.mkdir(exist_ok=True)
STATUS = ROOT / "results" / "event_capture.json"
POLL_S = 1.0
WS_HOST = "ws-feed.exchange.coinbase.com"
KALSHI = ("https://api.elections.kalshi.com/trade-api/v2/markets"
          "?series_ticker=KXBTC15M&status=open&limit=5")

_lock = threading.Lock()
_counts = {"cb_trade": 0, "cb_ticker": 0, "k_quote": 0, "errors": 0}


def _shard():
    return OUT / (datetime.now(timezone.utc).strftime("%Y%m%d_%H")
                  + ".jsonl")


def emit(rec):
    rec["persist_ts"] = time.time()
    with _lock, _shard().open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def heartbeat():
    while True:
        try:
            STATUS.write_text(json.dumps(
                {"alive_at": time.time(), **_counts}))
        except OSError:
            pass
        time.sleep(10)


def coinbase_loop():
    import websocket                     # websocket-client 1.8.0
    while True:
        try:
            ws = websocket.create_connection(
                f"wss://{WS_HOST}", timeout=20)
            ws.send(json.dumps({"type": "subscribe",
                                "product_ids": ["BTC-USD"],
                                "channels": ["matches", "ticker"]}))
            while True:
                msg = ws.recv()
                rt = time.time()
                try:
                    m = json.loads(msg)
                except (ValueError, TypeError):
                    continue
                t = m.get("type")
                if t == "match":
                    _counts["cb_trade"] += 1
                    emit({"src": "coinbase", "kind": "trade",
                          "exchange_ts": m.get("time"),
                          "receive_ts": rt,
                          "price": m.get("price"),
                          "size": m.get("size"),
                          "side": m.get("side")})
                elif t == "ticker":
                    _counts["cb_ticker"] += 1
                    emit({"src": "coinbase", "kind": "l1",
                          "exchange_ts": m.get("time"),
                          "receive_ts": rt,
                          "bid": m.get("best_bid"),
                          "ask": m.get("best_ask"),
                          "bid_sz": m.get("best_bid_size"),
                          "ask_sz": m.get("best_ask_size"),
                          "last": m.get("price")})
        except Exception:
            _counts["errors"] += 1
            time.sleep(3)


def kalshi_loop():
    while True:
        rt = time.time()
        try:
            with urllib.request.urlopen(KALSHI, timeout=5) as r:
                mkts = json.loads(r.read()).get("markets", [])
            def cents(m, k):
                v = m.get(k)
                try:
                    return round(float(v) * 100, 1) \
                        if v is not None else None
                except (TypeError, ValueError):
                    return None
            for m in mkts:
                emit({"src": "kalshi", "kind": "quote",
                      "exchange_ts": None,   # REST: no event time
                      "receive_ts": rt,
                      "ticker": m.get("ticker"),
                      # 2026 API: prices arrive as *_dollars strings
                      "yes_bid": cents(m, "yes_bid_dollars"),
                      "yes_ask": cents(m, "yes_ask_dollars"),
                      "no_bid": cents(m, "no_bid_dollars"),
                      "no_ask": cents(m, "no_ask_dollars"),
                      "yes_bid_sz": m.get("yes_bid_size_fp"),
                      "yes_ask_sz": m.get("yes_ask_size_fp"),
                      "last": cents(m, "last_price_dollars"),
                      "volume": m.get("volume"),
                      "close_time": m.get("close_time"),
                      "strike": (m.get("floor_strike")
                                 or m.get("cap_strike"))})
                _counts["k_quote"] += 1
        except Exception:
            _counts["errors"] += 1
        time.sleep(max(0.0, POLL_S - (time.time() - rt)))


if __name__ == "__main__":
    threading.Thread(target=heartbeat, daemon=True).start()
    threading.Thread(target=kalshi_loop, daemon=True).start()
    coinbase_loop()
