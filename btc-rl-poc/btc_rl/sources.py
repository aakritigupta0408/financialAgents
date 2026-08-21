"""Open, no-auth data streams for BTC (verified reachable 2026-08-19).

Primary: Coinbase Exchange (1m OHLCV). Sentiment: alternative.me Fear & Greed.
Derivatives context: OKX funding rate. Others cataloged in the README.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from . import config

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

_session = requests.Session()
_session.headers["User-Agent"] = "btc-rl-poc/0.1"


def _reset_session() -> None:
    """Rebuild the shared session. After a network blip, requests' keep-alive
    pool can wedge into endless read-timeouts while a fresh session works
    (observed 2026-08-21: 1h of timeouts, fresh session answered in 0.4s)."""
    global _session
    try:
        _session.close()
    except Exception:
        pass
    _session = requests.Session()
    _session.headers["User-Agent"] = "btc-rl-poc/0.1"


def fetch_coinbase_candles(start: datetime, end: datetime) -> list[dict]:
    """Fetch 1m BTC-USD candles [start, end) from Coinbase Exchange (max 300/req).

    Returns bars sorted by time: {"ts", "open", "high", "low", "close", "volume"}.
    """
    url = f"{config.COINBASE_BASE}/products/BTC-USD/candles"
    params = {
        "granularity": 60,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    for attempt in range(4):
        try:
            resp = _session.get(url, params=params, timeout=15)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            _reset_session()
            time.sleep(1.0 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        rows = resp.json()
        bars = [
            {"ts": r[0], "low": r[1], "high": r[2], "open": r[3],
             "close": r[4], "volume": r[5]}
            for r in rows
        ]
        return sorted(bars, key=lambda b: b["ts"])
    raise RuntimeError("Coinbase rate limit persisted after retries")


def fetch_range(start: datetime, end: datetime) -> list[dict]:
    """Fetch an arbitrary 1m-bar range by chunking Coinbase's 300-candle cap."""
    bars: dict[int, dict] = {}
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(hours=4), end)
        for b in fetch_coinbase_candles(cur, nxt):
            bars[b["ts"]] = b
        cur = nxt
        time.sleep(0.12)
    return [bars[ts] for ts in sorted(bars)]


def fetch_day_window(day_pacific: datetime) -> list[dict]:
    """Fetch (and disk-cache) the configured Pacific-time window for one day."""
    cache = CACHE_DIR / f"bars_{day_pacific:%Y-%m-%d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    start = day_pacific.replace(hour=config.DAY_WINDOW_START_HHMM[0],
                                minute=config.DAY_WINDOW_START_HHMM[1],
                                second=0, microsecond=0)
    end = day_pacific.replace(hour=config.DAY_WINDOW_END_HHMM[0],
                              minute=config.DAY_WINDOW_END_HHMM[1],
                              second=0, microsecond=0)
    bars = fetch_coinbase_candles(start, end)
    # Only cache completed days so partial data never poisons the cache.
    now_pacific = datetime.now(tz=config.PACIFIC)
    if end < now_pacific:
        cache.write_text(json.dumps(bars))
    time.sleep(0.15)  # stay far below Coinbase's 10 req/s public limit
    return bars


def fetch_fear_greed(limit: int = 200) -> dict[str, int]:
    """Daily Fear & Greed index keyed by ISO date (UTC)."""
    cache = CACHE_DIR / "fng.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
        return json.loads(cache.read_text())
    resp = _session.get(config.FNG_URL, params={"limit": limit}, timeout=15)
    resp.raise_for_status()
    out = {}
    for row in resp.json()["data"]:
        day = datetime.utcfromtimestamp(int(row["timestamp"])).date().isoformat()
        out[day] = int(row["value"])
    cache.write_text(json.dumps(out))
    return out


def fetch_okx_funding_rate() -> float | None:
    """Current BTC perp funding rate from OKX (live-mode feature)."""
    try:
        resp = _session.get(config.OKX_FUNDING_URL, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["data"][0]["fundingRate"])
    except Exception:
        return None


def fetch_deribit_mark() -> float | None:
    """Deribit BTC-PERPETUAL mark price (for perp-vs-spot basis)."""
    try:
        resp = _session.get("https://www.deribit.com/api/v2/public/ticker"
                            "?instrument_name=BTC-PERPETUAL", timeout=6)
        resp.raise_for_status()
        return float(resp.json()["result"]["mark_price"])
    except Exception:
        return None


def fetch_mempool_fee() -> float | None:
    """mempool.space fastest fee (sat/vB) — on-chain congestion."""
    try:
        resp = _session.get("https://mempool.space/api/v1/fees/recommended",
                            timeout=6)
        resp.raise_for_status()
        return float(resp.json()["fastestFee"])
    except Exception:
        return None


def fetch_book_stats(depth: int = 20) -> dict | None:
    """Coinbase L2 order-book snapshot → bid/ask imbalance and spread (bp)."""
    try:
        resp = _session.get(f"{config.COINBASE_BASE}/products/BTC-USD/book"
                            "?level=2", timeout=8)
        resp.raise_for_status()
        j = resp.json()
        bid_sz = sum(float(b[1]) for b in j["bids"][:depth])
        ask_sz = sum(float(a[1]) for a in j["asks"][:depth])
        best_bid = float(j["bids"][0][0])
        best_ask = float(j["asks"][0][0])
        mid = (best_bid + best_ask) / 2
        return {"imb": (bid_sz - ask_sz) / (bid_sz + ask_sz)
                if bid_sz + ask_sz else 0.0,
                "spread_bp": (best_ask - best_bid) / mid * 1e4}
    except Exception:
        return None


def fetch_recent_trades(limit: int = 1000) -> list[dict]:
    """Recent Coinbase BTC-USD trades with the TAKER side resolved.

    Coinbase's `side` field is the MAKER's side, so side=="sell" means the
    resting order was a sell — i.e., the aggressor was a BUYER (up-tick).
    Getting this backwards inverts the order-flow signal.
    """
    try:
        resp = _session.get(f"{config.COINBASE_BASE}/products/BTC-USD/trades",
                            params={"limit": limit}, timeout=8)
        resp.raise_for_status()
        out = []
        for t in resp.json():
            ts = datetime.fromisoformat(
                t["time"].replace("Z", "+00:00")).timestamp()
            out.append({"id": t["trade_id"], "ts": ts,
                        "size": float(t["size"]),
                        "taker_buy": t["side"] == "sell"})
        return out
    except Exception:
        return []


def fetch_kalshi_btc15() -> dict | None:
    """Current Robinhood/Kalshi 'BTC price up in next 15 mins?' market.

    Public no-auth endpoint; these contracts settle on 60-second averages of
    CF Benchmarks' BRTI — the index our composite approximates. Prices are
    in cents (probability x 100).
    """
    try:
        resp = _session.get(
            "https://api.elections.kalshi.com/trade-api/v2/markets",
            params={"limit": 1, "status": "open",
                    "series_ticker": "KXBTC15M"}, timeout=8)
        resp.raise_for_status()
        ms = resp.json().get("markets") or []
        if not ms:
            return None
        m = ms[0]
        yes_bid, yes_ask = m.get("yes_bid"), m.get("yes_ask")
        if not yes_bid or not yes_ask:
            # markets endpoint often carries no quotes — derive the touch
            # from the orderbook (best yes bid; best no bid => 100 - yes ask)
            try:
                raw = _session.get(
                    "https://api.elections.kalshi.com/trade-api/v2/markets/"
                    f"{m['ticker']}/orderbook",
                    params={"depth": 1}, timeout=8).json()
                book = raw.get("orderbook") or raw.get("orderbook_fp") or {}
                if "yes_dollars" in book or "no_dollars" in book:
                    yes_lvls = [float(p) * 100 for p, _ in
                                book.get("yes_dollars") or []]
                    no_lvls = [float(p) * 100 for p, _ in
                               book.get("no_dollars") or []]
                else:  # legacy shape: integer-cent [price, qty] levels
                    yes_lvls = [lvl[0] for lvl in book.get("yes") or []]
                    no_lvls = [lvl[0] for lvl in book.get("no") or []]
                # resting YES buys: best bid = highest; resting NO buys
                # imply the YES ask at 100 - best no bid
                if yes_lvls:
                    yes_bid = yes_bid or round(max(yes_lvls), 1)
                if no_lvls:
                    yes_ask = yes_ask or round(100 - max(no_lvls), 1)
            except Exception:
                pass
        return {"ticker": m["ticker"], "title": m.get("title"),
                "strike": m.get("floor_strike"),
                "close_time": m.get("close_time"),
                "yes_bid": yes_bid, "yes_ask": yes_ask,
                "last_price": m.get("last_price")}
    except Exception:
        return None


def fetch_brti_composite() -> dict | None:
    """BRTI-style live BTC price: volume-weighted across CME CF BRTI
    constituent exchanges that expose open, no-auth tickers.

    The real BRTI (CF Benchmarks / CME) is licensed; this is an honest
    open approximation from 4 of its 6 constituents (no itBit/LMAX).
    """
    tickers = {
        "coinbase": ("https://api.exchange.coinbase.com/products/BTC-USD/ticker",
                     lambda j: (float(j["price"]), float(j["volume"]))),
        "kraken": ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
                   lambda j: (float(j["result"]["XXBTZUSD"]["c"][0]),
                              float(j["result"]["XXBTZUSD"]["v"][1]))),
        "bitstamp": ("https://www.bitstamp.net/api/v2/ticker/btcusd/",
                     lambda j: (float(j["last"]), float(j["volume"]))),
        "gemini": ("https://api.gemini.com/v1/pubticker/btcusd",
                   lambda j: (float(j["last"]), float(j["volume"]["BTC"]))),
    }
    quotes: dict[str, tuple[float, float]] = {}
    for name, (url, parse) in tickers.items():
        try:
            resp = _session.get(url, timeout=6)
            resp.raise_for_status()
            quotes[name] = parse(resp.json())
        except Exception:
            continue
    if not quotes:
        return None
    total_vol = sum(v for _, v in quotes.values())
    price = sum(p * v for p, v in quotes.values()) / total_vol
    return {"price": round(price, 2),
            "constituents": {n: p for n, (p, _) in quotes.items()},
            "method": "volume-weighted composite of open CME CF BRTI constituents"}


def fetch_history(days: int) -> dict[str, list[dict]]:
    """Fetch the daily windows for the last `days` completed days."""
    today = datetime.now(tz=config.PACIFIC).replace(hour=0, minute=0, second=0,
                                                    microsecond=0)
    out: dict[str, list[dict]] = {}
    for i in range(days, 0, -1):
        day = today - timedelta(days=i)
        try:
            bars = fetch_day_window(day)
        except Exception as exc:  # one bad day shouldn't sink the run
            print(f"  skip {day:%Y-%m-%d}: {exc}")
            continue
        if bars:
            out[f"{day:%Y-%m-%d}"] = bars
    return out
