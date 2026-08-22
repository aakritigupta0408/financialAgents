"""Mine weeks of TRUE Kalshi KXBTC15M history: per-minute yes_bid/yes_ask
candlesticks for expired contracts + strike + settlement, joined with
Coinbase 1-min bars for path/momentum features.

Outputs
  results/kalshi_history.jsonl  — one row per (window, minute):
    ticker, ts, mins_left, strike, outcome, yes_bid_c, yes_ask_c,
    price_c (last), btc_close, z (above-strike sigma), pf (path feats)
  results/best_bids.jsonl       — one row per window: the hindsight-optimal
    entry at TRUE ask (side, minute, price, net) = hard positives for the
    mandatory bidder's entry policy.

Read-only vs the live system: new files only, public no-auth endpoints,
throttled. Usage: python3 tests/kalshi_history_miner.py [days]
"""
import calendar
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.elections.kalshi.com/trade-api/v2"
CB = "https://api.exchange.coinbase.com"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
THROTTLE = 0.13


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "btc-rl-poc"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def fetch_settled_markets(since_ts: int) -> list[dict]:
    out, cursor = [], ""
    while True:
        url = (f"{API}/markets?series_ticker=KXBTC15M&status=settled"
               f"&limit=200" + (f"&cursor={cursor}" if cursor else ""))
        d = get(url)
        batch = d.get("markets", [])
        for m in batch:
            try:
                close_ts = calendar.timegm(time.strptime(
                    m["close_time"], "%Y-%m-%dT%H:%M:%SZ"))
                if (m.get("floor_strike") and m.get("expiration_value")
                        and close_ts >= since_ts
                        and close_ts <= time.time()):
                    out.append({"ticker": m["ticker"], "close_ts": close_ts,
                                "strike": float(m["floor_strike"]),
                                "settle": float(m["expiration_value"])})
            except Exception:
                continue
        cursor = d.get("cursor")
        time.sleep(THROTTLE)
        if not cursor or not batch:
            break
        # markets come newest-first; stop paging once past the horizon
        oldest = min((x["close_ts"] for x in out), default=None)
        if oldest is not None and oldest < since_ts + 60:
            break
    return out


def fetch_candles(ticker: str, open_ts: int, close_ts: int) -> list[dict]:
    url = (f"{API}/series/KXBTC15M/markets/{ticker}/candlesticks"
           f"?start_ts={open_ts}&end_ts={close_ts}&period_interval=1")
    try:
        return get(url).get("candlesticks", [])
    except Exception:
        return []


def fetch_btc_bars(start_ts: int, end_ts: int) -> dict[int, float]:
    """Coinbase 1-min closes keyed by bucket start, chunked <=300 bars."""
    closes = {}
    t = start_ts
    while t < end_ts:
        t2 = min(t + 300 * 60, end_ts)
        url = (f"{CB}/products/BTC-USD/candles?granularity=60"
               f"&start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))}"
               f"&end={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t2))}")
        try:
            for r in get(url):
                closes[int(r[0])] = float(r[4])
        except Exception:
            pass
        t = t2
        time.sleep(THROTTLE)
    return closes


def fee_c(price_c: float) -> float:
    p = price_c / 100.0
    return float(math.ceil(7.0 * p * (1.0 - p)))


def main() -> None:
    since = int(time.time()) - DAYS * 86400
    print(f"fetching settled KXBTC15M markets, last {DAYS} days …")
    mkts = fetch_settled_markets(since)
    mkts.sort(key=lambda m: m["close_ts"])
    print(f"  {len(mkts)} settled windows")
    if not mkts:
        return
    print("fetching BTC 1-min bars for the whole span …")
    bars = fetch_btc_bars(mkts[0]["close_ts"] - 1200,
                          mkts[-1]["close_ts"] + 60)
    print(f"  {len(bars)} bars")

    hist, best = [], []
    for i, m in enumerate(mkts):
        open_ts = m["close_ts"] - 900
        outcome = int(m["settle"] >= m["strike"])
        candles = fetch_candles(m["ticker"], open_ts, m["close_ts"])
        if not candles:
            continue
        win_rows = []
        for c in candles:
            ts = int(c["end_period_ts"])
            if not open_ts < ts <= m["close_ts"]:
                continue
            try:
                yb = float(c["yes_bid"]["close_dollars"]) * 100
                ya = float(c["yes_ask"]["close_dollars"]) * 100
                px = float(c["price"]["close_dollars"]) * 100
            except (KeyError, TypeError):
                continue
            btc = bars.get(ts - 60)
            # path features from bars inside this window so far
            seg = [bars[t] for t in range(open_ts, ts, 60) if t in bars]
            if seg:
                above = sum(1 for v in seg if v >= m["strike"]) / len(seg)
                signs = [v >= m["strike"] for v in seg]
                cross = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
                drift = seg[-1] - seg[-4] if len(seg) >= 4 else 0.0
            else:
                above, cross, drift = 0.5, 0, 0.0
            win_rows.append({
                "ticker": m["ticker"], "ts": ts,
                "mins_left": round((m["close_ts"] - ts) / 60, 1),
                "strike": m["strike"], "outcome": outcome,
                "yes_bid_c": round(yb, 1), "yes_ask_c": round(ya, 1),
                "price_c": round(px, 1),
                "btc_close": btc,
                "pf": [round(above - 0.5, 4), min(cross, 4) / 4,
                       round(drift, 2)],
            })
        if not win_rows:
            continue
        hist.extend(win_rows)
        # hindsight-optimal entry at TRUE ask: winning side, cheapest ask
        side = "yes" if outcome else "no"
        cands = []
        for r in win_rows:
            ask = r["yes_ask_c"] if side == "yes" else 100 - r["yes_bid_c"]
            if 1.0 <= ask < 80.0:
                cands.append((ask, r))
        if cands:
            ask, r = min(cands, key=lambda t: t[0])
            best.append({"ticker": m["ticker"], "side": side,
                         "price_c": round(ask, 1),
                         "net_c": round(100 - ask - fee_c(ask), 1),
                         "mins_left": r["mins_left"], "ts": r["ts"],
                         "strike": m["strike"], "outcome": outcome,
                         "pf": r["pf"]})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(mkts)} windows …")

    (ROOT / "results" / "kalshi_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in hist))
    (ROOT / "results" / "best_bids.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in best))
    n_win = len({r['ticker'] for r in hist})
    print(f"wrote {len(hist)} minute-rows across {n_win} windows "
          f"-> results/kalshi_history.jsonl")
    print(f"wrote {len(best)} hindsight-optimal entries "
          f"-> results/best_bids.jsonl")
    if best:
        med = sorted(b["mins_left"] for b in best)[len(best) // 2]
        avg = sum(b["net_c"] for b in best) / len(best)
        print(f"optimal entries: median {med:.0f} min left, "
              f"avg net +{avg:.0f}c at TRUE ask")


if __name__ == "__main__":
    main()
