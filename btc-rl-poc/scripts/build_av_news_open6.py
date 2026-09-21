"""Build strict-PIT AV NEWS_SENTIMENT features at t_decide=open+6min for the
open6 BTC windows.

News store = AV NEWS_SENTIMENT for CRYPTO:BTC (time_published is UTC), fetched
the same way as scripts/news_backfill.py::fetch_news (dedup syndicated copies by
title, keep earliest publication). We extend the fetch span to cover the open6
window range plus a 24h lookback before the earliest window.

For each window in results/open6_dataset.jsonl we take t_decide = row['ts']
(already open+6min, epoch UTC) and compute features using ONLY articles with
published_epoch <= t_decide. Nothing from the future.
"""
import json
import math
import statistics
import urllib.parse
import urllib.request
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = (Path.home() / ".alphavantage_key").read_text().strip()
BASE = "https://www.alphavantage.co/query"
DS = ROOT / "results" / "open6_dataset.jsonl"
OUT = ROOT / "results" / "av_news_open6.jsonl"
STORE = ROOT / "results" / "av_news_store.jsonl"

# Fetch span: earliest window is 2026-08-30 08:06 UTC; go back to Aug 28 for a
# full trailing-24h lookback, forward to Sep 16 to cover the last window.
MONTHS = [("20260828T0000", "20260901T0000"),
          ("20260901T0000", "20260908T0000"),
          ("20260908T0000", "20260916T0000")]
KH = [1, 4, 24]  # trailing-window hours for count/mean features


def _call(p):
    p["apikey"] = KEY
    with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(p), timeout=45) as r:
        return json.load(r)


def _epoch(s):  # "YYYYMMDDTHHMMSS" (UTC)
    return int(datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp())


def fetch_news():
    by_title = {}
    info = []
    for tf, tt in MONTHS:
        try:
            d = _call({"function": "NEWS_SENTIMENT", "tickers": "CRYPTO:BTC",
                       "time_from": tf, "time_to": tt, "limit": "1000", "sort": "EARLIEST"})
        except Exception as e:
            info.append(f"{tf}-{tt}: ERR {e}")
            continue
        feed = d.get("feed")
        if feed is None:
            info.append(f"{tf}-{tt}: NOFEED {(d.get('Information') or d.get('Note') or '')[:60]}")
            continue
        info.append(f"{tf}-{tt}: {len(feed)} raw")
        for f in feed:
            title = (f.get("title") or "").strip().lower()
            tp = f.get("time_published")
            if not title or not tp:
                continue
            try:
                ts = _epoch(tp)
            except ValueError:
                continue
            sent = float(f.get("overall_sentiment_score") or 0.0)
            src = f.get("source") or "?"
            # BTC-specific sentiment if present in ticker_sentiment
            btc_sent = None
            for t in (f.get("ticker_sentiment") or []):
                if t.get("ticker") == "CRYPTO:BTC":
                    try:
                        btc_sent = float(t.get("ticker_sentiment_score"))
                    except (TypeError, ValueError):
                        btc_sent = None
                    break
            if title not in by_title or ts < by_title[title][0]:
                by_title[title] = (ts, sent, src, title, btc_sent)
    arts = sorted(by_title.values())
    return arts, info


def build():
    arts, info = fetch_news()
    print("fetch:", " | ".join(info))
    a_ts = [a[0] for a in arts]
    a_sent = [a[1] for a in arts]
    if arts:
        print(f"store: {len(arts)} deduped articles, span "
              f"{datetime.utcfromtimestamp(a_ts[0])} .. {datetime.utcfromtimestamp(a_ts[-1])} UTC")
    STORE.write_text("".join(json.dumps(
        {"time_published_epoch": a[0], "overall_sentiment_score": a[1],
         "source": a[2], "title": a[3], "btc_sentiment_score": a[4]}) + "\n" for a in arts))

    rows = [json.loads(l) for l in DS.open() if l.strip()]
    out = []
    cover = {"n": len(rows)}
    any1h = any4h = any24h = anyever = 0
    for w in rows:
        t = float(w["ts"])  # t_decide = open+6min, epoch UTC
        hi = bisect_right(a_ts, t)  # articles with published_epoch <= t_decide (PIT)

        def trailing(hours):
            lo = bisect_left(a_ts, t - hours * 3600)
            seg = a_sent[lo:hi]
            return len(seg), (round(statistics.fmean(seg), 4) if seg else 0.0)

        feat = {}
        for k in KH:
            c, m = trailing(k)
            feat[f"cnt_{k}h"] = c
            feat[f"sent_mean_{k}h"] = m
        # most-recent headline <= t_decide
        if hi > 0:
            feat["last_sent"] = round(a_sent[hi - 1], 4)
            feat["secs_since_last"] = int(t - a_ts[hi - 1])
        else:
            feat["last_sent"] = 0.0
            feat["secs_since_last"] = -1  # no prior news at all
        feat["has_news_4h"] = 1 if feat["cnt_4h"] > 0 else 0
        # exp-decayed sentiment over trailing 4h (1h half-life-ish, tau=3600s)
        lo4 = bisect_left(a_ts, t - 4 * 3600)
        feat["sent_decayed_4h"] = round(
            sum(a_sent[i] * math.exp(-(t - a_ts[i]) / 3600.0) for i in range(lo4, hi)), 4)

        any1h += feat["cnt_1h"] > 0
        any4h += feat["cnt_4h"] > 0
        any24h += feat["cnt_24h"] > 0
        anyever += hi > 0
        out.append({"ticker": w["ticker"], "y": w["y"], "news": feat})

    OUT.write_text("".join(json.dumps(r) + "\n" for r in out))
    cover.update({"windows_with_news_1h": any1h, "windows_with_news_4h": any4h,
                  "windows_with_news_24h": any24h, "windows_with_any_prior_news": anyever,
                  "pct_4h": round(100 * any4h / len(rows), 1),
                  "pct_24h": round(100 * any24h / len(rows), 1),
                  "articles_in_store": len(arts)})
    print("coverage:", json.dumps(cover))
    return cover


if __name__ == "__main__":
    build()
