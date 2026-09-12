"""R2 — NEWS ARRIVAL capture (Alpha Vantage NEWS_SENTIMENT), prospective &
PIT-safe. Captures STRUCTURED event arrival, not an LLM narrative. The
decision-critical timestamp is available_for_decision_ts = when WE first
observed the article (received_ts); time_published is the vendor's event time.

Runtime: poll every few minutes; append new article ids to
results/news_capture.jsonl (dedup by url). Derived per-decision features
(time-since-last-relevant-article, burst count, rolling sentiment) are built
downstream from this arrival log — never by reaching into the future.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
KEYFILE = Path.home() / ".alphavantage_key"
OUT = ROOT / "results" / "news_capture.jsonl"
HEALTH = ROOT / "results" / "news_health.json"
SEEN = ROOT / "results" / ".news_seen.json"
BASE = "https://www.alphavantage.co/query"
SCHEMA = "news-v1"


def _key():
    return KEYFILE.read_text().strip()


def _pub_epoch(s):
    # AV format: 20260912T142530
    try:
        return time.mktime(time.strptime(s, "%Y%m%dT%H%M%S"))
    except Exception:
        return None


def capture_once(tickers="CRYPTO:BTC", limit=50):
    # NB: mixing CRYPTO:* and equity tickers in one call is rejected by AV;
    # poll crypto and equities as separate calls if both are wanted.
    now = time.time()
    params = {"function": "NEWS_SENTIMENT", "tickers": tickers,
              "limit": str(limit), "apikey": _key()}
    url = BASE + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            doc = json.loads(r.read().decode())
    except Exception as e:
        HEALTH.write_text(json.dumps({"source": "av-news", "state": "UNAVAILABLE",
                                      "error": str(e)[:80], "last_ts": now}, indent=1))
        return 0, "UNAVAILABLE"
    if "feed" not in doc:
        HEALTH.write_text(json.dumps({"source": "av-news", "state": "DEGRADED",
            "note": (doc.get("Information") or doc.get("Note") or "no feed")[:100],
            "last_ts": now}, indent=1))
        return 0, "DEGRADED"
    try:
        seen = set(json.loads(SEEN.read_text()))
    except Exception:
        seen = set()
    new = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        for a in doc["feed"]:
            u = a.get("url")
            if not u or u in seen:
                continue
            seen.add(u); new += 1
            rec = {"schema_version": SCHEMA, "url": u,
                   "received_ts": now, "available_for_decision_ts": now,
                   "time_published": a.get("time_published"),
                   "published_epoch": _pub_epoch(a.get("time_published", "")),
                   "title": a.get("title"), "source": a.get("source"),
                   "overall_sentiment_score": a.get("overall_sentiment_score"),
                   "topics": [t.get("topic") for t in (a.get("topics") or [])],
                   "ticker_sentiment": [{"t": t.get("ticker"),
                                         "rel": t.get("relevance_score"),
                                         "sent": t.get("ticker_sentiment_score")}
                                        for t in (a.get("ticker_sentiment") or [])]}
            fh.write(json.dumps(rec) + "\n")
    SEEN.write_text(json.dumps(sorted(seen)[-5000:]))
    HEALTH.write_text(json.dumps({"source": "av-news", "state": "CONNECTED",
        "last_ts": now, "new_articles": new, "total_seen": len(seen),
        "schema_version": SCHEMA}, indent=1))
    return new, "CONNECTED"


if __name__ == "__main__":
    n, state = capture_once()
    print(f"av-news: {state}, {n} new articles captured")
