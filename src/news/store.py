"""JSONL-based local news store.

Layout:
  data/store/{TICKER}/news/YYYY-MM.jsonl

Each line is a JSON-serialised NewsArticle (Pydantic model_dump).
Deduplication is by (title, time_published).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from schemas.news import NewsArticle, NewsFeed

log = logging.getLogger(__name__)

try:
    from src.data_store.paths import DATA_STORE_DIR
except Exception:
    DATA_STORE_DIR = Path("data") / "store"

_NEWS_SUBDIR = "news"


def _news_dir(ticker: str) -> Path:
    d = DATA_STORE_DIR / ticker.upper() / _NEWS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _partition_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def store_feed(feed: NewsFeed) -> int:
    """
    Persist articles from a NewsFeed to JSONL partition files.
    Returns the number of NEW articles written (deduped).
    """
    if not feed.articles:
        return 0

    ticker = feed.ticker.upper()
    news_dir = _news_dir(ticker)

    # Group articles by partition
    by_partition: dict[str, list[NewsArticle]] = {}
    for article in feed.articles:
        key = _partition_key(article.time_published)
        by_partition.setdefault(key, []).append(article)

    written = 0
    for partition, articles in by_partition.items():
        path = news_dir / f"{partition}.jsonl"

        # Load existing dedup keys
        existing_keys: set[tuple[str, str]] = set()
        if path.exists():
            try:
                with path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            existing_keys.add((obj.get("title", ""), obj.get("time_published", "")))
                        except Exception:
                            pass
            except Exception as e:
                log.warning("news_store.read_error %s: %s", path, e)

        new_articles = []
        for article in articles:
            tp_str = article.time_published.isoformat()
            key = (article.title, tp_str)
            if key not in existing_keys:
                new_articles.append(article)
                existing_keys.add(key)

        if new_articles:
            try:
                with path.open("a") as f:
                    for article in new_articles:
                        d = article.model_dump()
                        # Serialize datetime fields to ISO strings for JSON
                        d["time_published"] = article.time_published.isoformat()
                        # TickerSentiment objects are already dicts after model_dump
                        f.write(json.dumps(d) + "\n")
                written += len(new_articles)
            except Exception as e:
                log.warning("news_store.write_error %s: %s", path, e)

    return written


def load_feed(ticker: str, since: datetime | None = None, until: datetime | None = None) -> NewsFeed:
    """
    Load stored news articles for a ticker.

    Parameters
    ----------
    ticker : ticker symbol
    since  : only return articles at or after this datetime
    until  : only return articles before or at this datetime (defaults to now)
    """
    news_dir = _news_dir(ticker.upper())
    until = until or datetime.now(timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)

    articles: list[NewsArticle] = []

    # Scan all partition files
    for path in sorted(news_dir.glob("*.jsonl")):
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        article = NewsArticle.model_validate(obj)
                        tp = article.time_published
                        if tp.tzinfo is None:
                            tp = tp.replace(tzinfo=timezone.utc)
                            article = article.model_copy(update={"time_published": tp})
                        if since is not None:
                            if_since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                            if tp < if_since:
                                continue
                        if tp > until:
                            continue
                        articles.append(article)
                    except Exception as e:
                        log.debug("news_store.skip_line: %s", e)
        except Exception as e:
            log.warning("news_store.read_error %s: %s", path, e)

    articles.sort(key=lambda a: a.time_published, reverse=True)
    return NewsFeed(ticker=ticker.upper(), fetched_at=until, articles=articles)
