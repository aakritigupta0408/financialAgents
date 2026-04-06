"""compute_news_features — derives structured numeric features from a NewsFeed."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from schemas.news import NewsFeed
from schemas.news_features import NewsFeatures


def compute_news_features(feed: NewsFeed, ticker: str, as_of: datetime | None = None) -> NewsFeatures:
    """
    Derive NewsFeatures from a NewsFeed for the given ticker.

    Parameters
    ----------
    feed    : NewsFeed returned by a NewsProvider
    ticker  : the ticker we are computing features for
    as_of   : reference time for recency calculations (defaults to feed.fetched_at)
    """
    as_of = as_of or feed.fetched_at
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    # articles within 1h / 24h
    articles_1h = feed.articles_since(1)
    articles_24h = feed.articles_since(24)

    news_count_1h = len(articles_1h)
    news_count_24h = len(articles_24h)

    # Sentiment scores — prefer ticker-specific score, fall back to overall
    def _ticker_score(article):
        ts = article.sentiment_for_ticker(ticker)
        if ts is not None:
            return ts.ticker_sentiment_score
        return article.overall_sentiment_score

    scores_1h = [_ticker_score(a) for a in articles_1h]
    scores_24h = [_ticker_score(a) for a in articles_24h]

    sentiment_mean_1h = float(sum(scores_1h) / len(scores_1h)) if scores_1h else 0.0
    sentiment_mean_24h = float(sum(scores_24h) / len(scores_24h)) if scores_24h else 0.0

    # std for 24h
    if len(scores_24h) >= 2:
        mean24 = sentiment_mean_24h
        variance = sum((s - mean24) ** 2 for s in scores_24h) / len(scores_24h)
        sentiment_std_24h = math.sqrt(variance)
    else:
        sentiment_std_24h = 0.0

    # Minutes since last news article
    if feed.articles:
        sorted_articles = sorted(feed.articles, key=lambda a: a.time_published, reverse=True)
        latest = sorted_articles[0].time_published
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        delta_minutes = (as_of - latest).total_seconds() / 60.0
        minutes_since_last_news = max(0.0, delta_minutes)
    else:
        minutes_since_last_news = 9999.0

    # Major event detection from topics
    _EARNINGS_KEYWORDS = {"earnings", "financial results", "eps", "revenue", "guidance"}
    _MERGER_KEYWORDS = {"mergers & acquisitions", "merger", "acquisition", "m&a", "buyout"}
    _REGULATORY_KEYWORDS = {"regulation", "regulatory", "sec", "antitrust", "ftc", "doj", "compliance"}
    _MACRO_KEYWORDS = {"federal reserve", "interest rates", "inflation", "gdp", "economy", "macro", "fed"}

    has_major_event = False
    event_type = "none"

    all_recent = articles_1h if articles_1h else articles_24h[:5]
    for article in all_recent:
        topics_lower = {t.lower() for t in article.topics}
        title_lower = article.title.lower()

        if topics_lower & _EARNINGS_KEYWORDS or any(k in title_lower for k in _EARNINGS_KEYWORDS):
            has_major_event = True
            event_type = "earnings"
            break
        elif topics_lower & _MERGER_KEYWORDS or any(k in title_lower for k in _MERGER_KEYWORDS):
            has_major_event = True
            event_type = "merger"
            break
        elif topics_lower & _REGULATORY_KEYWORDS or any(k in title_lower for k in _REGULATORY_KEYWORDS):
            has_major_event = True
            event_type = "regulatory"
            break
        elif topics_lower & _MACRO_KEYWORDS or any(k in title_lower for k in _MACRO_KEYWORDS):
            has_major_event = True
            event_type = "macro"
            break
    else:
        # Check if any 24h article has strong sentiment
        if scores_24h and max(abs(s) for s in scores_24h) > 0.5:
            has_major_event = True
            event_type = "other"

    # Headline shock score: max abs deviation from mean in 1h window
    if scores_1h:
        mean1h = sentiment_mean_1h
        headline_shock_score = max(abs(s - mean1h) for s in scores_1h)
    else:
        headline_shock_score = 0.0

    data_available = len(feed.articles) > 0

    return NewsFeatures(
        ticker=ticker,
        computed_at=as_of,
        news_count_1h=news_count_1h,
        news_count_24h=news_count_24h,
        sentiment_mean_1h=sentiment_mean_1h,
        sentiment_mean_24h=sentiment_mean_24h,
        sentiment_std_24h=sentiment_std_24h,
        minutes_since_last_news=minutes_since_last_news,
        has_major_event=has_major_event,
        event_type=event_type,
        headline_shock_score=headline_shock_score,
        data_available=data_available,
    )
