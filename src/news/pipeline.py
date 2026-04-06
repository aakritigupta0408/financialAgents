"""High-level news pipeline — single entry point for the live loop and backtest."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from schemas.news_features import NewsFeatures
from src.news.features import compute_news_features
from src.news.provider import get_news_provider

log = logging.getLogger(__name__)


def get_news_features(
    ticker: str,
    as_of: datetime | None = None,
    mode: str | None = None,
    lookback_hours: int | None = None,
) -> NewsFeatures:
    """
    Fetch news and compute NewsFeatures for `ticker`.

    Parameters
    ----------
    ticker         : ticker symbol
    as_of          : reference time for recency features (defaults to now)
    mode           : override NEWS_MODE from config
    lookback_hours : override NEWS_LOOKBACK_HOURS from config

    Returns NewsFeatures with data_available=False if news is disabled or fetch fails.
    """
    from config import settings

    effective_mode = mode or settings.NEWS_MODE
    effective_lookback = lookback_hours or settings.NEWS_LOOKBACK_HOURS
    as_of = as_of or datetime.now(timezone.utc)

    if not settings.NEWS_ENABLED or effective_mode == "disabled":
        return NewsFeatures(
            ticker=ticker,
            computed_at=as_of,
            data_available=False,
        )

    try:
        provider = get_news_provider(mode=effective_mode)
        feed = provider.fetch(ticker=ticker, lookback_hours=effective_lookback)
        features = compute_news_features(feed=feed, ticker=ticker, as_of=as_of)
        return features
    except Exception as e:
        log.warning("news_pipeline.get_news_features failed for %s: %s", ticker, e)
        return NewsFeatures(
            ticker=ticker,
            computed_at=as_of,
            data_available=False,
        )


def news_features_to_dict(features: NewsFeatures) -> dict:
    """
    Flatten NewsFeatures into a plain dict of numeric/bool scalars.
    Suitable for appending to a meta-model feature vector.
    """
    return {
        "news_count_1h": features.news_count_1h,
        "news_count_24h": features.news_count_24h,
        "sentiment_mean_1h": features.sentiment_mean_1h,
        "sentiment_mean_24h": features.sentiment_mean_24h,
        "sentiment_std_24h": features.sentiment_std_24h,
        "minutes_since_last_news": features.minutes_since_last_news,
        "has_major_event": int(features.has_major_event),
        "event_type_encoded": _encode_event_type(features.event_type),
        "headline_shock_score": features.headline_shock_score,
        "news_data_available": int(features.data_available),
    }


_EVENT_TYPE_MAP = {
    "none": 0,
    "earnings": 1,
    "merger": 2,
    "regulatory": 3,
    "macro": 4,
    "other": 5,
}


def _encode_event_type(event_type: str) -> int:
    return _EVENT_TYPE_MAP.get(event_type.lower(), 5)
