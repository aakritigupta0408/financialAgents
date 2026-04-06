"""src.news — news data fetch, storage, and feature computation."""

from src.news.provider import NewsProvider, AlphaVantageNewsProvider, NullNewsProvider, get_news_provider
from src.news.features import compute_news_features
from src.news.pipeline import get_news_features, news_features_to_dict
from src.news.store import store_feed, load_feed

__all__ = [
    "NewsProvider",
    "AlphaVantageNewsProvider",
    "NullNewsProvider",
    "get_news_provider",
    "compute_news_features",
    "get_news_features",
    "news_features_to_dict",
    "store_feed",
    "load_feed",
]
