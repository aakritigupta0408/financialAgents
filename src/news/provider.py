"""News provider ABC and Alpha Vantage implementation."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import requests

from schemas.news import NewsArticle, NewsFeed, TickerSentiment

log = logging.getLogger(__name__)

_AV_NEWS_URL = "https://www.alphavantage.co/query"


def _parse_av_time(s: str) -> datetime:
    """
    Parse Alpha Vantage time_published string to UTC datetime.
    Handles formats: "YYYYMMDDTHHMM", "YYYYMMDDTHHMM00", "YYYYMMDDTHHMMSS"
    """
    # Remove the "T" separator and take exactly 12 chars YYYYMMDDHHMM
    clean = s.replace("T", "")
    clean = clean.ljust(12, "0")[:12]
    return datetime.strptime(clean, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


class NewsProvider(ABC):
    @abstractmethod
    def fetch(self, ticker: str, limit: int = 50, lookback_hours: int = 24) -> NewsFeed:
        """Fetch news for a ticker. Returns NewsFeed (may be empty if no API key)."""


class AlphaVantageNewsProvider(NewsProvider):
    """
    Fetches NEWS_SENTIMENT from Alpha Vantage REST API.
    API key from ALPHA_VANTAGE_API_KEY env var.

    If no API key: returns empty NewsFeed (graceful degradation).
    Rate limit: 1 req/sec on free tier, ~25 req/day.
    """

    def __init__(self, api_key: str | None = None):
        self._key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")

    def fetch(self, ticker: str, limit: int = 50, lookback_hours: int = 24) -> NewsFeed:
        """
        Fetch news. If no API key, return empty NewsFeed.
        Parse the JSON response structure from Alpha Vantage NEWS_SENTIMENT endpoint.
        """
        now = datetime.now(timezone.utc)

        if not self._key:
            log.warning("alpha_vantage_news.no_api_key: returning empty feed for %s", ticker)
            return NewsFeed(ticker=ticker, fetched_at=now, articles=[])

        # Compute time_from: now - lookback_hours
        time_from_dt = now - timedelta(hours=lookback_hours)
        time_from = time_from_dt.strftime("%Y%m%dT%H%M")

        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "sort": "LATEST",
                "limit": str(limit),
                "time_from": time_from,
                "apikey": self._key,
            }
            resp = requests.get(_AV_NEWS_URL, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()

            # AV may return error messages in the body
            if "Error Message" in payload:
                log.warning("alpha_vantage_news.error: %s", payload["Error Message"])
                return NewsFeed(ticker=ticker, fetched_at=now, articles=[])
            if "Note" in payload or "Information" in payload:
                msg = payload.get("Note") or payload.get("Information", "")
                log.warning("alpha_vantage_news.rate_limit: %s", msg[:100])
                return NewsFeed(ticker=ticker, fetched_at=now, articles=[])

            raw_articles = payload.get("feed", [])
            articles: list[NewsArticle] = []
            for item in raw_articles:
                try:
                    tp = item.get("time_published", "")
                    time_published = _parse_av_time(tp)

                    # Extract topic names only
                    topics = [t["topic"] for t in item.get("topics", []) if "topic" in t]

                    # Parse ticker_sentiment
                    ticker_sentiments: list[TickerSentiment] = []
                    for ts_raw in item.get("ticker_sentiment", []):
                        try:
                            ticker_sentiments.append(TickerSentiment(
                                ticker=ts_raw["ticker"],
                                relevance_score=float(ts_raw.get("relevance_score", 0.0)),
                                ticker_sentiment_score=float(ts_raw.get("ticker_sentiment_score", 0.0)),
                                ticker_sentiment_label=ts_raw.get("ticker_sentiment_label", "Neutral"),
                            ))
                        except Exception as e:
                            log.debug("alpha_vantage_news.skip_ticker_sentiment: %s", e)

                    article = NewsArticle(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        time_published=time_published,
                        source=item.get("source", ""),
                        summary=item.get("summary", ""),
                        overall_sentiment_score=float(item.get("overall_sentiment_score", 0.0)),
                        overall_sentiment_label=item.get("overall_sentiment_label", "Neutral"),
                        topics=topics,
                        ticker_sentiment=ticker_sentiments,
                    )
                    articles.append(article)
                except Exception as e:
                    log.debug("alpha_vantage_news.skip_article: %s", e)

            return NewsFeed(ticker=ticker, fetched_at=now, articles=articles)

        except Exception as e:
            log.warning("alpha_vantage_news.fetch_failed for %s: %s", ticker, e)
            return NewsFeed(ticker=ticker, fetched_at=now, articles=[])


class NullNewsProvider(NewsProvider):
    """Returns empty NewsFeed. Used when news is disabled."""

    def fetch(self, ticker: str, limit: int = 50, lookback_hours: int = 24) -> NewsFeed:
        return NewsFeed(ticker=ticker, fetched_at=datetime.now(timezone.utc), articles=[])


def get_news_provider(mode: str | None = None) -> NewsProvider:
    """
    Return appropriate provider based on NEWS_MODE config.
    mode="disabled" -> NullNewsProvider
    mode=anything else -> AlphaVantageNewsProvider (falls back to null if no key)
    """
    from config.settings import NEWS_MODE
    effective_mode = mode or NEWS_MODE
    if effective_mode == "disabled":
        return NullNewsProvider()
    return AlphaVantageNewsProvider()
