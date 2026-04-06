"""News data schemas — Pydantic v2 models for Alpha Vantage news sentiment."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TickerSentiment(BaseModel):
    ticker: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    ticker_sentiment_score: float = Field(ge=-1.0, le=1.0)
    ticker_sentiment_label: str


class NewsArticle(BaseModel):
    title: str
    url: str
    time_published: datetime        # parsed from YYYYMMDDTHHMM
    source: str
    summary: str = ""
    overall_sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    overall_sentiment_label: str = "Neutral"
    topics: list[str] = []          # just topic names, not scores
    ticker_sentiment: list[TickerSentiment] = []

    def sentiment_for_ticker(self, ticker: str) -> Optional[TickerSentiment]:
        """Return TickerSentiment for a specific ticker, or None."""
        for ts in self.ticker_sentiment:
            if ts.ticker.upper() == ticker.upper():
                return ts
        return None


class NewsFeed(BaseModel):
    ticker: str
    fetched_at: datetime
    articles: list[NewsArticle]

    def articles_since(self, hours: float) -> list[NewsArticle]:
        """Return articles published within the last `hours` hours of fetched_at."""
        from datetime import timedelta
        cutoff = self.fetched_at - timedelta(hours=hours)
        return [a for a in self.articles if a.time_published >= cutoff]
