"""NewsFeatures schema — structured news signal output for meta-model input."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class NewsFeatures(BaseModel):
    ticker: str
    computed_at: datetime
    news_count_1h: int = 0
    news_count_24h: int = 0
    sentiment_mean_1h: float = 0.0      # ticker-specific sentiment score mean
    sentiment_mean_24h: float = 0.0
    sentiment_std_24h: float = 0.0
    minutes_since_last_news: float = 9999.0   # 9999 if no recent news
    has_major_event: bool = False
    event_type: str = "none"            # "none"|"earnings"|"merger"|"regulatory"|"macro"|"other"
    headline_shock_score: float = 0.0   # abs(max ticker_sentiment_score - mean) in last 1h; 0 if no news
    data_available: bool = False        # False if news fetch failed or disabled
