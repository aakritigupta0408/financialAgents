"""Phase 16 — News MCP integration and timeframe adaptation policy tests."""
from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Schema tests ────────────────────────────────────────────────────────────

def test_news_article_schema():
    from schemas.news import NewsArticle, TickerSentiment
    art = NewsArticle(
        title="Test Article",
        url="http://example.com",
        time_published=datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
        source="Reuters",
        overall_sentiment_score=0.35,
        overall_sentiment_label="Somewhat-Bullish",
        topics=["Technology", "Earnings"],
        ticker_sentiment=[
            TickerSentiment(
                ticker="AAPL",
                relevance_score=0.9,
                ticker_sentiment_score=0.42,
                ticker_sentiment_label="Bullish",
            )
        ],
    )
    assert art.title == "Test Article"
    assert art.overall_sentiment_score == 0.35
    ts = art.sentiment_for_ticker("AAPL")
    assert ts is not None
    assert ts.ticker_sentiment_score == pytest.approx(0.42)
    assert art.sentiment_for_ticker("MSFT") is None


def test_news_feed_articles_since():
    from schemas.news import NewsArticle, NewsFeed
    now = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)
    articles = [
        NewsArticle(title="Old", url="u1", time_published=now - timedelta(hours=25), source="X"),
        NewsArticle(title="Day", url="u2", time_published=now - timedelta(hours=12), source="X"),
        NewsArticle(title="Recent", url="u3", time_published=now - timedelta(minutes=30), source="X"),
    ]
    feed = NewsFeed(ticker="AAPL", fetched_at=now, articles=articles)
    assert len(feed.articles_since(1)) == 1     # only "Recent"
    assert len(feed.articles_since(24)) == 2    # "Day" + "Recent"
    assert len(feed.articles_since(48)) == 3    # all


def test_news_features_schema():
    from schemas.news_features import NewsFeatures
    nf = NewsFeatures(
        ticker="NVDA",
        computed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        news_count_1h=3,
        sentiment_mean_1h=0.25,
        has_major_event=True,
        event_type="earnings",
    )
    assert nf.ticker == "NVDA"
    assert nf.has_major_event is True
    assert nf.event_type == "earnings"
    assert nf.data_available is False   # default


# ── Provider tests ──────────────────────────────────────────────────────────

def test_null_news_provider_returns_empty_feed():
    from src.news.provider import NullNewsProvider
    p = NullNewsProvider()
    feed = p.fetch("AAPL")
    assert feed.ticker == "AAPL"
    assert feed.articles == []


def test_alpha_vantage_news_provider_no_key_returns_empty():
    from src.news.provider import AlphaVantageNewsProvider
    p = AlphaVantageNewsProvider(api_key="")
    feed = p.fetch("MSFT", lookback_hours=24)
    assert feed.ticker == "MSFT"
    assert feed.articles == []


def test_get_news_provider_disabled_returns_null():
    from src.news.provider import NullNewsProvider, get_news_provider
    provider = get_news_provider(mode="disabled")
    assert isinstance(provider, NullNewsProvider)


def test_alpha_vantage_provider_parses_mock_response():
    """Test that the provider correctly parses the AV JSON structure."""
    from src.news.provider import AlphaVantageNewsProvider

    mock_payload = {
        "feed": [
            {
                "title": "AAPL Earnings Beat",
                "url": "http://example.com/1",
                "time_published": "20260401T1430",
                "source": "Reuters",
                "summary": "Apple beats Q2 expectations.",
                "overall_sentiment_score": "0.420",
                "overall_sentiment_label": "Bullish",
                "topics": [{"topic": "Earnings", "relevance_score": "0.9"}],
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.95",
                        "ticker_sentiment_score": "0.480",
                        "ticker_sentiment_label": "Bullish",
                    }
                ],
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_payload
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        provider = AlphaVantageNewsProvider(api_key="TESTKEY")
        feed = provider.fetch("AAPL", lookback_hours=48)

    assert len(feed.articles) == 1
    art = feed.articles[0]
    assert art.title == "AAPL Earnings Beat"
    assert art.overall_sentiment_score == pytest.approx(0.42)
    assert art.time_published == datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc)
    ts = art.sentiment_for_ticker("AAPL")
    assert ts is not None
    assert ts.ticker_sentiment_score == pytest.approx(0.48)


# ── Feature computation tests ───────────────────────────────────────────────

def test_compute_news_features_empty_feed():
    from schemas.news import NewsFeed
    from src.news.features import compute_news_features
    now = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)
    feed = NewsFeed(ticker="TSLA", fetched_at=now, articles=[])
    features = compute_news_features(feed, "TSLA", as_of=now)
    assert features.news_count_1h == 0
    assert features.news_count_24h == 0
    assert features.sentiment_mean_1h == 0.0
    assert features.minutes_since_last_news == pytest.approx(9999.0)
    assert features.data_available is False


def test_compute_news_features_with_articles():
    from schemas.news import NewsArticle, NewsFeed, TickerSentiment
    from src.news.features import compute_news_features

    now = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)
    articles = [
        NewsArticle(
            title="TSLA Earnings",
            url="u1",
            time_published=now - timedelta(minutes=20),
            source="X",
            overall_sentiment_score=0.5,
            topics=["Earnings"],
            ticker_sentiment=[
                TickerSentiment(
                    ticker="TSLA",
                    relevance_score=0.9,
                    ticker_sentiment_score=0.6,
                    ticker_sentiment_label="Bullish",
                )
            ],
        ),
        NewsArticle(
            title="Market Update",
            url="u2",
            time_published=now - timedelta(hours=10),
            source="Y",
            overall_sentiment_score=-0.1,
            topics=["Economy"],
        ),
    ]
    feed = NewsFeed(ticker="TSLA", fetched_at=now, articles=articles)
    features = compute_news_features(feed, "TSLA", as_of=now)

    assert features.news_count_1h == 1
    assert features.news_count_24h == 2
    assert features.sentiment_mean_1h == pytest.approx(0.6)   # ticker-specific score
    assert features.has_major_event is True
    assert features.event_type == "earnings"
    assert features.data_available is True
    assert features.minutes_since_last_news == pytest.approx(20.0, abs=1.0)


# ── Store tests ─────────────────────────────────────────────────────────────

def test_news_store_roundtrip(tmp_path, monkeypatch):
    from schemas.news import NewsArticle, NewsFeed
    import src.news.store as store_mod
    monkeypatch.setattr(store_mod, "DATA_STORE_DIR", tmp_path)

    now = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)
    articles = [
        NewsArticle(title="A1", url="u1", time_published=now - timedelta(hours=2), source="S"),
        NewsArticle(title="A2", url="u2", time_published=now - timedelta(hours=1), source="S"),
    ]
    feed = NewsFeed(ticker="AMD", fetched_at=now, articles=articles)

    written = store_mod.store_feed(feed)
    assert written == 2

    loaded = store_mod.load_feed("AMD")
    assert len(loaded.articles) == 2
    titles = {a.title for a in loaded.articles}
    assert titles == {"A1", "A2"}


def test_news_store_deduplication(tmp_path, monkeypatch):
    from schemas.news import NewsArticle, NewsFeed
    import src.news.store as store_mod
    monkeypatch.setattr(store_mod, "DATA_STORE_DIR", tmp_path)

    now = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)
    article = NewsArticle(title="Dup", url="u1", time_published=now - timedelta(hours=1), source="S")
    feed = NewsFeed(ticker="AMD", fetched_at=now, articles=[article])

    w1 = store_mod.store_feed(feed)
    w2 = store_mod.store_feed(feed)
    assert w1 == 1
    assert w2 == 0   # deduped

    loaded = store_mod.load_feed("AMD")
    assert len(loaded.articles) == 1


# ── Pipeline tests ──────────────────────────────────────────────────────────

def test_pipeline_disabled_returns_no_data():
    from src.news.pipeline import get_news_features
    with patch("config.settings.NEWS_ENABLED", False):
        features = get_news_features("AAPL", mode="disabled")
    assert features.data_available is False
    assert features.ticker == "AAPL"


def test_news_features_to_dict_keys():
    from src.news.pipeline import news_features_to_dict
    from schemas.news_features import NewsFeatures
    nf = NewsFeatures(
        ticker="SPY",
        computed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        news_count_1h=2,
        sentiment_mean_1h=0.15,
        has_major_event=False,
    )
    d = news_features_to_dict(nf)
    expected_keys = {
        "news_count_1h", "news_count_24h", "sentiment_mean_1h", "sentiment_mean_24h",
        "sentiment_std_24h", "minutes_since_last_news", "has_major_event",
        "event_type_encoded", "headline_shock_score", "news_data_available",
    }
    assert set(d.keys()) == expected_keys
    assert d["news_count_1h"] == 2
    assert d["has_major_event"] == 0
    assert d["event_type_encoded"] == 0   # "none"


# ── Timeframe policy tests ──────────────────────────────────────────────────

def test_policy_daily_only_default():
    from src.timeframe_policy.policy import TimeframePolicy
    policy = TimeframePolicy()
    decision = policy.decide(
        configured_policy="daily_only",
        has_real_intraday_data=False,
        intraday_trade_count=0,
    )
    assert decision.effective_mode == "daily_only"
    assert decision.fell_back is False
    assert decision.prediction_timeframe == "1d"


def test_policy_falls_back_when_no_intraday_data():
    from src.timeframe_policy.policy import TimeframePolicy
    policy = TimeframePolicy()
    with patch("config.settings.REQUIRE_REAL_INTRADAY_DATA", True):
        decision = policy.decide(
            configured_policy="daily_plus_intraday_calibration",
            has_real_intraday_data=False,
            intraday_trade_count=100,
        )
    assert decision.effective_mode == "daily_only"
    assert decision.fell_back is True
    assert "no_real_intraday_data" in decision.reason
    assert len(decision.warnings) > 0


def test_policy_falls_back_insufficient_trades():
    from src.timeframe_policy.policy import TimeframePolicy
    policy = TimeframePolicy()
    with patch("config.settings.REQUIRE_REAL_INTRADAY_DATA", False), \
         patch("config.settings.MIN_INTRADAY_TRADES_FOR_ADAPTATION", 50):
        decision = policy.decide(
            configured_policy="daily_plus_intraday_calibration",
            has_real_intraday_data=True,
            intraday_trade_count=10,
        )
    assert decision.effective_mode == "daily_only"
    assert decision.fell_back is True


def test_policy_finetune_blocked_without_flag():
    from src.timeframe_policy.policy import TimeframePolicy
    policy = TimeframePolicy()
    with patch("config.settings.REQUIRE_REAL_INTRADAY_DATA", False), \
         patch("config.settings.MIN_INTRADAY_TRADES_FOR_ADAPTATION", 5), \
         patch("config.settings.ALLOW_EXPERIMENTAL_FINETUNE", False):
        decision = policy.decide(
            configured_policy="intraday_experimental_finetune",
            has_real_intraday_data=True,
            intraday_trade_count=100,
        )
    assert decision.effective_mode == "daily_plus_intraday_calibration"
    assert len(decision.warnings) > 0


def test_policy_reporting_output():
    from src.timeframe_policy.policy import PolicyDecision
    from src.timeframe_policy.reporting import policy_report
    dec = PolicyDecision(
        effective_mode="daily_only",
        prediction_timeframe="1d",
        reason="daily_only",
        warnings=["test warning"],
        fell_back=False,
    )
    report = policy_report(dec)
    assert "daily_only" in report
    assert "test warning" in report


def test_news_reporting_disabled():
    from schemas.news_features import NewsFeatures
    from src.timeframe_policy.reporting import news_report
    nf = NewsFeatures(
        ticker="SPY",
        computed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        data_available=False,
    )
    report = news_report(nf)
    assert "disabled" in report.lower() or "unavailable" in report.lower()
