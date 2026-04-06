"""
Phase 2 tests — market-data layer.

Run all tests:
    cd /Users/aakritigupta/trading-system
    python -m pytest tests/test_phase2_data.py -v

Run only non-live tests:
    python -m pytest tests/test_phase2_data.py -v -m "not live"
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schemas.market_data import OHLCVBar, OHLCVSeries, MarketSnapshot
from src.data.provider import DataFetchError, MarketDataProvider
from src.data.alpha_vantage import AlphaVantageProvider
from src.data.cache import CachedProvider, _bucket_for, _cache_key, _ttl_for
from src.data.retry import retry_call, with_retry, _is_retryable


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_bar(
    ticker: str = "AAPL",
    timeframe: str = "5m",
    timestamp: datetime | None = None,
    open: float = 150.0,
    high: float = 151.0,
    low: float = 149.0,
    close: float = 150.5,
    volume: float = 1000.0,
) -> OHLCVBar:
    if timestamp is None:
        timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    return OHLCVBar(
        timestamp=timestamp,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        ticker=ticker,
        timeframe=timeframe,  # type: ignore[arg-type]
    )


def _make_series(
    ticker: str = "AAPL",
    timeframe: str = "5m",
    n_bars: int = 3,
) -> OHLCVSeries:
    bars = [
        _make_bar(
            ticker=ticker,
            timeframe=timeframe,
            timestamp=datetime(2024, 1, 15, 14, i * 5, tzinfo=timezone.utc),
        )
        for i in range(n_bars)
    ]
    return OHLCVSeries(ticker=ticker, timeframe=timeframe, bars=bars)  # type: ignore[arg-type]


@pytest.fixture()
def av_provider() -> AlphaVantageProvider:
    return AlphaVantageProvider(api_key="test_key_demo")


@pytest.fixture()
def tmp_cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture()
def cached_provider(tmp_cache_dir):
    mock_inner = MagicMock(spec=MarketDataProvider)
    mock_inner.name = "mock_provider"
    mock_inner.supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "1d"]
    return CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)


# ── Test 1: Provider interface ─────────────────────────────────────────────

class TestProviderInterface:
    """AlphaVantageProvider correctly implements the MarketDataProvider ABC."""

    def test_is_subclass(self):
        assert issubclass(AlphaVantageProvider, MarketDataProvider)

    def test_name_property(self, av_provider):
        assert av_provider.name == "alpha_vantage"
        assert isinstance(av_provider.name, str)

    def test_supported_timeframes_property(self, av_provider):
        tfs = av_provider.supported_timeframes
        assert isinstance(tfs, list)
        assert len(tfs) > 0
        for tf in tfs:
            assert isinstance(tf, str)

    def test_supported_timeframes_includes_expected(self, av_provider):
        tfs = av_provider.supported_timeframes
        for expected in ("5m", "1h", "1d"):
            assert expected in tfs, f"Expected {expected!r} in supported_timeframes"

    def test_fetch_ohlcv_is_callable(self, av_provider):
        assert callable(av_provider.fetch_ohlcv)

    def test_fetch_snapshot_is_callable(self, av_provider):
        assert callable(av_provider.fetch_snapshot)

    def test_validate_timeframe_raises_on_bad_input(self, av_provider):
        with pytest.raises(ValueError, match="not supported"):
            av_provider._validate_timeframe("99d")

    def test_data_fetch_error_retryable_5xx(self):
        err = DataFetchError("server error", status_code=503)
        assert err.is_retryable() is True

    def test_data_fetch_error_not_retryable_4xx(self):
        err = DataFetchError("auth error", status_code=401)
        assert err.is_retryable() is False

    def test_data_fetch_error_no_status_is_retryable(self):
        err = DataFetchError("network timeout")
        assert err.is_retryable() is True

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            MarketDataProvider()  # type: ignore[abstract]


# ── Test 2: Live OHLCV fetch ───────────────────────────────────────────────

@pytest.mark.live
class TestOHLCVFetchLive:
    """Live integration test — requires ALPHA_VANTAGE_API_KEY to be set."""

    def test_ohlcv_fetch_live(self):
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            pytest.skip("ALPHA_VANTAGE_API_KEY not set; skipping live test")

        provider = AlphaVantageProvider(api_key=api_key)
        series = provider.fetch_ohlcv("AAPL", "5m", limit=30)

        assert isinstance(series, OHLCVSeries)
        assert series.ticker == "AAPL"
        assert series.timeframe == "5m"
        # Data may be empty outside market hours — just verify type is correct.
        for bar in series.bars:
            assert isinstance(bar, OHLCVBar)
            assert bar.ticker == "AAPL"
            assert bar.timeframe == "5m"
            assert bar.high >= bar.low
            assert bar.open >= bar.low
            assert bar.close >= bar.low
            assert bar.volume >= 0


# ── Test 3: Cache hit ──────────────────────────────────────────────────────

class TestCacheHit:
    """Second fetch for the same key must be served from cache."""

    def test_cache_hit(self, tmp_cache_dir):
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["5m"]
        mock_inner.fetch_ohlcv.return_value = _make_series(n_bars=5)

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)

        result1 = provider.fetch_ohlcv("AAPL", "5m", limit=100)
        result2 = provider.fetch_ohlcv("AAPL", "5m", limit=100)

        # Underlying provider must have been called exactly once.
        assert mock_inner.fetch_ohlcv.call_count == 1

        # Both results must be identical.
        assert result1.ticker == result2.ticker
        assert result1.timeframe == result2.timeframe
        assert len(result1.bars) == len(result2.bars)

        provider.close()

    def test_cache_hit_returns_ohlcvseries(self, tmp_cache_dir):
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["1d"]
        mock_inner.fetch_ohlcv.return_value = _make_series(timeframe="1d", n_bars=10)

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)
        result = provider.fetch_ohlcv("MSFT", "1d", limit=10)

        assert isinstance(result, OHLCVSeries)
        provider.close()


# ── Test 4: Cache miss then hit ────────────────────────────────────────────

class TestCacheMissThenHit:
    """Clear cache → fetch → verify stored → fetch again → verify cache hit."""

    def test_cache_miss_then_hit(self, tmp_cache_dir):
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["5m"]
        expected_series = _make_series(n_bars=7)
        mock_inner.fetch_ohlcv.return_value = expected_series

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)

        # Clear to guarantee a miss.
        provider.clear()

        # First call — must be a cache miss.
        result1 = provider.fetch_ohlcv("AAPL", "5m", limit=100)
        assert mock_inner.fetch_ohlcv.call_count == 1
        assert len(result1.bars) == 7

        # Second call — must be a cache hit.
        result2 = provider.fetch_ohlcv("AAPL", "5m", limit=100)
        assert mock_inner.fetch_ohlcv.call_count == 1  # still 1
        assert len(result2.bars) == 7

        provider.close()

    def test_force_refresh_bypasses_cache(self, tmp_cache_dir):
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["5m"]
        mock_inner.fetch_ohlcv.return_value = _make_series(n_bars=3)

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)

        provider.fetch_ohlcv("AAPL", "5m")  # populate cache
        provider.fetch_ohlcv("AAPL", "5m", force_refresh=True)  # bypass

        assert mock_inner.fetch_ohlcv.call_count == 2  # both calls hit provider
        provider.close()


# ── Test 5: Retry on transient error ──────────────────────────────────────

class TestRetryOnTransientError:
    """Mock a 500 error then success; verify retry fires."""

    def test_retry_on_transient_error(self):
        call_count = 0
        expected = _make_series(n_bars=2)

        def flaky_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DataFetchError("server error", status_code=503)
            return expected

        result = retry_call(
            flaky_fetch,
            max_retries=3,
            base_delay=0.0,  # no real sleep in tests
        )

        assert call_count == 3
        assert result is expected

    def test_retry_fires_correct_number_of_times(self):
        calls = []

        def always_fail(*args, **kwargs):
            calls.append(1)
            raise DataFetchError("temporary failure", status_code=500)

        with pytest.raises(DataFetchError):
            retry_call(always_fail, max_retries=2, base_delay=0.0)

        # 1 initial + 2 retries = 3 total attempts
        assert len(calls) == 3

    def test_retry_decorator(self):
        attempts = []

        @with_retry(max_retries=2, base_delay=0.0)
        def fragile():
            attempts.append(1)
            if len(attempts) < 2:
                raise DataFetchError("transient", status_code=500)
            return "ok"

        result = fragile()
        assert result == "ok"
        assert len(attempts) == 2


# ── Test 6: No retry on 4xx ────────────────────────────────────────────────

class TestNoRetryOn4xx:
    """4xx errors must not be retried."""

    def test_no_retry_on_401(self):
        calls = []

        def auth_fail(*args, **kwargs):
            calls.append(1)
            raise DataFetchError("unauthorized", status_code=401)

        with pytest.raises(DataFetchError) as exc_info:
            retry_call(auth_fail, max_retries=3, base_delay=0.0)

        # Must have been called exactly once.
        assert len(calls) == 1
        assert exc_info.value.status_code == 401

    def test_no_retry_on_404(self):
        calls = []

        def not_found(*args, **kwargs):
            calls.append(1)
            raise DataFetchError("not found", status_code=404)

        with pytest.raises(DataFetchError):
            retry_call(not_found, max_retries=3, base_delay=0.0)

        assert len(calls) == 1

    def test_no_retry_on_400(self):
        calls = []

        def bad_request(*args, **kwargs):
            calls.append(1)
            raise DataFetchError("bad symbol", status_code=400)

        with pytest.raises(DataFetchError):
            retry_call(bad_request, max_retries=3, base_delay=0.0)

        assert len(calls) == 1

    def test_is_retryable_helper(self):
        assert _is_retryable(DataFetchError("net", status_code=503)) is True
        assert _is_retryable(DataFetchError("auth", status_code=403)) is False
        assert _is_retryable(DataFetchError("unknown")) is True
        assert _is_retryable(ConnectionError("dropped")) is True
        assert _is_retryable(ValueError("bad value")) is False


# ── Test 7: Missing / empty data ───────────────────────────────────────────

class TestMissingDataHandling:
    """Empty or malformed AV responses must produce an empty OHLCVSeries, not crash."""

    def test_empty_response_returns_empty_series(self, av_provider):
        """
        When AV returns a valid JSON with no time series key, we get an empty series.
        """
        empty_payload: dict = {"Meta Data": {"1. Information": "Intraday"}}
        with patch.object(av_provider, "_fetch_with_retry", return_value=empty_payload):
            result = av_provider.fetch_ohlcv("AAPL", "5m", limit=100)

        assert isinstance(result, OHLCVSeries)
        assert result.ticker == "AAPL"
        assert result.timeframe == "5m"
        assert result.bars == []

    def test_partial_bar_skipped_gracefully(self, av_provider):
        """A bar missing the 'close' field should be skipped, not crash the parse."""
        payload = {
            "Time Series (5min)": {
                "2024-01-15 14:30:00": {
                    "1. open": "150.0",
                    "2. high": "151.0",
                    "3. low": "149.0",
                    # "4. close" intentionally missing
                    "5. volume": "1000",
                },
                "2024-01-15 14:25:00": {
                    "1. open": "149.0",
                    "2. high": "150.0",
                    "3. low": "148.5",
                    "4. close": "149.5",
                    "5. volume": "800",
                },
            }
        }
        with patch.object(av_provider, "_fetch_with_retry", return_value=payload):
            result = av_provider.fetch_ohlcv("AAPL", "5m", limit=100)

        # Only the valid bar should be present.
        assert isinstance(result, OHLCVSeries)
        assert len(result.bars) == 1
        assert result.bars[0].close == 149.5

    def test_completely_empty_bars_list(self):
        series = OHLCVSeries(ticker="AAPL", timeframe="5m")  # type: ignore[arg-type]
        assert series.bars == []
        assert series.latest_close is None
        df = series.to_dataframe()
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_data_fetch_error_on_rate_limit(self, av_provider):
        """AV 'Note' field (rate limit) must raise DataFetchError."""
        payload = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute..."
        }
        with patch.object(av_provider, "_fetch_with_retry", side_effect=DataFetchError("rate limit", status_code=429)):
            with pytest.raises(DataFetchError) as exc_info:
                av_provider.fetch_ohlcv("AAPL", "5m")
            assert exc_info.value.status_code == 429

    def test_4h_returns_empty_series(self, av_provider):
        """4h is not supported by AV; must return empty OHLCVSeries without crashing."""
        result = av_provider.fetch_ohlcv("AAPL", "4h", limit=100)
        assert isinstance(result, OHLCVSeries)
        assert result.bars == []

    def test_unsupported_timeframe_raises_value_error(self, av_provider):
        with pytest.raises(ValueError):
            av_provider.fetch_ohlcv("AAPL", "3d", limit=10)


# ── Test 8: Snapshot multi-timeframe ──────────────────────────────────────

class TestSnapshotMultiTimeframe:
    """MarketSnapshot must have the expected timeframe fields populated."""

    def test_snapshot_fields_populated(self, tmp_cache_dir):
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["5m", "1h", "1d"]

        def side_effect(ticker, timeframe, limit=100, **kwargs):
            return _make_series(ticker=ticker, timeframe=timeframe, n_bars=5)

        mock_inner.fetch_ohlcv.side_effect = side_effect

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)
        snapshot = provider.fetch_snapshot("AAPL", timeframes=["5m", "1h", "1d"])

        assert isinstance(snapshot, MarketSnapshot)
        assert snapshot.ticker == "AAPL"
        assert snapshot.tf_5m is not None
        assert snapshot.tf_1h is not None
        assert snapshot.tf_1d is not None
        assert snapshot.tf_1m is None   # not requested
        assert snapshot.tf_4h is None   # not requested

        provider.close()

    def test_snapshot_failed_timeframe_is_none(self, tmp_cache_dir):
        """A failed timeframe sets that field to None without raising."""
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["5m", "1d"]

        def side_effect(ticker, timeframe, limit=100, **kwargs):
            if timeframe == "1d":
                raise DataFetchError("server down", status_code=503)
            return _make_series(ticker=ticker, timeframe=timeframe, n_bars=5)

        mock_inner.fetch_ohlcv.side_effect = side_effect

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)
        snapshot = provider.fetch_snapshot("AAPL", timeframes=["5m", "1d"])

        assert snapshot.tf_5m is not None
        assert snapshot.tf_1d is None   # failed → None

        provider.close()

    def test_snapshot_returns_correct_ticker(self, tmp_cache_dir):
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["1d"]
        mock_inner.fetch_ohlcv.return_value = _make_series(ticker="TSLA", timeframe="1d")

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)
        snapshot = provider.fetch_snapshot("TSLA", timeframes=["1d"])

        assert snapshot.ticker == "TSLA"
        assert isinstance(snapshot.snapshot_time, datetime)
        provider.close()

    def test_snapshot_get_method(self, tmp_cache_dir):
        """MarketSnapshot.get(timeframe) returns the correct OHLCVSeries."""
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = ["5m"]

        series_5m = _make_series(timeframe="5m", n_bars=4)
        mock_inner.fetch_ohlcv.return_value = series_5m

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)
        snapshot = provider.fetch_snapshot("AAPL", timeframes=["5m"])

        assert snapshot.get("5m") is not None
        assert snapshot.get("1m") is None
        assert snapshot.get("invalid") is None

        provider.close()


# ── Additional cache utility tests ────────────────────────────────────────

class TestCacheUtilities:
    """Unit tests for cache helper functions."""

    def test_ttl_intraday(self):
        for tf in ("1m", "5m", "15m", "30m", "1h", "4h"):
            assert _ttl_for(tf) == 300

    def test_ttl_daily(self):
        assert _ttl_for("1d") == 21600

    def test_bucket_intraday(self):
        dt = datetime(2024, 3, 10, 14, 35, tzinfo=timezone.utc)
        bucket = _bucket_for("5m", now=dt)
        assert bucket == "2024-03-10-14"

    def test_bucket_daily(self):
        dt = datetime(2024, 3, 10, 14, 35, tzinfo=timezone.utc)
        bucket = _bucket_for("1d", now=dt)
        assert bucket == "2024-03-10"

    def test_cache_key_is_deterministic(self):
        k1 = _cache_key("AAPL", "5m", 100, "2024-03-10-14")
        k2 = _cache_key("AAPL", "5m", 100, "2024-03-10-14")
        assert k1 == k2

    def test_cache_key_differs_for_different_limit(self):
        k1 = _cache_key("AAPL", "5m", 100, "2024-03-10-14")
        k2 = _cache_key("AAPL", "5m", 50, "2024-03-10-14")
        assert k1 != k2

    def test_cache_stats_returns_dict(self, tmp_cache_dir):
        mock_inner = MagicMock(spec=MarketDataProvider)
        mock_inner.name = "mock"
        mock_inner.supported_timeframes = []

        provider = CachedProvider(provider=mock_inner, cache_dir=tmp_cache_dir)
        stats = provider.cache_stats
        assert "volume" in stats
        assert "size_limit" in stats
        assert "directory" in stats
        provider.close()


# ── AlphaVantage parser unit tests ────────────────────────────────────────

class TestAlphaVantageParser:
    """Unit tests for AV response parsing."""

    def test_parse_intraday_bars(self, av_provider):
        payload = {
            "Meta Data": {},
            "Time Series (5min)": {
                "2024-01-15 14:30:00": {
                    "1. open": "150.00",
                    "2. high": "151.50",
                    "3. low": "149.50",
                    "4. close": "151.00",
                    "5. volume": "5000",
                },
                "2024-01-15 14:25:00": {
                    "1. open": "149.00",
                    "2. high": "150.00",
                    "3. low": "148.50",
                    "4. close": "149.80",
                    "5. volume": "3000",
                },
            },
        }
        bars = av_provider._parse_bars(payload, ticker="AAPL", timeframe="5m")
        assert len(bars) == 2
        closes = {b.close for b in bars}
        assert 151.0 in closes
        assert 149.8 in closes

    def test_parse_daily_bars(self, av_provider):
        payload = {
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "180.00",
                    "2. high": "185.00",
                    "3. low": "179.50",
                    "4. close": "184.00",
                    "5. volume": "80000000",
                },
            }
        }
        bars = av_provider._parse_bars(payload, ticker="MSFT", timeframe="1d")
        assert len(bars) == 1
        assert bars[0].ticker == "MSFT"
        assert bars[0].timeframe == "1d"
        assert bars[0].close == 184.0

    def test_find_series_key_intraday(self, av_provider):
        payload = {"Meta Data": {}, "Time Series (5min)": {}}
        assert av_provider._find_series_key(payload) == "Time Series (5min)"

    def test_find_series_key_daily(self, av_provider):
        payload = {"Meta Data": {}, "Time Series (Daily)": {}}
        assert av_provider._find_series_key(payload) == "Time Series (Daily)"

    def test_find_series_key_missing(self, av_provider):
        payload = {"Meta Data": {}, "Some Other Key": {}}
        assert av_provider._find_series_key(payload) is None

    def test_parse_timestamp_datetime(self, av_provider):
        dt = av_provider._parse_timestamp("2024-01-15 14:30:00")
        assert dt.year == 2024
        assert dt.hour == 14
        assert dt.minute == 30

    def test_parse_timestamp_date(self, av_provider):
        dt = av_provider._parse_timestamp("2024-01-15")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_timestamp_invalid_raises(self, av_provider):
        with pytest.raises(ValueError):
            av_provider._parse_timestamp("not-a-date")

    def test_build_params_intraday(self, av_provider):
        params = av_provider._build_params("AAPL", "5m")
        assert params["function"] == "TIME_SERIES_INTRADAY"
        assert params["interval"] == "5min"
        assert params["symbol"] == "AAPL"

    def test_build_params_daily(self, av_provider):
        params = av_provider._build_params("AAPL", "1d")
        assert params["function"] == "TIME_SERIES_DAILY"
        assert "interval" not in params

    def test_fetch_ohlcv_full_pipeline_mocked(self, av_provider):
        """End-to-end parse from a realistic (mocked) AV payload."""
        payload = {
            "Meta Data": {"1. Information": "Intraday (5min) open, high, low, close prices"},
            "Time Series (5min)": {
                f"2024-01-15 14:{i:02d}:00": {
                    "1. open": f"{150 + i}.00",
                    "2. high": f"{151 + i}.00",
                    "3. low": f"{149 + i}.00",
                    "4. close": f"{150 + i}.50",
                    "5. volume": "1000",
                }
                for i in range(0, 30, 5)  # 0, 5, 10, 15, 20, 25 minutes
            },
        }
        with patch.object(av_provider, "_fetch_with_retry", return_value=payload):
            result = av_provider.fetch_ohlcv("AAPL", "5m", limit=10)

        assert isinstance(result, OHLCVSeries)
        assert result.ticker == "AAPL"
        assert result.timeframe == "5m"
        assert len(result.bars) == 6

        # Verify bars are sorted ascending by timestamp.
        timestamps = [b.timestamp for b in result.bars]
        assert timestamps == sorted(timestamps)
