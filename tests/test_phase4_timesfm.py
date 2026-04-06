"""
tests/test_phase4_timesfm.py — Phase 4: TimesFM integration tests.

All tests use synthetic OHLCVSeries only — no API calls.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from schemas.forecast import ForecastOutput
from schemas.market_data import OHLCVBar, OHLCVSeries
from src.timesfm import (
    BaseForecaster,
    StatisticalForecaster,
    TimesFMForecaster,
    get_forecaster,
    run_forecast,
)
from src.timesfm.forecaster import _forecaster_instance  # noqa: F401 — import test


# ── Synthetic data helpers ────────────────────────────────────────────────────


def _make_series(
    prices: list[float],
    ticker: str = "TEST",
    timeframe: str = "1h",
) -> OHLCVSeries:
    """
    Build a synthetic OHLCVSeries from a list of close prices.
    High = close + 0.5, Low = close - 0.5, Open = close.
    """
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    bars = []
    for i, price in enumerate(prices):
        bars.append(
            OHLCVBar(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 0.5,
                low=max(price - 0.5, 0.01),
                close=price,
                volume=10_000.0,
                ticker=ticker,
                timeframe=timeframe,
            )
        )
    return OHLCVSeries(ticker=ticker, timeframe=timeframe, bars=bars)


def _uptrend(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + i * step for i in range(n)]


def _downtrend(n: int = 60, start: float = 130.0, step: float = 0.5) -> list[float]:
    return [max(start - i * step, 0.5) for i in range(n)]


def _flat(n: int = 60, price: float = 100.0) -> list[float]:
    return [price] * n


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestBaseForecasterContract:
    """test_base_forecaster_contract"""

    def test_statistical_forecaster_is_base_forecaster(self):
        sf = StatisticalForecaster()
        assert isinstance(sf, BaseForecaster)

    def test_has_forecast_method(self):
        sf = StatisticalForecaster()
        assert callable(sf.forecast)

    def test_has_is_available_classmethod(self):
        assert callable(StatisticalForecaster.is_available)
        assert StatisticalForecaster.is_available() is True

    def test_has_name_property(self):
        sf = StatisticalForecaster()
        assert isinstance(sf.name, str)
        assert len(sf.name) > 0


class TestForecastOutputSchema:
    """test_forecast_output_schema"""

    def test_schema_all_required_fields(self):
        series = _make_series(_uptrend())
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=5, ticker="TEST", timeframe="1h")

        assert isinstance(out, ForecastOutput)
        assert out.ticker == "TEST"
        assert out.timeframe == "1h"
        assert out.direction in ("up", "down")
        assert isinstance(out.expected_return, float)
        assert isinstance(out.confidence, float)
        assert isinstance(out.horizon, int)
        assert isinstance(out.quantile_50, list)
        assert isinstance(out.quantile_10, list)
        assert isinstance(out.quantile_90, list)

    def test_horizon_field_matches_input(self):
        series = _make_series(_uptrend())
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=7, ticker="TEST", timeframe="1h")
        assert out.horizon == 7


class TestDirectionMatchesReturn:
    """test_direction_matches_return"""

    def test_positive_return_means_up(self):
        # Strong uptrend should give positive expected_return and direction "up".
        series = _make_series(_uptrend(n=60, step=1.0))
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=10, ticker="TEST", timeframe="1h")
        if out.expected_return > 0:
            assert out.direction == "up"
        elif out.expected_return < 0:
            assert out.direction == "down"

    def test_direction_consistency_general(self):
        for prices in [_uptrend(), _downtrend(), _flat()]:
            series = _make_series(prices)
            sf = StatisticalForecaster()
            out = sf.forecast(series, horizon=5, ticker="X", timeframe="1h")
            if out.expected_return > 0:
                assert out.direction == "up", f"return={out.expected_return} but direction={out.direction}"
            elif out.expected_return < 0:
                assert out.direction == "down", f"return={out.expected_return} but direction={out.direction}"
            # expected_return == 0.0 → direction is "up" by convention (no strict requirement).


class TestConfidenceClamped:
    """test_confidence_clamped_0_1"""

    def test_confidence_always_0_to_1(self):
        test_cases = [_uptrend(), _downtrend(), _flat()]
        sf = StatisticalForecaster()
        for prices in test_cases:
            series = _make_series(prices)
            out = sf.forecast(series, horizon=10, ticker="T", timeframe="1h")
            assert 0.0 <= out.confidence <= 1.0, f"confidence={out.confidence} out of range"

    def test_pydantic_clamp_extreme_values(self):
        # ForecastOutput validator should clamp even if we push extreme values.
        out = ForecastOutput(
            ticker="T", timeframe="1h",
            direction="up", expected_return=0.1,
            confidence=5.0,  # would be clamped to 1.0
            horizon=5,
        )
        assert out.confidence == 1.0

        out2 = ForecastOutput(
            ticker="T", timeframe="1h",
            direction="down", expected_return=-0.1,
            confidence=-3.0,  # would be clamped to 0.0
            horizon=5,
        )
        assert out2.confidence == 0.0


class TestNoLookahead:
    """test_no_lookahead — forecast uses only data up to last bar."""

    def test_last_bar_is_known_reference(self):
        """
        Build a series with a known final price.  Verify that q50[0] is
        derived from (and close to) the last price, not some future value.
        """
        prices = _uptrend(n=50, start=200.0, step=0.0)  # flat at 200
        series = _make_series(prices)
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=5, ticker="TEST", timeframe="1h")

        last_price = 200.0
        # q50[0] should be near last_price (one EWM step from 200.0).
        assert len(out.quantile_50) == 5
        # For a flat series, returns are 0 → q50 should stay near 200.
        assert abs(out.quantile_50[0] - last_price) < 5.0, (
            f"q50[0]={out.quantile_50[0]} is far from last_price={last_price}"
        )

    def test_adding_future_bar_changes_forecast(self):
        """
        Extending a downtrend series by one strongly up bar should shift direction.
        """
        base = _downtrend(n=50, start=100.0, step=0.2)
        series_short = _make_series(base)

        # Add a spike bar that pushes price way up.
        extended = base + [base[-1] * 5.0]
        series_long = _make_series(extended)

        sf = StatisticalForecaster(context_len=32)
        out_short = sf.forecast(series_short, horizon=5, ticker="T", timeframe="1h")
        out_long = sf.forecast(series_long, horizon=5, ticker="T", timeframe="1h")

        # The spike should be reflected — at minimum, outputs differ.
        assert out_short.expected_return != out_long.expected_return, (
            "Expected forecast to change after adding a spike bar."
        )


class TestInsufficientDataFallback:
    """test_insufficient_data_fallback"""

    def test_short_series_returns_low_confidence(self):
        # 5 bars < default min_context_len=32
        series = _make_series(_uptrend(n=5))
        sf = StatisticalForecaster(min_context_len=32)
        out = sf.forecast(series, horizon=5, ticker="TEST", timeframe="1h")

        assert isinstance(out, ForecastOutput)
        assert out.confidence == 0.05  # low-confidence sentinel

    def test_empty_series_returns_low_confidence(self):
        series = OHLCVSeries(ticker="EMPTY", timeframe="1h", bars=[])
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=5, ticker="EMPTY", timeframe="1h")

        assert isinstance(out, ForecastOutput)
        assert out.confidence == 0.05

    def test_no_crash_on_short_series(self):
        for n in [0, 1, 5, 10, 31]:
            series = _make_series(_uptrend(n=n) if n > 0 else [], ticker="T")
            sf = StatisticalForecaster(min_context_len=32)
            out = sf.forecast(series, horizon=5, ticker="T", timeframe="1h")
            assert isinstance(out, ForecastOutput)


class TestUptrendForecastsUp:
    """test_uptrend_series_forecasts_up"""

    def test_strong_uptrend(self):
        # Very steep uptrend — both EWM and LR should point up clearly.
        prices = _uptrend(n=64, start=100.0, step=2.0)
        series = _make_series(prices)
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=10, ticker="AAPL", timeframe="1h")

        assert out.direction == "up", f"Expected 'up' for uptrend, got '{out.direction}'"
        assert out.expected_return > 0, f"Expected positive return for uptrend, got {out.expected_return}"
        assert out.confidence > 0.1


class TestDowntrendForecastsDown:
    """test_downtrend_series_forecasts_down"""

    def test_strong_downtrend(self):
        prices = _downtrend(n=64, start=200.0, step=2.0)
        series = _make_series(prices)
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=10, ticker="TSLA", timeframe="1h")

        assert out.direction == "down", f"Expected 'down' for downtrend, got '{out.direction}'"
        assert out.expected_return < 0, f"Expected negative return for downtrend, got {out.expected_return}"
        assert out.confidence > 0.1


class TestTimesFMWrapperUnavailable:
    """test_timesfm_wrapper_unavailable"""

    def test_is_available_false(self):
        # In Python 3.12 environment, TimesFM cannot be installed.
        assert TimesFMForecaster.is_available() is False

    def test_instantiation_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="TimesFM is not installed"):
            TimesFMForecaster()


class TestGetForecasterReturnsStatistical:
    """test_get_forecaster_returns_statistical"""

    def test_returns_statistical_when_timesfm_unavailable(self):
        forecaster = get_forecaster(prefer_timesfm=True)
        # In this environment TimesFM is unavailable, so we always get Statistical.
        assert isinstance(forecaster, StatisticalForecaster)

    def test_returns_statistical_when_prefer_false(self):
        forecaster = get_forecaster(prefer_timesfm=False)
        assert isinstance(forecaster, StatisticalForecaster)


class TestQuantilePathsLength:
    """test_quantile_paths_length"""

    def test_all_quantile_paths_have_length_horizon(self):
        series = _make_series(_uptrend())
        sf = StatisticalForecaster()
        for horizon in [1, 5, 10, 20]:
            out = sf.forecast(series, horizon=horizon, ticker="T", timeframe="1h")
            assert len(out.quantile_50) == horizon, f"q50 len {len(out.quantile_50)} != horizon {horizon}"
            assert len(out.quantile_10) == horizon, f"q10 len {len(out.quantile_10)} != horizon {horizon}"
            assert len(out.quantile_90) == horizon, f"q90 len {len(out.quantile_90)} != horizon {horizon}"

    def test_q10_le_q50_le_q90_for_uptrend(self):
        series = _make_series(_uptrend())
        sf = StatisticalForecaster()
        out = sf.forecast(series, horizon=10, ticker="T", timeframe="1h")
        for i in range(len(out.quantile_50)):
            assert out.quantile_10[i] <= out.quantile_50[i], (
                f"q10[{i}]={out.quantile_10[i]} > q50[{i}]={out.quantile_50[i]}"
            )
            assert out.quantile_50[i] <= out.quantile_90[i], (
                f"q50[{i}]={out.quantile_50[i]} > q90[{i}]={out.quantile_90[i]}"
            )


class TestRunForecastConvenience:
    """test_run_forecast_convenience"""

    def test_run_forecast_returns_forecast_output(self):
        series = _make_series(_uptrend())
        out = run_forecast(series, horizon=10)
        assert isinstance(out, ForecastOutput)

    def test_ticker_and_timeframe_passthrough(self):
        series = _make_series(_uptrend(), ticker="NVDA", timeframe="1h")
        out = run_forecast(series, horizon=5, ticker="NVDA", timeframe="1h")
        assert out.ticker == "NVDA"
        assert out.timeframe == "1h"

    def test_defaults_use_series_ticker_and_timeframe(self):
        series = _make_series(_uptrend(), ticker="SPY", timeframe="1d")
        out = run_forecast(series)  # No ticker/timeframe args.
        assert out.ticker == "SPY"
        assert out.timeframe == "1d"


class TestBatchMultipleTickers:
    """test_batch_multiple_tickers"""

    def test_three_different_tickers(self):
        configs = [
            ("AAPL", "1h", _uptrend()),
            ("MSFT", "1d", _downtrend()),
            ("SPY",  "1h", _flat()),
        ]
        for ticker, tf, prices in configs:
            series = _make_series(prices, ticker=ticker, timeframe=tf)
            out = run_forecast(series, horizon=10, ticker=ticker, timeframe=tf)

            assert isinstance(out, ForecastOutput), f"Expected ForecastOutput for {ticker}"
            assert out.ticker == ticker
            assert out.timeframe == tf
            assert out.direction in ("up", "down")
            assert 0.0 <= out.confidence <= 1.0
            assert out.horizon == 10
            assert len(out.quantile_50) == 10
            assert math.isfinite(out.expected_return)
