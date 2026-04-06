"""
Phase 1 validation: import all schemas, instantiate each contract type,
verify field validation, and confirm settings load without error.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime, timezone


# ── Schema import smoke test ──────────────────────────────────────────────

def test_market_data_imports():
    from schemas.market_data import OHLCVBar, OHLCVSeries, MarketSnapshot
    assert OHLCVBar and OHLCVSeries and MarketSnapshot


def test_feature_imports():
    from schemas.features import (
        StructureFeatures, LevelFeatures,
        VolatilityFeatures, LiquidityFeatures,
    )
    assert StructureFeatures and LevelFeatures and VolatilityFeatures and LiquidityFeatures


def test_forecast_imports():
    from schemas.forecast import ForecastOutput
    assert ForecastOutput


def test_fta_imports():
    from schemas.fta import FTAInput, FTAOutput, FTAVerdict, FTACandidate
    assert FTAInput and FTAOutput and FTAVerdict and FTACandidate


def test_meta_model_imports():
    from schemas.meta_model import MetaModelInput, MetaModelOutput
    assert MetaModelInput and MetaModelOutput


def test_portfolio_imports():
    from schemas.portfolio import TradeOrder, Position, PortfolioState
    assert TradeOrder and Position and PortfolioState


def test_signals_imports():
    from schemas.signals import CandidateTrade, RankedTrade
    assert CandidateTrade and RankedTrade


def test_top_level_schemas_import():
    import schemas
    assert hasattr(schemas, "OHLCVBar")
    assert hasattr(schemas, "FTAInput")
    assert hasattr(schemas, "RankedTrade")


# ── Instantiation and validation ──────────────────────────────────────────

def test_ohlcv_bar_valid():
    from schemas.market_data import OHLCVBar
    bar = OHLCVBar(
        timestamp=datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc),
        open=150.0, high=152.0, low=149.0, close=151.0,
        volume=1_000_000,
        ticker="AAPL",
        timeframe="5m",
    )
    assert bar.close == 151.0


def test_ohlcv_bar_rejects_invalid_ohlc():
    from schemas.market_data import OHLCVBar
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        OHLCVBar(
            timestamp=datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc),
            open=200.0,  # above high — invalid
            high=152.0, low=149.0, close=151.0,
            volume=1_000_000,
            ticker="AAPL",
            timeframe="5m",
        )


def test_ohlcv_series_to_dataframe():
    from schemas.market_data import OHLCVBar, OHLCVSeries
    bars = [
        OHLCVBar(
            timestamp=datetime(2024, 1, 2, 9, 30 + i, tzinfo=timezone.utc),
            open=150.0, high=152.0, low=149.0, close=150.0 + i * 0.3,
            volume=500_000 + i * 1000,
            ticker="AAPL",
            timeframe="1m",
        )
        for i in range(5)
    ]
    series = OHLCVSeries(ticker="AAPL", timeframe="1m", bars=bars)
    df = series.to_dataframe()
    assert len(df) == 5
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_forecast_output_clamps_confidence():
    from schemas.forecast import ForecastOutput
    f = ForecastOutput(
        ticker="MSFT",
        timeframe="1h",
        direction="up",
        expected_return=0.015,
        confidence=1.5,  # out of range — must be clamped to 1.0
        horizon=12,
    )
    assert f.confidence == 1.0


def test_portfolio_state_return_pct():
    from schemas.portfolio import PortfolioState
    state = PortfolioState(
        starting_capital=100_000,
        cash=95_000,
        equity=105_000,
        peak_equity=105_000,
    )
    assert abs(state.total_return_pct - 5.0) < 1e-6


def test_settings_load():
    from config.settings import STARTING_CAPITAL, SCAN_TICKERS, CACHE_DIR
    assert STARTING_CAPITAL > 0
    assert len(SCAN_TICKERS) > 0
    assert CACHE_DIR.exists()


def test_src_module_stubs_importable():
    """All src module stubs must be importable."""
    import src.data
    import src.features
    import src.fta
    import src.timesfm
    import src.meta_model
    import src.portfolio
    import src.backtest
    import src.loop
    import src.reports
