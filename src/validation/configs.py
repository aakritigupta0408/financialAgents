"""
src.validation.configs — Benchmark configurations, slippage variants, ticker
universe, and regime configs used throughout Phase 11 validation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Benchmark configs: name -> BacktestEngine kwargs overrides
# "buy_and_hold" is special and computed analytically in BenchmarkRunner.
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS: dict[str, dict] = {
    "buy_and_hold":                 {},
    "forecast_only":                {"fta_enabled": False, "meta_model_enabled": False},
    "forecast_plus_fta":            {"fta_enabled": True,  "meta_model_enabled": False},
    "full_system":                  {"fta_enabled": True,  "meta_model_enabled": True},
    "full_system_no_meta_model":    {"fta_enabled": True,  "meta_model_enabled": False},
    "full_system_heuristic_meta":   {"fta_enabled": False, "meta_model_enabled": True},
    "full_system_no_adaptation":    {"fta_enabled": True,  "meta_model_enabled": True},
    "full_system_with_adaptation":  {"fta_enabled": True,  "meta_model_enabled": True},
}

# ---------------------------------------------------------------------------
# Slippage / fee variants (fraction of entry price / trade value)
# ---------------------------------------------------------------------------
SLIPPAGE_VARIANTS: list[float] = [0.0, 0.0005, 0.001, 0.002]
FEE_VARIANTS: list[float] = [0.0, 0.0001, 0.0005, 0.001]

# ---------------------------------------------------------------------------
# Ticker universe — synthetic proxies via make_synthetic_ohlcv
# ---------------------------------------------------------------------------
TICKER_UNIVERSE: list[dict] = [
    {"ticker": "AAPL", "trend": 0.0003,  "volatility": 0.012, "seed": 1},
    {"ticker": "MSFT", "trend": 0.0002,  "volatility": 0.010, "seed": 2},
    {"ticker": "NVDA", "trend": 0.0005,  "volatility": 0.020, "seed": 3},
    {"ticker": "TSLA", "trend": 0.0001,  "volatility": 0.025, "seed": 4},
    {"ticker": "AMD",  "trend": 0.0002,  "volatility": 0.018, "seed": 5},
    {"ticker": "META", "trend": 0.0004,  "volatility": 0.015, "seed": 6},
    {"ticker": "SPY",  "trend": 0.0001,  "volatility": 0.007, "seed": 7},
    {"ticker": "QQQ",  "trend": 0.0002,  "volatility": 0.009, "seed": 8},
    {"ticker": "IWM",  "trend": 0.00005, "volatility": 0.011, "seed": 9},
]

# ---------------------------------------------------------------------------
# Regime splits for robustness testing
# ---------------------------------------------------------------------------
REGIME_CONFIGS: list[dict] = [
    {"name": "trending_bull",   "trend":  0.0005, "volatility": 0.012, "seed": 10},
    {"name": "trending_bear",   "trend": -0.0004, "volatility": 0.012, "seed": 11},
    {"name": "ranging",         "trend":  0.00001,"volatility": 0.008, "seed": 12},
    {"name": "high_volatility", "trend":  0.0001, "volatility": 0.030, "seed": 13},
    {"name": "low_volatility",  "trend":  0.0001, "volatility": 0.004, "seed": 14},
]
