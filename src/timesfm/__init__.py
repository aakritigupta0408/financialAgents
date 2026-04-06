"""
src.timesfm — TimesFM forecasting wrapper.

Exports
-------
  get_forecaster    : factory returning the active BaseForecaster instance.
  run_forecast      : convenience wrapper for one-line forecast calls.
  BaseForecaster    : abstract base class (for type hints and subclassing).
  StatisticalForecaster : pure-Python/NumPy fallback forecaster.
  TimesFMForecaster : real TimesFM wrapper (guarded by import; raises if unavailable).

Usage
-----
  from src.timesfm import run_forecast
  forecast = run_forecast(series, horizon=10)
"""

from src.timesfm.base import BaseForecaster
from src.timesfm.fallback import StatisticalForecaster
from src.timesfm.forecaster import get_forecaster, run_forecast
from src.timesfm.timesfm_wrapper import TimesFMForecaster

__all__ = [
    "get_forecaster",
    "run_forecast",
    "BaseForecaster",
    "StatisticalForecaster",
    "TimesFMForecaster",
]
