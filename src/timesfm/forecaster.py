"""
src.timesfm.forecaster — Forecaster factory and convenience wrapper.

Public API
----------
  get_forecaster(prefer_timesfm=True) -> BaseForecaster
      Returns TimesFMForecaster if available and preferred, else StatisticalForecaster.

  run_forecast(series, horizon, ticker, timeframe) -> ForecastOutput
      Thin convenience wrapper: get_forecaster().forecast(...).
"""

from __future__ import annotations

import logging

from schemas.forecast import ForecastOutput
from schemas.market_data import OHLCVSeries
from src.timesfm.base import BaseForecaster
from src.timesfm.fallback import StatisticalForecaster
from src.timesfm.timesfm_wrapper import TimesFMForecaster

logger = logging.getLogger(__name__)

# Module-level singleton so get_forecaster() doesn't re-instantiate on every call.
_forecaster_instance: BaseForecaster | None = None
_forecaster_prefer: bool | None = None  # Track what preference was used.


def get_forecaster(prefer_timesfm: bool = True) -> BaseForecaster:
    """
    Return the active forecaster instance.

    Logic:
      1. If prefer_timesfm=True and TimesFMForecaster.is_available(): use TimesFMForecaster.
      2. Otherwise: use StatisticalForecaster.
      3. Log which forecaster is active (once per preference change).

    Parameters
    ----------
    prefer_timesfm : bool
        Set to True to use real TimesFM when available (default).
        Set to False to force the statistical fallback regardless.

    Returns
    -------
    BaseForecaster instance (singleton per preference setting).
    """
    global _forecaster_instance, _forecaster_prefer

    if _forecaster_instance is not None and _forecaster_prefer == prefer_timesfm:
        return _forecaster_instance

    if prefer_timesfm and TimesFMForecaster.is_available():
        try:
            instance: BaseForecaster = TimesFMForecaster()
            logger.info(
                "[TimesFM] Active forecaster: %s (TimesFM backend available).",
                instance.name,
            )
        except Exception as exc:
            logger.warning(
                "[TimesFM] TimesFMForecaster instantiation failed (%s); "
                "falling back to StatisticalForecaster.",
                exc,
            )
            instance = StatisticalForecaster()
            logger.info("[TimesFM] Active forecaster: %s (statistical fallback).", instance.name)
    else:
        instance = StatisticalForecaster()
        if prefer_timesfm:
            logger.info(
                "[TimesFM] TimesFM not available in this environment "
                "(Python 3.12 / no timesfm package). "
                "Active forecaster: %s (statistical fallback).",
                instance.name,
            )
        else:
            logger.info(
                "[TimesFM] prefer_timesfm=False. "
                "Active forecaster: %s (statistical fallback).",
                instance.name,
            )

    _forecaster_instance = instance
    _forecaster_prefer = prefer_timesfm
    return _forecaster_instance


def run_forecast(
    series: OHLCVSeries,
    horizon: int = 10,
    ticker: str = "",
    timeframe: str = "",
) -> ForecastOutput:
    """
    Convenience wrapper: obtain the active forecaster and run inference.

    Parameters
    ----------
    series    : OHLCVSeries — past OHLCV bars (ascending).
    horizon   : int — bars ahead to forecast (default 10).
    ticker    : str — ticker symbol; uses series.ticker if empty.
    timeframe : str — timeframe label; uses series.timeframe if empty.

    Returns
    -------
    ForecastOutput — standardised forecast feature.
    """
    resolved_ticker = ticker or series.ticker
    resolved_tf = timeframe or series.timeframe
    forecaster = get_forecaster()
    return forecaster.forecast(series, horizon, resolved_ticker, resolved_tf)
