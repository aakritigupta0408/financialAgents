"""
src.timesfm.timesfm_wrapper — Real TimesFM forecaster (guarded by import).

Install blocker (as of 2026-04-05):
--------------------------------------
TimesFM requires Python <3.12.  Current environment is Python 3.12.2 on arm64 macOS.
All PyPI releases up to timesfm 1.3.0 are incompatible.

TODO (activate when TimesFM supports Python 3.12):
  1. Create a Python 3.10 or 3.11 virtual environment:
       python3.11 -m venv .venv_timesfm && source .venv_timesfm/bin/activate
  2. Install TimesFM with PyTorch backend:
       pip install timesfm[torch]
  3. Verify:
       python -c "import timesfm; print(timesfm.__version__)"
  4. Run smoke test:
       python scripts/smoke_timesfm.py
  5. Set PREFER_TIMESFM=true in .env to activate this wrapper at runtime.

TimesFM public API reference (from google-research/timesfm README):
  tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="torch",
          per_core_batch_size=32,
          horizon_len=horizon,
          context_len=context_len,
          num_heads=16,
          num_layers=20,
          model_dims=1280,
          ...
      ),
      checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
  )
  output_df = tfm.forecast_on_df(
      inputs=input_df,        # columns: ["unique_id", "ds", "y"]
      freq="h",               # "h"=hourly, "d"=daily, "min"=minute
      value_name="y",
      num_jobs=-1,
  )
  # output_df columns: unique_id, ds, timesfm, timesfm-q-0.1, timesfm-q-0.5, timesfm-q-0.9
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from schemas.forecast import ForecastOutput
from schemas.market_data import OHLCVSeries
from src.timesfm.base import BaseForecaster

logger = logging.getLogger(__name__)

# ── Guarded import ────────────────────────────────────────────────────────────
try:
    import timesfm as _timesfm  # type: ignore[import-not-found]
    _TIMESFM_AVAILABLE = True
    logger.info("TimesFM backend loaded successfully (version: %s).", getattr(_timesfm, "__version__", "unknown"))
except ImportError:
    _timesfm = None  # type: ignore[assignment]
    _TIMESFM_AVAILABLE = False
    logger.info(
        "TimesFM not installed — TimesFMForecaster is unavailable. "
        "Falling back to StatisticalForecaster. "
        "See TODO in src/timesfm/timesfm_wrapper.py for install instructions."
    )


class TimesFMForecaster(BaseForecaster):
    """
    TimesFM-backed forecaster.

    Wraps the google/timesfm-1.0-200m-pytorch checkpoint (or any compatible
    HuggingFace checkpoint).  Raises RuntimeError on instantiation if TimesFM
    is not installed.

    Input contract (TimesFM nixtla-style):
      DataFrame with columns ["unique_id", "ds", "y"]
        unique_id : ticker string
        ds        : datetime index (timestamp for each bar)
        y         : close price value

    Output contract:
      ForecastOutput with quantile_50/10/90 paths of length == horizon.
    """

    DEFAULT_CHECKPOINT = "google/timesfm-1.0-200m-pytorch"
    DEFAULT_CONTEXT_LEN = 512
    DEFAULT_HORIZON = 10

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        context_len: int = DEFAULT_CONTEXT_LEN,
        horizon: int = DEFAULT_HORIZON,
        min_context_len: int = 32,
    ) -> None:
        if not _TIMESFM_AVAILABLE:
            raise RuntimeError(
                "TimesFM is not installed. "
                "TODO: install with `pip install timesfm[torch]` in a Python 3.10/3.11 environment. "
                "Exact blocker: timesfm requires Python <3.12; current env is Python 3.12.2."
            )

        super().__init__(min_context_len=min_context_len)
        self.checkpoint = checkpoint
        self.context_len = context_len
        self._horizon = horizon
        self._tfm = None  # Lazy-initialised on first forecast call.

    # ── Abstract interface ────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        return _TIMESFM_AVAILABLE

    @property
    def name(self) -> str:
        return f"TimesFMForecaster({self.checkpoint})"

    # ── Lazy model initialisation ─────────────────────────────────────────

    def _ensure_model(self) -> None:
        """
        Lazy-load the TimesFM model on first use.

        TODO (activate when TimesFM is installed):
          self._tfm = _timesfm.TimesFm(
              hparams=_timesfm.TimesFmHparams(
                  backend="torch",
                  per_core_batch_size=32,
                  horizon_len=self._horizon,
                  context_len=self.context_len,
              ),
              checkpoint=_timesfm.TimesFmCheckpoint(
                  huggingface_repo_id=self.checkpoint,
              ),
          )
        """
        if self._tfm is not None:
            return

        # TODO: uncomment the block below once TimesFM supports Python 3.12.
        # self._tfm = _timesfm.TimesFm(
        #     hparams=_timesfm.TimesFmHparams(
        #         backend="torch",
        #         per_core_batch_size=32,
        #         horizon_len=self._horizon,
        #         context_len=self.context_len,
        #     ),
        #     checkpoint=_timesfm.TimesFmCheckpoint(
        #         huggingface_repo_id=self.checkpoint,
        #     ),
        # )
        raise RuntimeError(
            "TimesFM model initialisation reached but _TIMESFM_AVAILABLE is True — "
            "this should not happen.  Please file a bug."
        )

    # ── Forecast ──────────────────────────────────────────────────────────

    def forecast(
        self,
        series: OHLCVSeries,
        horizon: int,
        ticker: str,
        timeframe: str,
    ) -> ForecastOutput:
        """
        Run TimesFM inference on the given OHLCV series.

        Steps (TODOs activate when TimesFM is installed):
          1. Validate and extract close prices via _prepare_prices().
          2. Build nixtla-style input DataFrame:
               input_df = pd.DataFrame({
                   "unique_id": ticker,
                   "ds": timestamps[-context_len:],
                   "y": prices[-context_len:],
               })
          3. Call tfm.forecast_on_df(input_df, freq=freq_code, value_name="y").
          4. Parse output columns:
               q50 = output_df["timesfm-q-0.5"].tolist()
               q10 = output_df["timesfm-q-0.1"].tolist()
               q90 = output_df["timesfm-q-0.9"].tolist()
          5. Compute expected_return = (q50[-1] - last_price) / last_price.
          6. Confidence = 1 - (spread / last_price), clamped to [0.05, 0.95].
          7. Return ForecastOutput.
        """
        prices, err = self._prepare_prices(series)
        if prices is None:
            return self._low_confidence_output(ticker, timeframe, horizon, err)

        self._ensure_model()  # Will raise until TimesFM is installed.

        # TODO: build input DataFrame
        # df = series.to_dataframe()
        # timestamps = df.index[-self.context_len:]
        # context_prices = prices[-self.context_len:]
        # input_df = pd.DataFrame({
        #     "unique_id": ticker,
        #     "ds": timestamps,
        #     "y": context_prices,
        # })

        # TODO: map timeframe to TimesFM freq code
        # _FREQ_MAP = {"1m": "min", "5m": "5min", "15m": "15min",
        #              "30m": "30min", "1h": "h", "4h": "4h", "1d": "d"}
        # freq = _FREQ_MAP.get(timeframe, "h")

        # TODO: run inference
        # output_df = self._tfm.forecast_on_df(
        #     inputs=input_df,
        #     freq=freq,
        #     value_name="y",
        #     num_jobs=1,
        # )

        # TODO: parse quantile outputs
        # q50 = output_df["timesfm-q-0.5"].tolist()[:horizon]
        # q10 = output_df["timesfm-q-0.1"].tolist()[:horizon]
        # q90 = output_df["timesfm-q-0.9"].tolist()[:horizon]

        # TODO: derive ForecastOutput fields
        # last_price = float(prices[-1])
        # expected_return = (q50[-1] - last_price) / last_price if q50 else 0.0
        # direction = "up" if expected_return > 0 else "down"
        # spread_pct = (q90[-1] - q10[-1]) / last_price if q90 and q10 else 1.0
        # confidence = max(0.05, min(0.95, 1.0 - spread_pct))

        # TODO: return ForecastOutput(...)
        raise NotImplementedError(
            "TimesFMForecaster.forecast() is a skeleton awaiting TimesFM installation. "
            "See TODO comments in this method."
        )

    # ── Frequency mapping helper ──────────────────────────────────────────

    @staticmethod
    def _timeframe_to_freq(timeframe: str) -> str:
        """
        Convert trading-system timeframe label to TimesFM freq string.

        TODO: extend as TimesFM freq codes are clarified in the library.
        """
        mapping = {
            "1m": "min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "h",
            "4h": "4h",
            "1d": "d",
        }
        freq = mapping.get(timeframe)
        if freq is None:
            logger.warning("Unknown timeframe '%s'; defaulting to 'h'.", timeframe)
            freq = "h"
        return freq
