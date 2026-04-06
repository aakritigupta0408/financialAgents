"""Live loop configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import META_MODEL_MIN_CONFIDENCE, STARTING_CAPITAL


@dataclass
class LoopConfig:
    ticker: str = "AAPL"
    timeframe: str = "1h"
    starting_capital: float = field(default_factory=lambda: STARTING_CAPITAL)
    context_bars: int = 100
    min_bars_required: int = 50
    forecast_horizon: int = 10
    atr_stop_multiple: float = 1.5
    atr_target_multiple: float = 3.0
    meta_model_threshold: float = field(default_factory=lambda: META_MODEL_MIN_CONFIDENCE)
    fta_enabled: bool = True
    meta_model_enabled: bool = True
    eod_retrain: bool = False           # default False — only retrain when explicitly enabled
    verbose: bool = False
    bar_sleep_seconds: float = 0.0      # 0 = replay mode; > 0 = near-real-time
