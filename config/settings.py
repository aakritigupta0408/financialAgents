"""
Central runtime configuration.
Override any value via environment variables or a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# ── Capital ────────────────────────────────────────────────────────────────
STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "100000"))

# ── Risk limits ────────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))   # 1% of equity
MAX_TRADES_PER_DAY: int = int(os.getenv("MAX_TRADES_PER_DAY", "5"))
MAX_CONCURRENT_POSITIONS: int = int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
MAX_DAILY_DRAWDOWN_PCT: float = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.03"))  # 3%
MAX_TICKER_EXPOSURE_PCT: float = float(os.getenv("MAX_TICKER_EXPOSURE_PCT", "0.10"))

# ── Meta-model thresholds ──────────────────────────────────────────────────
META_MODEL_MIN_CONFIDENCE: float = float(os.getenv("META_MODEL_MIN_CONFIDENCE", "0.60"))

# ── FTA thresholds ─────────────────────────────────────────────────────────
FTA_MIN_REWARD_RISK: float = float(os.getenv("FTA_MIN_REWARD_RISK", "2.0"))
FTA_MIN_DISTANCE_TO_FTA_PCT: float = float(os.getenv("FTA_MIN_DISTANCE_TO_FTA_PCT", "0.005"))

# ── Data paths ─────────────────────────────────────────────────────────────
CACHE_DIR: Path = ROOT / "data" / "cache"
LOG_DIR: Path = ROOT / "data" / "logs"
MODEL_DIR: Path = ROOT / "data" / "models"
JOURNAL_DIR: Path = ROOT / "data" / "journal"

for _d in (CACHE_DIR, LOG_DIR, MODEL_DIR, JOURNAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Tickers to scan ────────────────────────────────────────────────────────
SCAN_TICKERS: list[str] = os.getenv(
    "SCAN_TICKERS", "AAPL,MSFT,NVDA,TSLA,SPY,QQQ"
).split(",")

# ── Timeframes used across the system ─────────────────────────────────────
TIMEFRAMES: list[str] = ["1m", "5m", "1h", "1d"]

# ── News settings (Phase 16) ───────────────────────────────────────────────
NEWS_ENABLED: bool = bool(os.getenv("NEWS_ENABLED", "false").lower() == "true")
NEWS_MODE: str = os.getenv("NEWS_MODE", "disabled")  # disabled|reporting_only|risk_filter|meta_model_feature
NEWS_PROVIDER: str = os.getenv("NEWS_PROVIDER", "alpha_vantage")
NEWS_LOOKBACK_HOURS: int = int(os.getenv("NEWS_LOOKBACK_HOURS", "24"))

# ── Timeframe policy settings (Phase 16) ──────────────────────────────────
TIMEFRAME_POLICY: str = os.getenv("TIMEFRAME_POLICY", "daily_only")  # daily_only|daily_plus_intraday_calibration|intraday_experimental_finetune
DEFAULT_PREDICTION_TIMEFRAME: str = os.getenv("DEFAULT_PREDICTION_TIMEFRAME", "1d")
INTRADAY_TARGET_TIMEFRAME: str = os.getenv("INTRADAY_TARGET_TIMEFRAME", "1h")
REQUIRE_REAL_INTRADAY_DATA: bool = bool(os.getenv("REQUIRE_REAL_INTRADAY_DATA", "true").lower() == "true")
MIN_INTRADAY_TRADES_FOR_ADAPTATION: int = int(os.getenv("MIN_INTRADAY_TRADES_FOR_ADAPTATION", "50"))
ALLOW_EXPERIMENTAL_FINETUNE: bool = bool(os.getenv("ALLOW_EXPERIMENTAL_FINETUNE", "false").lower() == "true")

# ── Risk appetite (Phase 17) ───────────────────────────────────────────────
RISK_APPETITE_MODE: str = os.getenv("RISK_APPETITE_MODE", "moderate")  # conservative|moderate|aggressive|custom
