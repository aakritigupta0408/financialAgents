"""
src.adaptive.context — AdaptiveContext dataclass persisted as JSON.

CONTEXT_DIR = Path(MODEL_DIR) / "adaptive"
File: CONTEXT_DIR / "adaptive_context.json"

Bounds enforced via clamp_thresholds():
    meta_model_min_confidence  ∈ [0.30, 0.90]
    min_reward_risk            ∈ [1.0,  5.0]
    forecast_confidence_min    ∈ [0.10, 0.90]
    fta_score_min              ∈ [0.0,  1.0]
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import META_MODEL_MIN_CONFIDENCE, FTA_MIN_REWARD_RISK, MODEL_DIR

log = logging.getLogger(__name__)

CONTEXT_DIR: Path = Path(MODEL_DIR) / "adaptive"
_CONTEXT_FILE: str = "adaptive_context.json"

# ── Threshold bounds ──────────────────────────────────────────────────────────
_BOUNDS: dict[str, tuple[float, float]] = {
    "meta_model_min_confidence": (0.30, 0.90),
    "min_reward_risk": (1.0, 5.0),
    "forecast_confidence_min": (0.10, 0.90),
    "fta_score_min": (0.0, 1.0),
}


@dataclass
class BestThresholds:
    meta_model_min_confidence: float = META_MODEL_MIN_CONFIDENCE
    min_reward_risk: float = FTA_MIN_REWARD_RISK
    forecast_confidence_min: float = 0.50
    fta_score_min: float = 0.0


@dataclass
class RecentPerformance:
    n_sessions: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class RegimeStats:
    regime: str = "unknown"
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0


@dataclass
class TickerStats:
    ticker: str = ""
    n_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0


@dataclass
class AdaptiveContext:
    version: int = 0
    updated_at: str = ""
    best_thresholds: BestThresholds = field(default_factory=BestThresholds)
    recent_performance: RecentPerformance = field(default_factory=RecentPerformance)
    regime_stats: dict[str, RegimeStats] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    per_ticker_stats: dict[str, TickerStats] = field(default_factory=dict)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "AdaptiveContext":
        """Return a fresh AdaptiveContext with sensible defaults from config."""
        ctx = cls(
            version=0,
            updated_at=_now_iso(),
            best_thresholds=BestThresholds(),
            recent_performance=RecentPerformance(),
            regime_stats={},
            feature_importance={},
            per_ticker_stats={},
        )
        ctx.clamp_thresholds()
        return ctx

    @classmethod
    def load(cls) -> "AdaptiveContext":
        """
        Load AdaptiveContext from disk.

        Returns default() if file is missing or corrupt.
        """
        path = CONTEXT_DIR / _CONTEXT_FILE
        if not path.exists():
            log.info("AdaptiveContext: no file at %s — using defaults", path)
            return cls.default()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            ctx = _from_dict(raw)
            ctx.clamp_thresholds()
            return ctx
        except Exception as exc:
            log.warning("AdaptiveContext: failed to load %s: %s — using defaults", path, exc)
            return cls.default()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write context to CONTEXT_DIR / adaptive_context.json (create dir if needed)."""
        CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
        path = CONTEXT_DIR / _CONTEXT_FILE
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(dataclasses.asdict(self), fh, indent=2)
            log.info("AdaptiveContext: saved version %d to %s", self.version, path)
        except Exception as exc:
            log.warning("AdaptiveContext: failed to save to %s: %s", path, exc)

    # ── Bounds enforcement ────────────────────────────────────────────────────

    def clamp_thresholds(self) -> None:
        """Enforce all threshold bounds in-place."""
        t = self.best_thresholds
        lo, hi = _BOUNDS["meta_model_min_confidence"]
        t.meta_model_min_confidence = max(lo, min(hi, t.meta_model_min_confidence))
        lo, hi = _BOUNDS["min_reward_risk"]
        t.min_reward_risk = max(lo, min(hi, t.min_reward_risk))
        lo, hi = _BOUNDS["forecast_confidence_min"]
        t.forecast_confidence_min = max(lo, min(hi, t.forecast_confidence_min))
        lo, hi = _BOUNDS["fta_score_min"]
        t.fta_score_min = max(lo, min(hi, t.fta_score_min))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _from_dict(raw: dict) -> AdaptiveContext:
    """Reconstruct AdaptiveContext from a plain dict (graceful on missing keys)."""
    bt_raw = raw.get("best_thresholds") or {}
    best_thresholds = BestThresholds(
        meta_model_min_confidence=float(bt_raw.get("meta_model_min_confidence", META_MODEL_MIN_CONFIDENCE)),
        min_reward_risk=float(bt_raw.get("min_reward_risk", FTA_MIN_REWARD_RISK)),
        forecast_confidence_min=float(bt_raw.get("forecast_confidence_min", 0.50)),
        fta_score_min=float(bt_raw.get("fta_score_min", 0.0)),
    )

    rp_raw = raw.get("recent_performance") or {}
    recent_performance = RecentPerformance(
        n_sessions=int(rp_raw.get("n_sessions", 0)),
        total_trades=int(rp_raw.get("total_trades", 0)),
        win_rate=float(rp_raw.get("win_rate", 0.0)),
        total_return_pct=float(rp_raw.get("total_return_pct", 0.0)),
        max_drawdown_pct=float(rp_raw.get("max_drawdown_pct", 0.0)),
        sharpe_ratio=float(rp_raw.get("sharpe_ratio", 0.0)),
    )

    regime_stats: dict[str, RegimeStats] = {}
    for k, v in (raw.get("regime_stats") or {}).items():
        if isinstance(v, dict):
            regime_stats[k] = RegimeStats(
                regime=str(v.get("regime", k)),
                n_trades=int(v.get("n_trades", 0)),
                win_rate=float(v.get("win_rate", 0.0)),
                avg_pnl=float(v.get("avg_pnl", 0.0)),
            )

    feature_importance: dict[str, float] = {
        k: float(v) for k, v in (raw.get("feature_importance") or {}).items()
    }

    per_ticker_stats: dict[str, TickerStats] = {}
    for k, v in (raw.get("per_ticker_stats") or {}).items():
        if isinstance(v, dict):
            per_ticker_stats[k] = TickerStats(
                ticker=str(v.get("ticker", k)),
                n_trades=int(v.get("n_trades", 0)),
                win_rate=float(v.get("win_rate", 0.0)),
                total_pnl=float(v.get("total_pnl", 0.0)),
            )

    return AdaptiveContext(
        version=int(raw.get("version", 0)),
        updated_at=str(raw.get("updated_at", _now_iso())),
        best_thresholds=best_thresholds,
        recent_performance=recent_performance,
        regime_stats=regime_stats,
        feature_importance=feature_importance,
        per_ticker_stats=per_ticker_stats,
    )
