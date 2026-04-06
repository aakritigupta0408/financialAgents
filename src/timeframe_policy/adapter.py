"""Calibration adapter — applies intraday-derived thresholds to the live config.

Currently a passthrough stub (Phase 15 dual-calibration writes thresholds to
AdaptiveContext; this module reads them back and optionally patches runtime config).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CalibrationAdapterResult:
    applied: bool
    fta_rr_used: float
    fta_dist_pct_used: float
    source: str       # "intraday_calibration" | "defaults" | "adaptive_context"
    notes: str = ""


def apply_calibration(policy_mode: str) -> CalibrationAdapterResult:
    """
    Apply calibration thresholds appropriate for `policy_mode`.

    For daily_only: no-op, use config defaults.
    For daily_plus_intraday_calibration / intraday_experimental_finetune:
      attempt to load best thresholds from AdaptiveContext (Phase 15).
      Falls back to config defaults if context unavailable.
    """
    from config import settings

    default_rr = settings.FTA_MIN_REWARD_RISK
    default_dist = settings.FTA_MIN_DISTANCE_TO_FTA_PCT

    if policy_mode == "daily_only":
        return CalibrationAdapterResult(
            applied=False,
            fta_rr_used=default_rr,
            fta_dist_pct_used=default_dist,
            source="defaults",
            notes="daily_only mode; calibration skipped",
        )

    # Try to load from AdaptiveContext
    try:
        from src.adaptive.context import AdaptiveContext
        ctx = AdaptiveContext()
        best = ctx.load_best_thresholds()
        if best is not None:
            log.info(
                "calibration_adapter.loaded: rr=%.2f dist=%.4f from adaptive_context",
                best.fta_min_rr,
                best.fta_min_dist_pct,
            )
            return CalibrationAdapterResult(
                applied=True,
                fta_rr_used=best.fta_min_rr,
                fta_dist_pct_used=best.fta_min_dist_pct,
                source="adaptive_context",
                notes=f"loaded from context; score={best.score:.4f}",
            )
    except Exception as e:
        log.debug("calibration_adapter.context_unavailable: %s", e)

    # Fallback to defaults with a note
    return CalibrationAdapterResult(
        applied=False,
        fta_rr_used=default_rr,
        fta_dist_pct_used=default_dist,
        source="defaults",
        notes=f"policy={policy_mode} but no calibration data; using defaults",
    )
