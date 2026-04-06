"""Timeframe adaptation policy.

Three modes (from config.settings.TIMEFRAME_POLICY):

  daily_only
      Always use daily bars. Safe default. No 1h data required.

  daily_plus_intraday_calibration
      Use daily bars but calibrate FTA thresholds using 1h data if available.
      Requires REQUIRE_REAL_INTRADAY_DATA=False OR real 1h data present.
      Requires at least MIN_INTRADAY_TRADES_FOR_ADAPTATION trades in 1h backtest.

  intraday_experimental_finetune
      Use 1h bars as primary timeframe and attempt meta-model fine-tuning.
      Requires ALLOW_EXPERIMENTAL_FINETUNE=True AND data+sample requirements.

Decision hierarchy (enforce top-down):
  1. If no real 1h data AND REQUIRE_REAL_INTRADAY_DATA → force daily_only
  2. If insufficient 1h trades (<MIN_INTRADAY_TRADES_FOR_ADAPTATION) → at most calibration
  3. If ALLOW_EXPERIMENTAL_FINETUNE=False → at most calibration
  4. Otherwise: use configured policy
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class TimeframePolicyMode(str, Enum):
    DAILY_ONLY = "daily_only"
    DAILY_PLUS_INTRADAY_CALIBRATION = "daily_plus_intraday_calibration"
    INTRADAY_EXPERIMENTAL_FINETUNE = "intraday_experimental_finetune"


@dataclass
class PolicyDecision:
    effective_mode: str           # one of TimeframePolicyMode values
    prediction_timeframe: str     # "1d" or "1h"
    reason: str                   # human-readable explanation
    warnings: list[str] = field(default_factory=list)
    fell_back: bool = False       # True if policy was downgraded from configured


class TimeframePolicy:
    """
    Stateless policy evaluator. Call .decide() with current system state
    to get a PolicyDecision.
    """

    def decide(
        self,
        configured_policy: str | None = None,
        has_real_intraday_data: bool = False,
        intraday_trade_count: int = 0,
    ) -> PolicyDecision:
        """
        Evaluate policy given current data state.

        Parameters
        ----------
        configured_policy      : override for config.settings.TIMEFRAME_POLICY
        has_real_intraday_data : True if the local store has at least one real 1h bar
        intraday_trade_count   : number of 1h backtest trades available for calibration
        """
        from config import settings

        policy = configured_policy or settings.TIMEFRAME_POLICY
        require_real = settings.REQUIRE_REAL_INTRADAY_DATA
        min_trades = settings.MIN_INTRADAY_TRADES_FOR_ADAPTATION
        allow_finetune = settings.ALLOW_EXPERIMENTAL_FINETUNE
        pred_tf = settings.DEFAULT_PREDICTION_TIMEFRAME
        intraday_tf = settings.INTRADAY_TARGET_TIMEFRAME

        warnings: list[str] = []

        # Normalise
        try:
            mode = TimeframePolicyMode(policy)
        except ValueError:
            warnings.append(f"Unknown TIMEFRAME_POLICY '{policy}'; defaulting to daily_only")
            mode = TimeframePolicyMode.DAILY_ONLY

        # Gating rule 1: no real data + require_real → force daily_only
        if mode != TimeframePolicyMode.DAILY_ONLY and require_real and not has_real_intraday_data:
            warnings.append(
                f"Configured policy '{mode.value}' requires real 1h data "
                f"(REQUIRE_REAL_INTRADAY_DATA=True) but none found. Falling back to daily_only."
            )
            return PolicyDecision(
                effective_mode=TimeframePolicyMode.DAILY_ONLY.value,
                prediction_timeframe=pred_tf,
                reason="no_real_intraday_data",
                warnings=warnings,
                fell_back=True,
            )

        # Gating rule 2: insufficient trades for calibration
        if mode in (
            TimeframePolicyMode.DAILY_PLUS_INTRADAY_CALIBRATION,
            TimeframePolicyMode.INTRADAY_EXPERIMENTAL_FINETUNE,
        ) and intraday_trade_count < min_trades:
            warnings.append(
                f"Intraday trade count ({intraday_trade_count}) < "
                f"MIN_INTRADAY_TRADES_FOR_ADAPTATION ({min_trades}). "
                f"Falling back to daily_only."
            )
            return PolicyDecision(
                effective_mode=TimeframePolicyMode.DAILY_ONLY.value,
                prediction_timeframe=pred_tf,
                reason="insufficient_intraday_trades",
                warnings=warnings,
                fell_back=True,
            )

        # Gating rule 3: finetune requires flag
        if mode == TimeframePolicyMode.INTRADAY_EXPERIMENTAL_FINETUNE and not allow_finetune:
            warnings.append(
                "ALLOW_EXPERIMENTAL_FINETUNE=False. Downgrading to daily_plus_intraday_calibration."
            )
            mode = TimeframePolicyMode.DAILY_PLUS_INTRADAY_CALIBRATION

        # Determine prediction timeframe
        if mode == TimeframePolicyMode.INTRADAY_EXPERIMENTAL_FINETUNE:
            effective_tf = intraday_tf
            reason = "intraday_finetune_enabled"
        elif mode == TimeframePolicyMode.DAILY_PLUS_INTRADAY_CALIBRATION:
            effective_tf = pred_tf   # still daily predictions, but calibrated with 1h
            reason = "daily_with_intraday_calibration"
        else:
            effective_tf = pred_tf
            reason = "daily_only"

        return PolicyDecision(
            effective_mode=mode.value,
            prediction_timeframe=effective_tf,
            reason=reason,
            warnings=warnings,
            fell_back=False,
        )


def get_timeframe_policy() -> TimeframePolicy:
    """Factory — returns a TimeframePolicy instance."""
    return TimeframePolicy()
