"""RecommendationEngine — converts pipeline outputs into a TradeRecommendation.

Decision hierarchy (mirrors CLAUDE.md):
  1. Risk appetite outer envelope — hard limits reject first
  2. FTA structural filter — must be accepted
  3. Meta-model learned gate — probability threshold
  4. Forecast confidence gate
  5. News/event risk behavior
  6. Position sizing with all multipliers
  7. Emit recommendation

The engine is deterministic and stateless given its inputs.
It does NOT modify any portfolio state itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from schemas.fta import FTAOutput
from schemas.forecast import ForecastOutput
from schemas.meta_model import MetaModelOutput
from schemas.news_features import NewsFeatures
from schemas.recommendation import TradeRecommendation
from schemas.risk_appetite import RiskAppetiteConfig

log = logging.getLogger(__name__)


@dataclass
class RecommendationContext:
    """
    All inputs the engine needs to produce a recommendation.

    portfolio_equity: current portfolio equity (for sizing)
    open_position_size: shares held in ticker (0 if no position)
    losing_streak: consecutive losing trades in the current session
    trades_today: trades already opened today
    """

    ticker: str
    current_price: float
    forecast: ForecastOutput
    fta_output: FTAOutput
    meta_output: MetaModelOutput
    risk_appetite: RiskAppetiteConfig
    news_features: Optional[NewsFeatures] = None
    timeframe_mode: str = "daily_only"
    news_mode: str = "disabled"
    portfolio_equity: float = 100_000.0
    open_position_size: float = 0.0     # shares currently held (>0 long, <0 short)
    losing_streak: int = 0
    trades_today: int = 0


class RecommendationEngine:
    """Stateless recommendation engine."""

    def evaluate(self, ctx: RecommendationContext) -> TradeRecommendation:
        """
        Produce a TradeRecommendation from the given context.

        All gates are checked in order; the first failure produces a HOLD.
        """
        ra = ctx.risk_appetite
        fta = ctx.fta_output
        meta = ctx.meta_output
        forecast = ctx.forecast

        # Shared metadata for every outcome
        base_kw = dict(
            ticker=ctx.ticker,
            risk_profile=ra.mode,
            timeframe_mode=ctx.timeframe_mode,
            news_mode=ctx.news_mode,
            fta_score=fta.verdict.score if fta else None,
            forecast_confidence=forecast.confidence if forecast else None,
            probability_of_success=meta.probability_of_success if meta else None,
            reward_risk=fta.reward_risk if fta else None,
        )

        # ── Check for existing position exit signals ───────────────────────
        if ctx.open_position_size != 0:
            # Pass a copy of base_kw without ticker to avoid duplicate kwarg
            _exit_kw = {k: v for k, v in base_kw.items() if k != "ticker"}
            close_rec = self._check_exit_signals(ctx, **_exit_kw)
            if close_rec is not None:
                return close_rec

        # ── Gate 1: daily trade count (risk appetite) ──────────────────────
        if ctx.trades_today >= ra.max_trades_per_day:
            return self._hold(
                rejection_reason="max_trades_per_day reached for risk profile",
                rationale=f"Risk profile '{ra.mode}' allows {ra.max_trades_per_day} trades/day; "
                          f"already opened {ctx.trades_today}.",
                **base_kw,
            )

        # ── Gate 2: FTA hard filter ────────────────────────────────────────
        if not fta.verdict.accepted:
            reasons = ", ".join(r.code for r in fta.rejection_reasons) or "structure_rejected"
            return self._hold(
                rejection_reason=f"fta_rejected: {reasons}",
                rationale=f"FTA structural filter rejected the setup: {fta.verdict.summary}",
                **base_kw,
            )

        # ── Gate 3: FTA score threshold ────────────────────────────────────
        if fta.verdict.score < ra.min_fta_score:
            return self._hold(
                rejection_reason=f"fta_score_below_threshold: {fta.verdict.score:.2f} < {ra.min_fta_score}",
                rationale=f"FTA quality score {fta.verdict.score:.2f} is below the "
                          f"'{ra.mode}' minimum of {ra.min_fta_score}.",
                **base_kw,
            )

        # ── Gate 4: Reward:Risk threshold ─────────────────────────────────
        rr = fta.reward_risk or 0.0
        if rr < ra.min_reward_risk:
            return self._hold(
                rejection_reason=f"reward_risk_below_threshold: {rr:.2f} < {ra.min_reward_risk}",
                rationale=f"Reward:Risk {rr:.2f} is below the '{ra.mode}' minimum of {ra.min_reward_risk}.",
                **base_kw,
            )

        # ── Gate 5: Meta-model probability ────────────────────────────────
        prob = meta.probability_of_success if meta else 0.0
        if prob < ra.min_meta_model_probability:
            return self._hold(
                rejection_reason=f"meta_model_probability_below_threshold: {prob:.2f} < {ra.min_meta_model_probability}",
                rationale=f"Meta-model probability {prob:.2f} is below the "
                          f"'{ra.mode}' minimum of {ra.min_meta_model_probability}.",
                **base_kw,
            )

        # ── Gate 6: Forecast confidence ────────────────────────────────────
        conf = forecast.confidence if forecast else 0.0
        if conf < ra.min_forecast_confidence:
            return self._hold(
                rejection_reason=f"forecast_confidence_below_threshold: {conf:.2f} < {ra.min_forecast_confidence}",
                rationale=f"Forecast confidence {conf:.2f} is below the "
                          f"'{ra.mode}' minimum of {ra.min_forecast_confidence}.",
                **base_kw,
            )

        # ── Gate 7: News / event risk ──────────────────────────────────────
        has_event = (
            ctx.news_features is not None
            and ctx.news_features.data_available
            and ctx.news_features.has_major_event
        )
        event_size_mult = 1.0
        if has_event:
            if ra.event_risk_behavior == "skip":
                event_type = ctx.news_features.event_type if ctx.news_features else "unknown"
                return self._hold(
                    rejection_reason=f"event_risk_skip: {event_type}",
                    rationale=f"'{ra.mode}' profile skips new trades around major events "
                              f"(detected: {event_type}).",
                    **base_kw,
                )
            elif ra.event_risk_behavior == "reduce":
                event_size_mult = ra.event_risk_size_multiplier

        # ── All gates passed — build OPEN recommendation ───────────────────
        entry = ctx.current_price
        stop = fta.fta_output_stop(ctx.current_price) if hasattr(fta, "fta_output_stop") else None
        # Derive stop from FTA candidate if available
        if stop is None and fta.nearest_trouble_price is not None:
            # Use trouble price as stop proxy: entry - (entry - trouble) padded 10%
            dist = abs(entry - fta.nearest_trouble_price)
            stop = entry - dist * 1.10 if entry > fta.nearest_trouble_price else entry + dist * 1.10

        # Targets from FTA or forecast
        target_1 = None
        target_2 = None
        if stop is not None:
            risk_per_share = abs(entry - stop)
            target_1 = entry + risk_per_share * ra.min_reward_risk
            target_2 = entry + risk_per_share * ra.min_reward_risk * 1.5
        elif forecast.expected_return != 0:
            target_1 = entry * (1.0 + forecast.expected_return)

        # Expected return
        expected_return: float | None = None
        if target_1 is not None and entry > 0:
            expected_return = (target_1 - entry) / entry

        # Position sizing
        size = self._compute_size(
            equity=ctx.portfolio_equity,
            entry=entry,
            stop=stop,
            ra=ra,
            timeframe_mode=ctx.timeframe_mode,
            event_size_mult=event_size_mult,
            losing_streak=ctx.losing_streak,
        )

        # Trade style
        trade_style = (
            "day_trade"
            if "intraday" in ctx.timeframe_mode or ctx.timeframe_mode == "intraday_experimental_finetune"
            else "swing_trade"
        )

        # Action direction
        if forecast.direction == "up":
            action = "BUY"
        else:
            action = "SELL"

        rationale_parts = [
            f"FTA accepted (score={fta.verdict.score:.2f}, R:R={rr:.2f}).",
            f"Meta-model probability={prob:.2f}.",
            f"Forecast confidence={conf:.2f}, direction={forecast.direction}.",
        ]
        if has_event and event_size_mult < 1.0:
            rationale_parts.append(f"Position reduced to {event_size_mult*100:.0f}% due to major event.")
        if ctx.losing_streak > 0:
            rationale_parts.append(f"Size reduced by losing streak ({ctx.losing_streak} consecutive losses).")

        # base_kw already contains: ticker, risk_profile, timeframe_mode, news_mode,
        # fta_score, forecast_confidence, probability_of_success, reward_risk
        # Override reward_risk in base_kw if we have a computed value
        kw = {**base_kw, "reward_risk": rr if rr > 0 else None}
        return TradeRecommendation(
            action=action,
            position_action="OPEN",
            trade_style=trade_style,
            entry_price=entry,
            stop_price=stop,
            target_1=target_1,
            target_2=target_2,
            expected_return=expected_return,
            recommended_position_size=size,
            **kw,
            rationale=" ".join(rationale_parts),
            rejection_reason=None,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _check_exit_signals(
        self, ctx: RecommendationContext, **base_kw
    ) -> TradeRecommendation | None:
        """
        Inspect open position for exit/reduce signals.
        Returns a recommendation if exit warranted, else None.
        """
        fta = ctx.fta_output
        forecast = ctx.forecast

        is_long = ctx.open_position_size > 0

        # If FTA structure breaks against our position → CLOSE
        if fta.verdict.accepted is False and not fta.verdict.accepted:
            structure_against = False
            for r in fta.rejection_reasons:
                if r.code in ("weak_structure", "bos_against_position"):
                    structure_against = True
            if structure_against:
                action = "SELL" if is_long else "BUY"
                return TradeRecommendation(
                    ticker=ctx.ticker,
                    action=action,
                    position_action="CLOSE",
                    trade_style=None,
                    rationale="FTA structure broken; existing position flagged for exit.",
                    rejection_reason=None,
                    **base_kw,
                )

        # If forecast flips direction against position → REDUCE
        if forecast is not None:
            if is_long and forecast.direction == "down" and forecast.confidence >= 0.65:
                return TradeRecommendation(
                    ticker=ctx.ticker,
                    action="SELL",
                    position_action="REDUCE",
                    trade_style=None,
                    recommended_position_size=abs(ctx.open_position_size) * 0.5,
                    rationale=f"Forecast turned bearish (confidence={forecast.confidence:.2f}); "
                              "reducing long exposure by 50%.",
                    rejection_reason=None,
                    **base_kw,  # base_kw here excludes ticker (caller strips it)
                )
            if not is_long and forecast.direction == "up" and forecast.confidence >= 0.65:
                return TradeRecommendation(
                    ticker=ctx.ticker,
                    action="BUY",
                    position_action="REDUCE",
                    trade_style=None,
                    recommended_position_size=abs(ctx.open_position_size) * 0.5,
                    rationale=f"Forecast turned bullish (confidence={forecast.confidence:.2f}); "
                              "reducing short exposure by 50%.",
                    rejection_reason=None,
                    **base_kw,
                )

        return None

    def _compute_size(
        self,
        equity: float,
        entry: float,
        stop: float | None,
        ra: RiskAppetiteConfig,
        timeframe_mode: str,
        event_size_mult: float,
        losing_streak: int,
    ) -> float | None:
        """
        Compute recommended position size in shares.

        Base size = (equity × risk_per_trade_pct) / risk_per_share
        Then apply: trade style multiplier × event multiplier × streak reduction.
        """
        if entry <= 0:
            return None

        if stop is not None and abs(entry - stop) > 0:
            risk_per_share = abs(entry - stop)
            base_dollars = equity * ra.risk_per_trade_pct
            base_shares = base_dollars / risk_per_share
        else:
            # Fallback: 1% move as risk proxy
            risk_per_share = entry * 0.01
            base_dollars = equity * ra.risk_per_trade_pct
            base_shares = base_dollars / risk_per_share

        # Trade style multiplier
        if "intraday" in timeframe_mode:
            style_mult = ra.day_trade_size_multiplier
        else:
            style_mult = ra.swing_trade_size_multiplier

        # Losing streak reduction
        streak_mult = max(0.10, 1.0 - losing_streak * ra.losing_streak_size_reduction)

        size = base_shares * style_mult * event_size_mult * streak_mult
        return max(1.0, round(size, 2))

    @staticmethod
    def _hold(rejection_reason: str, rationale: str, **base_kw) -> TradeRecommendation:
        return TradeRecommendation(
            action="HOLD",
            position_action="HOLD_POSITION",
            trade_style=None,
            rejection_reason=rejection_reason,
            rationale=rationale,
            **base_kw,
        )
