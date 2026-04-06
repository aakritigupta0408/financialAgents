"""Human-readable formatting for TradeRecommendation."""
from __future__ import annotations

from schemas.recommendation import TradeRecommendation


def format_recommendation(rec: TradeRecommendation) -> str:
    """Return a multi-line text report of a TradeRecommendation."""

    def _fmt(v, fmt=".4f", prefix="", suffix="") -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{prefix}{v:{fmt}}{suffix}"
        return f"{prefix}{v}{suffix}"

    action_line = f"{rec.action} / {rec.position_action}"
    if rec.trade_style:
        action_line += f" [{rec.trade_style}]"

    lines = [
        "═" * 56,
        f"  RECOMMENDATION  {rec.ticker}",
        "═" * 56,
        f"  Action          : {action_line}",
        f"  Risk profile    : {rec.risk_profile}",
        f"  Timeframe mode  : {rec.timeframe_mode}",
        f"  News mode       : {rec.news_mode}",
        "─" * 56,
        f"  Entry price     : {_fmt(rec.entry_price, '.2f', '$')}",
        f"  Stop price      : {_fmt(rec.stop_price, '.2f', '$')}",
        f"  Target 1        : {_fmt(rec.target_1, '.2f', '$')}",
        f"  Target 2        : {_fmt(rec.target_2, '.2f', '$')}",
        f"  Expected return : {_fmt(rec.expected_return, '.2%')}",
        f"  Reward:Risk     : {_fmt(rec.reward_risk, '.2f')}",
        f"  Position size   : {_fmt(rec.recommended_position_size, '.0f', suffix=' shares')}",
        "─" * 56,
        f"  FTA score       : {_fmt(rec.fta_score, '.3f')}",
        f"  Meta-model prob : {_fmt(rec.probability_of_success, '.3f')}",
        f"  Forecast conf   : {_fmt(rec.forecast_confidence, '.3f')}",
        "─" * 56,
        f"  Rationale: {rec.rationale}",
    ]
    if rec.rejection_reason:
        lines.append(f"  Rejection: {rec.rejection_reason}")
    lines.append("═" * 56)
    return "\n".join(lines)
