"""Human-readable reporting for timeframe policy and news state."""
from __future__ import annotations

from datetime import datetime, timezone

from schemas.news_features import NewsFeatures
from src.timeframe_policy.policy import PolicyDecision


def policy_report(decision: PolicyDecision) -> str:
    """Format a PolicyDecision as a multi-line report string."""
    lines = [
        "── Timeframe Policy ───────────────────────────────────",
        f"  Effective mode      : {decision.effective_mode}",
        f"  Prediction timeframe: {decision.prediction_timeframe}",
        f"  Reason              : {decision.reason}",
        f"  Fell back           : {decision.fell_back}",
    ]
    if decision.warnings:
        lines.append("  Warnings:")
        for w in decision.warnings:
            lines.append(f"    • {w}")
    lines.append("────────────────────────────────────────────────────")
    return "\n".join(lines)


def news_report(features: NewsFeatures) -> str:
    """Format NewsFeatures as a concise summary string."""
    if not features.data_available:
        return (
            "── News Features ─────────────────────────────────────\n"
            "  Status: disabled or unavailable\n"
            "────────────────────────────────────────────────────"
        )

    now = datetime.now(timezone.utc)
    age_min = features.minutes_since_last_news

    lines = [
        "── News Features ─────────────────────────────────────",
        f"  Ticker              : {features.ticker}",
        f"  Computed at         : {features.computed_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Articles (1h / 24h) : {features.news_count_1h} / {features.news_count_24h}",
        f"  Sentiment mean 1h   : {features.sentiment_mean_1h:+.3f}",
        f"  Sentiment mean 24h  : {features.sentiment_mean_24h:+.3f}",
        f"  Sentiment std 24h   : {features.sentiment_std_24h:.3f}",
        f"  Min since last news : {age_min:.0f}",
        f"  Major event         : {features.has_major_event} ({features.event_type})",
        f"  Headline shock score: {features.headline_shock_score:.3f}",
        "────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)
