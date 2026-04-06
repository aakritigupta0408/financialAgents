"""Phase 17 — Recommendation Engine and Configurable Risk Appetite tests."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from schemas.fta import FTAOutput, FTAVerdict, FTARejectionReason
from schemas.forecast import ForecastOutput
from schemas.meta_model import MetaModelOutput
from schemas.news_features import NewsFeatures
from schemas.recommendation import TradeRecommendation
from schemas.risk_appetite import RiskAppetiteConfig
from src.recommendation.engine import RecommendationEngine, RecommendationContext
from src.risk_appetite.presets import CONSERVATIVE, MODERATE, AGGRESSIVE, get_preset
from src.risk_appetite.loader import load_risk_appetite


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_fta(accepted=True, score=0.72, rr=2.5, dist_pct=0.008, trouble_price=195.0,
              rejection_codes=None) -> FTAOutput:
    rejection_reasons = [FTARejectionReason(code=c, detail="") for c in (rejection_codes or [])]
    return FTAOutput(
        ticker="AAPL",
        evaluated_at=datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc),
        verdict=FTAVerdict(
            accepted=accepted,
            score=score,
            summary="accepted" if accepted else "rejected",
        ),
        nearest_trouble_price=trouble_price,
        distance_to_fta_pct=dist_pct,
        reward_risk=rr if accepted else None,
        structure_score=0.75,
        liquidity_score=0.80,
        volatility_ok=True,
        rejection_reasons=rejection_reasons,
    )


def _make_forecast(direction="up", confidence=0.62, expected_return=0.04) -> ForecastOutput:
    return ForecastOutput(
        ticker="AAPL",
        timeframe="1d",
        horizon=5,
        direction=direction,
        confidence=confidence,
        expected_return=expected_return,
    )


def _make_meta(prob=0.65, should_trade=True) -> MetaModelOutput:
    return MetaModelOutput(
        ticker="AAPL",
        evaluated_at=datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc),
        probability_of_success=prob,
        confidence=prob,
        should_trade=should_trade,
    )


def _ctx(ra: RiskAppetiteConfig, **overrides) -> RecommendationContext:
    defaults = dict(
        ticker="AAPL",
        current_price=200.0,
        forecast=_make_forecast(),
        fta_output=_make_fta(),
        meta_output=_make_meta(),
        risk_appetite=ra,
        news_features=None,
        timeframe_mode="daily_only",
        news_mode="disabled",
        portfolio_equity=100_000.0,
        open_position_size=0.0,
        losing_streak=0,
        trades_today=0,
    )
    defaults.update(overrides)
    return RecommendationContext(**defaults)


# ── Risk appetite schema tests ────────────────────────────────────────────────

def test_risk_appetite_conservative_fields():
    assert CONSERVATIVE.mode == "conservative"
    assert CONSERVATIVE.risk_per_trade_pct < MODERATE.risk_per_trade_pct
    assert CONSERVATIVE.min_meta_model_probability > MODERATE.min_meta_model_probability
    assert CONSERVATIVE.max_trades_per_day < MODERATE.max_trades_per_day
    assert CONSERVATIVE.event_risk_behavior == "skip"


def test_risk_appetite_aggressive_fields():
    assert AGGRESSIVE.mode == "aggressive"
    assert AGGRESSIVE.risk_per_trade_pct > MODERATE.risk_per_trade_pct
    assert AGGRESSIVE.min_fta_score < MODERATE.min_fta_score
    assert AGGRESSIVE.max_trades_per_day > MODERATE.max_trades_per_day
    assert AGGRESSIVE.event_risk_behavior == "normal"


def test_get_preset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown risk appetite"):
        get_preset("ultra_yolo")


def test_load_risk_appetite_from_settings():
    with patch("config.settings.RISK_APPETITE_MODE", "conservative"):
        cfg = load_risk_appetite()
    assert cfg.mode == "conservative"


def test_load_risk_appetite_unknown_falls_back_to_moderate():
    with patch("config.settings.RISK_APPETITE_MODE", "bogus"):
        cfg = load_risk_appetite()
    assert cfg.mode == "moderate"


def test_risk_appetite_to_risk_config():
    rc = MODERATE.to_risk_config()
    assert rc.risk_per_trade_pct == MODERATE.risk_per_trade_pct
    assert rc.max_trades_per_day == MODERATE.max_trades_per_day


# ── Recommendation schema tests ───────────────────────────────────────────────

def test_recommendation_is_actionable():
    rec = TradeRecommendation(
        ticker="AAPL",
        action="BUY",
        position_action="OPEN",
        risk_profile="moderate",
        timeframe_mode="daily_only",
        news_mode="disabled",
        rationale="test",
    )
    assert rec.is_actionable is True


def test_recommendation_hold_not_actionable():
    rec = TradeRecommendation(
        ticker="AAPL",
        action="HOLD",
        position_action="HOLD_POSITION",
        risk_profile="moderate",
        timeframe_mode="daily_only",
        news_mode="disabled",
        rationale="test",
        rejection_reason="fta_rejected",
    )
    assert rec.is_actionable is False


# ── Engine: HOLD scenarios ────────────────────────────────────────────────────

def test_hold_when_fta_rejected():
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE, fta_output=_make_fta(accepted=False, score=0.4))
    rec = engine.evaluate(ctx)
    assert rec.action == "HOLD"
    assert rec.rejection_reason is not None
    assert "fta_rejected" in rec.rejection_reason


def test_hold_when_meta_model_below_threshold():
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE, meta_output=_make_meta(prob=0.40))
    rec = engine.evaluate(ctx)
    assert rec.action == "HOLD"
    assert "meta_model_probability" in rec.rejection_reason


def test_hold_when_trades_today_at_limit():
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE, trades_today=MODERATE.max_trades_per_day)
    rec = engine.evaluate(ctx)
    assert rec.action == "HOLD"
    assert "max_trades_per_day" in rec.rejection_reason


def test_hold_when_reward_risk_below_threshold():
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE, fta_output=_make_fta(accepted=True, rr=1.0, score=0.75))
    rec = engine.evaluate(ctx)
    assert rec.action == "HOLD"
    assert "reward_risk" in rec.rejection_reason


def test_hold_when_forecast_confidence_low():
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE, forecast=_make_forecast(confidence=0.30))
    rec = engine.evaluate(ctx)
    assert rec.action == "HOLD"
    assert "forecast_confidence" in rec.rejection_reason


# ── Engine: OPEN recommendation ───────────────────────────────────────────────

def test_open_recommendation_all_gates_pass():
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE)
    rec = engine.evaluate(ctx)
    assert rec.action == "BUY"
    assert rec.position_action == "OPEN"
    assert rec.entry_price == pytest.approx(200.0)
    assert rec.stop_price is not None
    assert rec.target_1 is not None
    assert rec.recommended_position_size is not None and rec.recommended_position_size >= 1.0
    assert rec.rejection_reason is None


def test_open_sell_when_forecast_down():
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE, forecast=_make_forecast(direction="down", confidence=0.65))
    rec = engine.evaluate(ctx)
    assert rec.action == "SELL"
    assert rec.position_action == "OPEN"


# ── Engine: Part E validation — same trade under three profiles ───────────────

def test_same_candidate_conservative_vs_moderate_vs_aggressive():
    """
    Part E: show the same candidate trade evaluated under all three profiles.
    Conservative may reject what moderate and aggressive accept.
    """
    engine = RecommendationEngine()

    # FTA score and meta-model prob set between conservative (0.70) and moderate (0.60) thresholds
    fta = _make_fta(accepted=True, score=0.65, rr=2.2)
    meta = _make_meta(prob=0.63)
    forecast = _make_forecast(confidence=0.65)

    def run(ra):
        return engine.evaluate(_ctx(ra, fta_output=fta, meta_output=meta, forecast=forecast))

    rec_cons = run(CONSERVATIVE)
    rec_mod = run(MODERATE)
    rec_agg = run(AGGRESSIVE)

    # Conservative rejects (score 0.65 < 0.70, prob 0.63 < 0.70)
    assert rec_cons.action == "HOLD", f"Expected HOLD for conservative, got {rec_cons.action}"
    # Moderate accepts (score 0.65 >= 0.60, prob 0.63 >= 0.60)
    assert rec_mod.action == "BUY"
    # Aggressive accepts (thresholds 0.50)
    assert rec_agg.action == "BUY"

    # Sizing: aggressive should give larger position than moderate
    assert rec_agg.recommended_position_size is not None
    assert rec_mod.recommended_position_size is not None
    assert rec_agg.recommended_position_size >= rec_mod.recommended_position_size


def test_conservative_skips_event_risk():
    """Conservative profile with event_risk_behavior='skip' must HOLD on major event.
    Use signals that clear ALL conservative thresholds so only event gate can block."""
    engine = RecommendationEngine()
    news = NewsFeatures(
        ticker="AAPL",
        computed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        has_major_event=True,
        event_type="earnings",
        data_available=True,
    )
    # All signals above conservative thresholds: fta_score>=0.70, meta>=0.70, rr>=2.5, conf>=0.70
    fta = _make_fta(accepted=True, score=0.75, rr=2.6)
    meta = _make_meta(prob=0.75)
    forecast = _make_forecast(confidence=0.75)
    ctx = _ctx(
        CONSERVATIVE,
        fta_output=fta,
        meta_output=meta,
        forecast=forecast,
        news_features=news,
        news_mode="risk_filter",
    )
    rec = engine.evaluate(ctx)
    assert rec.action == "HOLD"
    assert "event_risk_skip" in rec.rejection_reason


def test_moderate_reduces_size_on_event():
    """Moderate profile with event_risk_behavior='reduce' opens but with smaller size."""
    engine = RecommendationEngine()
    news = NewsFeatures(
        ticker="AAPL",
        computed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        has_major_event=True,
        event_type="earnings",
        data_available=True,
    )
    ctx_no_event = _ctx(MODERATE)
    ctx_event = _ctx(MODERATE, news_features=news, news_mode="risk_filter")

    rec_no_event = engine.evaluate(ctx_no_event)
    rec_event = engine.evaluate(ctx_event)

    assert rec_no_event.action == "BUY"
    assert rec_event.action == "BUY"  # moderate still opens
    # Event position should be smaller due to event_risk_size_multiplier=0.5
    assert rec_event.recommended_position_size < rec_no_event.recommended_position_size


# ── Engine: CLOSE / REDUCE recommendations ────────────────────────────────────

def test_reduce_when_forecast_flips_against_long():
    """Forecast turning bearish with high confidence on an open long → REDUCE."""
    engine = RecommendationEngine()
    ctx = _ctx(
        MODERATE,
        open_position_size=100.0,  # long position
        forecast=_make_forecast(direction="down", confidence=0.70),
    )
    rec = engine.evaluate(ctx)
    assert rec.position_action == "REDUCE"
    assert rec.action == "SELL"
    assert rec.recommended_position_size == pytest.approx(50.0)


def test_no_reduce_when_forecast_confidence_low():
    """Forecast flip with low confidence should not trigger REDUCE."""
    engine = RecommendationEngine()
    ctx = _ctx(
        MODERATE,
        open_position_size=100.0,
        forecast=_make_forecast(direction="down", confidence=0.55),
    )
    rec = engine.evaluate(ctx)
    # Should not reduce — confidence below the 0.65 reduce trigger
    # May still HOLD if FTA score / meta score pass all gates (which they do in default)
    assert rec.position_action != "REDUCE"


# ── Engine: losing streak reduces size ────────────────────────────────────────

def test_losing_streak_reduces_position_size():
    engine = RecommendationEngine()
    ctx_0 = _ctx(MODERATE, losing_streak=0)
    ctx_3 = _ctx(MODERATE, losing_streak=3)

    rec_0 = engine.evaluate(ctx_0)
    rec_3 = engine.evaluate(ctx_3)

    assert rec_0.action == "BUY"
    assert rec_3.action == "BUY"
    assert rec_3.recommended_position_size < rec_0.recommended_position_size


# ── Reporter ──────────────────────────────────────────────────────────────────

def test_format_recommendation_open():
    from src.recommendation.reporter import format_recommendation
    engine = RecommendationEngine()
    rec = engine.evaluate(_ctx(MODERATE))
    report = format_recommendation(rec)
    assert "AAPL" in report
    assert "BUY" in report
    assert "OPEN" in report
    assert "moderate" in report


def test_format_recommendation_hold():
    from src.recommendation.reporter import format_recommendation
    engine = RecommendationEngine()
    ctx = _ctx(MODERATE, fta_output=_make_fta(accepted=False, score=0.3))
    rec = engine.evaluate(ctx)
    report = format_recommendation(rec)
    assert "HOLD" in report
    assert "Rejection" in report
