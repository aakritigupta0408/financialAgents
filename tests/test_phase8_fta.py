"""Phase 8 — FTA engine tests.

All tests use synthetic Pydantic model instances only.
No API calls, no live data.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from schemas.fta import FTACandidate, FTAInput, FTAOutput
from schemas.features import (
    BOSEvent,
    LevelFeatures,
    LiquidityFeatures,
    PriceZone,
    StructureFeatures,
    VolatilityFeatures,
)
from schemas.forecast import ForecastOutput
from src.fta import build_fta_input, evaluate

# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 4, 5, 9, 30, 0)


def _make_structure(
    trend_state: str = "uptrend",
    trend_strength: float = 0.8,
    bos_direction: str | None = "bullish",
) -> StructureFeatures:
    bos_events = []
    if bos_direction is not None:
        bos_events = [
            BOSEvent(
                timestamp=_TS,
                direction=bos_direction,
                broken_level=100.0,
                confirmation_close=101.0,
            )
        ]
    return StructureFeatures(
        ticker="AAPL",
        timeframe="1d",
        trend_state=trend_state,
        trend_strength=trend_strength,
        bos_events=bos_events,
    )


def _make_levels(
    resistance_zones: list[PriceZone] | None = None,
    support_zones: list[PriceZone] | None = None,
) -> LevelFeatures:
    return LevelFeatures(
        ticker="AAPL",
        timeframe="1d",
        resistance_zones=resistance_zones or [],
        support_zones=support_zones or [],
    )


def _make_volatility(
    atr: float = 2.0,
    atr_pct: float = 0.015,
    regime: str = "normal",
) -> VolatilityFeatures:
    return VolatilityFeatures(
        ticker="AAPL",
        timeframe="1d",
        atr=atr,
        atr_pct=atr_pct,
        volatility_regime=regime,
    )


def _make_liquidity(
    relative_volume: float = 1.5,
    spread_estimate: float = 0.01,
) -> LiquidityFeatures:
    return LiquidityFeatures(
        ticker="AAPL",
        timeframe="1d",
        avg_volume=1_000_000,
        relative_volume=relative_volume,
        spread_estimate=spread_estimate,
    )


def _make_forecast(
    direction: str = "up",
    expected_return: float = 0.04,
    confidence: float = 0.75,
) -> ForecastOutput:
    return ForecastOutput(
        ticker="AAPL",
        timeframe="1d",
        direction=direction,
        expected_return=expected_return,
        confidence=confidence,
        horizon=5,
    )


def _make_candidate(
    side: str = "long",
    entry_price: float = 150.0,
    stop_price: float = 145.0,
) -> FTACandidate:
    return FTACandidate(
        ticker="AAPL",
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
    )


def _good_long_input() -> FTAInput:
    """A well-formed long setup that should be accepted."""
    # Resistance zone at 160–162 (midpoint 161 > 150)
    res_zone = PriceZone(low=160.0, high=162.0, strength=0.8, zone_type="resistance")
    return FTAInput(
        candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
        structure=_make_structure(trend_state="uptrend", trend_strength=0.8, bos_direction="bullish"),
        levels=_make_levels(resistance_zones=[res_zone]),
        volatility=_make_volatility(atr=2.0, atr_pct=0.015, regime="normal"),
        liquidity=_make_liquidity(relative_volume=1.5, spread_estimate=0.01),
        # expected_return=0.04 → expected_price=156.0, trouble=162.0
        # margin = (156.0 - 162.0) / 150 = -0.04  → fta NOT cleared unless we overshoot
        # Use a bigger expected_return so expected_price exceeds zone high
        forecast=_make_forecast(direction="up", expected_return=0.10),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFTAEvaluateReturnsOutput:
    """1. evaluate() returns FTAOutput with all required fields populated."""

    def test_fta_evaluate_returns_ftaoutput(self):
        fta_input = _good_long_input()
        result = evaluate(fta_input)

        assert isinstance(result, FTAOutput)
        assert result.ticker == "AAPL"
        assert result.verdict is not None
        assert isinstance(result.verdict.accepted, bool)
        assert isinstance(result.verdict.score, float)
        assert isinstance(result.verdict.summary, str)
        assert result.nearest_trouble_price is not None
        assert result.distance_to_fta_pct is not None
        assert result.reward_risk is not None
        assert result.structure_score is not None
        assert result.liquidity_score is not None
        assert result.volatility_ok is not None
        assert isinstance(result.rejection_reasons, list)


class TestFTAAcceptsGoodSetup:
    """2. Good uptrend long setup → accepted."""

    def test_fta_long_accepts_good_setup(self):
        # Resistance at 165–167 (midpoint 166 > 150), entry=150, stop=145
        # expected_return=0.15 → expected_price=172.5, trouble=167
        # margin = (172.5 - 167) / 150 = 0.0367 > 0.005 ✓
        # risk = 5, expected_move = 150*0.15=22.5, fta_dist=167-150=17 → reward=max(22.5, 17*0.8)=max(22.5,13.6)=22.5
        # RR = 22.5/5 = 4.5 > 2.0 ✓
        res_zone = PriceZone(low=165.0, high=167.0, strength=0.9, zone_type="resistance")
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(trend_state="uptrend", trend_strength=0.8, bos_direction="bullish"),
            levels=_make_levels(resistance_zones=[res_zone]),
            volatility=_make_volatility(atr=2.0, atr_pct=0.015, regime="normal"),
            liquidity=_make_liquidity(relative_volume=1.5, spread_estimate=0.01),
            forecast=_make_forecast(direction="up", expected_return=0.15),
        )
        result = evaluate(fta_input)
        assert result.verdict.accepted is True
        assert result.verdict.summary == "ACCEPTED"
        assert result.rejection_reasons == []


class TestFTARejectsPoorRR:
    """3. Low reward:risk → REWARD_RISK_BELOW_THRESHOLD."""

    def test_fta_rejects_poor_rr(self):
        # entry=150, stop=148 → risk=2
        # expected_return=0.005 → expected_move=0.75; fta_dist small → reward tiny
        # Also set large resistance zone very close so fta_dist is small
        res_zone = PriceZone(low=150.5, high=151.0, strength=0.5, zone_type="resistance")
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=148.0),
            structure=_make_structure(trend_state="uptrend", trend_strength=0.8),
            levels=_make_levels(resistance_zones=[res_zone]),
            volatility=_make_volatility(atr=2.0, atr_pct=0.015, regime="normal"),
            liquidity=_make_liquidity(relative_volume=1.5, spread_estimate=0.01),
            # expected_return tiny → reward tiny relative to risk=2
            forecast=_make_forecast(direction="up", expected_return=0.002),
        )
        result = evaluate(fta_input)
        codes = [r.code for r in result.rejection_reasons]
        assert "REWARD_RISK_BELOW_THRESHOLD" in codes
        assert result.verdict.accepted is False


class TestFTARejectsWeakStructure:
    """4. Unknown trend + zero trend_strength → WEAK_STRUCTURE."""

    def test_fta_rejects_weak_structure(self):
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(trend_state="unknown", trend_strength=0.0, bos_direction=None),
            levels=_make_levels(),
            volatility=_make_volatility(atr=2.0, atr_pct=0.015, regime="normal"),
            liquidity=_make_liquidity(relative_volume=1.5, spread_estimate=0.01),
            forecast=_make_forecast(direction="up", expected_return=0.10),
        )
        result = evaluate(fta_input)
        codes = [r.code for r in result.rejection_reasons]
        assert "WEAK_STRUCTURE" in codes
        assert result.verdict.accepted is False


class TestFTARejectsPoorLiquidity:
    """5. relative_volume=0.1 → POOR_LIQUIDITY."""

    def test_fta_rejects_poor_liquidity(self):
        res_zone = PriceZone(low=165.0, high=167.0, strength=0.9, zone_type="resistance")
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(trend_state="uptrend", trend_strength=0.8, bos_direction="bullish"),
            levels=_make_levels(resistance_zones=[res_zone]),
            volatility=_make_volatility(atr=2.0, atr_pct=0.015, regime="normal"),
            liquidity=_make_liquidity(relative_volume=0.1, spread_estimate=0.01),
            forecast=_make_forecast(direction="up", expected_return=0.15),
        )
        result = evaluate(fta_input)
        codes = [r.code for r in result.rejection_reasons]
        assert "POOR_LIQUIDITY" in codes
        assert result.verdict.accepted is False


class TestFTARejectsUnsuitableVolatility:
    """6. volatility_regime='extreme' → UNSUITABLE_VOLATILITY."""

    def test_fta_rejects_unsuitable_volatility(self):
        res_zone = PriceZone(low=165.0, high=167.0, strength=0.9, zone_type="resistance")
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(trend_state="uptrend", trend_strength=0.8, bos_direction="bullish"),
            levels=_make_levels(resistance_zones=[res_zone]),
            volatility=_make_volatility(atr=2.0, atr_pct=0.015, regime="extreme"),
            liquidity=_make_liquidity(relative_volume=1.5, spread_estimate=0.01),
            forecast=_make_forecast(direction="up", expected_return=0.15),
        )
        result = evaluate(fta_input)
        codes = [r.code for r in result.rejection_reasons]
        assert "UNSUITABLE_VOLATILITY" in codes
        assert result.verdict.accepted is False


class TestFTAComputesFTADistance:
    """7. nearest_trouble_price and distance_to_fta_pct are floats > 0."""

    def test_fta_computes_fta_distance(self):
        fta_input = _good_long_input()
        result = evaluate(fta_input)
        assert isinstance(result.nearest_trouble_price, float)
        assert isinstance(result.distance_to_fta_pct, float)
        assert result.nearest_trouble_price > 0.0
        assert result.distance_to_fta_pct > 0.0

    def test_fta_distance_long_uses_zone_high(self):
        res_zone = PriceZone(low=160.0, high=163.0, strength=0.7, zone_type="resistance")
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(),
            levels=_make_levels(resistance_zones=[res_zone]),
            volatility=_make_volatility(),
            liquidity=_make_liquidity(),
            forecast=_make_forecast(expected_return=0.10),
        )
        result = evaluate(fta_input)
        # trouble = zone.high = 163.0
        assert result.nearest_trouble_price == pytest.approx(163.0)
        assert result.distance_to_fta_pct == pytest.approx((163.0 - 150.0) / 150.0)

    def test_fta_distance_long_open_air_proxy(self):
        """No resistance above entry → proxy at entry * 1.10."""
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(),
            levels=_make_levels(resistance_zones=[]),
            volatility=_make_volatility(),
            liquidity=_make_liquidity(),
            forecast=_make_forecast(expected_return=0.15),
        )
        result = evaluate(fta_input)
        assert result.nearest_trouble_price == pytest.approx(150.0 * 1.10)

    def test_fta_distance_short_uses_zone_low(self):
        sup_zone = PriceZone(low=138.0, high=140.0, strength=0.7, zone_type="support")
        fta_input = FTAInput(
            candidate=_make_candidate(side="short", entry_price=150.0, stop_price=155.0),
            structure=_make_structure(trend_state="downtrend", bos_direction="bearish"),
            levels=_make_levels(support_zones=[sup_zone]),
            volatility=_make_volatility(),
            liquidity=_make_liquidity(),
            forecast=_make_forecast(direction="down", expected_return=-0.12),
        )
        result = evaluate(fta_input)
        # trouble = zone.low = 138.0
        assert result.nearest_trouble_price == pytest.approx(138.0)
        assert result.distance_to_fta_pct == pytest.approx((150.0 - 138.0) / 150.0)

    def test_fta_distance_short_open_air_proxy(self):
        """No support below entry → proxy at entry * 0.90."""
        fta_input = FTAInput(
            candidate=_make_candidate(side="short", entry_price=150.0, stop_price=155.0),
            structure=_make_structure(trend_state="downtrend", bos_direction="bearish"),
            levels=_make_levels(support_zones=[]),
            volatility=_make_volatility(),
            liquidity=_make_liquidity(),
            forecast=_make_forecast(direction="down", expected_return=-0.15),
        )
        result = evaluate(fta_input)
        assert result.nearest_trouble_price == pytest.approx(150.0 * 0.90)


class TestFTAStructureScoreUptrend:
    """8. Uptrend + high trend_strength → structure_score > 0.5."""

    def test_fta_structure_score_uptrend(self):
        res_zone = PriceZone(low=165.0, high=167.0, strength=0.9, zone_type="resistance")
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(trend_state="uptrend", trend_strength=0.9, bos_direction="bullish"),
            levels=_make_levels(resistance_zones=[res_zone]),
            volatility=_make_volatility(),
            liquidity=_make_liquidity(),
            forecast=_make_forecast(expected_return=0.15),
        )
        result = evaluate(fta_input)
        assert result.structure_score is not None
        assert result.structure_score > 0.5

    def test_fta_structure_score_downtrend_long_is_low(self):
        """Downtrend + long → trend_component=0.1 → score should be low."""
        fta_input = FTAInput(
            candidate=_make_candidate(side="long", entry_price=150.0, stop_price=145.0),
            structure=_make_structure(trend_state="downtrend", trend_strength=0.0, bos_direction="bearish"),
            levels=_make_levels(),
            volatility=_make_volatility(),
            liquidity=_make_liquidity(),
            forecast=_make_forecast(expected_return=0.10),
        )
        result = evaluate(fta_input)
        # trend_component=0.1, trend_strength=0.0, bos_component=-0.2
        # score = 0.1*0.5 + 0*0.3 + (-0.2)*0.2 = 0.05 - 0.04 = 0.01 → clamped to 0.01
        assert result.structure_score < 0.35


class TestBuildFTAInput:
    """9. build_fta_input() constructs valid FTAInput from feature dicts."""

    def test_build_fta_input_from_dicts(self):
        structure = _make_structure()
        levels = _make_levels(
            resistance_zones=[PriceZone(low=160.0, high=162.0, strength=0.7, zone_type="resistance")]
        )
        volatility = _make_volatility()
        liquidity = _make_liquidity()
        forecast = _make_forecast(expected_return=0.10)

        features = {
            "structure": structure,
            "levels": levels,
            "volatility": volatility,
            "liquidity": liquidity,
        }
        candidate = {
            "side": "long",
            "entry": 150.0,
            "stop": 145.0,
            "target": 165.0,
            "reward_risk": 3.0,
            "forecast_confidence": 0.75,
        }

        fta_input = build_fta_input(features, forecast, candidate, ticker="AAPL")

        assert isinstance(fta_input, FTAInput)
        assert fta_input.candidate.ticker == "AAPL"
        assert fta_input.candidate.side == "long"
        assert fta_input.candidate.entry_price == 150.0
        assert fta_input.candidate.stop_price == 145.0
        assert fta_input.structure is structure
        assert fta_input.levels is levels
        assert fta_input.volatility is volatility
        assert fta_input.liquidity is liquidity
        assert fta_input.forecast is forecast


class TestFTADeterministic:
    """10. Same input produces identical FTAOutput (idempotent)."""

    def test_fta_deterministic(self):
        fta_input = _good_long_input()
        result_a = evaluate(fta_input)
        result_b = evaluate(fta_input)

        assert result_a.verdict.accepted == result_b.verdict.accepted
        assert result_a.verdict.score == pytest.approx(result_b.verdict.score)
        assert result_a.verdict.summary == result_b.verdict.summary
        assert result_a.nearest_trouble_price == pytest.approx(result_b.nearest_trouble_price)
        assert result_a.distance_to_fta_pct == pytest.approx(result_b.distance_to_fta_pct)
        assert result_a.reward_risk == pytest.approx(result_b.reward_risk)
        assert result_a.structure_score == pytest.approx(result_b.structure_score)
        assert result_a.liquidity_score == pytest.approx(result_b.liquidity_score)
        assert result_a.volatility_ok == result_b.volatility_ok
        assert [r.code for r in result_a.rejection_reasons] == [
            r.code for r in result_b.rejection_reasons
        ]
