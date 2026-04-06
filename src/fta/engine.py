"""FTA engine — deterministic, side-effect free trade filter.

Implements an 8-step structural analysis pipeline:
  1. First Trouble Area (FTA)
  2. Reward:Risk
  3. Structure Score
  4. Liquidity Score
  5. Volatility Check
  6. FTA Clearance
  7. Rejection Reason Collection
  8. Composite Score + Verdict
"""

from __future__ import annotations

from config.settings import FTA_MIN_DISTANCE_TO_FTA_PCT, FTA_MIN_REWARD_RISK
from schemas.fta import FTAInput, FTAOutput, FTARejectionReason, FTAVerdict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate(fta_input: FTAInput) -> FTAOutput:
    """Evaluate a trade candidate and return a fully populated FTAOutput.

    The function is deterministic and has no side effects.
    """
    candidate = fta_input.candidate
    entry_price: float = candidate.entry_price
    side: str = candidate.side

    # ------------------------------------------------------------------
    # STEP 1 — First Trouble Area
    # ------------------------------------------------------------------
    if side == "long":
        nearest_trouble_price, fta_distance, distance_to_fta_pct = _compute_fta_long(
            entry_price, fta_input
        )
    else:
        nearest_trouble_price, fta_distance, distance_to_fta_pct = _compute_fta_short(
            entry_price, fta_input
        )

    # ------------------------------------------------------------------
    # STEP 2 — Reward:Risk
    # ------------------------------------------------------------------
    risk = abs(entry_price - candidate.stop_price)

    forecast = fta_input.forecast
    expected_price_move = entry_price * abs(forecast.expected_return)
    reward = max(expected_price_move, fta_distance * 0.8)

    reward_risk = reward / risk if risk > 0.0 else 0.0

    # ------------------------------------------------------------------
    # STEP 3 — Structure Score (0–1)
    # ------------------------------------------------------------------
    structure_score = _compute_structure_score(fta_input, side)

    # ------------------------------------------------------------------
    # STEP 4 — Liquidity Score (0–1)
    # ------------------------------------------------------------------
    liq_score = _compute_liquidity_score(fta_input)

    # ------------------------------------------------------------------
    # STEP 5 — Volatility Check
    # ------------------------------------------------------------------
    volatility_ok = _check_volatility(fta_input)

    # ------------------------------------------------------------------
    # STEP 6 — FTA Clearance Check
    # ------------------------------------------------------------------
    fta_cleared = _check_fta_clearance(
        fta_input, entry_price, nearest_trouble_price, side
    )

    # ------------------------------------------------------------------
    # STEP 7 — Collect ALL rejection reasons before verdict
    # ------------------------------------------------------------------
    rejection_reasons: list[FTARejectionReason] = []

    if reward_risk < FTA_MIN_REWARD_RISK:
        rejection_reasons.append(
            FTARejectionReason(
                code="REWARD_RISK_BELOW_THRESHOLD",
                detail=(
                    f"reward_risk={reward_risk:.3f} is below the minimum "
                    f"threshold of {FTA_MIN_REWARD_RISK:.1f}"
                ),
            )
        )

    if not fta_cleared:
        rejection_reasons.append(
            FTARejectionReason(
                code="FTA_NOT_CLEARED",
                detail=(
                    f"Expected move does not clear the First Trouble Area at "
                    f"{nearest_trouble_price:.4f} with sufficient margin "
                    f"(min_pct={FTA_MIN_DISTANCE_TO_FTA_PCT})"
                ),
            )
        )

    if structure_score < 0.35:
        rejection_reasons.append(
            FTARejectionReason(
                code="WEAK_STRUCTURE",
                detail=f"structure_score={structure_score:.3f} is below 0.35",
            )
        )

    if liq_score < 0.25:
        rejection_reasons.append(
            FTARejectionReason(
                code="POOR_LIQUIDITY",
                detail=f"liquidity_score={liq_score:.3f} is below 0.25",
            )
        )

    if not volatility_ok:
        rejection_reasons.append(
            FTARejectionReason(
                code="UNSUITABLE_VOLATILITY",
                detail=(
                    f"volatility_regime='{fta_input.volatility.volatility_regime}', "
                    f"atr_pct={fta_input.volatility.atr_pct:.4f}"
                ),
            )
        )

    if distance_to_fta_pct < FTA_MIN_DISTANCE_TO_FTA_PCT:
        rejection_reasons.append(
            FTARejectionReason(
                code="INSUFFICIENT_FTA_DISTANCE",
                detail=(
                    f"distance_to_fta_pct={distance_to_fta_pct:.5f} is below "
                    f"the minimum {FTA_MIN_DISTANCE_TO_FTA_PCT}"
                ),
            )
        )

    # ------------------------------------------------------------------
    # STEP 8 — Composite Score + Verdict
    # ------------------------------------------------------------------
    composite_score = (
        0.35 * min(reward_risk / 4.0, 1.0)
        + 0.25 * (1.0 if fta_cleared else 0.0)
        + 0.20 * structure_score
        + 0.15 * liq_score
        + 0.05 * (1.0 if volatility_ok else 0.0)
    )

    accepted = len(rejection_reasons) == 0
    summary = (
        "ACCEPTED"
        if accepted
        else "REJECTED: " + ", ".join(r.code for r in rejection_reasons)
    )

    return FTAOutput(
        ticker=candidate.ticker,
        verdict=FTAVerdict(accepted=accepted, score=composite_score, summary=summary),
        nearest_trouble_price=nearest_trouble_price,
        distance_to_fta_pct=distance_to_fta_pct,
        reward_risk=reward_risk,
        structure_score=structure_score,
        liquidity_score=liq_score,
        volatility_ok=volatility_ok,
        rejection_reasons=rejection_reasons,
    )


# ---------------------------------------------------------------------------
# Step 1 helpers
# ---------------------------------------------------------------------------

def _compute_fta_long(
    entry_price: float, fta_input: FTAInput
) -> tuple[float, float, float]:
    """Return (nearest_trouble_price, fta_distance, distance_to_fta_pct) for a long."""
    # Find the lowest resistance zone whose midpoint is strictly above entry.
    best_zone = None
    best_midpoint = float("inf")

    for zone in fta_input.levels.resistance_zones:
        midpoint = (zone.low + zone.high) / 2.0
        if midpoint > entry_price and midpoint < best_midpoint:
            best_midpoint = midpoint
            best_zone = zone

    if best_zone is None:
        nearest_trouble_price = entry_price * 1.10
    else:
        nearest_trouble_price = best_zone.high

    fta_distance = nearest_trouble_price - entry_price
    distance_to_fta_pct = fta_distance / entry_price
    return nearest_trouble_price, fta_distance, distance_to_fta_pct


def _compute_fta_short(
    entry_price: float, fta_input: FTAInput
) -> tuple[float, float, float]:
    """Return (nearest_trouble_price, fta_distance, distance_to_fta_pct) for a short."""
    # Find the highest support zone whose midpoint is strictly below entry.
    best_zone = None
    best_midpoint = float("-inf")

    for zone in fta_input.levels.support_zones:
        midpoint = (zone.low + zone.high) / 2.0
        if midpoint < entry_price and midpoint > best_midpoint:
            best_midpoint = midpoint
            best_zone = zone

    if best_zone is None:
        nearest_trouble_price = entry_price * 0.90
    else:
        nearest_trouble_price = best_zone.low

    fta_distance = entry_price - nearest_trouble_price
    distance_to_fta_pct = fta_distance / entry_price
    return nearest_trouble_price, fta_distance, distance_to_fta_pct


# ---------------------------------------------------------------------------
# Step 3 helper
# ---------------------------------------------------------------------------

_TREND_COMPONENT_LONG: dict[str, float] = {
    "uptrend": 1.0,
    "ranging": 0.4,
    "downtrend": 0.1,
    "unknown": 0.2,
}

_TREND_COMPONENT_SHORT: dict[str, float] = {
    "downtrend": 1.0,
    "ranging": 0.4,
    "uptrend": 0.1,
    "unknown": 0.2,
}


def _compute_structure_score(fta_input: FTAInput, side: str) -> float:
    structure = fta_input.structure

    if side == "long":
        trend_component = _TREND_COMPONENT_LONG.get(structure.trend_state, 0.2)
    else:
        trend_component = _TREND_COMPONENT_SHORT.get(structure.trend_state, 0.2)

    bos_component = 0.0
    if structure.bos_events:
        last_bos = structure.bos_events[-1]
        if side == "long" and last_bos.direction == "bullish":
            bos_component = 0.2
        elif side == "short" and last_bos.direction == "bearish":
            bos_component = 0.2
        elif side == "long" and last_bos.direction == "bearish":
            bos_component = -0.2
        elif side == "short" and last_bos.direction == "bullish":
            bos_component = -0.2

    score = (
        trend_component * 0.5
        + structure.trend_strength * 0.3
        + bos_component * 0.2
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Step 4 helper
# ---------------------------------------------------------------------------

def _compute_liquidity_score(fta_input: FTAInput) -> float:
    liquidity = fta_input.liquidity
    liq_score = min(liquidity.relative_volume / 2.0, 1.0)

    if (
        liquidity.spread_estimate > 0
        and fta_input.volatility.atr > 0
        and liquidity.spread_estimate > 0.05 * fta_input.volatility.atr
    ):
        liq_score *= 0.7

    return liq_score


# ---------------------------------------------------------------------------
# Step 5 helper
# ---------------------------------------------------------------------------

def _check_volatility(fta_input: FTAInput) -> bool:
    vol = fta_input.volatility
    if vol.volatility_regime not in ("low", "normal"):
        return False
    if vol.atr_pct > 0.05:
        return False
    return True


# ---------------------------------------------------------------------------
# Step 6 helper
# ---------------------------------------------------------------------------

def _check_fta_clearance(
    fta_input: FTAInput,
    entry_price: float,
    nearest_trouble_price: float,
    side: str,
) -> bool:
    expected_return = fta_input.forecast.expected_return

    if side == "long":
        if expected_return <= 0:
            return False
        expected_price = entry_price * (1.0 + expected_return)
        margin = (expected_price - nearest_trouble_price) / entry_price
        return margin > FTA_MIN_DISTANCE_TO_FTA_PCT
    else:  # short
        if expected_return >= 0:
            return False
        expected_price = entry_price * (1.0 + expected_return)  # moves down
        margin = (nearest_trouble_price - expected_price) / entry_price
        return margin > FTA_MIN_DISTANCE_TO_FTA_PCT
