"""One-step episodic environment (contextual bandit) for integer price prediction.

Each episode: at minute t the agent sees the feature state, picks an integer
dollar delta from the current price, and is rewarded on whether
int(current + delta) == int(price at t + horizon).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .features import compute_features, discretize


@dataclass
class Episode:
    day: str
    minute_ts: int            # unix ts of the decision bar
    state: tuple[int, ...]    # discretized state for tabular agents
    features: dict            # continuous features (for later, fancier agents)
    price_now: float
    price_future: float       # actual price at decision + horizon
    horizon_min: int
    is_target_slot: bool      # True when the future bar is 7:00 or 7:15 PM PT


def reward(pred_price: float, actual_price: float, shaped: bool,
           band: float | None = None) -> float:
    """Within ±band dollars of the actual => full hit reward.

    Callers with volatility context pass band = max(HIT_BAND,
    HIT_BAND_VOL * sigma_h) so the precision bar scales with the horizon;
    without one the flat HIT_BAND floor applies. (Originally exact-integer
    match, which fired on <1% of predictions — far too sparse to shape
    learning.)

    Shaped mode measures the miss in BAND UNITS (u = |error|/band) and
    ramps 2-u from +1 at the band edge down to the spec miss value -1 at
    three bands out: continuous at the boundary, bounded in the spec's
    [-1, 1] range, and horizon-fair — a dollar penalty made h30 arms
    MAE-chasers (sigma ~ $300 => every miss ~ -3) while h1 arms
    hit-chased, i.e. two different objectives from one reward.
    """
    b = config.HIT_BAND if band is None else band
    u = abs(pred_price - actual_price) / max(b, 1e-9)
    if u <= 1.0:
        return config.REWARD_HIT
    if shaped:
        return max(config.REWARD_MISS, 2.0 - u)
    return config.REWARD_MISS


def build_episodes(history: dict[str, list[dict]],
                   fng_by_day: dict[str, int]) -> list[Episode]:
    """Slice every day's bars into one-step episodes for both horizons."""
    from datetime import datetime

    episodes: list[Episode] = []
    target_hhmm = set(config.TARGETS_HHMM)
    for day, bars in sorted(history.items()):
        by_ts = {b["ts"]: b for b in bars}
        fng = fng_by_day.get(day)
        for i, bar in enumerate(bars):
            if i < config.LOOKBACK_MIN:
                continue
            for horizon in config.HORIZONS_MIN:
                fut = by_ts.get(bar["ts"] + horizon * 60)
                if fut is None:
                    continue
                feat = compute_features(bars[: i + 1], fng)
                fut_dt = datetime.fromtimestamp(fut["ts"], tz=config.PACIFIC)
                episodes.append(Episode(
                    day=day,
                    minute_ts=bar["ts"],
                    state=discretize(feat),
                    features=feat,
                    price_now=bar["close"],
                    price_future=fut["close"],
                    horizon_min=horizon,
                    is_target_slot=(fut_dt.hour, fut_dt.minute) in target_hhmm,
                ))
    return episodes
