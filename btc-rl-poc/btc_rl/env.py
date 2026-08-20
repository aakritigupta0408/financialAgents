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


def reward(pred_price: float, actual_price: float, shaped: bool) -> float:
    """Within ±HIT_BAND dollars of the actual => full hit reward.

    (Originally exact-integer match, which fired on <1% of predictions —
    far too sparse to shape learning.) Shaped mode otherwise pays
    -|error|/scale so the gradient toward the right neighborhood is visible.
    """
    if abs(pred_price - actual_price) <= config.HIT_BAND:
        return config.REWARD_HIT
    if shaped:
        return -abs(pred_price - actual_price) / config.SHAPED_SCALE
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
