"""
src.features.levels — support and resistance level computation.

Derives horizontal support/resistance zones from the confirmed swing points
produced by compute_structure().

Algorithm
---------
1. Collect all swing high prices as resistance candidates.
   Collect all swing low prices as support candidates.
2. Cluster nearby prices (within zone_margin_pct of each other) into zones.
   Greedy single-pass clustering: sort prices, start a new zone whenever
   the current price is more than zone_margin_pct away from the cluster mean.
3. For each zone compute a strength score:
   strength = (touches / max_touches) clamped to [0, 1].
   A touch is counted when any bar's high/low comes within zone_margin_pct
   of the zone midpoint.  More recent touches receive a 1.1x weight.

Timezone note
-------------
Timestamps are passed through unchanged from StructureFeatures.
"""

from __future__ import annotations

import pandas as pd

from schemas.features import LevelFeatures, PriceZone, StructureFeatures

# Maximum number of touches used to normalise strength to 1.0.
_MAX_TOUCHES = 5


def compute_levels(
    df: pd.DataFrame,
    structure: StructureFeatures,
    ticker: str,
    timeframe: str,
    zone_margin_pct: float = 0.002,
) -> LevelFeatures:
    """
    Compute support and resistance zones from confirmed swing points.

    Parameters
    ----------
    df : pd.DataFrame
        Timestamp-indexed OHLCV DataFrame (ascending, same as structure input).
    structure : StructureFeatures
        Output from compute_structure().
    ticker : str
    timeframe : str
    zone_margin_pct : float
        Fractional margin for clustering and touch detection (default 0.2%).

    Returns
    -------
    LevelFeatures
    """
    resistance_prices = [sp.price for sp in structure.swing_highs]
    support_prices = [sp.price for sp in structure.swing_lows]

    resistance_clusters = _cluster_prices(resistance_prices, zone_margin_pct)
    support_clusters = _cluster_prices(support_prices, zone_margin_pct)

    resistance_zones = [
        _build_zone(cluster, "resistance", df, zone_margin_pct)
        for cluster in resistance_clusters
    ]
    support_zones = [
        _build_zone(cluster, "support", df, zone_margin_pct)
        for cluster in support_clusters
    ]

    # Sort by price: resistance ascending (nearest to price at top), support descending.
    resistance_zones.sort(key=lambda z: z.low)
    support_zones.sort(key=lambda z: z.high, reverse=True)

    return LevelFeatures(
        ticker=ticker,
        timeframe=timeframe,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
    )


# ── Clustering ─────────────────────────────────────────────────────────────


def _cluster_prices(
    prices: list[float],
    zone_margin_pct: float,
) -> list[list[float]]:
    """
    Group prices into clusters where adjacent prices are within
    zone_margin_pct of the cluster mean.

    Returns a list of clusters; each cluster is a list of prices.
    """
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters: list[list[float]] = []
    current_cluster: list[float] = [sorted_prices[0]]

    for price in sorted_prices[1:]:
        cluster_mean = sum(current_cluster) / len(current_cluster)
        # Check if the new price is within zone_margin_pct of the cluster mean.
        if abs(price - cluster_mean) / cluster_mean <= zone_margin_pct:
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]

    clusters.append(current_cluster)
    return clusters


# ── Zone construction ──────────────────────────────────────────────────────


def _build_zone(
    cluster: list[float],
    zone_type: str,  # "support" or "resistance"
    df: pd.DataFrame,
    zone_margin_pct: float,
) -> PriceZone:
    """
    Build a PriceZone from a price cluster and compute its strength.

    Zone bounds:
      zone.low  = min(cluster) * (1 - zone_margin_pct / 2)
      zone.high = max(cluster) * (1 + zone_margin_pct / 2)

    Strength:
      Count touches where bar high (for support) or bar low (for resistance)
      comes within zone_margin_pct of the zone midpoint.
      Recent touches (latter half of df) are weighted 1.1x.
    """
    min_price = min(cluster)
    max_price = max(cluster)
    zone_low = min_price * (1.0 - zone_margin_pct / 2.0)
    zone_high = max_price * (1.0 + zone_margin_pct / 2.0)
    zone_mid = (zone_low + zone_high) / 2.0
    tolerance = zone_mid * zone_margin_pct

    strength = _compute_zone_strength(df, zone_mid, tolerance)

    return PriceZone(
        low=round(zone_low, 6),
        high=round(zone_high, 6),
        strength=round(strength, 4),
        zone_type=zone_type,  # type: ignore[arg-type]
    )


def _compute_zone_strength(
    df: pd.DataFrame,
    zone_mid: float,
    tolerance: float,
) -> float:
    """
    Compute a normalised touch score for a zone.

    A bar "touches" the zone if its high or low is within *tolerance*
    of *zone_mid*.  Bars in the latter half of the series receive a 1.1x
    recency weight.

    Returns a float in [0, 1].
    """
    if df.empty:
        return 0.0

    n = len(df)
    half = n // 2
    weighted_touches = 0.0

    highs = df["high"].values
    lows = df["low"].values

    for i in range(n):
        weight = 1.1 if i >= half else 1.0
        touched = (
            abs(highs[i] - zone_mid) <= tolerance
            or abs(lows[i] - zone_mid) <= tolerance
        )
        if touched:
            weighted_touches += weight

    # Normalise to [0, 1] using _MAX_TOUCHES as the ceiling.
    strength = min(weighted_touches / _MAX_TOUCHES, 1.0)
    return strength
