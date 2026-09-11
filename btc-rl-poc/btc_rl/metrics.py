"""Industry-standard evaluation metrics — pure functions, no I/O.

The reporting pages and evaluators share these so every surface computes a
given metric exactly one way. References: MASE (Hyndman/M-competitions),
Pesaran–Timmermann directional test, pinball loss / PICP (energy-forecast
competitions), Brier skill scores (Murphy), Kalshi's published fee schedule.
"""
from __future__ import annotations

import math


def mase(abs_errs: list[float], naive_abs_errs: list[float]) -> float | None:
    """Mean Absolute Scaled Error vs a naive (persistence) forecast on the
    SAME slots. <1 beats the naive; this is the scale-free industry framing
    of the 'MAE/floor ratio'."""
    if not abs_errs or not naive_abs_errs:
        return None
    naive = sum(naive_abs_errs) / len(naive_abs_errs)
    if naive <= 0:
        return None
    return (sum(abs_errs) / len(abs_errs)) / naive


def msse(abs_errs: list[float], naive_abs_errs: list[float]) -> float | None:
    """Mean Squared Scaled Error — the squared-loss analog of MASE:
    mean(err^2) / mean(naive err^2) on the same slots. <1 beats the
    naive floor. Added 2026-08-28 when the project's headline error
    metric switched MAE -> MSE (user-directed); MASE is kept alongside
    because squared loss is outlier-sensitive in fat-tailed series."""
    if not abs_errs or not naive_abs_errs:
        return None
    naive = sum(e * e for e in naive_abs_errs) / len(naive_abs_errs)
    if naive <= 0:
        return None
    return (sum(e * e for e in abs_errs) / len(abs_errs)) / naive


def rmse(errs: list[float]) -> float | None:
    if not errs:
        return None
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def pinball(actual: float, lo: float, hi: float,
            q_lo: float = 0.1, q_hi: float = 0.9) -> float:
    """Mean pinball (quantile) loss of the two band edges — the standard
    score for interval forecasts; rewards narrow bands that still cover."""
    def one(q, pred):
        return q * (actual - pred) if actual >= pred else (1 - q) * (pred - actual)
    return (one(q_lo, lo) + one(q_hi, hi)) / 2


def sharpness(los: list[float], his: list[float]) -> float | None:
    """Mean interval width — coverage's necessary companion (a huge band
    covers trivially; sharp AND calibrated is the goal)."""
    if not los:
        return None
    return sum(h - l for l, h in zip(los, his)) / len(los)


def pt_test(pred_up: list[bool], actual_up: list[bool]) -> float | None:
    """Pesaran–Timmermann z-statistic: is directional accuracy better than
    the no-skill benchmark implied by the marginal up/down frequencies?
    |z| > 1.96 => significant at 5%."""
    n = len(pred_up)
    if n < 20 or len(actual_up) != n:
        return None
    p_hat = sum(1 for p, a in zip(pred_up, actual_up) if p == a) / n
    py = sum(pred_up) / n
    pa = sum(actual_up) / n
    p_star = py * pa + (1 - py) * (1 - pa)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * pa - 1) ** 2 * py * (1 - py) / n
              + (2 * py - 1) ** 2 * pa * (1 - pa) / n
              + 4 * pa * py * (1 - pa) * (1 - py) / n ** 2)
    if v_hat - v_star <= 0:
        return None
    return (p_hat - p_star) / math.sqrt(v_hat - v_star)


def brier_skill(brier: float, reference: float) -> float | None:
    """Brier Skill Score vs a reference forecaster (market, or 0.25 for the
    coin-flip climatology). >0 beats the reference; 1 is perfect."""
    if reference <= 0:
        return None
    return 1 - brier / reference


def calibration_bins(ps: list[float], ys: list[int],
                     n_bins: int = 10) -> list[dict]:
    """Reliability-diagram bins: forecast probability vs observed frequency."""
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sel = [(p, y) for p, y in zip(ps, ys)
               if lo <= p < hi or (i == n_bins - 1 and p == 1.0)]
        if sel:
            bins.append({
                "mid": round((lo + hi) / 2, 3),
                "p_mean": round(sum(p for p, _ in sel) / len(sel), 4),
                "y_freq": round(sum(y for _, y in sel) / len(sel), 4),
                "n": len(sel),
            })
    return bins


def kalshi_fee_c(price_c: float) -> float:
    """Kalshi trading fee per contract in cents: 7% x P x (1-P) x 100,
    rounded UP to the next whole cent (their published general schedule).
    7 * p * (1-p) is already in cents for one contract."""
    p = price_c / 100
    return float(math.ceil(7 * p * (1 - p)))


def max_drawdown(cum: list[float]) -> float:
    """Largest peak-to-trough fall of a cumulative P&L series (>= 0)."""
    worst = 0.0
    peak = -math.inf
    for v in cum:
        peak = max(peak, v)
        worst = max(worst, peak - v)
    return worst
