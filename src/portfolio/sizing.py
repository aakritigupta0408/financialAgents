"""
src.portfolio.sizing — Risk-based position sizing.

Formula assumptions
-------------------
- equity is the current total portfolio equity (cash + unrealised value).
- risk_pct is a fraction of equity to risk per trade (e.g. 0.01 = 1%).
- risk_per_share = |entry_price - stop_price|.
- raw_size = (equity * risk_pct) / risk_per_share.
- confidence_scaling optionally scales down size for lower-confidence signals.
- max_ticker_exposure_pct caps the notional weight of any single ticker.
- available_cash caps size by what can actually be funded today.
- Final size is floored to a whole share and clamped to a minimum of 1.

No slippage or commissions are modelled here (paper trading).
"""

from __future__ import annotations

import math


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float,
    confidence: float = 1.0,
    confidence_scaling: bool = False,
    max_ticker_exposure_pct: float = 0.10,
    available_cash: float | None = None,
) -> float:
    """
    Compute the number of whole shares to buy/sell.

    Parameters
    ----------
    equity:
        Current total portfolio equity (cash + unrealised positions).
    entry_price:
        Intended fill price (exact — no slippage).
    stop_price:
        Stop-loss price. Used to derive per-share risk.
    risk_pct:
        Fraction of equity to risk on this trade (e.g. 0.01).
    confidence:
        Meta-model confidence score in [0, 1]. Used only if
        confidence_scaling=True.
    confidence_scaling:
        When True, scales raw_size by max(0.1, confidence) so that
        high-confidence signals receive full size and low-confidence
        signals receive reduced size.
    max_ticker_exposure_pct:
        Maximum fraction of equity that can be allocated to a single
        ticker. Hard cap on notional exposure.
    available_cash:
        Cash currently available. Prevents over-committing capital.
        Pass None to skip this cap.

    Returns
    -------
    float
        Number of whole shares (>= 1.0).
    """
    risk_amount = equity * risk_pct
    risk_per_share = abs(entry_price - stop_price)

    # Guard: zero-risk-per-share means stop == entry; default to 1 share.
    if risk_per_share == 0:
        return 1.0

    raw_size = risk_amount / risk_per_share

    if confidence_scaling:
        raw_size = raw_size * max(0.1, confidence)

    # Exposure cap: no single ticker may exceed max_ticker_exposure_pct * equity
    max_by_exposure = (equity * max_ticker_exposure_pct) / entry_price
    size = min(raw_size, max_by_exposure)

    # Cash cap: cannot spend more than what is available
    if available_cash is not None:
        size = min(size, available_cash / entry_price)

    return max(1.0, math.floor(size))


def required_capital(entry_price: float, quantity: float) -> float:
    """
    Capital required to open a position.

    Formula: entry_price * quantity

    No margin or leverage is modelled (paper trading, cash account).
    """
    return entry_price * quantity
