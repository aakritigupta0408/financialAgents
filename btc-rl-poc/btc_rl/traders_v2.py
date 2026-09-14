"""P1 — CLEAN TRADER FAMILY (T0-T3). One brain (Oracle), multiple policies.

All four consume the SAME frozen Oracle probability, the SAME contract state, the
SAME executable-price model, the SAME exact-BRTI settlement. They differ ONLY in
the one registered mechanism named below — nothing else — so an A/B measures that
mechanism and only that.

  T0 BASELINE FOLLOWER (CONTROL): trade fixed stake when locked & EV clears min.
  T1 SELECTIVE EDGE (TREATMENT):  T0 + require |p_oracle - executable_mkt| >= tau.
  T2 TIMING AWARE (SHADOW):       T0 side/eligibility; pick best entry in the lock
                                  window (execution timing only; never flips side).
  T3 RISK AWARE (SHADOW):         T0 entries; capped edge-proportional sizing only.

Frozen params live here as the single source of truth (P11). Contract: $1 binary,
stake in CONTRACTS; cost is the executable ask on the chosen side.
"""
from __future__ import annotations

# ── frozen policy parameters (P11) ────────────────────────────────────────────
LOCK_MAX_S = 720          # only act inside the final 12 min (tradeable envelope)
MIN_EDGE = 0.02           # executable EV must clear 2c/$1 to be eligible
FIXED_STAKE = 1           # contracts (T0/T1/T2)
T1_EDGE_TAU = 0.08        # selective-edge disagreement threshold (frozen on dev+val)
T3_MAX_STAKE = 3          # cap for edge-proportional sizing
T3_EDGE_FULL = 0.12       # edge at which T3 reaches max stake
FEE_RATE = 0.07           # kalshi-style per-contract fee = 0.07*p*(1-p)


def _fee(price):
    return FEE_RATE * price * (1 - price)


def side_and_cost(p_oracle, k_prob, half_spread):
    """Oracle-chosen side + executable ask cost on that side (per $1 contract)."""
    if p_oracle >= 0.5:
        return "yes", min(0.99, k_prob + half_spread)
    return "no", min(0.99, (1 - k_prob) + half_spread)


def edge_ev(p_oracle, k_prob, half_spread):
    """Executable expected value on the Oracle-favored side (per $1)."""
    side, cost = side_and_cost(p_oracle, k_prob, half_spread)
    p_win = p_oracle if side == "yes" else (1 - p_oracle)
    return side, cost, p_win - cost


def eligible(tte_s, ev):
    return tte_s <= LOCK_MAX_S and ev >= MIN_EDGE


def t1_qualifies(p_oracle, k_prob, tau=T1_EDGE_TAU):
    """Selective-edge: Oracle materially disagrees with the executable market."""
    return abs(p_oracle - k_prob) >= tau


def t3_stake(ev):
    """Edge-proportional stake, aggressively capped (sizing research only)."""
    frac = max(0.0, min(1.0, ev / T3_EDGE_FULL))
    return max(1, round(1 + frac * (T3_MAX_STAKE - 1)))


def settle_pnl(side, cost, contracts, exact_yes):
    """Realized paper P&L in cents for a settled position ($1=100c payoff)."""
    payoff = 1.0 if ((side == "yes") == (exact_yes == 1)) else 0.0
    per_contract = (payoff - cost - _fee(cost)) * 100.0
    return round(per_contract * contracts, 4), int(payoff > 0)


FROZEN = {
    "lock_max_s": LOCK_MAX_S, "min_edge": MIN_EDGE, "fixed_stake": FIXED_STAKE,
    "t1_edge_tau": T1_EDGE_TAU, "t3_max_stake": T3_MAX_STAKE,
    "t3_edge_full": T3_EDGE_FULL, "fee_rate": FEE_RATE,
}
