"""Directive check 2 — T1 SIDE-SEMANTICS AUDIT.

Prove the frozen disagreement rule and economics are correct for BOTH an UP (YES)
lock and a DOWN (NO) lock, i.e. no `abs(p_oracle_up - p_market_up)` implementation
mis-prices or mis-directs DOWN positions.

Facts about the frozen implementation (btc_rl/traders_v2.py):
  side_and_cost: side = YES iff p_oracle >= 0.5 (Oracle argmax), else NO;
                 cost_yes = k_prob + half_spread ; cost_no = (1-k_prob)+half_spread.
  edge_ev:       p_win = p_oracle (YES) or 1-p_oracle (NO); ev = p_win - cost.
  t1_qualifies:  |p_oracle - k_prob| >= tau.
  eligible:      requires ev >= MIN_EDGE on the CHOSEN side.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btc_rl import traders_v2 as T  # noqa: E402


def test_disagreement_is_side_symmetric():
    # |P(YES)_oracle - P(YES)_mkt| == |P(NO)_oracle - P(NO)_mkt| for any inputs
    for po, km in [(0.6, 0.5), (0.4, 0.5), (0.45, 0.30), (0.55, 0.70), (0.9, 0.1)]:
        yes_gap = abs(po - km)
        no_gap = abs((1 - po) - (1 - km))
        assert abs(yes_gap - no_gap) < 1e-12


def test_side_matches_oracle_argmax():
    assert T.side_and_cost(0.6, 0.5, 0.005)[0] == "yes"
    assert T.side_and_cost(0.4, 0.5, 0.005)[0] == "no"


def test_cost_is_correct_per_side():
    # YES ask ~ mid + half-spread; NO ask ~ (1-mid) + half-spread
    _, cost_yes = T.side_and_cost(0.6, 0.55, 0.01)
    assert abs(cost_yes - (0.55 + 0.01)) < 1e-9
    _, cost_no = T.side_and_cost(0.4, 0.55, 0.01)
    assert abs(cost_no - (0.45 + 0.01)) < 1e-9


def test_down_position_uses_down_economics():
    # DOWN lock: settle_pnl must pay when exact_yes==0 (NO wins), not when YES wins
    pnl_win, w1 = T.settle_pnl("no", cost=0.40, contracts=1, exact_yes=0)
    pnl_lose, w0 = T.settle_pnl("no", cost=0.40, contracts=1, exact_yes=1)
    assert w1 == 1 and pnl_win > 0
    assert w0 == 0 and pnl_lose < 0


def test_ev_gate_blocks_wrong_direction_disagreement():
    # oracle 0.45 vs market 0.30: disagreement 0.15 points to YES underpriced, but
    # oracle argmax is NO; EV on NO is negative -> eligibility must REJECT it, so T1
    # never trades a direction its own EV doesn't support.
    po, km, hs = 0.45, 0.30, 0.005
    assert T.t1_qualifies(po, km)              # disagreement clears tau
    side, cost, ev = T.edge_ev(po, km, hs)     # but EV on the chosen (NO) side...
    assert side == "no"
    assert ev < 0                              # ...is negative
    assert not T.eligible(300, ev)             # -> not eligible, correctly skipped


def test_symmetric_profitable_case_both_sides():
    # a clean UP edge and a mirror-image DOWN edge should both be eligible & +EV
    up = T.edge_ev(0.70, 0.55, 0.005)          # oracle bullish, market cheaper
    dn = T.edge_ev(0.30, 0.45, 0.005)          # mirror: oracle bearish, market cheaper
    assert up[0] == "yes" and up[2] > 0
    assert dn[0] == "no" and dn[2] > 0
    assert abs(up[2] - dn[2]) < 1e-9           # identical economics by symmetry
