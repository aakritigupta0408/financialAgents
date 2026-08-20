"""Unit tests for btc_rl core logic. Run: python3 -m pytest tests/ -q
(or python3 tests/test_core.py for a dependency-free run)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_rl import config
from btc_rl.agents import DistDQNAgent, LinearQAgent, LinUCBAgent
from btc_rl.env import reward
from btc_rl.features import (BOOK_DIM, LIVE_DIM, LLM_DIM, OFI_DIM, TREND_DIM,
                             book_feature_vector, compute_features,
                             feature_vector, live_feature_vector,
                             llm_feature_vector, ofi_feature_vector,
                             trend_feature_vector)


def _bars(n=300, start=70000.0):
    return [{"ts": 1000 + 60 * i, "open": start, "high": start + 5,
             "low": start - 5, "close": start + 0.5 * i, "volume": 1.0}
            for i in range(n)]


def test_reward_spec():
    assert reward(68000.4, 68000.9, shaped=False) == config.REWARD_HIT
    assert reward(68000.0, 68005.0, shaped=False) == config.REWARD_HIT  # $5 floor
    assert reward(68000.0, 68005.2, shaped=False) == config.REWARD_MISS
    assert abs(reward(68000.0, 68050.0, shaped=True) + 0.5) < 1e-9
    # vol-scaled band: a +15m sigma of $200 widens the hit band to $20
    assert reward(68000.0, 68018.0, shaped=False, band=20.0) == config.REWARD_HIT
    assert reward(68000.0, 68021.0, shaped=False, band=20.0) == config.REWARD_MISS


def test_feature_vector_dims():
    feat = compute_features(_bars(), fng=55)
    assert len(feature_vector(feat)) == 10
    assert len(trend_feature_vector(feat)) == TREND_DIM
    assert len(live_feature_vector(None)) == LIVE_DIM
    assert len(book_feature_vector(None)) == BOOK_DIM
    assert len(llm_feature_vector(None)) == LLM_DIM
    assert len(ofi_feature_vector(None)) == OFI_DIM
    # missing snapshots must read neutral, not crash
    assert live_feature_vector(None) == [0.0] * LIVE_DIM


def test_action_grid_is_median_bounded():
    assert max(abs(k) for k in config.K_FACTORS) <= 1.5
    assert 0.0 in config.K_FACTORS


def test_linucb_learns_signed_reward():
    agent = LinUCBAgent(dim=3, alpha=0.3, n_arms=len(config.K_FACTORS))
    x = [1.0, 0.5, -0.2]
    good = config.K_FACTORS.index(0.0)
    for _ in range(50):
        agent.update(x, good, 1.0)
    assert agent.select(x, greedy=True) == good


def test_linear_q_shapes():
    agent = LinearQAgent(dim=4)
    a = agent.select([1, 0, 0, 0], greedy=True)
    agent.update([1, 0, 0, 0], a, 0.5)
    assert 0 <= a < agent.n_arms
    d = agent.to_dict()
    agent2 = LinearQAgent.from_dict(d)
    assert agent2.select([1, 0, 0, 0], greedy=True) == \
        agent.select([1, 0, 0, 0], greedy=True)


def test_dist_dqn_overflow_bins_never_act():
    agent = DistDQNAgent(dim=5)
    for _ in range(20):
        a = agent.select([0.1] * 5, greedy=False)
        assert abs(agent.bins[a]) <= 1.5 + 1e-9
    assert agent.target_bin(2.6) is not None  # tails are valid TARGETS
    assert abs(agent.bins[agent.target_bin(2.6)] - 2.5) < 0.51


def test_replay_baseline_math():
    bars = _bars(100)
    h = 5
    pred_delta = bars[-1]["close"] - bars[-1 - h]["close"]
    assert abs(pred_delta - 0.5 * h) < 1e-9  # linear ramp of 0.5/min


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
