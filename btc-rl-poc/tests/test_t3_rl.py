"""Track D integrity — T3 RL side-immutability + sanity + honest (non-leverage) verdict."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "research" / "traders" / "t3_rl_result.json"


def _r():
    return json.loads(RESULT.read_text())


def test_rl_side_immutability():
    # the action space must NOT encode a direction/side; side comes from the Oracle
    import t3_rl
    for a in t3_rl.ACTIONS:
        assert a in ("SKIP", "BUY_SMALL", "BUY_MEDIUM", "BUY_LARGE")
        assert "UP" not in a and "DOWN" not in a and "SELL" not in a and "YES" not in a and "NO" not in a
    assert _r()["side_immutable"] is True


def test_sanity_checks_present():
    s = _r()["sanity"]
    for k in ("reward_shuffle_control_ev", "edge_vs_reward_shuffle_c", "seed_stable",
              "action_collapse", "never_trades", "always_trades", "action_distribution"):
        assert k in s


def test_verdict_from_allowed_set():
    assert _r()["verdict"] in {
        "QUALIFIED_FOR_PROSPECTIVE_SHADOW", "SHADOW_ONLY", "REJECTED",
        "INFORMATION_LIMITED", "OPTIMIZATION_FAILURE", "INSUFFICIENT_EVIDENCE"}


def test_leverage_only_not_qualified():
    # if the policy just bets max size, it must NOT be QUALIFIED (leverage != skill)
    d = _r()
    if d["risk_adjusted"]["leverage_only"]:
        assert d["verdict"] != "QUALIFIED_FOR_PROSPECTIVE_SHADOW"


def test_honest_contextual_bandit_label():
    # one-shot per-window decision is stated as a bandit, not dressed as deep RL
    assert "CONTEXTUAL_BANDIT" in _r()["formulation"]


def test_leverage_matched_baselines_present():
    h = _r()["holdout"]
    assert "baseline_T0_fixed_large" in h and "baseline_edge_prop" in h
