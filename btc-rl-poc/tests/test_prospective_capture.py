import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btc_rl import prospective_capture as pc  # noqa: E402


def _state(**kw):
    base = dict(market_window_id="KXBTC15M-TEST", decision_time=1.0,
                current_brti=77250.0, official_target=77240.0,
                time_remaining_s=420, k_prob=0.55)
    base.update(kw)
    return base


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("PROSPECTIVE_CAPTURE_ENABLED", raising=False)
    assert pc.enabled() is False
    assert pc.capture(_state()) is None


def test_leak_guard_blocks_outcome(monkeypatch):
    monkeypatch.setenv("PROSPECTIVE_CAPTURE_ENABLED", "1")
    for bad in ("exact_yes", "expiration_value", "result", "outcome", "actual"):
        try:
            pc.capture(_state(**{bad: 1}))
            assert False, f"leak-guard failed to block {bad}"
        except AssertionError as e:
            assert "leak-guard" in str(e)


def test_oracle_is_probability_and_deterministic():
    p1 = pc.p_oracle(77250.0, 77240.0, 420)
    p2 = pc.p_oracle(77250.0, 77240.0, 420)
    assert p1 == p2
    assert 0.0 < p1 < 1.0


def test_oracle_monotone_in_distance():
    # more above target => higher P(YES), all else equal
    lo = pc.p_oracle(77245.0, 77240.0, 420)
    hi = pc.p_oracle(77270.0, 77240.0, 420)
    assert hi > lo


def test_no_kalshi_input_in_oracle():
    # p_oracle signature takes no market price — structurally independent of Kalshi
    import inspect
    params = set(inspect.signature(pc.p_oracle).parameters)
    assert "k_prob" not in params and "p_market" not in params
