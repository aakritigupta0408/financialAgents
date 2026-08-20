import math

from btc_rl.metrics import (brier_skill, calibration_bins, kalshi_fee_c,
                            mase, max_drawdown, pinball, pt_test, rmse,
                            sharpness)


def test_mase():
    assert mase([10, 10], [20, 20]) == 0.5           # half the naive error
    assert mase([20], [20]) == 1.0                   # ties persistence
    assert mase([], [1]) is None and mase([1], []) is None


def test_rmse():
    assert rmse([3, -4]) == math.sqrt(12.5)
    assert rmse([]) is None


def test_pinball():
    # actual inside the band: small loss; outside: penalized by 1-q
    inside = pinball(100, lo=95, hi=105)
    below = pinball(80, lo=95, hi=105)
    assert below > inside > 0
    # perfect quantiles on a degenerate band
    assert pinball(100, lo=100, hi=100) == 0


def test_sharpness():
    assert sharpness([90, 80], [110, 120]) == 30
    assert sharpness([], []) is None


def test_pt_test():
    # perfectly skilled forecaster over balanced outcomes -> large positive z
    up = [True, False] * 30
    z = pt_test(up, up)
    assert z is not None and z > 3
    # anti-skilled -> strongly negative
    z2 = pt_test(up, [not a for a in up])
    assert z2 is not None and z2 < -3
    assert pt_test(up[:10], up[:10]) is None         # too small


def test_brier_skill():
    assert brier_skill(0.125, 0.25) == 0.5           # halves the reference
    assert brier_skill(0.25, 0.25) == 0
    assert brier_skill(0.3, 0.25) < 0


def test_calibration_bins():
    ps = [0.05, 0.15, 0.95, 1.0]
    ys = [0, 0, 1, 1]
    bins = calibration_bins(ps, ys, n_bins=10)
    assert bins[0]["n"] == 1 and bins[0]["y_freq"] == 0
    assert bins[-1]["n"] == 2 and bins[-1]["y_freq"] == 1  # 1.0 lands in top bin


def test_kalshi_fee():
    assert kalshi_fee_c(50) == 2.0                   # ceil(7 * .5 * .5) cents
    assert kalshi_fee_c(1) == 1.0                    # rounds up to a whole cent
    assert kalshi_fee_c(99) == 1.0


def test_max_drawdown():
    assert max_drawdown([0, 10, 4, 12, 3]) == 9      # 12 -> 3
    assert max_drawdown([1, 2, 3]) == 0
