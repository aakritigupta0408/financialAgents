import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btc_rl import economics as E  # noqa: E402


def test_empty_safe():
    assert E.realized_pnl([]) is None
    assert E.ev_per_trade([]) is None
    assert E.win_rate([]) is None
    assert E.drawdown([]) is None


def test_basic_pnl_and_ev():
    pnls = [50, -30, 20]
    assert E.realized_pnl(pnls) == 40
    assert abs(E.ev_per_trade(pnls) - 40 / 3) < 1e-9
    assert E.ev_per_eligible_window(pnls, 6) == 40 / 6


def test_coverage_and_rates():
    assert E.coverage(3, 6) == 0.5
    assert E.coverage(0, 0) is None
    assert E.win_rate([1, 0, 1, 1]) == 0.75
    assert E.bad_entry_rate([50, -30, -10, 20]) == 0.5


def test_equity_and_drawdown():
    eq = E.equity_curve(1000, [100, -300, 50])
    assert eq == [1100, 800, 850]
    assert E.drawdown(eq) is not None and E.drawdown(eq) >= 0


def test_counterfactuals_unavailable_not_faked():
    assert E.profitable_trade_recall(None, None) is None
    assert E.missed_profit_rate(None, None) is None
    assert E.profitable_trade_recall(3, 10) == 0.3


def test_paired_delta_ci():
    ctrl = [0.0] * 40
    trt = [0.05] * 40
    d = E.paired_delta(ctrl, trt)
    assert d is not None
    assert abs(d["paired_delta"] - 0.05) < 1e-9
    assert d["ci95"][0] <= 0.05 <= d["ci95"][1]
    assert d["n_pairs"] == 40


def test_summarize_trades_shape():
    trades = [{"pnl_c": 40, "win": 1}, {"pnl_c": -60, "win": 0},
              {"pnl_c": None, "skipped": True}]
    s = E.summarize_trades(trades, 1000, n_eligible=5)
    assert s["n_trades"] == 2
    assert s["realized_pnl_c"] == -20
    assert s["coverage"] == 2 / 5
    assert s["profitable_trade_recall"] is None  # honest UNAVAILABLE
