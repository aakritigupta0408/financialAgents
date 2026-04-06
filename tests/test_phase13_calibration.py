"""
tests/test_phase13_calibration.py

Phase 13: 15 tests covering intraday synthetic data generation, FTA calibration,
and the Phase 13 runner pipeline.

All test data uses make_structured_1h_series for a realistic 1h intraday series
with genuine swing structure detectable by compute_structure(swing_window=5).
"""
from __future__ import annotations

import pytest

from src.validation.intraday_synthetic import make_structured_1h_series
from src.validation.fta_calibration import (
    run_fta_calibration,
    CalibrationSummary,
    FTACalibrationResult,
)
from src.validation.phase13_runner import run_phase13, Phase13Result
from src.features.structure import compute_structure


# ---------------------------------------------------------------------------
# Module-level fixtures — generated once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def series_600():
    """600-bar series for generator tests."""
    return make_structured_1h_series(ticker="SIM", n_bars=600, seed=42, base_price=200.0)


@pytest.fixture(scope="module")
def series_300():
    """300-bar series for speed-sensitive tests."""
    return make_structured_1h_series(ticker="SIM300", n_bars=300, seed=99, base_price=150.0)


@pytest.fixture(scope="module")
def calibration_result_2tickers():
    """Calibration result for 2 tickers at n_bars=300."""
    aapl = make_structured_1h_series(ticker="AAPL_1H", n_bars=300, seed=1, base_price=255.0)
    msft = make_structured_1h_series(ticker="MSFT_1H", n_bars=300, seed=2, base_price=400.0)
    return run_fta_calibration(
        series_map={"AAPL_1H": aapl, "MSFT_1H": msft},
        rr_thresholds=[1.25, 1.50, 1.75, 2.00],
        starting_capital=100_000.0,
        min_bars_required=40,
    )


@pytest.fixture(scope="module")
def phase13_result_aapl():
    """Phase13Result for a single ticker, n_bars=400, for fast pipeline tests."""
    return run_phase13(tickers=["AAPL_1H"], n_bars=400, min_bars_required=40)


# ---------------------------------------------------------------------------
# 1. make_structured_1h_series returns series with n_bars bars
# ---------------------------------------------------------------------------

def test_structured_series_length(series_600):
    assert len(series_600.bars) == 600


# ---------------------------------------------------------------------------
# 2. bars are sorted ascending by timestamp
# ---------------------------------------------------------------------------

def test_structured_series_ascending(series_600):
    timestamps = [b.timestamp for b in series_600.bars]
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], (
            f"Bar {i} timestamp {timestamps[i]} is not after bar {i-1} {timestamps[i-1]}"
        )


# ---------------------------------------------------------------------------
# 3. all bars satisfy OHLCV validity constraints
# ---------------------------------------------------------------------------

def test_structured_series_ohlcv_valid(series_600):
    for i, bar in enumerate(series_600.bars):
        assert bar.high >= bar.open >= bar.low, (
            f"Bar {i}: high={bar.high} open={bar.open} low={bar.low}"
        )
        assert bar.high >= bar.close >= bar.low, (
            f"Bar {i}: high={bar.high} close={bar.close} low={bar.low}"
        )
        assert bar.high > 0
        assert bar.low > 0
        assert bar.volume > 0


# ---------------------------------------------------------------------------
# 4. compute_structure() finds >= 8 swing points in a 600-bar series
# ---------------------------------------------------------------------------

def test_structured_series_has_swings(series_600):
    df = series_600.to_dataframe()
    feat = compute_structure(df, "SIM", "1h", swing_window=5)
    total_swings = len(feat.swing_highs) + len(feat.swing_lows)
    assert total_swings >= 8, (
        f"Expected >= 8 swing points, found {total_swings} "
        f"(highs={len(feat.swing_highs)}, lows={len(feat.swing_lows)})"
    )


# ---------------------------------------------------------------------------
# 5. mean ATR is between 0.003 and 0.012 of mean close price
# ---------------------------------------------------------------------------

def test_structured_series_atr_range(series_600):
    df = series_600.to_dataframe()
    atr_pct = ((df["high"] - df["low"]) / df["close"]).mean()
    assert 0.003 <= atr_pct <= 0.012, (
        f"Mean ATR % = {atr_pct:.5f}, expected [0.003, 0.012]"
    )


# ---------------------------------------------------------------------------
# 6. same seed produces identical close prices (reproducibility)
# ---------------------------------------------------------------------------

def test_structured_series_reproducible():
    s1 = make_structured_1h_series(ticker="REP1", n_bars=100, seed=7, base_price=100.0)
    s2 = make_structured_1h_series(ticker="REP2", n_bars=100, seed=7, base_price=100.0)
    closes1 = [b.close for b in s1.bars]
    closes2 = [b.close for b in s2.bars]
    assert closes1 == closes2, "Same seed must produce identical close prices"


# ---------------------------------------------------------------------------
# 7. different seeds produce different close sequences
# ---------------------------------------------------------------------------

def test_structured_series_different_seeds():
    s1 = make_structured_1h_series(ticker="DIFF1", n_bars=100, seed=10, base_price=100.0)
    s2 = make_structured_1h_series(ticker="DIFF2", n_bars=100, seed=20, base_price=100.0)
    closes1 = [b.close for b in s1.bars]
    closes2 = [b.close for b in s2.bars]
    assert closes1 != closes2, "Different seeds must produce different close sequences"


# ---------------------------------------------------------------------------
# 8. run_fta_calibration returns dict with expected tickers
# ---------------------------------------------------------------------------

def test_fta_calibration_returns_summaries(calibration_result_2tickers):
    result = calibration_result_2tickers
    assert isinstance(result, dict)
    assert "AAPL_1H" in result
    assert "MSFT_1H" in result
    for ticker, summary in result.items():
        assert isinstance(summary, CalibrationSummary)
        assert summary.ticker == ticker


# ---------------------------------------------------------------------------
# 9. each CalibrationSummary has 4 FTACalibrationResults (one per threshold)
# ---------------------------------------------------------------------------

def test_fta_calibration_rr_sweep_counts(calibration_result_2tickers):
    for ticker, summary in calibration_result_2tickers.items():
        assert len(summary.results) == 4, (
            f"{ticker}: expected 4 threshold results, got {len(summary.results)}"
        )
        thresholds = [r.rr_threshold for r in summary.results]
        for expected in [1.25, 1.50, 1.75, 2.00]:
            assert expected in thresholds, (
                f"{ticker}: threshold {expected} missing from {thresholds}"
            )


# ---------------------------------------------------------------------------
# 10. at RR=1.25 at least ONE ticker produces n_trades > 0
# ---------------------------------------------------------------------------

def test_fta_calibration_trades_exist_at_low_rr():
    """
    Test with 3 tickers to maximize chance of getting trades at RR=1.25.
    Uses n_bars=600 and min_bars_required=40 to give the engine more opportunities.
    """
    tickers_spec = [
        ("AAPL_1H", 1, 255.0),
        ("MSFT_1H", 2, 400.0),
        ("NVDA_1H", 3, 185.0),
    ]
    series_map = {
        t: make_structured_1h_series(ticker=t, n_bars=600, seed=s, base_price=p)
        for t, s, p in tickers_spec
    }
    result = run_fta_calibration(
        series_map=series_map,
        rr_thresholds=[1.25, 1.50, 1.75, 2.00],
        starting_capital=100_000.0,
        min_bars_required=40,
    )
    at_125 = []
    for ticker, summary in result.items():
        for r in summary.results:
            if r.rr_threshold == 1.25:
                at_125.append(r.n_trades)

    # At least one ticker should have > 0 trades at the most permissive threshold
    assert max(at_125) > 0, (
        f"Expected >= 1 trade at RR=1.25 in at least one ticker, "
        f"got trade counts: {at_125}. "
        "The FTA filter may be too restrictive even for synthetic data."
    )


# ---------------------------------------------------------------------------
# 11. recommended_rr in [1.25, 2.00] for each ticker
# ---------------------------------------------------------------------------

def test_fta_calibration_recommended_rr_in_range(calibration_result_2tickers):
    for ticker, summary in calibration_result_2tickers.items():
        assert 1.25 <= summary.recommended_rr <= 2.00, (
            f"{ticker}: recommended_rr={summary.recommended_rr} "
            "is not in [1.25, 2.00]"
        )


# ---------------------------------------------------------------------------
# 12. run_phase13() completes without error on 1 ticker
# ---------------------------------------------------------------------------

def test_phase13_runs_without_error(phase13_result_aapl):
    assert isinstance(phase13_result_aapl, Phase13Result)


# ---------------------------------------------------------------------------
# 13. acceptance_verdict is one of the 3 valid strings
# ---------------------------------------------------------------------------

def test_phase13_verdict_is_valid(phase13_result_aapl):
    valid_verdicts = {
        "READY_FOR_PAPER_TRADING",
        "NEEDS_CALIBRATION",
        "FAILS_CURRENT_ACCEPTANCE",
    }
    assert phase13_result_aapl.acceptance_verdict in valid_verdicts, (
        f"Unexpected verdict: {phase13_result_aapl.acceptance_verdict!r}"
    )


# ---------------------------------------------------------------------------
# 14. recommended_settings has key "FTA_MIN_REWARD_RISK"
# ---------------------------------------------------------------------------

def test_phase13_recommended_settings_keys(phase13_result_aapl):
    settings = phase13_result_aapl.recommended_settings
    assert "FTA_MIN_REWARD_RISK" in settings, (
        f"'FTA_MIN_REWARD_RISK' missing from recommended_settings: {list(settings.keys())}"
    )
    rr = settings["FTA_MIN_REWARD_RISK"]
    assert isinstance(rr, float), f"FTA_MIN_REWARD_RISK should be float, got {type(rr)}"
    assert 1.0 <= rr <= 5.0, f"FTA_MIN_REWARD_RISK={rr} out of sensible range"


# ---------------------------------------------------------------------------
# 15. remaining_todos is a non-empty list
# ---------------------------------------------------------------------------

def test_phase13_remaining_todos_populated(phase13_result_aapl):
    todos = phase13_result_aapl.remaining_todos
    assert isinstance(todos, list), f"remaining_todos should be list, got {type(todos)}"
    assert len(todos) > 0, "remaining_todos must not be empty"
    assert all(isinstance(t, str) for t in todos), "All todos must be strings"
