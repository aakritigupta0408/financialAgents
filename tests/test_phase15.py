"""
Phase 15: 1h Data Store + Dual Calibration + Dataset Expansion — 15 tests.

All tests use tmp_path + monkeypatch to isolate data store paths,
following the same pattern as test_phase14_data_store.py.

Speed note: calibration tests use n_bars=200, 1 ticker, 2 rr x 2 dist
to stay under 60s per test.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Store isolation fixture (copied from test_phase14_data_store.py)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    """Redirect all data_store paths to tmp_path so tests don't pollute production data."""
    import src.data_store.paths as paths_mod
    import src.data_store.store as store_mod
    import src.data_store.inventory as inv_mod

    store_dir = tmp_path / "store"
    ingest_dir = tmp_path / "ingest_log"
    db_path = tmp_path / "metadata.sqlite"

    monkeypatch.setattr(paths_mod, "DATA_STORE_DIR", store_dir)
    monkeypatch.setattr(paths_mod, "INGEST_LOG_DIR", ingest_dir)
    monkeypatch.setattr(paths_mod, "METADATA_DB_PATH", db_path)
    monkeypatch.setattr(store_mod, "DATA_STORE_DIR", store_dir)
    monkeypatch.setattr(inv_mod, "_paths_module", paths_mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_series_map(n_bars: int = 200) -> dict:
    """Return a small series_map with one ticker for speed."""
    from src.validation.intraday_synthetic import make_structured_1h_series
    series = make_structured_1h_series(
        ticker="AAPL_1H", n_bars=n_bars, seed=1, base_price=255.0
    )
    return {"AAPL_1H": series}


def _make_csv_content(n_bars: int = 50) -> str:
    """Generate a minimal valid 1h CSV string."""
    from datetime import timedelta

    lines = ["timestamp,open,high,low,close,volume"]
    base = datetime(2025, 1, 2, 14, 0, 0, tzinfo=timezone.utc)
    price = 100.0
    for i in range(n_bars):
        ts = (base + timedelta(hours=i)).isoformat()
        lo = price * 0.99
        hi = price * 1.01
        lines.append(f"{ts},{price:.4f},{hi:.4f},{lo:.4f},{price:.4f},1000000")
        price += 0.1
    return "\n".join(lines)


# ===========================================================================
# Group 1: Intraday Ingest
# ===========================================================================

class TestIntradayIngest:

    def test_populate_1h_store_writes_rows(self, tmp_path):
        """T1: populate_1h_store writes >0 rows for each ticker into tmp store."""
        from src.data_store.inventory import DataInventory
        from src.data_store.store import DataStore
        from src.phase15.intraday_ingest import populate_1h_store

        store = DataStore(store_dir=tmp_path / "store")
        inventory = DataInventory(db_path=tmp_path / "meta.sqlite")

        results = populate_1h_store(
            store=store,
            inventory=inventory,
            n_bars=200,
            tickers=["AAPL_1H", "MSFT_1H"],
        )

        assert "AAPL_1H" in results
        assert "MSFT_1H" in results
        assert results["AAPL_1H"] > 0
        assert results["MSFT_1H"] > 0

        # Verify data is actually in store
        loaded = store.load("AAPL_1H", "1h")
        assert len(loaded.bars) > 0

    def test_populate_1h_store_inventory_updated(self, tmp_path):
        """T2: inventory has 1h entries for each ticker after populate."""
        from src.data_store.inventory import DataInventory
        from src.data_store.store import DataStore
        from src.phase15.intraday_ingest import populate_1h_store

        store = DataStore(store_dir=tmp_path / "store")
        inventory = DataInventory(db_path=tmp_path / "meta.sqlite")

        populate_1h_store(
            store=store,
            inventory=inventory,
            n_bars=200,
            tickers=["AAPL_1H"],
        )

        cov = inventory.get("AAPL_1H", "1h")
        assert cov is not None
        assert cov.ticker == "AAPL_1H"
        assert cov.timeframe == "1h"
        assert cov.row_count > 0
        # Sentinel: simulation data has last_fetch_at in year 2000
        assert cov.last_fetch_at is not None
        assert cov.last_fetch_at.year == 2000

    def test_import_real_1h_csv_valid(self, tmp_path):
        """T3: import_real_1h_csv with a valid CSV writes rows into store."""
        from src.data_store.inventory import DataInventory
        from src.data_store.store import DataStore
        from src.phase15.intraday_ingest import import_real_1h_csv

        store = DataStore(store_dir=tmp_path / "store")
        inventory = DataInventory(db_path=tmp_path / "meta.sqlite")

        csv_content = _make_csv_content(n_bars=50)
        csv_path = tmp_path / "real_aapl.csv"
        csv_path.write_text(csv_content)

        rows = import_real_1h_csv(
            csv_path=csv_path,
            ticker="AAPL_REAL",
            store=store,
            inventory=inventory,
        )

        assert rows > 0

        # Verify inventory was updated with a real timestamp (not sentinel)
        cov = inventory.get("AAPL_REAL", "1h")
        assert cov is not None
        assert cov.row_count > 0
        # Real import: year should NOT be 2000
        assert cov.last_fetch_at is not None
        assert cov.last_fetch_at.year != 2000

    def test_inventory_summary_structure(self, tmp_path):
        """T4: get_1h_inventory_summary returns list with required keys."""
        from src.data_store.inventory import DataInventory
        from src.data_store.store import DataStore
        from src.phase15.intraday_ingest import get_1h_inventory_summary, populate_1h_store

        store = DataStore(store_dir=tmp_path / "store")
        inventory = DataInventory(db_path=tmp_path / "meta.sqlite")

        populate_1h_store(
            store=store,
            inventory=inventory,
            n_bars=200,
            tickers=["AAPL_1H"],
        )

        summary = get_1h_inventory_summary(inventory=inventory, tickers=["AAPL_1H"])

        assert isinstance(summary, list)
        assert len(summary) == 1

        item = summary[0]
        required_keys = {"ticker", "timeframe", "first_date", "last_date", "row_count", "is_fresh", "source"}
        assert required_keys.issubset(item.keys()), f"Missing keys: {required_keys - item.keys()}"
        assert item["ticker"] == "AAPL_1H"
        assert item["timeframe"] == "1h"
        # Simulation data
        assert item["source"] == "simulation"
        assert item["row_count"] > 0


# ===========================================================================
# Group 2: Dual Calibration
# ===========================================================================

class TestDualCalibration:

    def test_dual_calibration_grid_size(self, tmp_path):
        """T5: run_dual_calibration returns 4x4=16 grid points per ticker."""
        from src.phase15.dual_calibration import run_dual_calibration

        series_map = _make_small_series_map(n_bars=200)

        results = run_dual_calibration(
            series_map=series_map,
            rr_thresholds=[1.25, 1.50, 1.75, 2.00],
            distance_pcts=[0.001, 0.002, 0.003, 0.005],
            starting_capital=100_000.0,
            min_bars_required=40,
        )

        assert "AAPL_1H" in results
        cal = results["AAPL_1H"]
        assert len(cal.grid) == 16, f"Expected 16 grid points, got {len(cal.grid)}"

    def test_dual_calibration_distance_pct_matters(self, tmp_path):
        """T6: at lower distance_pct, n_trades >= n_trades at dist_pct=0.005."""
        from src.phase15.dual_calibration import run_dual_calibration

        series_map = _make_small_series_map(n_bars=200)

        results = run_dual_calibration(
            series_map=series_map,
            rr_thresholds=[1.25, 1.50, 1.75, 2.00],
            distance_pcts=[0.001, 0.005],
            starting_capital=100_000.0,
            min_bars_required=40,
        )

        cal = results["AAPL_1H"]

        # For a given RR, lower distance_pct should yield >= trades than 0.005
        for rr in [1.25, 1.50]:
            trades_001 = next(
                (p.n_trades for p in cal.grid if p.rr_threshold == rr and abs(p.distance_pct - 0.001) < 1e-9),
                0,
            )
            trades_005 = next(
                (p.n_trades for p in cal.grid if p.rr_threshold == rr and abs(p.distance_pct - 0.005) < 1e-9),
                0,
            )
            assert trades_001 >= trades_005, (
                f"RR={rr}: trades at dist=0.001 ({trades_001}) < trades at dist=0.005 ({trades_005})"
            )

    def test_dual_calibration_recommended_in_range(self, tmp_path):
        """T7: recommended_rr in [1.25, 2.0], recommended_distance_pct in [0.001, 0.005]."""
        from src.phase15.dual_calibration import run_dual_calibration

        series_map = _make_small_series_map(n_bars=200)

        results = run_dual_calibration(
            series_map=series_map,
            rr_thresholds=[1.25, 1.50, 1.75, 2.00],
            distance_pcts=[0.001, 0.002, 0.003, 0.005],
            starting_capital=100_000.0,
            min_bars_required=40,
        )

        cal = results["AAPL_1H"]
        assert 1.25 <= cal.recommended_rr <= 2.00, f"recommended_rr={cal.recommended_rr} out of range"
        assert 0.001 <= cal.recommended_distance_pct <= 0.005, (
            f"recommended_distance_pct={cal.recommended_distance_pct} out of range"
        )

    def test_dual_calibration_print_runs(self, tmp_path):
        """T8: print_dual_calibration_table runs without error."""
        from src.phase15.dual_calibration import print_dual_calibration_table, run_dual_calibration

        series_map = _make_small_series_map(n_bars=200)

        results = run_dual_calibration(
            series_map=series_map,
            rr_thresholds=[1.25, 1.50],
            distance_pcts=[0.001, 0.005],
            starting_capital=100_000.0,
            min_bars_required=40,
        )

        # Should not raise
        print_dual_calibration_table(results)


# ===========================================================================
# Group 3: Dataset Expansion
# ===========================================================================

class TestDatasetExpansion:

    def test_dataset_expansion_collects_trades(self, tmp_path):
        """T9: expand_dataset returns DatasetExpansionResult with total_trades >= 0."""
        from src.phase15.dataset_expansion import DatasetExpansionResult, expand_dataset

        series_map = _make_small_series_map(n_bars=200)

        result = expand_dataset(
            series_map=series_map,
            calibrated_rr=1.25,
            calibrated_distance_pct=0.001,
            starting_capital=100_000.0,
            min_bars_required=40,
        )

        assert isinstance(result, DatasetExpansionResult)
        assert result.total_trades >= 0
        assert result.model_decision in ("trained_model", "heuristic_fallback", "insufficient_data")

    def test_dataset_expansion_label_keys(self, tmp_path):
        """T10: result has positive_labels, negative_labels, label_balance."""
        from src.phase15.dataset_expansion import expand_dataset

        series_map = _make_small_series_map(n_bars=200)

        result = expand_dataset(
            series_map=series_map,
            calibrated_rr=1.25,
            calibrated_distance_pct=0.001,
            starting_capital=100_000.0,
            min_bars_required=40,
        )

        assert hasattr(result, "positive_labels")
        assert hasattr(result, "negative_labels")
        assert hasattr(result, "label_balance")
        assert result.positive_labels + result.negative_labels == result.total_trades
        assert 0.0 <= result.label_balance <= 1.0

    def test_dataset_expansion_per_ticker(self, tmp_path):
        """T11: per_ticker_counts has an entry for each input ticker."""
        from src.validation.intraday_synthetic import make_structured_1h_series
        from src.phase15.dataset_expansion import expand_dataset

        series_map = {
            "AAPL_1H": make_structured_1h_series("AAPL_1H", n_bars=200, seed=1, base_price=255.0),
            "MSFT_1H": make_structured_1h_series("MSFT_1H", n_bars=200, seed=2, base_price=400.0),
        }

        result = expand_dataset(
            series_map=series_map,
            calibrated_rr=1.25,
            calibrated_distance_pct=0.001,
            starting_capital=100_000.0,
            min_bars_required=40,
        )

        assert "AAPL_1H" in result.per_ticker_counts
        assert "MSFT_1H" in result.per_ticker_counts


# ===========================================================================
# Group 4: Phase 15 Runner
# ===========================================================================

class TestPhase15Runner:

    def test_phase15_runs_without_error(self, tmp_path):
        """T12: run_phase15(tickers=["AAPL_1H"], n_bars=300) completes."""
        from src.phase15.phase15_runner import run_phase15

        result = run_phase15(
            tickers=["AAPL_1H"],
            n_bars=300,
            starting_capital=100_000.0,
            min_bars_required=40,
            rr_thresholds=[1.25, 1.50],
            distance_pcts=[0.001, 0.005],
            save_model=False,
        )

        assert result is not None
        assert "AAPL_1H" in result.tickers_loaded
        assert result.total_1h_bars_stored > 0

    def test_phase15_verdict_valid(self, tmp_path):
        """T13: verdict is one of the four allowed strings."""
        from src.phase15.phase15_runner import run_phase15

        result = run_phase15(
            tickers=["AAPL_1H"],
            n_bars=300,
            starting_capital=100_000.0,
            min_bars_required=40,
            rr_thresholds=[1.25, 1.50],
            distance_pcts=[0.001, 0.005],
            save_model=False,
        )

        allowed_verdicts = {
            "READY_FOR_PAPER_TRADING",
            "NEEDS_MORE_INTRADAY_DATA",
            "NEEDS_CALIBRATION",
            "FAILS_CURRENT_ACCEPTANCE",
        }
        assert result.verdict in allowed_verdicts, (
            f"Invalid verdict: {result.verdict!r}"
        )

    def test_phase15_recommended_config_keys(self, tmp_path):
        """T14: recommended_config has FTA_MIN_REWARD_RISK and FTA_MIN_DISTANCE_TO_FTA_PCT."""
        from src.phase15.phase15_runner import run_phase15

        result = run_phase15(
            tickers=["AAPL_1H"],
            n_bars=300,
            starting_capital=100_000.0,
            min_bars_required=40,
            rr_thresholds=[1.25, 1.50],
            distance_pcts=[0.001, 0.005],
            save_model=False,
        )

        assert "FTA_MIN_REWARD_RISK" in result.recommended_config
        assert "FTA_MIN_DISTANCE_TO_FTA_PCT" in result.recommended_config
        assert isinstance(result.recommended_config["FTA_MIN_REWARD_RISK"], float)
        assert isinstance(result.recommended_config["FTA_MIN_DISTANCE_TO_FTA_PCT"], float)

    def test_phase15_report_prints(self, tmp_path):
        """T15: print_phase15_report runs without error on a result."""
        from src.phase15.phase15_runner import print_phase15_report, run_phase15

        result = run_phase15(
            tickers=["AAPL_1H"],
            n_bars=300,
            starting_capital=100_000.0,
            min_bars_required=40,
            rr_thresholds=[1.25, 1.50],
            distance_pcts=[0.001, 0.005],
            save_model=False,
        )

        # Should not raise
        print_phase15_report(result)
