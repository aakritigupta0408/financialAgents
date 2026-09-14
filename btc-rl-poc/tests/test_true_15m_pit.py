"""OPEN_ORACLE_15M falsification invariants (directive §4, §13, §53).

TRUE_15M_NO_POST_OPEN_INFORMATION — every feature in the canonical T0 dataset is
computed only from windows that closed at or before the target window's open.
ONE_PREDICTION_PER_WINDOW      — exactly one row per market_window_id; the
                                 independent unit is the window, never a tick.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "results" / "open_oracle_15m_dataset.jsonl"
META = ROOT / "results" / "open_oracle_15m_dataset.meta.json"


def _rows():
    assert DS.exists(), "run scripts/build_open_oracle_dataset.py first"
    return [json.loads(l) for l in DS.open() if l.strip()]


def test_true_15m_no_post_open_information():
    bad = [r["market_window_id"] for r in _rows() if r["max_source_close_ts"] > r["T0"]]
    assert not bad, f"TRUE_15M_NO_POST_OPEN_INFORMATION: {len(bad)} rows use post-open data"


def test_meta_reports_zero_violations():
    m = json.loads(META.read_text())
    assert m["post_open_information_violations"] == 0


def test_one_prediction_per_window():
    rows = _rows()
    ids = [r["market_window_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate market_window_id — not one row per window"
    m = json.loads(META.read_text())
    assert m["raw_observation_n"] == m["market_window_n"], "tick inflation: raw_n != window_n"


def test_windows_are_15_minutes_and_labeled():
    for r in _rows():
        assert r["T1"] - r["T0"] == 900, f"{r['market_window_id']} not a 900s window"
        assert r["official_outcome"] in (0, 1), "label must be exact 0/1"
        assert r["p_mech_15m"] == 0.5, "mechanics control at the open must be 0.5 (driftless)"
