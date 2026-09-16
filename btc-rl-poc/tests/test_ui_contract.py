"""DT-07 §41/§42 — UI/backend contract + smoke. The HOME/ORACLE snapshots must
carry the fields the renderers consume, referential integrity must hold, and the
pages must degrade safely on missing data."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _regen():
    for s in ("emit_home_snapshot.py", "emit_oracle_snapshot.py",
              "emit_experiments_snapshot.py"):
        subprocess.run([sys.executable, f"scripts/{s}"], cwd=ROOT,
                       capture_output=True, timeout=120)


def _load(name):
    _regen()
    return json.loads((RES / name).read_text())


def test_home_snapshot_contract():
    d = _load("home_snapshot.json")
    for k in ("schema_version", "metric_definition_version", "generated_at",
              "source_versions", "paper_banner", "oracle_strip", "traders", "bottleneck"):
        assert k in d, k
    assert len(d["traders"]) <= 5                      # §11 roster cap
    control = [t for t in d["traders"] if t.get("role") == "CONTROL"]
    assert len(control) == 1                           # exactly one control
    for t in d["traders"]:
        if t.get("trader_id"):
            o = t["online_summary"]
            for k in ("realized_pnl_c", "coverage", "win_rate", "bad_entry_rate",
                      "max_drawdown_c", "total_return"):
                assert k in o, k
            # counterfactual metrics honestly UNAVAILABLE, not fabricated
            assert o["profitable_trade_recall"] is None


def test_oracle_snapshot_six_subtabs():
    d = _load("oracle_snapshot.json")
    subs = d["subtabs"]
    for k in ("input_data", "feature_processing", "modelling", "output_processing",
              "sevs_and_tickets", "graveyard"):
        assert k in subs, k
    # BRTI prominent + role CONTRACT_TRUTH (§19)
    assert subs["input_data"]["brti"]["role"] == "CONTRACT_TRUTH"
    # DT-01 stays visible until activation (§51)
    aw = subs["sevs_and_tickets"]["architecture_watch"]
    assert any(i["id"] == "DT-01" for i in aw)
    # model cards carry backend-computed aggregate, not raw folds for the browser
    for m in subs["modelling"]["model_cards"]:
        assert "offline_mean_bss" in m


def test_paper_banner_present():
    d = _load("home_snapshot.json")
    assert "PAPER" in d["paper_banner"] and "REAL MONEY DISABLED" in d["paper_banner"]


def test_runtime_truth_not_falsely_active():
    # §12/§51 — while the flag is off, HOME must not claim exact-BRTI runtime
    d = _load("home_snapshot.json")
    assert d["oracle_strip"]["runtime_contract_truth"].startswith("LEGACY")


def test_experiments_platform_contract():
    d = _load("experiments_snapshot.json")
    # §6 unit must be market_window_id; §7 primary metric must not be win rate
    assert d["unit_default"] == "market_window_id"
    pm = d["primary_trader_metric"].upper()
    assert "WIN RATE" not in pm and "WIN_RATE" not in pm and "BRIER" not in pm
    assert "EV_PER_ELIGIBLE_WINDOW" in pm
    for e in d["experiments"]:
        s = e["sample_sizes"]
        # raw observations and windows reported SEPARATELY (§6)
        for k in ("raw_observations_control", "windows_control", "paired_windows",
                  "effective_n"):
            assert k in s
        assert e["unit"] == "market_window_id"
        assert "integrity" in e and "sequential" in e
        # mechanism decomposition never fabricates components (§9)
        md = e["mechanism_decomposition"]
        assert md["gross_price_improvement"] == "UNAVAILABLE"


def test_trader_family_backtest_live_separated():
    d = _load("home_snapshot.json")
    fam = d["trader_family"]
    ids = [t["id"] for t in fam]
    # 2026-09-15: the home board = T0 CONTROL first, then the REAL live treatments
    # (ob, cg33, fm). The old generic T1-T4 backtest placeholder cards are retired from
    # the board; retired arms (cg5/cg10/tv/pt6) must NOT appear either.
    assert ids[0] == "T0"
    assert "ob" in ids and "cg33" in ids and "fm" in ids
    assert not ({"T1", "T2", "T3", "T4", "cg5", "cg10", "tv", "pt6"} & set(ids))
    roles = {t["id"]: t["role"] for t in fam}
    assert roles["T0"] == "CONTROL"
    assert all(roles[a] == "TREATMENT" for a in ("ob", "cg33", "fm"))
    for t in fam:
        # backtest and live are SEPARATE objects, never blended (integrity invariant)
        assert "backtest" in t and "live" in t


def test_experiments_provenance_labeled():
    # §27 — retrospective/proxy-settled evidence must be labeled, not sold as proof
    d = _load("experiments_snapshot.json")
    assert d["evidence_class"] == "ONLINE_PAPER_RETROSPECTIVE"
    assert "settlement_provenance" in d and d["caveats"]
