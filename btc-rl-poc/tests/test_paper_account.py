import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btc_rl import paper_account as PA  # noqa: E402


def test_new_account_has_no_retroactive_pnl():
    a = PA.create_account("pa1", "pt", "v1", started_at=1000.0, starting_capital_c=100000)
    assert a["realized_pnl_c"] == 0.0
    assert a["equity_c"] == 100000
    assert a["history"] == []


def test_retroactive_event_rejected():
    a = PA.create_account("pa1", "pt", "v1", started_at=1000.0, starting_capital_c=100000)
    with pytest.raises(ValueError):
        PA.apply_settlement(a, staked_c=200, payout_c=300, fee_c=1, ts=999.0)


def test_settlement_and_reconciliation():
    a = PA.create_account("pa1", "pt", "v1", started_at=1000.0, starting_capital_c=100000)
    PA.apply_settlement(a, staked_c=200, payout_c=300, fee_c=1, ts=1001.0)  # +99 net
    r = PA.reconcile(a)
    assert r["reconciled"] is True
    assert abs(a["realized_pnl_c"] - 99) < 1e-9


def test_reconciliation_detects_residual():
    a = PA.create_account("pa1", "pt", "v1", started_at=1000.0, starting_capital_c=100000)
    PA.apply_settlement(a, staked_c=200, payout_c=300, fee_c=1, ts=1001.0)
    a["equity_c"] += 500          # inject inconsistency
    r = PA.reconcile(a)
    assert r["reconciled"] is False
    assert r["accounting_incident"]["category"] == "ACCOUNTING"


def test_service_activation_unavailable_not_faked():
    s = PA.start_service_status()
    assert s["activation"] == "UNAVAILABLE"
