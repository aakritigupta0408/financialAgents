"""§13-14 — PAPER FOLLOW ACCOUNT contract + reconciliation.

A "Follow — Paper" action creates an independent account from NOW. There are NO
retroactive historical profits: only trader decisions AFTER started_at affect the
account (§13). Economics use the canonical owner (btc_rl/economics). Every account
maintains a reconciliation identity (§14); an unexplained residual is an accounting
incident.

The live follow SERVICE (applying future trader decisions to open accounts) is not
yet wired into the running daemon, so start_service_status() reports UNAVAILABLE
honestly rather than fabricating fills (§13/§16).
"""
from __future__ import annotations

from . import economics as E

SCHEMA = "paper-account-1"


def create_account(paper_account_id: str, trader_id: str, trader_version: str,
                   started_at: float, starting_capital_c: float) -> dict:
    """New account, funded, no positions, no retroactive P&L."""
    return {
        "schema": SCHEMA,
        "paper_account_id": paper_account_id,
        "trader_id": trader_id,
        "trader_version": trader_version,
        "started_at": started_at,
        "starting_capital_c": starting_capital_c,
        "cash_c": starting_capital_c,
        "open_exposure_c": 0.0,
        "realized_pnl_c": 0.0,
        "fees_c": 0.0,
        "settled_positions": 0,
        "equity_c": starting_capital_c,
        "history": [],           # only events with ts >= started_at may be appended
    }


def apply_settlement(account: dict, staked_c: float, payout_c: float, fee_c: float,
                     ts: float) -> dict:
    """Apply one settled position that occurred AFTER the account started. Rejects
    retroactive events (§13)."""
    if ts < account["started_at"]:
        raise ValueError("retroactive event rejected: ts < started_at (§13)")
    pnl = payout_c - staked_c - fee_c
    account["cash_c"] += payout_c - fee_c        # stake was already debited at entry
    account["realized_pnl_c"] += pnl
    account["fees_c"] += fee_c
    account["settled_positions"] += 1
    account["equity_c"] = account["cash_c"] + account["open_exposure_c"]
    account["history"].append({"ts": ts, "staked_c": staked_c, "payout_c": payout_c,
                               "fee_c": fee_c, "pnl_c": pnl})
    return account


def reconcile(account: dict, tol_c: float = 0.5) -> dict:
    """§14 reconciliation identity:
       equity == cash + open_exposure
       realized_pnl == sum(history pnl)
    A residual beyond tolerance is an accounting incident."""
    id_equity = account["cash_c"] + account["open_exposure_c"]
    residual_equity = round(account["equity_c"] - id_equity, 4)
    pnl_sum = E.realized_pnl([h["pnl_c"] for h in account["history"]]) or 0.0
    residual_pnl = round(account["realized_pnl_c"] - pnl_sum, 4)
    ok = abs(residual_equity) <= tol_c and abs(residual_pnl) <= tol_c
    doc = {"reconciled": ok,
           "residual_equity_c": residual_equity, "residual_pnl_c": residual_pnl}
    if not ok:
        doc["accounting_incident"] = {
            "severity": "SEV-2", "category": "ACCOUNTING",
            "detail": "paper account reconciliation identity violated",
            "paper_account_id": account["paper_account_id"]}
    return doc


def start_service_status() -> dict:
    """Live follow service (applying future decisions to open accounts) is not yet
    wired into the daemon — reported UNAVAILABLE, never faked."""
    return {"schema": SCHEMA, "service": "paper_follow",
            "activation": "UNAVAILABLE",
            "reason": "follow service is not wired into btc_rl/online.py; the contract "
                      "and reconciliation exist and are tested, but no live account "
                      "accrues until the daemon applies future decisions to accounts.",
            "provenance": "backend contract only"}
