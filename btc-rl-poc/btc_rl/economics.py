"""DT-08 — CANONICAL TRADER / PAPER-ECONOMICS METRICS (evaluation/economics owner).

Single backend owner for every economic number the UI shows, so no browser
JavaScript ever recomputes P&L / EV / drawdown / rates. Probability metrics live in
btc_rl/metrics.py; this module owns the economic layer. All functions are pure,
empty-safe (return None on empty), and never fabricate counterfactuals — metrics
that need ground truth we don't have return None so the UI can render UNAVAILABLE
rather than a fake number (§5/§16).

Units: cents unless noted (matches the daemon's *_bankroll_c / pnl_c ledgers).
"""
from __future__ import annotations

from . import metrics as M


def _nz(xs):
    return [x for x in xs if x is not None]


def realized_pnl(pnls: list[float]) -> float | None:
    """Total realized paper P&L (sum of settled trade pnls)."""
    xs = _nz(pnls)
    return float(sum(xs)) if xs else None


def ev_per_trade(pnls: list[float]) -> float | None:
    """Mean realized P&L per settled trade."""
    xs = _nz(pnls)
    return float(sum(xs) / len(xs)) if xs else None


def ev_per_eligible_window(pnls: list[float], n_eligible: int) -> float | None:
    """Primary trader metric (§36): realized paper EV per ELIGIBLE window — untraded
    eligible windows contribute 0, so this rewards coverage-adjusted profitability."""
    if not n_eligible or n_eligible <= 0:
        return None
    return float(sum(_nz(pnls)) / n_eligible)


def win_rate(wins: list[int]) -> float | None:
    xs = _nz(wins)
    return float(sum(1 for w in xs if w) / len(xs)) if xs else None


def coverage(n_traded: int, n_eligible: int) -> float | None:
    if not n_eligible or n_eligible <= 0:
        return None
    return float(n_traded / n_eligible)


def equity_curve(starting_c: float, pnls: list[float]) -> list[float]:
    """Cumulative bankroll from a starting balance and ordered trade pnls."""
    cur, out = float(starting_c), []
    for p in pnls:
        cur += (p or 0.0)
        out.append(cur)
    return out


def drawdown(equity: list[float]) -> float | None:
    """Max peak-to-trough drawdown of an equity curve (delegates to metrics owner)."""
    if not equity:
        return None
    return M.max_drawdown(equity)


def total_return(starting_c: float, ending_c: float) -> float | None:
    if not starting_c:
        return None
    return float((ending_c - starting_c) / starting_c)


def bad_entry_rate(pnls: list[float]) -> float | None:
    """Fraction of settled trades that lost money (entered a losing position)."""
    xs = _nz(pnls)
    return float(sum(1 for p in xs if p < 0) / len(xs)) if xs else None


def profitable_trade_recall(captured_profitable: int | None,
                            total_profitable_available: int | None) -> float | None:
    """Of windows where a profitable entry was AVAILABLE (counterfactual ground
    truth), the fraction the trader captured. Returns None (UNAVAILABLE) unless the
    counterfactual is supplied — never fabricated."""
    if not total_profitable_available:
        return None
    return float((captured_profitable or 0) / total_profitable_available)


def missed_profit_rate(missed_profitable: int | None,
                       total_profitable_available: int | None) -> float | None:
    """Complement of recall over available profitable windows. None if unavailable."""
    r = profitable_trade_recall(
        (total_profitable_available or 0) - (missed_profitable or 0),
        total_profitable_available)
    return None if r is None else float(1.0 - r)


def paired_delta(control: list[float], treatment: list[float]) -> dict | None:
    """Paired treatment-minus-control effect on a per-unit metric (e.g. per-window
    pnl), with a moving-block bootstrap 95% CI that respects serial correlation.
    `control` and `treatment` must be aligned by unit (same window order)."""
    n = min(len(control), len(treatment))
    if n < 5:
        return None
    diffs = [treatment[i] - control[i] for i in range(n)]
    point = sum(diffs) / n
    # deterministic moving-block bootstrap (LCG; no RNG dependency)
    import math
    B, BLK = 2000, 6
    nblocks = int(math.ceil(n / BLK))
    boot = []
    for b in range(B):
        s = (1103515245 * (b + 1) + 12345) & 0x7fffffff
        acc, cnt = 0.0, 0
        for _ in range(nblocks):
            s = (1103515245 * s + 12345) & 0x7fffffff
            start = s % n
            for k in range(BLK):
                if cnt < n:
                    acc += diffs[(start + k) % n]; cnt += 1
        boot.append(acc / n)
    boot.sort()
    lo, hi = boot[int(0.025 * B)], boot[int(0.975 * B)]
    return {"paired_delta": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_pairs": n, "significant": lo > 0 or hi < 0}


def summarize_trades(trades: list[dict], starting_c: float, n_eligible: int,
                     pnl_key: str = "pnl_c", win_key: str = "win") -> dict:
    """Full canonical economic summary for one trader from its settled trade rows.
    Untraded/skipped rows (pnl None) are excluded from trade stats but the caller
    supplies n_eligible so coverage/EV-per-eligible are honest."""
    settled = [t for t in trades if t.get(pnl_key) is not None and not t.get("skipped")]
    pnls = [t[pnl_key] for t in settled]
    wins = [t.get(win_key) for t in settled]
    eq = equity_curve(starting_c, pnls)
    ending = eq[-1] if eq else starting_c
    return {
        "n_trades": len(settled),
        "n_eligible": n_eligible,
        "coverage": coverage(len(settled), n_eligible),
        "realized_pnl_c": realized_pnl(pnls),
        "ev_per_trade_c": ev_per_trade(pnls),
        "realized_ev_per_eligible_c": ev_per_eligible_window(pnls, n_eligible),
        "win_rate": win_rate(wins),
        "bad_entry_rate": bad_entry_rate(pnls),
        "starting_capital_c": starting_c,
        "ending_capital_c": ending,
        "total_return": total_return(starting_c, ending),
        "max_drawdown_c": drawdown(eq),
        "profitable_trade_recall": None,   # counterfactual not supplied -> UNAVAILABLE
        "missed_profit_rate": None,
    }
