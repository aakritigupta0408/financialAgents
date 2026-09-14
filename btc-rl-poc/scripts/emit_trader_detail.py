"""Trader-detail data for HOME's trader-family experience: per-trader backtest
equity curve + recent trades + policy architecture, all from the FROZEN exact-BRTI
replay holdout (static backtest evidence). Plus the live A/B block (T0 vs T1),
collecting until DT-01 activation. Writes results/trader_detail.json.

Numbers are computed by the canonical replay/economics owners; the frontend only
renders them.
"""
import calendar
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import replay_backtest as RB          # noqa: E402
from btc_rl import economics as E     # noqa: E402

OUT = ROOT / "results" / "trader_detail.json"

POLICY = {
    "T0": ["Check lock", "EV ≥ minimum", "Fixed stake", "Execute"],
    "T1": ["Check lock", "EV ≥ minimum", "|Oracle − market| ≥ 0.15", "Fixed stake", "Execute"],
    "T2": ["Check lock", "Meta-model P(profit)", "Take / skip", "Execute if take"],
    "T3": ["Check lock", "Read state", "Size: skip / S / M / L", "Execute"],
    "T4": ["Awaiting component qualification"],
}
ORACLE_STATE = ["p_oracle", "p_market", "delta", "lock state", "features"]
TRADE_STEPS = ["Order", "Fill", "Settle", "P&L"]


def _et(ts):
    # display timestamp (UTC label; the desk trades US markets, shown as captured)
    return time.strftime("%b %d, %H:%M", time.gmtime(ts))


def build():
    rows, _ = RB.load()
    dev, val, hold = RB.split(rows)
    devval = dev | val
    sigma, iso = RB.fit_oracle([r for r in rows if r["market_window_id"] in devval])
    RB.attach_oracle(rows, sigma, iso)
    wr = RB.by_window(rows, hold)
    # row lookup for entry BRTI/target
    row_at = {(r["market_window_id"], r["decision_time"]): r for r in rows}

    detail = {}
    for tid in ("T0", "T1", "T2", "T3"):
        res = RB.run_trader(wr, tid, RB.T.T1_EDGE_TAU)
        trades = [x["traded"] for x in res if x["traded"]]
        trades.sort(key=lambda t: t["close_ts"])
        pnls = [t["pnl_c"] for t in trades]
        eq = E.equity_curve(0.0, pnls)                 # cents, chronological
        curve = [{"i": i, "equity_c": round(v, 1)} for i, v in enumerate(eq)]
        recent = []
        for t in trades[-6:]:
            r = row_at.get((t["ticker"], t["close_ts"]), {})
            recent.append({
                "time": _et(t["close_ts"]),
                "brti": round(r.get("current_brti"), 0) if r.get("current_brti") else None,
                "target": round(r.get("official_target"), 0) if r.get("official_target") else None,
                "side": t["side"].upper(),
                "entry_price": round(t["cost"], 2),
                "result": "WIN" if t["win"] else "LOSS",
                "pnl_c": round(t["pnl_c"], 1),
                "status": "Settled"})
        detail[tid] = {
            "policy_steps": POLICY[tid],
            "oracle_state": ORACLE_STATE,
            "trade_steps": TRADE_STEPS,
            "equity_curve": curve,
            "n_trades": len(trades),
            "recent_trades": list(reversed(recent)),
        }
    detail["T4"] = {"policy_steps": POLICY["T4"], "oracle_state": ORACLE_STATE,
                    "trade_steps": TRADE_STEPS, "equity_curve": [], "n_trades": 0,
                    "recent_trades": []}

    freeze = json.loads((ROOT / "research" / "replay" / "prospective_freeze.json").read_text())
    exp = freeze["first_experiment"]
    ab = {
        "state": "NOT_STARTED",
        "control": {"id": "T0", "name": "Baseline Follower"},
        "treatment": {"id": "T1", "name": "Selective Edge"},
        "primary_metric": "Paired Δ EV / eligible window",
        "paired_delta": None, "live_n_paired": 0, "required_sample": None,
        "ci95": None, "sequential_progress_pct": 0,
        "spec_hash": exp.get("spec_hash"),
        "note": "Live trading begins once the exact-BRTI runtime is activated.",
    }
    doc = {"schema_version": "trader-detail-1", "generated_at": time.time(),
           "holdout_windows": len(hold),
           "settlement": "OFFICIAL_EXACT_BRTI (unseen historical holdout)",
           "traders": detail, "ab_experiment": ab,
           "note": "Backtest equity/trades from the frozen exact-BRTI replay holdout; "
                   "live metrics collect until DT-01 activation."}
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"trader_detail: {len(hold)} holdout windows; "
          + " ".join(f"{k}={detail[k]['n_trades']}tr" for k in ("T0", "T1", "T2", "T3")))


if __name__ == "__main__":
    build()
