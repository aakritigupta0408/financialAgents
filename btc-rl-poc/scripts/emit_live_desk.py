"""LIVE DESK — surface the paper desk's REAL live activity for HOME.

The always-on daemon (btc_rl.online) trades the Oracle's Follower policy live on
every Kalshi 15-minute window and settles it on official BRTI (exact-BRTI runtime,
DT-01). Those real paper bets are written to results/pt_trades.jsonl with realized
P&L and a running bankroll. The Follower is the lineage of trader T0; this is the
live half of the desk (the frozen T0-vs-T1 PAIRED A/B accrues separately and needs
the prospective-capture pipeline — this file does NOT claim that experiment).

Pure presentation adapter: it reads the desk's own ledger and reuses the canonical
metric owner (btc_rl.economics) for equity/drawdown. It computes no new science and
fabricates nothing — if there are no live trades it says so. Writes
results/live_desk.json.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")  # desk owner is US Pacific; show PT (PST/PDT)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import economics as E  # noqa: E402

RES = ROOT / "results"
OUT = RES / "live_desk.json"
PT = RES / "pt_trades.jsonl"
KB = RES / "kalshi_binary_log.jsonl"
STATUS = RES / "online_status.json"

# exact-BRTI runtime activation (paper_desk.sh restart, 2026-09-14 05:03 UTC).
# Windows settled at/after this instant are settled on official BRTI.
ACTIVATION_TS = 1789387391


def _rows(path):
    if not path.exists():
        return []
    out = []
    for l in path.open():
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return out


def _et(ts):
    # Pacific Time (PST/PDT) — the desk owner trades US markets
    return datetime.fromtimestamp(ts, PACIFIC).strftime("%b %d, %H:%M") if ts else None


def _stats(trades):
    """Canonical live stats over a list of settled pt trades."""
    pnls = [t["pnl_c"] for t in trades]
    wins = [1 if t.get("win") else 0 for t in trades]
    eq = [t.get("bankroll_c") for t in trades if t.get("bankroll_c") is not None]
    return {
        "n": len(trades),
        "wins": sum(wins),
        "hit_rate": round(E.win_rate(wins), 4) if wins else None,
        "pnl_c": round(E.realized_pnl(pnls), 1) if pnls else 0.0,
        "ev_per_trade_c": round(E.ev_per_trade(pnls), 2) if pnls else None,
        "max_drawdown_c": round(E.drawdown(eq), 1) if len(eq) > 1 else None,
        "bankroll_c": eq[-1] if eq else None,
    }


def build():
    pt = _rows(PT)
    pt.sort(key=lambda r: r.get("close_ts") or r.get("made_ts") or 0)
    settled = [r for r in pt if r.get("pnl_c") is not None]
    open_rows = [r for r in pt if r.get("pnl_c") is None]

    # current open window: prefer an open pt row; else the newest kalshi_binary
    # row that has not settled yet (enriched with oracle/market probs).
    kb = _rows(KB)
    kb_open = [r for r in kb if r.get("actual") is None]
    cur = None
    if kb_open:
        k = kb_open[-1]
        cur = {
            "ticker": k.get("ticker"),
            "oracle_p_up": k.get("p_up"),
            "market_p_up": k.get("mkt_p_up"),
            "call": "UP" if k.get("call") else "DOWN",
            "brti": round(k["base"], 0) if k.get("base") else None,
            "target": round(k["strike"], 0) if k.get("strike") else None,
            "mins_left": k.get("mins_left"),
        }
        po = [r for r in open_rows if r.get("ticker") == k.get("ticker")]
        if po:
            cur["side"] = po[-1].get("side", "").upper()
            cur["entry_c"] = po[-1].get("ask_c")
            cur["stake_c"] = po[-1].get("stake_c")

    # BRTI-at-entry per ticker from the binary log (pt rows carry only the strike)
    brti_at = {}
    for k in kb:
        if k.get("ticker") and k.get("base") is not None:
            brti_at.setdefault(k["ticker"], k["base"])  # first (earliest) seen

    def _trow(t, status):
        return {
            "sort_ts": t.get("close_ts") or t.get("made_ts") or 0,
            "time": _et(t.get("close_ts") or t.get("made_ts")),
            "ticker": t.get("ticker"),
            "brti": round(brti_at[t["ticker"]], 0) if t.get("ticker") in brti_at else None,
            "target": round(t["strike"], 0) if t.get("strike") else None,
            "side": (t.get("side") or "").upper(),
            "entry_c": t.get("ask_c"),
            "result": (None if status == "Open" else ("WIN" if t.get("win") else "LOSS")),
            "pnl_c": t.get("pnl_c"),
            "bankroll_c": t.get("bankroll_c"),
            "status": status,
        }

    # live trades table: open trade(s) pinned first, then settled most-recent-first
    opens = sorted((_trow(r, "Open") for r in open_rows[-2:]),
                   key=lambda x: x["sort_ts"], reverse=True)
    setts = sorted((_trow(r, "Settled") for r in settled),
                   key=lambda x: x["sort_ts"], reverse=True)[:14]
    recent = opens + setts

    # equity curve: real running bankroll over the recent session (last 80 trades)
    session = settled[-80:]
    curve = [{"i": i, "equity_c": r.get("bankroll_c")}
             for i, r in enumerate(session) if r.get("bankroll_c") is not None]

    since = [t for t in settled if t.get("close_ts", 0) >= ACTIVATION_TS]

    alive_age = None
    if STATUS.exists():
        try:
            alive_age = round(time.time() - json.loads(STATUS.read_text()).get("alive_at", 0), 1)
        except Exception:
            alive_age = None

    doc = {
        "schema_version": "live-desk-1",
        "generated_at": time.time(),
        "settlement": "OFFICIAL_EXACT_BRTI",
        "daemon_alive_age_s": alive_age,
        "activation_ts": ACTIVATION_TS,
        "current_window": cur,
        "since_activation": _stats(since),
        "session": _stats(session),
        "recent_trades": recent,
        "equity_curve": curve,
        "note": "Live paper desk — the Oracle's Follower policy (trader T0's lineage) "
                "trading live Kalshi 15-minute windows, settled on official BRTI. The "
                "formal T0-vs-T1 paired A/B is a separate experiment that accrues "
                "one window at a time.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"live_desk: {len(settled)} settled trades; "
          f"{doc['since_activation']['n']} since exact-BRTI activation; "
          f"bankroll={doc['session']['bankroll_c']}c; "
          f"open_window={cur['ticker'] if cur else None}")


if __name__ == "__main__":
    build()
