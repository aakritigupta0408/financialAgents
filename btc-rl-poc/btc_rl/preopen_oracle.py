"""PRE-OPEN ORACLE capture (handoff §11) — the cleanest test of independent
predictive ability: form a settlement view BEFORE Kalshi's window opens.

For each upcoming 15-min expiry we store independent feature snapshots at
lead times before T_open (= expiry - 15min). NO Kalshi price/probability is
ever stored in a pre-open snapshot. When the contract's strike is later
published, attach_strike() links it to the already-frozen snapshot; the
settlement label is attached after expiry. This makes premature knowledge
of the strike or the market impossible by construction.

Runtime: the daemon calls due_snapshots(now) each loop and, for any due
(expiry, lead), calls snapshot(expiry, lead, btc_state, deriv_state).
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "preopen_oracle.jsonl"
WINDOW_S = 900                     # 15-minute contract
LEADS_S = [300, 120, 60, 30, 10]   # snapshot at T_open - lead


def next_expiries(now, ahead=2):
    """Upcoming :00/:15/:30/:45 boundaries (epoch seconds)."""
    step = WINDOW_S
    base = (int(now) // step + 1) * step
    return [base + i * step for i in range(ahead)]


def due_snapshots(now, tol=3.0):
    """(expiry, lead_s) pairs whose scheduled pre-open time is ~now."""
    due = []
    for exp in next_expiries(now, ahead=3):
        t_open = exp - WINDOW_S
        for lead in LEADS_S:
            sched = t_open - lead
            if -tol <= (now - sched) <= tol:
                due.append((exp, lead))
    return due


def snapshot(expiry, lead_s, btc_state, deriv_state=None, options_state=None,
             news_state=None):
    """Freeze an independent pre-open snapshot. btc_state is a dict of
    BTC-only features (price, returns, realized vol, spot flow, cross-venue)
    computed by the daemon. Kalshi is NOT accepted here."""
    assert not any(k.startswith(("k_", "kalshi", "mkt_")) for k in btc_state), \
        "pre-open Oracle must not receive Kalshi inputs"
    now = time.time()
    rec = {"kind": "preopen_snapshot", "expiry_ts": expiry,
           "t_open_ts": expiry - WINDOW_S, "lead_s": lead_s,
           "captured_ts": now, "available_for_decision_ts": now,
           "btc": btc_state, "deriv": deriv_state or {},
           "options": options_state or {}, "news": news_state or {},
           "strike": None, "settlement": None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _rewrite(match, update):
    """Append-safe field attachment: rewrite matching rows in place (the file
    is small; history of the snapshot content itself is preserved — only the
    later-known strike/settlement are filled once)."""
    if not OUT.exists():
        return 0
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    n = 0
    for r in rows:
        if match(r):
            for k, v in update.items():
                if r.get(k) is None:
                    r[k] = v; n += 1
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    return n


def attach_strike(expiry, strike):
    return _rewrite(lambda r: r.get("expiry_ts") == expiry and r.get("strike") is None,
                    {"strike": strike})


def attach_settlement(expiry, settlement):
    return _rewrite(lambda r: r.get("expiry_ts") == expiry and r.get("settlement") is None,
                    {"settlement": int(settlement)})


if __name__ == "__main__":
    now = time.time()
    print("next expiries:", [time.strftime('%H:%M:%S', time.gmtime(e))
                             for e in next_expiries(now)])
    print("lead schedule (s before T_open):", LEADS_S)
    # self-test: reject Kalshi leakage
    try:
        snapshot(next_expiries(now)[0], 60, {"k_prob": 0.5})
        print("LEAK GUARD FAILED")
    except AssertionError:
        print("leak guard OK — Kalshi inputs rejected")
