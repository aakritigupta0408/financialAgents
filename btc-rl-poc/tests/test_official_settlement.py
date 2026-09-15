"""Guard: the whole paper desk settles ONLY on official Kalshi — never the candle proxy.

This makes the settlement-integrity fix permanent. If anyone reintroduces a proxy
settlement (contract_truth.resolve_outcome with a candle fallback) in the daemon, or a
live arm books a PROXY_DEGRADED outcome, these tests fail.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONLINE = ROOT / "btc_rl" / "online.py"
RES = ROOT / "results"
# arms whose settlement must be official. After the 2026-09-15 roster cut the live
# desk is T0 (pt) + T1 (cg33) + T2 (fm); the retired arms' ledgers are still checked
# so a stray proxy row can never reappear even in frozen history.
ACTIVE_ARMS = ["pt_trades.jsonl", "cg33_trades.jsonl", "fm_trades.jsonl",
               "pt3_trades.jsonl", "cg5_trades.jsonl", "cg10_trades.jsonl",
               "tv_trades.jsonl"]


def test_daemon_has_no_proxy_settlement():
    src = ONLINE.read_text()
    # the proxy-fallback resolver must not be called anywhere in the daemon
    assert "resolve_outcome(" not in src, \
        "contract_truth.resolve_outcome (candle-proxy fallback) reintroduced in online.py"
    # settlement must go through the single official source
    assert "_official_outcome(" in src, "official-settlement helper missing"
    assert '"OFFICIAL_KALSHI"' in src, "official settlement quality tag missing"


def test_no_proxy_outcome_expression_in_settles():
    # the specific bug shape: booking an outcome from the candle close vs strike
    src = ONLINE.read_text()
    assert 'settle_bar["close"] >= ' not in src, \
        "trader outcome derived from candle close (proxy) — must use official Kalshi"


def test_active_arm_ledgers_have_no_proxy_settlement():
    for log in ACTIVE_ARMS:
        p = RES / log
        if not p.exists():
            continue
        for l in p.open():
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r.get("actual") is not None:
                q = r.get("contract_truth_quality")
                assert q != "PROXY_DEGRADED", \
                    f"{log}: settled row on PROXY_DEGRADED ({r.get('ticker')}) — must be OFFICIAL_KALSHI"
