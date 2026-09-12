"""R2 — OPTIONS capture (Alpha Vantage REALTIME_OPTIONS), prospective &
PIT-safe. Captures the current option chain snapshot for BTC-proxy ETFs and
derives a small mechanistic state: near-ATM implied vol, 25-delta skew, and
put/call activity. HISTORICAL_OPTIONS is EOD-daily and is NOT used for
intraday PIT features (per handoff §9 — do not fake intraday history).

Runtime: poll a few times per session (equities are RTH-only); append to
results/options_capture.jsonl with received_ts as the decision-availability
time. Endpoint verified live (REALTIME_OPTIONS symbol=IBIT -> success). The
live smoke call is omitted here to conserve the AV daily quota.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
KEYFILE = Path.home() / ".alphavantage_key"
OUT = ROOT / "results" / "options_capture.jsonl"
HEALTH = ROOT / "results" / "options_health.json"
BASE = "https://www.alphavantage.co/query"
SCHEMA = "options-v1"
SYMS = ["IBIT", "COIN", "MSTR"]


def _key():
    return KEYFILE.read_text().strip()


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _summarize(rows, spot=None):
    """near-ATM IV + skew from the chain (mechanistic, compact)."""
    calls = [r for r in rows if r.get("type") == "call" and _f(r.get("implied_volatility"))]
    puts = [r for r in rows if r.get("type") == "put" and _f(r.get("implied_volatility"))]
    if not calls or not puts:
        return {}
    # nearest expiry
    exps = sorted({r.get("expiration") for r in rows if r.get("expiration")})
    exp = exps[0] if exps else None
    c = [r for r in calls if r.get("expiration") == exp]
    p = [r for r in puts if r.get("expiration") == exp]
    if spot is None:
        strikes = [_f(r.get("strike")) for r in c if _f(r.get("strike"))]
        spot = sorted(strikes)[len(strikes) // 2] if strikes else None
    def atm_iv(chain):
        chain = [r for r in chain if _f(r.get("strike"))]
        if not chain or spot is None:
            return None
        r = min(chain, key=lambda x: abs(_f(x["strike"]) - spot))
        return _f(r.get("implied_volatility"))
    def d25(chain, want):     # ~25-delta wing IV
        wing = [r for r in chain if _f(r.get("delta")) is not None
                and abs(abs(_f(r["delta"])) - 0.25) < 0.1]
        return (sum(_f(r["implied_volatility"]) for r in wing) / len(wing)) if wing else None
    atm_c, atm_p = atm_iv(c), atm_iv(p)
    put_wing, call_wing = d25(p, "put"), d25(c, "call")
    return {"expiration": exp, "atm_iv_call": atm_c, "atm_iv_put": atm_p,
            "skew_25d": (put_wing - call_wing) if (put_wing and call_wing) else None,
            "n_call": len(c), "n_put": len(p)}


def capture_once(symbols=None):
    symbols = symbols or SYMS
    now = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    got = {}
    errs = []
    for sym in symbols:
        params = {"function": "REALTIME_OPTIONS", "symbol": sym, "apikey": _key()}
        url = BASE + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                doc = json.loads(r.read().decode())
            rows = doc.get("data") or []
            if not rows:
                errs.append(f"{sym}:{(doc.get('message') or doc.get('Information') or 'empty')[:60]}")
                continue
            summ = _summarize(rows)
            rec = {"schema_version": SCHEMA, "symbol": sym, "received_ts": now,
                   "available_for_decision_ts": now, **summ}
            with OUT.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            got[sym] = summ
        except Exception as e:
            errs.append(f"{sym}:{str(e)[:50]}")
    state = "CONNECTED" if got else ("DEGRADED" if errs else "UNAVAILABLE")
    HEALTH.write_text(json.dumps({"source": "av-options", "state": state,
        "last_ts": now, "symbols": list(got), "errors": errs,
        "schema_version": SCHEMA}, indent=1))
    return got, state


if __name__ == "__main__":
    got, state = capture_once()
    print("av-options:", state, "symbols:", list(got))
    for s, v in got.items():
        print(" ", s, v)
