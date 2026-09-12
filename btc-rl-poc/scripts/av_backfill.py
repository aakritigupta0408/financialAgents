"""AV BACKFILL (F-AV1 screen) — cached, rate-limit-aware pull of
orthogonal exogenous features over the BTC window history.

SCREEN-ONLY / Research plane (D). Raw vendor responses are cached under
av_cache/ (gitignored, never published, never a decision input). This
script only *fetches and caches*; PIT alignment + the receipt-time
latency handicap live in av_features.py so that peeking is impossible
here (we cannot look at outcomes from a downloader).

Free-tier safe:
  - >=13s between live calls (<5 req/min)
  - single-key {Note|Information|Error Message} payloads = limit/error:
    NOT cached, loop stops gracefully, rerun resumes from cache
  - a slice already on disk is never re-pulled

Our history spans exactly 2026-08 and 2026-09 (T1 dataset Aug30-Sep05;
live windows Sep09-11), so intraday = 2 month-slices per symbol.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "av_cache"
KEYFILE = Path.home() / ".alphavantage_key"
BASE = "https://www.alphavantage.co/query"
SLEEP_S = 13
MONTHS = ["2026-08", "2026-09"]

# Tier A — orthogonal-to-BTC-price regime/vol/macro (pull first).
TIER_A_INTRADAY = ["SPY", "UUP"]          # equity risk regime + US dollar
TIER_A_ECON = [                            # dated macro series (full hist)
    ("CPI", {"function": "CPI", "interval": "monthly"}),
    ("TREASURY_YIELD", {"function": "TREASURY_YIELD",
                        "interval": "daily", "maturity": "10year"}),
    ("FEDERAL_FUNDS_RATE", {"function": "FEDERAL_FUNDS_RATE",
                            "interval": "daily"}),
]
# Tier B — crypto-beta equity proxies (pull only if Tier A leaves budget).
TIER_B_INTRADAY = ["COIN", "MSTR", "IBIT", "QQQ"]

LIMIT_KEYS = {"Note", "Information", "Error Message"}


def key():
    return KEYFILE.read_text().strip()


def is_limit(doc):
    """AV limit/error payloads are single-key {Note|Information|Error}."""
    return bool(doc) and set(doc).issubset(LIMIT_KEYS)


def fetch(params, cache_name, calls):
    """Return (doc, status). status in cache|fetched|LIMIT. Live calls
    are counted+rate-limited; cached slices cost nothing."""
    fp = CACHE / cache_name
    if fp.exists():
        return json.loads(fp.read_text()), "cache"
    if calls[0] > 0:
        time.sleep(SLEEP_S)
    calls[0] += 1
    url = BASE + "?" + urllib.parse.urlencode({**params, "apikey": key()})
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            doc = json.loads(r.read().decode())
    except Exception as e:                       # network/parse failure
        return {"Error Message": str(e)}, "LIMIT"
    if is_limit(doc):
        return doc, "LIMIT"
    fp.write_text(json.dumps(doc))
    return doc, "fetched"


def intraday_params(sym, month):
    return {"function": "TIME_SERIES_INTRADAY", "symbol": sym,
            "interval": "5min", "month": month, "outputsize": "full"}


def rows_of(doc):
    for k in doc:
        if "Time Series" in k or "data" in k:
            v = doc[k]
            return len(v) if isinstance(v, (list, dict)) else 0
    return 0


def main():
    tier_b = "--tier-b" in sys.argv
    CACHE.mkdir(exist_ok=True)
    calls = [0]
    plan = []
    for sym in TIER_A_INTRADAY + (TIER_B_INTRADAY if tier_b else []):
        for m in MONTHS:
            plan.append((f"intraday_{sym}_{m}.json", intraday_params(sym, m)))
    for name, p in TIER_A_ECON:
        plan.append((f"econ_{name}.json", p))

    stopped = False
    for cache_name, params in plan:
        if stopped:
            print(f"  SKIP (budget stopped): {cache_name}")
            continue
        doc, status = fetch(params, cache_name, calls)
        if status == "LIMIT":
            note = (doc.get("Information") or doc.get("Note")
                    or doc.get("Error Message") or "")[:90]
            print(f"  LIMIT on {cache_name}: {note}")
            print(f"  -> stopping; rerun tomorrow to resume from cache "
                  f"({calls[0]} live calls made this run)")
            stopped = True
            continue
        print(f"  {status:8s} {cache_name}  rows={rows_of(doc)}")

    cached = sorted(p.name for p in CACHE.glob("*.json"))
    print(f"\nlive_calls_this_run={calls[0]}  cached_slices={len(cached)}")
    print("cache:", ", ".join(cached) if cached else "(empty)")


if __name__ == "__main__":
    main()
