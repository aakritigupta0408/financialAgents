"""P0 — retrieve the EXACT KXBTC15M contract outcomes from Kalshi's own API
(authoritative). For every settled market: floor_strike (= 60s-BRTI average
before window open == the target), expiration_value (= 60s-BRTI average before
expiry == the settlement statistic), and result (official Yes/No).

YES iff expiration_value >= floor_strike, i.e. D = close_avg - open_avg >= 0.
Writes results/contract_outcomes.jsonl (append-only canonical) and freezes the
versioned rule at research/contract_specs/KXBTC15M_2026-09.json.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "contract_outcomes.jsonl"
SPEC = ROOT / "research" / "contract_specs" / "KXBTC15M_2026-09.json"
BASE = "https://api.elections.kalshi.com/trade-api/v2/markets"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "Chrome/124.0 Safari/537.36")


def _get(params):
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    rows = {}
    cursor = None
    pages = 0
    while pages < 60:
        p = {"series_ticker": "KXBTC15M", "status": "settled", "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = _get(p)
        ms = d.get("markets", [])
        for m in ms:
            fs, ev = m.get("floor_strike"), m.get("expiration_value")
            if fs is None or ev is None:
                continue
            try:
                fs = float(fs); ev = float(ev)
            except (TypeError, ValueError):
                continue
            rows[m["ticker"]] = {
                "ticker": m["ticker"], "open_time": m.get("open_time"),
                "close_time": m.get("close_time"),
                "floor_strike": fs, "expiration_value": ev,
                "D": round(ev - fs, 4), "result": m.get("result"),
                "exact_yes": int(ev >= fs)}
        pages += 1
        cursor = d.get("cursor")
        if not cursor or not ms:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in sorted(rows.values(), key=lambda x: x["close_time"] or ""):
            fh.write(json.dumps(r) + "\n")

    # consistency: does official result match expiration_value>=floor_strike?
    mism = sum(1 for r in rows.values()
               if r["result"] in ("yes", "no")
               and r["exact_yes"] != (1 if r["result"] == "yes" else 0))
    yes_rate = (sum(r["exact_yes"] for r in rows.values()) / len(rows)) if rows else 0

    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(json.dumps({
        "series": "KXBTC15M", "rule_source": "Kalshi trade-api v2 (authoritative)",
        "retrieved_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "benchmark": "CF Benchmarks BRTI (BTC Real-Time Index)",
        "benchmark_provider": "CF Benchmarks",
        "target_construction": "simple average of the 60 one-second BRTI values "
                               "in the minute before window OPEN (== floor_strike)",
        "settlement_construction": "simple average of the 60 one-second BRTI "
                                   "values in the minute before EXPIRY (== expiration_value)",
        "yes_condition": "expiration_value >= floor_strike  (D = close_avg - open_avg >= 0)",
        "comparison_operator": ">=",
        "frequency": "fifteen_min", "timezone": "EDT (America/New_York)",
        "window_open": "expiry - 15min", "expiration": "top of the 15-min block",
        "rounding": "custom_strike round_digits=2 (cents)",
        "fee_type": "quadratic", "contiguous_windows": True,
        "note": "each window's floor_strike == the prior window's expiration_value "
                "(open avg = prior close avg) -> pure up/down over the block",
        "outcome_source": "Kalshi official result (BRTI-based); our labels use this",
        "n_markets": len(rows), "official_yes_rate": round(yes_rate, 4),
        "result_vs_formula_mismatches": mism}, indent=1))
    print(f"contract_outcomes.jsonl: {len(rows)} settled markets")
    print(f"YES rate {yes_rate:.3f} · result==（ev>=fs) mismatches: {mism}")
    ex = list(rows.values())[:2]
    for r in ex:
        print(" ", r["ticker"], "D", r["D"], "->", r["result"])


if __name__ == "__main__":
    main()
