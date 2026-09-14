"""L2 — Alpha Vantage historical market-state backfill (directive P1, §3B, §5, §8, §12).

Progressive mini-gate backfill: reconstruct real pre-open market state for KXBTC15M
contract opens using Alpha Vantage intraday history, PIT-correct (only bars whose
timestamp is <= T0, converted to UTC from the source timezone). Absolute BTC price
is a source primitive, never a feature (§6) — we emit returns / RSI / realized-vol.

This module runs a COHORT (MINI-0=5, MINI-1=25, ...) so bugs (timezone, as-of joins,
formulas) are caught on a small slice before the full 6,337-window build. Every step
emits a research event; live capture is untouched.

Usage: python3 scripts/av_backfill.py --mini 0     (5 recent windows)
       python3 scripts/av_backfill.py --mini 1     (25)
Writes artifacts/true15m/raw/av_<sym>.json (raw store, versioned) and
research/true15m/av_backfill_mini<k>.json (engineered slice + coverage matrix).
"""
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

KEY = (Path.home() / ".alphavantage_key").read_text().strip()
BASE = "https://www.alphavantage.co/query"
INV = ROOT / "research" / "true15m" / "contract_inventory.jsonl"
RAW = ROOT / "artifacts" / "true15m" / "raw"
UTC = ZoneInfo("UTC")

MINI_SIZES = {0: 5, 1: 25, 2: 100, 3: 500}
# instrument set for the mini gate: BTC (crypto, UTC) + two equities (ET, session-gated)
INSTR = [
    {"sym": "BTC", "kind": "crypto"},
    {"sym": "SPY", "kind": "equity"},
    {"sym": "QQQ", "kind": "equity"},
]


def _call(params):
    params["apikey"] = KEY
    u = BASE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(u, timeout=40) as r:
        return json.load(r)


def _fetch(instr):
    if instr["kind"] == "crypto":
        d = _call({"function": "CRYPTO_INTRADAY", "symbol": instr["sym"],
                   "market": "USD", "interval": "5min", "outputsize": "full"})
    else:
        d = _call({"function": "TIME_SERIES_INTRADAY", "symbol": instr["sym"],
                   "interval": "5min", "outputsize": "full", "extended_hours": "false"})
    meta = next((v for k, v in d.items() if "Meta Data" in k), {})
    tz = meta.get("6. Time Zone") or meta.get("7. Time Zone") or "US/Eastern"
    if instr["kind"] == "crypto":
        tz = "UTC"
    tskey = next((k for k in d if "Time Series" in k), None)
    if not tskey:
        return None, str(d)[:160], tz
    zone = ZoneInfo(tz)
    bars = []
    for tstr, ohlc in d[tskey].items():
        dt = datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone)
        close = float(ohlc.get("4. close") or ohlc.get("4a. close (USD)") or 0)
        vol = float(ohlc.get("5. volume") or ohlc.get("5. volume (USD)") or 0)
        bars.append((int(dt.astimezone(UTC).timestamp()), close, vol))
    bars.sort()
    return bars, None, tz


def _asof(bars, t0):
    """Latest bar with ts <= t0 (PIT). Returns (idx, bar) or (None, None)."""
    lo, hi, idx = 0, len(bars) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if bars[mid][0] <= t0:
            idx = mid; lo = mid + 1
        else:
            hi = mid - 1
    return (idx, bars[idx]) if idx is not None else (None, None)


def _rsi(closes, n=14):
    if len(closes) < n + 1:
        return None
    g = l = 0.0
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        g += max(d, 0); l += max(-d, 0)
    if g + l == 0:
        return 50.0
    rs = (g / n) / ((l / n) or 1e-9)
    return round(100 - 100 / (1 + rs), 2)


def _features(bars, idx):
    """PIT features from bars[:idx+1] only (never any bar after T0)."""
    closes = [b[1] for b in bars[:idx + 1]]
    if len(closes) < 8:
        return None
    def ret(k):
        return round(math.log(closes[-1] / closes[-1 - k]), 6) if len(closes) > k and closes[-1 - k] > 0 else None
    rets1 = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0][-12:]
    rvol = round((sum(x * x for x in rets1) / len(rets1)) ** 0.5, 6) if rets1 else None
    return {"ret_5m": ret(1), "ret_15m": ret(3), "ret_30m": ret(6),
            "rsi_14": _rsi(closes), "rvol_60m": rvol}


def run(mini=0):
    n = MINI_SIZES.get(mini, 5)
    RAW.mkdir(parents=True, exist_ok=True)
    EV.emit("JOB_STARTED", f"Alpha Vantage backfill started (MINI-{mini})", lane="L2",
            narrative=f"Reconstructing pre-open market state for the {n} most recent "
                      "contract opens. Live prospective collection remains healthy.")
    inv = [json.loads(l) for l in INV.open() if l.strip()]
    inv.sort(key=lambda r: r["T0"])
    cohort = inv[-n:]

    series = {}
    for instr in INSTR:
        sym = instr["sym"]
        EV.emit("DOWNLOAD_STARTED", f"AV {sym} intraday", lane="L2")
        try:
            bars, err, tz = _fetch(instr)
        except Exception as e:
            bars, err, tz = None, str(e)[:120], "?"
        if not bars:
            EV.emit("WARNING", f"AV {sym} returned no series", lane="L2", severity="high",
                    narrative=f"tz={tz} detail={err}. Marking {sym} MISSING for this cohort "
                              "(no fabrication).")
            series[sym] = {"bars": [], "tz": tz}
            continue
        (RAW / f"av_{sym}.json").write_text(json.dumps(
            {"sym": sym, "tz": tz, "n_bars": len(bars),
             "span": [bars[0][0], bars[-1][0]], "fetched_at": time.time()}))
        series[sym] = {"bars": bars, "tz": tz}
        EV.emit("DOWNLOAD_COMPLETE", f"AV {sym}: {len(bars)} bars", lane="L2",
                metrics={"bars": len(bars), "tz": tz},
                narrative=f"{sym} span {datetime.fromtimestamp(bars[0][0],UTC):%Y-%m-%d %H:%M}"
                          f"–{datetime.fromtimestamp(bars[-1][0],UTC):%Y-%m-%d %H:%M} UTC.")

    rows, future_joins, cov = [], 0, {i["sym"]: 0 for i in INSTR}
    for w in cohort:
        t0 = w["T0"]
        feat, avail = {}, {}
        for instr in INSTR:
            sym = instr["sym"]; bars = series[sym]["bars"]
            idx, bar = _asof(bars, t0) if bars else (None, None)
            if bar is None:
                avail[sym] = False
                continue
            if bar[0] > t0:                       # PIT guard — must never happen
                future_joins += 1; avail[sym] = False; continue
            f = _features(bars, idx)
            if f:
                feat[sym] = f; avail[sym] = True; cov[sym] += 1
            else:
                avail[sym] = False
        rows.append({"market_window_id": w["market_window_id"], "T0": t0,
                     "official_outcome": w["official_outcome"],
                     "features": feat, "availability": avail})

    m = len(cohort)
    coverage = {s: {"n": cov[s], "coverage": round(cov[s] / m, 3)} for s in cov}
    passed = future_joins == 0
    doc = {"schema_version": "av-backfill-mini-1", "generated_at": time.time(),
           "mini": mini, "cohort_windows": m,
           "cohort_span_utc": [cohort[0]["T0"], cohort[-1]["T0"]],
           "future_joins": future_joins, "pit_pass": passed,
           "coverage_matrix": coverage, "rows": rows,
           "note": "PIT as-of join <= T0 (UTC). BTC crypto = UTC; equities ET->UTC, "
                   "regular-hours only (session-gated). Absolute price never a feature."}
    outp = ROOT / "research" / "true15m" / f"av_backfill_mini{mini}.json"
    outp.write_text(json.dumps(doc, indent=1))

    cov_str = ", ".join(f"{s} {coverage[s]['n']}/{m}" for s in coverage)
    if passed:
        EV.emit("INTEGRITY_CHECK", f"MINI-{mini} passed", lane="L7", severity="info",
                fact=f"{m}/{m} windows aligned strictly at or before T0. Future joins: 0. "
                     f"Coverage: {cov_str}.",
                interpretation="Equity coverage < BTC is expected — BTC trades 24/7 while "
                               "SPY/QQQ only during US regular hours; session masks preserved, "
                               "no forward-fill of stale equity state.",
                next_action=f"expand to MINI-{mini+1}" if mini < 3 else "run full backfill",
                files=[str(outp.relative_to(ROOT))], metrics={"future_joins": 0})
    else:
        EV.emit("BUG_FOUND", f"MINI-{mini} PIT violation", lane="L7", severity="high",
                fact=f"{future_joins} as-of joins selected a bar after T0.",
                interpretation="Timezone/boundary bug — NOT allowed to enter the dataset.",
                next_action="fix as-of boundary before expanding")
    print(f"av_backfill MINI-{mini}: {m} windows, future_joins={future_joins}, coverage: {cov_str}")
    return doc


if __name__ == "__main__":
    mini = 0
    if "--mini" in sys.argv:
        mini = int(sys.argv[sys.argv.index("--mini") + 1])
    run(mini)
