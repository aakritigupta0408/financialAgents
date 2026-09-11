"""F-AV1 PIT feature builder — align cached Alpha Vantage exogenous
series to the 192 settled BTC windows under a HANDICAPPED receipt-time
model, and emit the residual-target feature matrix for the screen.

Screen-only / Research plane. The one judgment this harness does NOT
make for you is receipt_available_ts() — how late a live poll would have
delivered each AV bar. That assumption is owned by a human because it
encodes a fact about our own pipeline, not a fitted quantity, and it is
the single knob that decides whether the backtest flatters us.

Timezone note: Alpha Vantage intraday timestamps are US Eastern. Our
window made_ts are UTC epoch. We convert ET->UTC assuming EDT (UTC-4),
valid for the Aug-Sep 2026 capture; the *latency* on top of bar close
is the human-owned part below.
"""
import calendar
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CACHE = ROOT / "av_cache"
EDT_OFFSET = 4 * 3600            # ET->UTC for EDT (Aug-Sep 2026)


# ------------------------------------------------------------------ #
#  TODO(human): receipt-time latency model — the crux assumption.
# ------------------------------------------------------------------ #
def receipt_available_ts(bar_close_epoch, source):
    """Return the UTC epoch at which we could FIRST have acted on an AV
    bar that closed at `bar_close_epoch`. `source` is one of
    "equity" (SPY/UUP 5-min) or "macro" (daily econ).

    This models our live disadvantage: a REST poll delivers a bar only
    after it closes AND after our poll cadence + network + parse. Set it
    too tight and the backtest flatters us (look-ahead); too loose and
    we discard real signal. It must reflect OUR pipeline honestly.

    Return an epoch (seconds). A bar is usable for a window only if this
    value <= the window's decision time.
    """
    raise NotImplementedError(
        "receipt_available_ts is unset — see the Learn-by-Doing handoff. "
        "Implement the receipt-time latency model here.")


# ------------------------------------------------------------------ #
#  Everything below is built; it depends only on the function above.
# ------------------------------------------------------------------ #
def et_to_epoch(s):
    """'YYYY-MM-DD HH:MM:SS' in US Eastern -> UTC epoch (EDT)."""
    st = time.strptime(s, "%Y-%m-%d %H:%M:%S")
    return calendar.timegm(st) + EDT_OFFSET


def date_to_epoch(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%d"))


def load_intraday(symbol):
    bars = []
    for m in ("2026-08", "2026-09"):
        fp = CACHE / f"intraday_{symbol}_{m}.json"
        if not fp.exists():
            continue
        doc = json.loads(fp.read_text())
        key = next((k for k in doc if "Time Series" in k), None)
        if not key:
            continue
        for ts, row in doc[key].items():
            bars.append((et_to_epoch(ts), float(row["4. close"])))
    bars.sort()
    return bars


def load_econ(name):
    fp = CACHE / f"econ_{name}.json"
    if not fp.exists():
        return []
    doc = json.loads(fp.read_text())
    out = []
    for d in doc.get("data", []):
        try:
            out.append((date_to_epoch(d["date"]), float(d["value"])))
        except (ValueError, KeyError):
            continue
    out.sort()
    return out


def settled_windows():
    seen = {}
    for l in (RES / "kalshi_binary_log.jsonl").open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("variant") != "kb2" or r.get("actual") is None:
            continue
        if r.get("made_ts") is None or r.get("mkt_p_up") is None:
            continue
        seen[r["ticker"]] = r
    return sorted(seen.values(), key=lambda r: r["made_ts"])


def avail(bars, source, t_decide):
    """Bars deliverable by t_decide under the receipt model, in order."""
    return [(e, v) for (e, v) in bars
            if receipt_available_ts(e, source) <= t_decide]


def bar_at_or_before(bars, target_epoch):
    prev = None
    for (e, v) in bars:
        if e <= target_epoch:
            prev = v
        else:
            break
    return prev


def pct(a, b):
    return round(100.0 * (a - b) / b, 4) if b else None


def std(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return round((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5, 5)


def equity_features(bars, source, t_decide):
    a = avail(bars, source, t_decide)
    if len(a) < 2:
        return {f"{source}_staleness_min": None, f"{source}_present": 0}
    last_e, last_v = a[-1]
    stale = round((t_decide - last_e) / 60.0, 1)
    r15 = pct(last_v, bar_at_or_before(a, last_e - 900) or last_v)
    r60 = pct(last_v, bar_at_or_before(a, last_e - 3600) or last_v)
    rets = []
    for i in range(max(1, len(a) - 6), len(a)):
        if a[i - 1][1]:
            rets.append((a[i][1] - a[i - 1][1]) / a[i - 1][1])
    return {f"{source}_ret_15m": r15, f"{source}_ret_60m": r60,
            f"{source}_rvol_30m": std([r * 100 for r in rets]),
            f"{source}_staleness_min": stale,
            f"{source}_present": 1,
            f"{source}_session_open": int(stale < 10)}


def macro_features(yld, ff, t_decide):
    # daily series: value dated D is knowable only after D (>=1d lag)
    lag = t_decide - 86400
    y_now = None
    y_prev = None
    for (e, v) in yld:
        if e <= lag:
            y_prev, y_now = y_now, v
        else:
            break
    f_now = bar_at_or_before(ff, lag)
    return {"yield_10y_level": y_now,
            "yield_10y_chg_1d": (round(y_now - y_prev, 3)
                                 if y_now is not None
                                 and y_prev is not None else None),
            "fed_funds_level": f_now}


def main():
    spy = load_intraday("SPY")
    uup = load_intraday("UUP")
    yld = load_econ("TREASURY_YIELD")
    ff = load_econ("FEDERAL_FUNDS_RATE")
    wins = settled_windows()
    rows, n_equity = [], 0
    for r in wins:
        t = r["made_ts"]
        feats = {}
        ef = equity_features(spy, "spy", t)
        feats.update(ef)
        feats.update(equity_features(uup, "uup", t))
        feats.update(macro_features(yld, ff, t))
        if ef.get("spy_present"):
            n_equity += 1
        rows.append({
            "ticker": r["ticker"], "t_decide": t,
            "outcome": int(r["actual"]), "mkt_p_up": r["mkt_p_up"],
            "residual": round(int(r["actual"]) - r["mkt_p_up"], 4),
            "features": feats})
    manifest = {
        "generated_ts": int(time.time()),
        "spec": "F_AV1_SPEC.yaml", "status": "SCREEN_ONLY",
        "receipt_model": "human-owned (av_features.receipt_available_ts)",
        "n_windows": len(rows),
        "n_with_fresh_equity": n_equity,
        "sources": {"spy_bars": len(spy), "uup_bars": len(uup),
                    "yield_pts": len(yld), "fed_funds_pts": len(ff)},
        "note": "residual = outcome - mkt_p_up; screen target"}
    (RES / "av_features.jsonl").write_text(
        "\n".join(json.dumps(x) for x in rows) + "\n")
    (RES / "av_features_manifest.json").write_text(
        json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
