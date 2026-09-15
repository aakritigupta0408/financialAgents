"""AV cross-asset + macro features at open+6min for the BTC 15-min
close-direction problem, in TWO latency variants.

For each settled window in results/open6_dataset.jsonl we take the
decision epoch t_decide = window ``ts`` (the recorded open+6min poll
time; always >= contract open_time + 360s) and, using the av_features
machinery, compute cross-asset equity features (recent returns /
momentum / realized vol for COIN, IBIT, MSTR, QQQ, SPY) plus macro
(10y yield level & 1d change, fed funds level).

Two variants, differing ONLY in the receipt-latency assumption -- the
single knob av_features.receipt_available_ts:

  av_latency   (latency_correct=True)  PIT / honest. Equity bars usable
                only when receipt_available_ts(close, sym) <= t_decide
                (equity: close+180s). Macro uses macro_features()'s
                built-in >=1-day PIT lag (value dated D usable a day
                later). Never touches data after t_decide.

  av_nolatency (latency_correct=False) leak-control / look-ahead.
                Receipt latency is set to zero: an equity bar is usable
                the instant it closes (close <= t_decide) and a macro
                daily value dated D is usable from D's epoch. Still
                never reads a bar that closes AFTER t_decide -- it only
                removes the delivery lag, which is what the leak-control
                is meant to measure.

Mechanism:
  * Equity  -- reuse equity_features() unchanged. For the nolatency
    pass we monkeypatch av.receipt_available_ts to the identity on the
    close epoch, so avail() keeps every bar with close <= t_decide while
    staleness is still measured to the true t_decide.
  * Macro   -- reuse macro_features() unchanged. Its availability filter
    is ``e <= t - 86400``; passing t_decide gives the honest 1-day PIT
    (latency), and passing t_decide + 86400 neutralizes that lag so the
    value dated the window's own day becomes usable (nolatency).

Output: results/av_open6_features.jsonl, one row per window:
  {"ticker", "y", "av_latency": {...}, "av_nolatency": {...}}
"""
import json
from pathlib import Path

import av_features as av

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Cross-asset equity proxies (the task's load_intraday list).
SYMS = ["COIN", "IBIT", "MSTR", "QQQ", "SPY"]

DAY = 86400


def equity_block(series, t_decide):
    """All five symbols' equity_features merged into one flat dict.
    Uses whatever av.receipt_available_ts currently is (the caller flips
    it for the nolatency pass)."""
    feats = {}
    for sym in SYMS:
        feats.update(av.equity_features(series[sym], sym.lower(), t_decide))
    return feats


def build_row(r, series, yld, ff):
    t = r["ts"]                      # decision epoch = open+6min (recorded)

    # ---- variant A: latency-correct (honest PIT) ----------------------
    lat = equity_block(series, t)
    lat.update(av.macro_features(yld, ff, t))          # built-in 1d PIT lag

    # ---- variant B: no receipt latency (leak-control / look-ahead) ----
    real = av.receipt_available_ts
    av.receipt_available_ts = lambda close, source: close   # zero latency
    try:
        nol = equity_block(series, t)
    finally:
        av.receipt_available_ts = real
    nol.update(av.macro_features(yld, ff, t + DAY))    # neutralize 1d lag

    return {"ticker": r["ticker"], "y": r["y"],
            "av_latency": lat, "av_nolatency": nol}


def main():
    series = {sym: av.load_intraday(sym) for sym in SYMS}
    yld = av.load_econ("TREASURY_YIELD")
    ff = av.load_econ("FEDERAL_FUNDS_RATE")

    rows = [json.loads(l) for l in (RES / "open6_dataset.jsonl").open()
            if l.strip()]
    out = [build_row(r, series, yld, ff) for r in rows]

    (RES / "av_open6_features.jsonl").write_text(
        "\n".join(json.dumps(x) for x in out) + "\n")

    # ---- console report (recipe + coverage) ---------------------------
    def present_count(key):
        return sum(1 for x in out
                   if x[key].get("spy_present") or x[key].get("coin_present")
                   or x[key].get("qqq_present") or x[key].get("ibit_present")
                   or x[key].get("mstr_present"))

    report = {
        "n_windows": len(out),
        "t_decide": "window ts (recorded open+6min; >= open_time+360s)",
        "equity_syms": SYMS,
        "equity_features": ["<sym>_ret_15m", "<sym>_ret_60m",
                            "<sym>_rvol_30m", "<sym>_staleness_min",
                            "<sym>_present", "<sym>_session_open"],
        "macro_features": ["yield_10y_level", "yield_10y_chg_1d",
                           "fed_funds_level"],
        "latency_model": {
            "av_latency": "receipt_available_ts as shipped "
                          "(equity close+180s, macro built-in >=1d)",
            "av_nolatency": "receipt latency = 0 (equity usable at close; "
                            "macro value dated D usable from D)"},
        "sources": {sym: len(series[sym]) for sym in SYMS} |
                   {"yield_pts": len(yld), "fed_funds_pts": len(ff)},
        "n_windows_with_any_fresh_equity_latency": present_count("av_latency"),
        "n_windows_with_any_fresh_equity_nolatency":
            present_count("av_nolatency"),
        "output": str((RES / "av_open6_features.jsonl").relative_to(ROOT)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
