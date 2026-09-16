"""EXOGENOUS (off-BRTI-path) feature extraction from the event tapes.

For each labeled 15-min window, using ONLY events in the PIT flow phase [open, entry),
entry = close - ENTRY_TR (default 11 min left), builds microstructure signals the price-path
foundation models cannot see:
  cb_ofi         Coinbase trade order-flow imbalance (buy-sell)/(buy+sell), by size
  cb_ofi_notional  same, $-notional weighted
  cb_trades      trade count in the flow phase (intensity)
  book_imb       mean Coinbase L1 top-of-book imbalance (bid_sz-ask_sz)/(bid_sz+ask_sz)
  spread_bps     mean Coinbase L1 spread in bps
  bn_ofi         Binance (cross-venue) trade OFI
  bn_trades      Binance trade count
  basis_bps      (last Binance px - last Coinbase px)/cb * 1e4  (lead-lag / basis)
  cb_mid_entry, cb_mid_open, floor_strike  -> for an on-path barrier baseline in the eval

Streams results/events/*.jsonl (Coinbase trade+l1) ONCE and pulls the cross-venue
Binance trade flow through xvenue_sync (the sole sanctioned shard reader/aligner),
writing results/exo_features.jsonl (ticker -> features + label).
PAPER / SIMULATION research; read-only w.r.t. the live desk.
"""
import glob, json, os, sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xvenue_sync  # the ONE sanctioned reader/aligner of the raw
# cross-venue shards (no-private-time-alignment law); we route the
# Binance flow read through it rather than globbing the shards here.

ROOT_RES = "results"
OUT = "results/exo_features.jsonl"
ENTRY_TR = 660.0     # entry at 11 min left (mid-window regime)


def _ep(t):
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def windows():
    """ticker -> dict(open, entry, close, floor, label) for labeled windows."""
    W = {}
    for l in open(f"{ROOT_RES}/contract_outcomes.jsonl"):
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") not in (0, 1):
            continue
        ot, ct = _ep(d.get("open_time")), _ep(d.get("close_time"))
        if not ot or not ct:
            continue
        W[d["ticker"]] = {"open": ot, "entry": ct - ENTRY_TR, "close": ct,
                          "floor": d.get("floor_strike"), "label": int(d["exact_yes"])}
    return W


def _acc0():
    return {"buy": 0.0, "sell": 0.0, "buy_n": 0.0, "sell_n": 0.0, "n": 0,
            "bimb": 0.0, "bimb_n": 0, "spread": 0.0,
            "bn_buy": 0.0, "bn_sell": 0.0, "bn_n": 0,
            "cb_first": None, "cb_last": None, "bn_last": None}


def run():
    W = windows()
    # index windows by the UTC hour(s) their flow phase [open, entry) touches
    by_hour = defaultdict(list)
    for tk, w in W.items():
        h0 = int(w["open"] // 3600); h1 = int((w["entry"]) // 3600)
        for h in range(h0, h1 + 1):
            by_hour[h].append(tk)
    acc = {tk: _acc0() for tk in W}

    def active(tk, ts):
        w = W[tk]
        return w["open"] <= ts < w["entry"]

    # ---- Coinbase tape (events/): trade + l1 ----
    for f in sorted(glob.glob(f"{ROOT_RES}/events/*.jsonl")):
        b = os.path.basename(f).replace(".jsonl", "")
        try:
            hbase = int(datetime.strptime(b, "%Y%m%d_%H").replace(tzinfo=timezone.utc).timestamp() // 3600)
        except Exception:
            continue
        cand = set(by_hour.get(hbase, [])) | set(by_hour.get(hbase + 1, []))
        if not cand:
            continue
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            ts = r.get("receive_ts")
            if ts is None:
                continue
            kind = r.get("kind"); src = r.get("src")
            for tk in cand:
                if not active(tk, ts):
                    continue
                a = acc[tk]
                if src == "coinbase" and kind == "trade":
                    try:
                        sz = float(r.get("size") or 0); px = float(r.get("price") or 0)
                    except Exception:
                        continue
                    a["n"] += 1
                    if a["cb_first"] is None:
                        a["cb_first"] = px
                    a["cb_last"] = px
                    if r.get("side") == "buy":
                        a["buy"] += sz; a["buy_n"] += sz * px
                    else:
                        a["sell"] += sz; a["sell_n"] += sz * px
                elif src == "coinbase" and kind == "l1":
                    try:
                        bs = float(r.get("bid_sz") or 0); as_ = float(r.get("ask_sz") or 0)
                        bid = float(r.get("bid") or 0); ask = float(r.get("ask") or 0)
                    except Exception:
                        continue
                    if bs + as_ > 0:
                        a["bimb"] += (bs - as_) / (bs + as_); a["bimb_n"] += 1
                    if bid and ask:
                        a["spread"] += (ask - bid) / ((ask + bid) / 2) * 1e4
                        a["cb_last"] = (ask + bid) / 2
                        if a["cb_first"] is None:
                            a["cb_first"] = a["cb_last"]

    # ---- Binance cross-venue tape: trades, via the sanctioned aligner ----
    # xvenue_sync is the ONE layer allowed to read + time-align the raw
    # cross-venue shards (no-private-time-alignment law). Hand it our PIT
    # flow windows [open, entry) and let it accumulate the trade flow.
    # venue is left unset so EVERY shard row is counted, exactly as this
    # pass did when it globbed the shards directly (no src filter).
    xwins = {tk: (w["open"], w["entry"]) for tk, w in W.items()}
    for tk, fl in xvenue_sync.trade_flow_in_windows(xwins).items():
        a = acc[tk]
        a["bn_n"] += fl["n"]
        a["bn_buy"] += fl["buy"]
        a["bn_sell"] += fl["sell"]
        if fl["last"] is not None:
            a["bn_last"] = fl["last"]

    n_out = 0
    with open(OUT, "w") as out:
        for tk, w in W.items():
            a = acc[tk]
            if a["n"] < 5 and a["bn_n"] < 5:
                continue     # no meaningful flow captured -> skip
            tot = a["buy"] + a["sell"]; totn = a["buy_n"] + a["sell_n"]
            bntot = a["bn_buy"] + a["bn_sell"]
            row = {
                "ticker": tk, "label": w["label"], "entry_ts": w["entry"], "floor": w["floor"],
                "cb_ofi": (a["buy"] - a["sell"]) / tot if tot else 0.0,
                "cb_ofi_notional": (a["buy_n"] - a["sell_n"]) / totn if totn else 0.0,
                "cb_trades": a["n"],
                "book_imb": a["bimb"] / a["bimb_n"] if a["bimb_n"] else 0.0,
                "spread_bps": a["spread"] / a["bimb_n"] if a["bimb_n"] else 0.0,
                "bn_ofi": (a["bn_buy"] - a["bn_sell"]) / bntot if bntot else 0.0,
                "bn_trades": a["bn_n"],
                "cb_mid_open": a["cb_first"], "cb_mid_entry": a["cb_last"],
                "bn_last": a["bn_last"],
                "basis_bps": ((a["bn_last"] - a["cb_last"]) / a["cb_last"] * 1e4)
                             if (a["bn_last"] and a["cb_last"]) else 0.0,
            }
            out.write(json.dumps(row) + "\n"); n_out += 1
    print(f"exo_features: {len(W)} labeled windows in span, {n_out} with flow -> {OUT}")


if __name__ == "__main__":
    run()
