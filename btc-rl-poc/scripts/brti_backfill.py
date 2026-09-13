"""§3-4 EXACT-BRTI BACKFILL for the research universe.

For every KXBTC15M window in the Oracle dataset, fetch the exact BRTI trajectory
(hour-cached from Kalshi's CF Benchmarks history endpoint) and emit a
decision-ready dataset that makes BRTI the CONTRACT-STATE source of truth:

  official_target            = 60s BRTI mean over [open-60, open)   (== floor_strike)
  official_expiration_value  = 60s BRTI mean over [close-60, close) (== settlement)
  current_brti (at decision) = nearest BRTI sample <= decision ts
  brti_distance_to_target    = current_brti - official_target
  time_remaining_s           = close_ts - decision_ts
  settlement seen/remaining  = BRTI samples already inside [close-60, close)
  required_remaining_average = target level the unknown remainder must average to

Coinbase is retained ONLY as an independent-predictor column (coinbase_spot), never
as contract state. Output: results/brti_decision_dataset.jsonl (append-only, schema
versioned). Boundary reconstruction parity is re-asserted per window (quality flag).

Run:  python3 scripts/brti_backfill.py
"""
import calendar
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data.adapters.brti as brti  # noqa: E402

DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
CO = ROOT / "results" / "contract_outcomes.jsonl"
OUT = ROOT / "results" / "brti_decision_dataset.jsonl"
META = ROOT / "results" / "brti_decision_dataset.meta.json"
SCHEMA = "brti-decision-v1"
SETTLE_S = 60  # settlement averaging interval


def epoch(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


def brti_at(ts):
    """Nearest BRTI sample at or before ts (within its calendar hour)."""
    samples = brti.hour_samples(ts)
    best = None
    for t, v in samples:
        if t <= ts and (best is None or t > best[0]):
            best = (t, v)
    return best  # (sample_ts, value) or None


def main():
    outmap = {json.loads(l)["ticker"]: json.loads(l) for l in CO.open() if l.strip()}
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    windows = {}
    for r in rows:
        windows.setdefault(r["ticker"], []).append(r)

    emitted, skipped, parity_bad, boundary_err = 0, 0, 0, []
    fetch_ts = time.time()
    with OUT.open("w") as fh:
        for tk, rws in sorted(windows.items()):
            o = outmap.get(tk)
            if not o:
                skipped += len(rws); continue
            To, Tc = epoch(o["open_time"]), epoch(o["close_time"])
            target, n_o = brti.avg_60s_ending(To)
            exp_val, n_c = brti.avg_60s_ending(Tc)
            if target is None or exp_val is None or n_o < 30 or n_c < 30:
                skipped += len(rws); continue
            # parity: does reconstruction match the official strikes?
            e_t = abs(target - o["floor_strike"])
            e_e = abs(exp_val - o["expiration_value"])
            boundary_err += [e_t, e_e]
            qual = "OK" if max(e_t, e_e) < 5.0 else "PARITY_DRIFT"
            if qual != "OK":
                parity_bad += 1
            settle_samples = [(t, v) for t, v in
                              (brti.hour_samples(Tc - 60) + brti.hour_samples(Tc))
                              if Tc - 60 <= t < Tc]
            settle_samples = sorted(set(settle_samples))
            for r in rws:
                ts = r["ts"]
                b = brti_at(ts)
                if b is None:
                    skipped += 1; continue
                cur = b[1]
                trem = max(0.0, Tc - ts)
                # settlement partial state (only meaningful in final 60s)
                seen = [v for (t, v) in settle_samples if t <= ts]
                remaining = max(0, n_c - len(seen))
                req_rem = None
                if remaining > 0 and ts >= Tc - SETTLE_S:
                    # avg of unknown remainder needed for exp_val >= target
                    req_rem = round((o["floor_strike"] * n_c - sum(seen)) / remaining, 2)
                rec = {
                    "schema": SCHEMA,
                    "market_window_id": tk,
                    "decision_time": ts,
                    "official_target": round(target, 4),
                    "official_target_kalshi": o["floor_strike"],
                    "official_expiration_value": round(exp_val, 4),
                    "official_expiration_kalshi": o["expiration_value"],
                    "exact_yes": o["exact_yes"],
                    "current_brti": round(cur, 4),
                    "brti_sample_ts": round(b[0], 3),
                    "brti_distance_to_target": round(cur - target, 4),
                    "coinbase_spot": r.get("cb_price"),
                    "time_remaining_s": round(trem, 1),
                    "settle_seen": len(seen),
                    "settle_remaining": remaining,
                    "required_remaining_average": req_rem,
                    "cb_rvol_30s": r.get("cb_rvol_30s"),
                    "k_prob": r.get("k_prob"),
                    "source": "kalshi-cfbenchmarks-BRTI",
                    "cadence_hz": round(n_c / 60.0, 1),
                    "fetch_ts": fetch_ts,
                    "quality": qual,
                }
                fh.write(json.dumps(rec) + "\n")
                emitted += 1

    be = sorted(boundary_err)
    meta = {
        "schema": SCHEMA,
        "windows": len(windows),
        "windows_emitted": len(windows) - parity_bad,
        "rows_emitted": emitted,
        "rows_skipped": skipped,
        "parity_drift_windows": parity_bad,
        "boundary_recon_abs_err_usd": {
            "mean": round(sum(be) / len(be), 3) if be else None,
            "p95": round(be[int(.95 * len(be))], 3) if be else None,
            "max": round(max(be), 3) if be else None},
        "span": "2026-09-08..2026-09-12 (~4.25d)",
        "note": "BRTI = contract-state truth; coinbase_spot retained as independent "
                "predictor only. Window is the statistical unit (do not count samples).",
        "generated_ts": fetch_ts,
    }
    META.write_text(json.dumps(meta, indent=1))
    print(f"backfill: {emitted} rows over {len(windows)} windows "
          f"(skipped {skipped}); parity_drift {parity_bad}")
    print(f"boundary recon err $: mean {meta['boundary_recon_abs_err_usd']['mean']}"
          f" max {meta['boundary_recon_abs_err_usd']['max']}")


if __name__ == "__main__":
    main()
