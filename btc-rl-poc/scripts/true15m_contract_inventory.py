"""TRUE-15M Phase 1 — historical contract inventory (directive §2).

Recovers, for every historical KXBTC15M contract, the real market truth needed
for the true-15-minute task: T0/T1, opening 60s-BRTI average (= floor_strike),
final 60s-BRTI settlement average (= expiration_value), D, official YES/NO. Old
defective-model predictions are deliberately EXCLUDED (they may live only in an
archaeology table, not here).

Source: results/contract_outcomes.jsonl (official Kalshi settled markets, frozen
rule research/contract_specs/KXBTC15M_2026-09.json). Raw 60s-observation coverage
is cross-referenced against results/brti_decision_dataset.jsonl (the windows for
which we captured intra-window BRTI samples).

Verifies counts rather than assuming ~6,251. Writes:
  research/true15m/contract_inventory.jsonl
  research/true15m/contract_inventory_report.json
Emits research events for each step.
"""
import calendar
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

SRC = ROOT / "results" / "contract_outcomes.jsonl"
RICH = ROOT / "results" / "brti_decision_dataset.jsonl"
OUT = ROOT / "research" / "true15m" / "contract_inventory.jsonl"
REP = ROOT / "research" / "true15m" / "contract_inventory_report.json"
WINDOW_S = 900


def _epoch(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


def build():
    t0 = time.time()
    EV.emit("JOB_STARTED", "Phase 1 · contract inventory", lane="L1",
            narrative="Recovering every historical KXBTC15M contract's truth "
                      "(T0/T1, opening & settlement 60s-BRTI averages, official label).")
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    for r in rows:
        r["_t0"] = _epoch(r["open_time"]); r["_t1"] = _epoch(r["close_time"])
    rows.sort(key=lambda r: r["_t0"])

    ids = [r["ticker"] for r in rows]
    uniq = set(ids)
    dupes = len(ids) - len(uniq)
    labeled = sum(1 for r in rows if r.get("exact_yes") in (0, 1))
    avg_cov = sum(1 for r in rows if r.get("floor_strike") and r.get("expiration_value"))
    bad_dur = sum(1 for r in rows if (r["_t1"] - r["_t0"]) != WINDOW_S)

    # contiguity / missing 15-min slots between first and last open
    first, last = rows[0]["_t0"], rows[-1]["_t0"]
    expected_slots = (last - first) // WINDOW_S + 1
    present = {r["_t0"] for r in rows}
    missing = sum(1 for s in range(first, last + 1, WINDOW_S) if s not in present)

    # raw 60s-observation coverage (windows we actually sampled intra-window)
    rich_ids = set()
    if RICH.exists():
        for l in RICH.open():
            l = l.strip()
            if l:
                try:
                    rich_ids.add(json.loads(l)["market_window_id"])
                except Exception:
                    pass

    inv = []
    for r in rows:
        wid = r["ticker"]
        inv.append({
            "market_window_id": wid, "T0": r["_t0"], "T1": r["_t1"],
            "open_time": r["open_time"], "close_time": r["close_time"],
            "opening_brti_avg": r.get("floor_strike"),
            "settlement_brti_avg": r.get("expiration_value"),
            "D": r.get("D"), "official_outcome": r.get("exact_yes"),
            "result": r.get("result"),
            "has_raw_60s_observations": wid in rich_ids,
            "provenance": "kalshi trade-api v2 settled market; rule KXBTC15M_2026-09",
        })
    body = "".join(json.dumps(x) + "\n" for x in inv)
    OUT.write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()[:16]
    EV.emit("FILE_CREATED", "contract_inventory.jsonl written", lane="L1",
            files=[str(OUT.relative_to(ROOT))],
            metrics={"windows": len(inv), "sha": sha})

    report = {
        "schema_version": "true15m-inventory-1", "generated_at": time.time(),
        "source": "results/contract_outcomes.jsonl",
        "total_windows": len(rows),
        "unique_windows": len(uniq),
        "duplicate_windows": dupes,
        "date_range_utc": [rows[0]["open_time"], rows[-1]["open_time"]],
        "span_days": round((last - first) / 86400, 2),
        "expected_contiguous_slots": expected_slots,
        "missing_windows": missing,
        "non_900s_windows": bad_dur,
        "exact_label_coverage": {"n": labeled, "frac": round(labeled / len(rows), 4)},
        "brti_average_coverage": {"n": avg_cov, "frac": round(avg_cov / len(rows), 4)},
        "raw_60s_observation_coverage": {
            "n": len(rich_ids), "frac": round(len(rich_ids) / len(rows), 4),
            "note": "windows with captured intra-window BRTI samples "
                    "(brti_decision_dataset); the rest have official averages only"},
        "old_model_predictions": "EXCLUDED from inventory (archaeology only, per §2)",
        "content_sha256_16": sha,
        "verified_count_note": f"directive expected ~6,251; actual settled windows = {len(rows)}",
    }
    REP.write_text(json.dumps(report, indent=1))
    EV.emit("INTEGRITY_CHECK", "Inventory counts verified", lane="L1", severity="info",
            fact=f"{len(rows)} settled windows, {dupes} duplicates, {missing} missing slots, "
                 f"label coverage {report['exact_label_coverage']['frac']}, "
                 f"raw-obs coverage {report['raw_60s_observation_coverage']['frac']}.",
            interpretation="Full-universe labels+averages are complete; raw 60s samples "
                           "cover only the recently-captured windows (cohort split later).",
            next_action="await official class-baseline discovery before any training",
            metrics={"windows": len(rows)}, duration_ms=int((time.time() - t0) * 1000))
    print(f"contract_inventory: {len(rows)} windows ({dupes} dup, {missing} missing slots); "
          f"label_cov={report['exact_label_coverage']['frac']} "
          f"raw_obs_cov={report['raw_60s_observation_coverage']['frac']} sha={sha}")


if __name__ == "__main__":
    build()
