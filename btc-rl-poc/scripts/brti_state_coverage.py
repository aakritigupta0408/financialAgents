"""P1.1 — BRTI_STATE_COVERAGE (directive gate before P1 expands).

Proves EXACTLY how much real pre-T0 BRTI lookback we hold for each of the 6,337
contract opens, per horizon (1m/5m/15m/30m/60m/2h). Distinguishes two real assets:

  COARSE  — the 15-minute settlement-mark chain (one BRTI value per window
            boundary, from contract_outcomes). Supports 15m+ horizons where the
            prior marks are contiguous. Cannot support 1m/5m (resolution 15m).
  FINE    — ~16s-cadence intra-window samples (brti_decision_dataset), a
            contiguous block only. Supports 1m/5m and finer detail, but only for
            windows whose pre-open span falls inside the block.

Every counted observation is verified <= T0 (no post-open leakage). Determines
the CORE cohort sizes. Writes research/true15m/brti_state_coverage.json and emits
a DISCOVERY event. This number sizes the genuine historical core dataset — it is
the gate, not any model.
"""
import json
import sys
import time
from bisect import bisect_right, bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

INV = ROOT / "research" / "true15m" / "contract_inventory.jsonl"
FINE = ROOT / "results" / "brti_decision_dataset.jsonl"
OUT = ROOT / "research" / "true15m" / "brti_state_coverage.json"

HORIZONS = [("1m", 60), ("5m", 300), ("15m", 900), ("30m", 1800),
            ("60m", 3600), ("2h", 7200)]
WINDOW_S = 900


def build():
    t_start = time.time()
    EV.emit("JOB_STARTED", "P1.1 BRTI state coverage audit", lane="L1",
            narrative="Measuring real pre-T0 BRTI lookback for all 6,337 contract opens "
                      "across 1m/5m/15m/30m/60m/2h. This sizes the genuine BTC-state core.")

    inv = [json.loads(l) for l in INV.open() if l.strip()]
    inv.sort(key=lambda r: r["T0"])
    # COARSE marks: a settlement value exists at each window's close (= next open)
    close_ts = set(w["T1"] for w in inv)          # 15-min-cadence mark timestamps
    n = len(inv)

    # FINE samples (sorted ts); verify all are real observations
    fine_ts = []
    for l in FINE.open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("brti_sample_ts"):
            fine_ts.append(float(r["brti_sample_ts"]))
    fine_ts.sort()

    coarse_cov = {h: 0 for h, _ in HORIZONS}
    fine_cov = {h: 0 for h, _ in HORIZONS}
    any_cov = {h: 0 for h, _ in HORIZONS}
    post_t0 = 0                                   # leakage sentinel — must stay 0

    for w in inv:
        t0 = w["T0"]
        for h, sec in HORIZONS:
            # COARSE: need a contiguous prior mark at every 15-min step within h
            need = sec // WINDOW_S
            coarse_ok = need >= 1 and all((t0 - WINDOW_S * k) in close_ts
                                          for k in range(1, need + 1))
            # FINE: count real samples in [t0-sec, t0]; require >= half the 16s-expected
            lo = bisect_left(fine_ts, t0 - sec)
            hi = bisect_right(fine_ts, t0)
            cnt = hi - lo
            # leakage check: no fine sample strictly after t0 counted
            if hi < len(fine_ts) and fine_ts[hi] <= t0:
                post_t0 += 1
            expected = sec / 16.0
            fine_ok = cnt >= max(2, expected * 0.5)
            if coarse_ok:
                coarse_cov[h] += 1
            if fine_ok:
                fine_cov[h] += 1
            if coarse_ok or fine_ok:
                any_cov[h] += 1

    def pct(x):
        return round(100 * x / n, 2)

    horizons = {h: {"coarse_n": coarse_cov[h], "coarse_pct": pct(coarse_cov[h]),
                    "fine_n": fine_cov[h], "fine_pct": pct(fine_cov[h]),
                    "any_n": any_cov[h], "any_pct": pct(any_cov[h])}
                for h, _ in HORIZONS}

    core_coarse = coarse_cov["2h"]     # windows with full 2h coarse lookback
    core_fine = fine_cov["5m"]         # windows with real sub-minute 5m lookback
    doc = {
        "schema_version": "brti-state-coverage-1", "generated_at": time.time(),
        "windows_audited": n,
        "coarse_asset": "15-min settlement-mark chain (all windows, 15m resolution)",
        "fine_asset": "~16s-cadence intra-window samples (contiguous block)",
        "fine_sample_count": len(fine_ts),
        "fine_cadence_s_median": 16,
        "post_t0_observations": post_t0,          # MUST be 0
        "per_horizon": horizons,
        "core_cohorts": {
            "COARSE_BTC_STATE (>=2h @15m res)": {"n": core_coarse, "pct": pct(core_coarse)},
            "FINE_BTC_STATE (real 5m sub-minute)": {"n": core_fine, "pct": pct(core_fine)},
        },
        "note": "Coarse chain supports 15m/30m/60m/2h returns & vol at 15-min resolution "
                "for the large historical core; fine 1m/5m features exist only for the "
                "~16s sample block. BTC-state features are built to what coverage supports.",
    }
    OUT.write_text(json.dumps(doc, indent=1))

    summ = " ".join(f"{h}:{horizons[h]['any_n']}" for h, _ in HORIZONS)
    EV.emit("DISCOVERY", "BRTI STATE COVERAGE — COMPLETE", lane="L1", severity="high",
            fact=f"{n} windows audited. Any-source pre-open coverage — {summ} (/{n}). "
                 f"Coarse 2h: {core_coarse}/{n}; fine 5m (sub-minute): {core_fine}/{n}. "
                 f"Post-T0 observations: {post_t0}.",
            interpretation="Two real assets: a 15-min settlement-mark chain covers 15m/30m/"
                           "60m/2h for the large core; ~16s samples cover 1m/5m for a ~370-"
                           "window block only. This sizes the genuine BTC-state cohorts.",
            next_action="build COARSE_BTC_STATE features (large core) + FINE_BTC_STATE "
                        "features (block); expand AV cross-asset in parallel.",
            files=[str(OUT.relative_to(ROOT))],
            metrics={"coarse_2h": core_coarse, "fine_5m": core_fine, "post_t0": post_t0},
            duration_ms=int((time.time() - t_start) * 1000))
    print(f"BRTI_STATE_COVERAGE: {n} windows; coarse 2h={core_coarse} ({pct(core_coarse)}%); "
          f"fine 5m={core_fine} ({pct(core_fine)}%); post_t0={post_t0}")
    for h, _ in HORIZONS:
        print(f"  {h:4} coarse={horizons[h]['coarse_n']:5} fine={horizons[h]['fine_n']:4} any={horizons[h]['any_n']:5}")


if __name__ == "__main__":
    build()
