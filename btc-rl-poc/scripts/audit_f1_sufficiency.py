"""F1 EARLY SUFFICIENCY AUDIT (PM 2026-09-03) — observation only.

Question: is the capture-to-date F-XVENUE dataset SCIENTIFICALLY
sufficient to power the registered T1.1 experiment — not merely
technically valid? Computes coverage/integrity AND statistical
resolution (effective sample size, minimum detectable improvement,
expected CI width) on the 15-min decision grid.

Since the GATE_F1_V2 amendment (PM 09-03) this module is also the
SHARED ANALYSIS CORE: emit_f1_gate.py imports scan()/analyze() and
applies the pre-registered gate thresholds to the clean eligible
window. One implementation, two consumers — no private formulas.

LAWS HONORED:
- No model training, no feature selection. The only reference
  predictor is the martingale (predict zero change) — a fixed
  benchmark, not a fitted model.
- "Data looks promising" is NOT a sufficiency criterion; every test
  is about whether signal could be distinguished from noise.
"""
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "results" / "events_xvenue"
OUT = ROOT / "results" / "f1_sufficiency_audit.json"

VENUES = ("binance", "okx", "kraken")
GRID_S = 900                 # 15-min decision grid (desk cadence)
HORIZONS = (5, 15, 30)       # minutes ahead (T1 label horizons)
CLOCK_TOL_S = 2.0            # ts_event may exceed ts_recv by skew
GAP_MATERIAL_S = 300
RHOS = (0.90, 0.95, 0.99)    # assumed control/treatment error corr
MIN_PER_FOLD = 48            # pre-declared: fold MAE needs >=48 obs
N_FOLDS = 5


def scan():
    """One pass over all shards -> per-venue per-minute aggregates,
    per-second presence, and integrity counters."""
    per_min = {v: {} for v in VENUES}     # minute -> [count, last_px]
    sec_present = {v: set() for v in VENUES}
    integ = {v: {"rows": 0, "pit_violations": 0, "max_skew_s": 0.0,
                 "out_of_order": 0, "bad_rows": 0}
             for v in VENUES}
    last_recv = {v: 0.0 for v in VENUES}
    for shard in sorted(SHARDS.glob("xvenue-*.jsonl")):
        for line in shard.open():
            try:
                r = json.loads(line)
                v = r["src"]
                tr, te, px = r["ts_recv"], r["ts_event"], r["px"]
            except Exception:
                for v2 in VENUES:
                    integ[v2]["bad_rows"] += 1 / len(VENUES)
                continue
            if v not in per_min:
                continue
            g = integ[v]
            g["rows"] += 1
            skew = te - tr
            if skew > g["max_skew_s"]:
                g["max_skew_s"] = round(skew, 3)
            if skew > CLOCK_TOL_S:
                g["pit_violations"] += 1
            if tr < last_recv[v]:
                g["out_of_order"] += 1
            last_recv[v] = tr
            m = int(tr // 60)
            cell = per_min[v].get(m)
            if cell is None:
                per_min[v][m] = [1, px]
            else:
                cell[0] += 1
                cell[1] = px
            sec_present[v].add(int(tr))
    return per_min, sec_present, integ


def machine_outages(per_min, min_gap_min=15):
    """ALL-VENUE simultaneous silence > min_gap_min = machine-side
    outage (sleep/crash). Venue-specific gaps are venue nature and
    count against coverage instead. Returns [(gap_start_min,
    gap_end_min), ...] in minute units."""
    union = set()
    for v in VENUES:
        union |= set(per_min[v])
    mins = sorted(union)
    out = []
    for a, b in zip(mins, mins[1:]):
        if b - a > min_gap_min:
            out.append((a, b))
    return out


def acf_ess(x):
    """Effective sample size via initial-positive-sequence ACF."""
    n = len(x)
    if n < 30:
        return n, []
    mu = sum(x) / n
    var = sum((a - mu) ** 2 for a in x) / n
    if var == 0:
        return 1, []
    rhos, s = [], 0.0
    for k in range(1, min(60, n // 4)):
        c = sum((x[i] - mu) * (x[i + k] - mu)
                for i in range(n - k)) / n / var
        if c <= 0:
            break
        rhos.append(round(c, 4))
        s += c
    return max(1.0, n / (1 + 2 * s)), rhos


def sd(x):
    n = len(x)
    mu = sum(x) / n
    return math.sqrt(sum((a - mu) ** 2 for a in x) / max(1, n - 1))


def analyze(per_min, sec_present, integ, m_lo=None, m_hi=None,
            outages=None):
    """Full sufficiency analysis restricted to minutes [m_lo, m_hi]
    (None = whole capture). Returns the analysis doc body.

    outages (PROPOSED F1 v2.1, stitched-segment mode): a list of
    (gap_start_min, gap_end_min) documented machine outages. When
    given, outage minutes leave every denominator (they are
    documented absences, not silent holes), a venue gap is material
    only by its non-outage portion, and a grid point is usable only
    if its whole lookback+label path [m-60, m+max(h)] avoids outage
    minutes."""
    all_minutes = set()
    for v in VENUES:
        all_minutes |= set(per_min[v])
    if m_lo is None:
        m_lo = min(all_minutes)
    if m_hi is None:
        m_hi = max(all_minutes)
    pm = {v: {m: c for m, c in per_min[v].items()
              if m_lo <= m <= m_hi} for v in VENUES}
    span_min = m_hi - m_lo + 1
    span_days = round(span_min / 1440, 2)
    out_min = set()
    for a, b in (outages or []):
        out_min.update(range(max(a + 1, m_lo), min(b, m_hi + 1)))
    span_eff = span_min - len(out_min)

    # ---- coverage & continuity ------------------------------------
    coverage, gaps = {}, {}
    eff_buckets = math.ceil(span_min / 5) - len(
        {m // 5 for m in out_min}) if out_min else math.ceil(
            span_min / 5)
    for v in VENUES:
        mins = sorted(pm[v])
        coverage[v] = {
            "minute_presence": round(len(mins) / max(1, span_eff), 4),
            "five_min_presence": round(
                len({m // 5 for m in mins})
                / max(1, eff_buckets), 4)}
        gg, prev = [], None
        for m in mins:
            if prev is not None:
                hole = (m - prev) - sum(1 for x in range(prev + 1, m)
                                        if x in out_min)
                if hole * 60 > GAP_MATERIAL_S:
                    gg.append({"start": time.strftime(
                        "%m-%d %H:%M", time.gmtime(prev * 60)),
                        "gap_min": hole})
            prev = m
        gaps[v] = sorted(gg, key=lambda g: -g["gap_min"])[:5]

    # ---- reference series (cross-venue median, per minute) --------
    ref = {}
    for m in range(m_lo, m_hi + 1):
        px = sorted(pm[v][m][1] for v in VENUES if m in pm[v])
        if px:
            ref[m] = px[len(px) // 2]

    # ---- decision grid + labels -----------------------------------
    hmax = HORIZONS[-1]
    grid = [m for m in range(m_lo + 60, m_hi - hmax, GRID_S // 60)
            if m in ref and not any(
                x in out_min for x in range(m - 60, m + hmax + 1))]
    labels = {}
    for h in HORIZONS:
        rows = []
        for m in grid:
            if m + h in ref:
                rows.append((m, 1e4 * math.log(ref[m + h] / ref[m])))
        labels[h] = rows

    # ---- feature availability at decision points ------------------
    avail = {}
    for v in VENUES:
        s = sec_present[v]
        ok5 = ok60 = 0
        for m in grid:
            t = m * 60
            if any((t - k) in s for k in range(0, 6)):
                ok5 += 1
            if any((t - k) in s for k in range(0, 61)):
                ok60 += 1
        avail[v] = {"lookback_5s": round(ok5 / max(1, len(grid)), 4),
                    "lookback_60s": round(ok60 / max(1, len(grid)), 4)}
    all3_5s = 0
    for m in grid:
        t = m * 60
        if all(any((t - k) in sec_present[v] for k in range(0, 6))
               for v in VENUES):
            all3_5s += 1
    avail["all_venues_5s"] = round(all3_5s / max(1, len(grid)), 4)

    # ---- venue diversity (redundancy + dispersion) ----------------
    vret = {v: {} for v in VENUES}
    for v in VENUES:
        mins = sorted(pm[v])
        for a, b in zip(mins, mins[1:]):
            if b - a == 1:
                vret[v][b] = 1e4 * math.log(
                    pm[v][b][1] / pm[v][a][1])
    vcorr = {}
    for i, va in enumerate(VENUES):
        for vb in VENUES[i + 1:]:
            common = sorted(set(vret[va]) & set(vret[vb]))
            if len(common) > 100:
                xa = [vret[va][m] for m in common]
                xb = [vret[vb][m] for m in common]
                ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
                num = sum((p - ma) * (q - mb)
                          for p, q in zip(xa, xb))
                den = math.sqrt(sum((p - ma) ** 2 for p in xa)
                                * sum((q - mb) ** 2 for q in xb))
                vcorr[f"{va}~{vb}"] = (round(num / den, 4)
                                       if den else None)
    disp = []
    for m in ref:
        px = [pm[v][m][1] for v in VENUES if m in pm[v]]
        if len(px) == 3:
            med = sorted(px)[1]
            disp.append(1e4 * max(abs(p - med) / med for p in px))
    disp.sort()
    dispersion_bps = {
        "median": round(disp[len(disp) // 2], 2) if disp else None,
        "p90": round(disp[int(len(disp) * .9)], 2) if disp else None}

    # ---- temporal diversity / regimes (6h blocks) -----------------
    blocks = {}
    for m, r in sorted(ref.items()):
        if m + 1 in ref:
            blocks.setdefault(m // 360, []).append(
                1e4 * math.log(ref[m + 1] / ref[m]))
    bstats = []
    for b, rs in sorted(blocks.items()):
        if len(rs) > 60:
            bstats.append({
                "block_utc": time.strftime(
                    "%m-%d %H:%M", time.gmtime(b * 360 * 60)),
                "block_id": b,
                "vol_bps": round(sd(rs) * math.sqrt(len(rs)), 1),
                "ret_bps": round(sum(rs), 1), "n_min": len(rs)})
    terc_share, trend_mix = {}, None
    if bstats:
        vols = sorted(x["vol_bps"] for x in bstats)
        t1 = vols[len(vols) // 3]
        t2 = vols[2 * len(vols) // 3]
        for x in bstats:
            x["vol_tercile"] = ("LOW" if x["vol_bps"] <= t1 else
                                "MID" if x["vol_bps"] <= t2
                                else "HIGH")
        terc_share = {t: round(sum(1 for x in bstats
                                   if x["vol_tercile"] == t)
                               / len(bstats), 3)
                      for t in ("LOW", "MID", "HIGH")}
        trend_mix = round(sum(1 for x in bstats
                              if x["ret_bps"] > 0) / len(bstats), 3)

    # ---- outcome diversity ----------------------------------------
    outcome = {}
    for h in HORIZONS:
        rs = [r for _, r in labels[h]]
        if rs:
            up = sum(1 for r in rs if r > 0)
            outcome[f"h{h}"] = {
                "up_frac": round(up / len(rs), 3),
                "median_abs_bps": round(
                    sorted(map(abs, rs))[len(rs) // 2], 1)}

    # ---- power block (THE critical output) ------------------------
    power = {}
    for h in HORIZONS:
        rs = [r for _, r in labels[h]]
        if len(rs) < N_FOLDS * 2:
            power[f"h{h}"] = {"usable_n": len(rs),
                              "state": "TOO_FEW_OBS"}
            continue
        abserr = [abs(r) for r in rs]        # martingale-ref errors
        mae = sum(abserr) / len(abserr)
        ess, rhos_acf = acf_ess(abserr)
        sde = sd(abserr)
        mde = {}
        for rho in RHOS:
            sdd = sde * math.sqrt(2 * (1 - rho))
            mde[f"rho_{rho}"] = {
                "mde80_bps": round(2.80 * sdd / math.sqrt(ess), 2),
                "mde80_pct_of_mae": round(
                    100 * 2.80 * sdd / math.sqrt(ess) / mae, 1),
                "ci95_width_bps": round(
                    2 * 1.96 * sdd / math.sqrt(ess), 2)}
        fold_sz = len(rs) // N_FOLDS
        fmae = [sum(abserr[i * fold_sz:(i + 1) * fold_sz]) / fold_sz
                for i in range(N_FOLDS)]
        power[f"h{h}"] = {
            "usable_n": len(rs), "ess": round(ess, 1),
            "ess_ratio": round(ess / len(rs), 3),
            "acf_head": rhos_acf[:5],
            "baseline_mae_bps": round(mae, 1),
            "sd_abs_err_bps": round(sde, 1),
            "min_detectable_improvement": mde,
            "folds": {"n_folds": N_FOLDS, "per_fold": fold_sz,
                      "min_required": MIN_PER_FOLD,
                      "viable": fold_sz >= MIN_PER_FOLD,
                      "fold_mae_bps": [round(x, 1) for x in fmae],
                      "fold_mae_cv": round(
                          sd(fmae) / (sum(fmae) / N_FOLDS), 3)}}

    # ---- regime concentration of usable windows -------------------
    terc_by_block = {x["block_id"]: x.get("vol_tercile")
                     for x in bstats}
    conc = {t: 0 for t in ("LOW", "MID", "HIGH")}
    for m, _ in labels[HORIZONS[1]]:
        t = terc_by_block.get(m // 360)
        if t:
            conc[t] += 1
    tot = sum(conc.values()) or 1
    regime_conc = {t: round(c / tot, 3) for t, c in conc.items()}

    checks = {
        "coverage_95": all(coverage[v]["five_min_presence"] >= 0.95
                           for v in VENUES),
        "pit_zero": all(g["pit_violations"] == 0
                        for g in integ.values()),
        "timestamps_ordered": all(
            g["out_of_order"] / max(1, g["rows"]) < 0.001
            for g in integ.values()),
        "continuity": all(not gaps[v] or gaps[v][0]["gap_min"] < 60
                          for v in VENUES),
        "fold_viability": all(
            power[f"h{h}"].get("folds", {}).get("viable", False)
            for h in HORIZONS),
        "feature_availability": avail["all_venues_5s"] >= 0.95,
        "temporal_diversity": (len(bstats) >= 12 and trend_mix
                               is not None
                               and 0.2 <= trend_mix <= 0.8),
        "venue_no_absence": all(
            coverage[v]["five_min_presence"] >= 0.5 for v in VENUES),
        "outcome_nondegenerate": bool(outcome) and all(
            0.3 <= o["up_frac"] <= 0.7 for o in outcome.values()),
        "regime_no_domination": max(regime_conc.values()) <= 0.60,
    }
    return {
        "window_utc": [time.strftime("%m-%d %H:%M",
                                     time.gmtime(m_lo * 60)),
                       time.strftime("%m-%d %H:%M",
                                     time.gmtime(m_hi * 60))],
        "span_days": span_days, "grid_s": GRID_S,
        "venues": list(VENUES),
        "coverage": coverage, "material_gaps": gaps,
        "feature_availability": avail,
        "venue_redundancy_corr_1min": vcorr,
        "cross_venue_dispersion_bps": dispersion_bps,
        "temporal_blocks_6h": {"n_blocks": len(bstats),
                               "vol_tercile_share": terc_share,
                               "up_block_frac": trend_mix,
                               "blocks": bstats},
        "outcome_diversity": outcome,
        "regime_concentration_of_usable": regime_conc,
        "power": power, "checks": checks}


def main():
    t0 = time.time()
    per_min, sec_present, integ = scan()
    body = analyze(per_min, sec_present, integ)
    checks = body["checks"]
    doc = {
        "generated_ts": int(time.time()),
        "status": "EXPLORATORY_OBSERVATION_ONLY",
        "f1_registration": "see GATE_F1_V2 in FEATURE_REGISTRY.yaml "
                           "— the gate applies these computations to "
                           "the clean eligible window via "
                           "emit_f1_gate.py",
        "no_training_no_feature_selection": True,
        "integrity": integ,
        **body,
        "sufficiency_summary": {
            "n_pass": sum(checks.values()),
            "n_total": len(checks),
            "verdict": ("SUFFICIENT" if all(checks.values())
                        else "INSUFFICIENT"),
            "failed": [k for k, v in checks.items() if not v]},
        "runtime_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in
                      ("span_days", "checks", "sufficiency_summary")},
                     indent=1))
    print("full artifact:", OUT.name)


if __name__ == "__main__":
    main()
