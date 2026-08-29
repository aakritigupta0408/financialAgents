"""Emit results/decision_board.json — the DECISION layer over the
champion/challenger registry (TA metrics review, 2026-08-29).

The registry already answers "is the challenger ahead?" (paired mean,
SPRT). This emitter answers the TA's question: "given ALL available
evidence, should treatment X replace control Y — and how certain are
we?" Every number is computed from the per-window paired log the
daemon itself writes (results/treatments.jsonl); nothing is estimated
from aggregates, nothing is invented.

Per treatment it adds, on top of the daemon's SPRT status:
  * 95% CI on the paired Δ (normal AND bootstrap — P&L is not normal)
  * median paired Δ + top-3 |Δ| concentration (jackpot detector)
  * P(Δ>0) and P(Δ>pre-registered edge)   [normal approx + bootstrap]
  * MDE at current n, and power to detect the pre-registered 2¢ edge
    (both at the FAMILY-WISE alpha the SPRT runs under, not a vanity
    0.05)
  * projected windows to an SPRT verdict at the current LLR drift
  * paired completeness (both / treatment-only / control-only)
  * veto decomposition for gate treatments: on the windows the
    treatment skipped, what did the CONTROL earn? -> losses avoided,
    wins forgone, net veto value per window
  * regime / leader / time-of-day slices of the treatment effect, the
    worst adequately-sampled slice, and sign consistency
  * an automatic decision state: PROMOTE / KEEP_TESTING / HOLD /
    KILL / INVALID (+ BASELINE), with the reasons spelled out.

Decision rules are PRE-REGISTERED here in code (not fitted to the
current standings):
  INVALID       log stale >3h, or paired completeness <90%
  PROMOTE       SPRT crossed upper AND completeness ok AND coverage
                >=25% AND worst powered slice not significantly ruinous
                (mean + 1.96*se > -0.10 per $1)
  KILL          SPRT crossed lower
  HOLD          past min_n but structurally unresolvable: coverage
                <25% (a different product) or <2 powered slices
  KEEP_TESTING  everything else — the honest default
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402
from btc_rl import treatments             # noqa: E402

ND = NormalDist()
ET = ZoneInfo("America/New_York")
BOOT_N = 2000
BOOT_SEED = 20260829          # fixed: the report must be reproducible
SLICE_MIN_N = 15              # below this a slice mean is noise
BETA = 0.10                   # SPRT's miss rate, reused for MDE/power


def _load_rows():
    p = ROOT / "results" / O.TREAT_LOG_NAME
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


def _load_treats():
    """Rebuild Treatment objects exactly as the daemon does, then load
    only the accumulated evidence — so llr/verdict here are
    field-identical to what the daemon publishes."""
    treats = {}
    for k, lab, fn, why in O._treat_policies():
        treats[k] = treatments.Treatment(
            k, lab, fn, why, edge=O.TREAT_EDGE, min_n=O.TREAT_MIN_N,
            alpha=getattr(O, "TREAT_ALPHA", 0.05),
            baseline=k in ("champion", "champion_real"))
    sp = ROOT / "results" / O.TREAT_STATE_NAME
    if sp.exists():
        st = json.loads(sp.read_text())
        for k, d in (st.get("treats") or {}).items():
            if k in treats:
                treats[k].load(d)
    return treats


def _pairs(rows, key, base_key):
    """Per-window (own, base, row) with the daemon's None->0 scoring
    convention applied for the diff, but Nones kept for completeness
    and veto accounting."""
    out = []
    for r in rows:
        ev = r.get("ev") or {}
        if key not in ev and base_key not in ev:
            continue
        out.append((ev.get(key), ev.get(base_key), r))
    return out


def _mean_se(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def _slice_stats(name, ds):
    m, se = _mean_se(ds)
    return {"name": name, "n": len(ds), "mean": round(m, 5),
            "se": round(se, 5) if se is not None else None}


def _analyze(key, tr, pairs):
    d = [(0.0 if o is None else o) - (0.0 if b is None else b)
         for o, b, _ in pairs]
    n = len(d)
    if n < 2:
        return None
    mean, se = _mean_se(d)
    sd = se * math.sqrt(n)
    dsort = sorted(d)
    median = dsort[n // 2] if n % 2 else 0.5 * (
        dsort[n // 2 - 1] + dsort[n // 2])

    # bootstrap on the paired per-window diffs (P&L is fat-tailed;
    # the normal CI is kept beside it for comparison, not instead)
    rng = random.Random(BOOT_SEED)
    boots = []
    for _ in range(BOOT_N):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    boot_lo = boots[int(0.025 * BOOT_N)]
    boot_hi = boots[int(0.975 * BOOT_N)]
    p_gt0_boot = sum(1 for b in boots if b > 0) / BOOT_N
    p_gtedge_boot = sum(1 for b in boots if b > O.TREAT_EDGE) / BOOT_N

    p_gt0 = ND.cdf(mean / se) if se else None
    alpha = getattr(O, "TREAT_ALPHA", 0.05)
    za = ND.inv_cdf(1 - alpha)          # one-sided, family-wise alpha
    zb = ND.inv_cdf(1 - BETA)
    mde = (za + zb) * sd / math.sqrt(n) if n else None
    power = ND.cdf(O.TREAT_EDGE / (sd / math.sqrt(n)) - za) \
        if sd > 0 else None

    # concentration: is the lift broad, or three jackpots?
    tot_abs = sum(abs(x) for x in d)
    top3 = sum(sorted((abs(x) for x in d), reverse=True)[:3])
    top3_share = top3 / tot_abs if tot_abs > 0 else None

    # Pair completeness vs activity overlap — deliberately distinct.
    # A None here is a policy's DECISION to stand down (scored 0 by
    # pre-registration), not a missing observation, so every logged
    # row is a complete pair. Completeness therefore measures
    # EVALUABILITY (was this treatment scorable on the row at all);
    # the both/only counts report ACTIVITY overlap, which the TA's
    # exposure accounting asks for but which must never invalidate.
    evaluable = sum(1 for _, _, r in pairs
                    if key in (r.get("ev") or {}))
    completeness = evaluable / len(pairs) if pairs else None
    both = sum(1 for o, b, _ in pairs if o is not None and b is not None)
    t_only = sum(1 for o, b, _ in pairs
                 if o is not None and b is None)
    c_only = sum(1 for o, b, _ in pairs
                 if o is None and b is not None)

    # veto decomposition: what did the control do on skipped windows?
    skipped = [b for o, b, _ in pairs if o is None and b is not None]
    veto = None
    if skipped:
        veto = {
            "skips_scored": len(skipped),
            "losses_avoided": sum(1 for x in skipped if x < 0),
            "wins_forgone": sum(1 for x in skipped if x > 0),
            "control_ev_on_skipped": round(sum(skipped), 4),
            # a veto's contribution to the paired mean is exactly
            # -control_ev on those windows, spread over all n
            "net_veto_per_window": round(-sum(skipped) / n, 5),
        }

    # slices of the treatment effect (not of control performance)
    slices = []
    by_regime = {"regime<floor": [], "regime>=floor": []}
    by_hour = {}
    by_leader = {}
    floor = getattr(O, "REGIME_FLOOR", 0.62)
    for di, (_, _, r) in zip(d, pairs):
        ra = r.get("regime_acc")
        if ra is not None:
            by_regime["regime>=floor" if ra >= floor
                      else "regime<floor"].append(di)
        ld = r.get("leader")
        if ld:
            by_leader.setdefault("leader:" + ld, []).append(di)
    from datetime import datetime
    for di, (_, _, r) in zip(d, pairs):
        ts = r.get("close_ts")
        if not ts:
            continue
        h = datetime.fromtimestamp(ts, ET).hour
        b = f"ET {6 * (h // 6):02d}-{6 * (h // 6) + 6:02d}"
        by_hour.setdefault(b, []).append(di)
    for grp in (by_regime, by_hour, by_leader):
        for name, ds in sorted(grp.items()):
            if ds:
                slices.append(_slice_stats(name, ds))
    powered = [s for s in slices if s["n"] >= SLICE_MIN_N]
    worst = min(powered, key=lambda s: s["mean"]) if powered else None
    sign_pos = sum(1 for s in powered if s["mean"] > 0)

    # projected windows to an SPRT verdict at current drift
    st = tr.status()
    llr, up, lo = st["llr"], st["upper"], st["lower"]
    denom_n = max(1, st["n"] - treatments.SPRT.WARMUP)
    slope = llr / denom_n
    proj = None
    if slope > 1e-6 and llr < up:
        proj = min(5000, int((up - llr) / slope))
    elif slope < -1e-6 and llr > lo:
        proj = min(5000, int((lo - llr) / slope))

    return {
        "n": n, "bets": st["bets"], "skips": st["skips"],
        "coverage": round(st["bets"] / n, 3) if n else None,
        "mean": round(mean, 5), "se": round(se, 5),
        "median": round(median, 5),
        "ci95": [round(mean - 1.96 * se, 5),
                 round(mean + 1.96 * se, 5)],
        "boot_ci95": [round(boot_lo, 5), round(boot_hi, 5)],
        "p_gt0": round(p_gt0, 3) if p_gt0 is not None else None,
        "p_gt0_boot": round(p_gt0_boot, 3),
        "p_gt_edge_boot": round(p_gtedge_boot, 3),
        "mde": round(mde, 5) if mde is not None else None,
        "power_at_edge": round(power, 3) if power is not None else None,
        "top3_share": round(top3_share, 3)
        if top3_share is not None else None,
        "completeness": {
            "evaluable": evaluable,
            "pct": round(completeness, 4)
            if completeness is not None else None,
            "activity": {"both_bet": both, "treatment_only_bet": t_only,
                         "control_only_bet": c_only,
                         "note": "skips are scored decisions (EV 0), "
                                 "not missing data"}},
        "veto": veto,
        "slices": slices,
        "worst_slice": worst,
        "powered_slices": len(powered),
        "sign_consistency": f"{sign_pos}/{len(powered)}"
        if powered else None,
        "llr": st["llr"], "upper": st["upper"], "lower": st["lower"],
        "sprt_verdict": st["verdict"],
        "own_ev": st["own_ev"],
        "proj_windows_to_decision": proj,
    }


def _decide(a, integrity_ok):
    """Pre-registered decision rules — see module docstring."""
    reasons = []
    if not integrity_ok:
        return "INVALID", ["treatment log stale — do not decide"]
    comp = (a["completeness"]["pct"]
            if a["completeness"]["pct"] is not None else 1.0)
    if comp < 0.90:
        return "INVALID", [f"evaluable on only {comp:.0%} of logged "
                           "windows < 90%"]
    if a["sprt_verdict"] == "reject":
        return "KILL", ["SPRT crossed the reject boundary"]
    if a["sprt_verdict"] == "promote":
        if a["coverage"] is not None and a["coverage"] < 0.25:
            return "HOLD", ["SPRT passed but coverage <25% — "
                            "different product, needs its own review"]
        w = a["worst_slice"]
        if w and w["se"] and w["mean"] + 1.96 * w["se"] < -0.10:
            return "HOLD", [f"worst slice {w['name']} significantly "
                            "ruinous — regime instability"]
        return "PROMOTE", ["SPRT passed with guardrails clean"]
    # still collecting
    if a["n"] >= O.TREAT_MIN_N:
        if a["coverage"] is not None and a["coverage"] < 0.25:
            reasons.append(f"coverage {a['coverage']:.0%} <25% — "
                           "too selective to compare like-for-like")
        if a["powered_slices"] < 2:
            reasons.append("fewer than 2 adequately-sampled slices")
        if reasons:
            return "HOLD", reasons
    trend = ("winning" if (a["p_gt0"] or 0.5) >= 0.8 else
             "losing" if (a["p_gt0"] or 0.5) <= 0.2 else "uncertain")
    return "KEEP_TESTING", [f"collecting — trend {trend}, "
                            f"P(Δ>0)≈{a['p_gt0']}"]


def main():
    rows = _load_rows()
    treats = _load_treats()
    now = int(time.time())
    last_ts = max((r.get("close_ts") or 0) for r in rows) if rows else 0
    age_min = (now - last_ts) / 60 if last_ts else None
    integrity_ok = bool(rows) and age_min is not None and age_min < 180

    fam = {"champion_real": "real", "t_exec": "real",
           "t_exec_reg": "real"}
    out = []
    for key, tr in treats.items():
        base = "champion_real" if fam.get(key) == "real" else "champion"
        basis = fam.get(key, "model")
        if tr.baseline:
            st = tr.status()
            m, se = _mean_se([
                (r.get("ev") or {}).get(key) or 0.0 for r in rows])
            out.append({"key": key, "label": st["label"],
                        "basis": basis, "state": "BASELINE",
                        "state_reasons": ["incumbent — the yardstick"],
                        "n": st["n"], "bets": st["bets"],
                        "own_ev": st["own_ev"],
                        "mean_ev_all_windows": round(m, 5)})
            continue
        a = _analyze(key, tr, _pairs(rows, key, base))
        if a is None:
            continue
        state, reasons = _decide(a, integrity_ok)
        a.update({"key": key, "label": tr.label, "basis": basis,
                  "rationale": tr.rationale,
                  "state": state, "state_reasons": reasons})
        out.append(a)

    # execution waterfall: same windows, model quote vs real fill
    both = [((r["ev"] or {}).get("champion"),
             (r["ev"] or {}).get("champion_real"))
            for r in rows]
    both = [(m, x) for m, x in both if m is not None and x is not None]
    waterfall = None
    if both:
        dm = sum(m for m, _ in both) / len(both)
        dr = sum(x for _, x in both) / len(both)
        waterfall = {"windows": len(both),
                     "decision_ev_per_1": round(dm, 5),
                     "realized_ev_per_1": round(dr, 5),
                     "execution_gap_per_1": round(dr - dm, 5)}

    states = {}
    for a in out:
        states[a["state"]] = states.get(a["state"], 0) + 1
    cands = [a for a in out if a["state"] in
             ("KEEP_TESTING", "PROMOTE") and a.get("p_gt0")]
    best = max(cands, key=lambda a: a["p_gt0"])["key"] if cands else None

    doc = {
        "generated_ts": now,
        "config": {
            "edge": O.TREAT_EDGE, "alpha": getattr(O, "TREAT_ALPHA",
                                                   0.05),
            "beta": BETA, "min_n": O.TREAT_MIN_N,
            "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
            "slice_min_n": SLICE_MIN_N,
            "note": "CI/P(gt0) use a normal approximation on the "
                    "paired mean plus a seeded bootstrap; MDE/power "
                    "use the family-wise alpha, one-sided."},
        "integrity": {"rows": len(rows), "last_close_ts": last_ts,
                      "age_min": round(age_min, 1)
                      if age_min is not None else None,
                      "health": "green" if integrity_ok else "red"},
        "summary": {"states": states, "best_candidate": best},
        "execution_waterfall": waterfall,
        "treatments": out,
    }
    (ROOT / "results" / "decision_board.json").write_text(
        json.dumps(doc, indent=1))
    print(f"decision_board.json: {len(out)} treatments, "
          f"states={states}, best={best}, "
          f"integrity={'green' if integrity_ok else 'RED'}")


if __name__ == "__main__":
    main()
