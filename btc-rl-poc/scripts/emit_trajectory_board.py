"""TRAJECTORY BOARD (PM 2026-09-07) — observation only.

Every surviving component gets a trajectory STATE, not just WIN/LOSE.
"Below target" is not "retire"; what matters is the distance to the
gate AND the rate that distance is closing. For components with a
clean independent-unit series we compute a robust slope, an ETA, and
a bootstrap P(hit the gate before max-N); where no metric time-series
exists we say PACE_UNKNOWN rather than invent a slope.

REGISTERED THRESHOLDS (frozen once, PM 09-07 — NOT per-experiment):
  P(hit gate before max-N) > 0.70          -> ON_TRACK
  0.30 <= P <= 0.70                        -> PROMISING / PACE UNKNOWN
  P < 0.30 with positive slope             -> IMPROVING_TOO_SLOW
  slope indistinguishable from 0           -> FLAT
  bootstrap mass strongly negative         -> REGRESSING
  waiting on a prerequisite                -> BLOCKED
For SPRT experiments the frozen SPRT remains the decision authority;
the trajectory here is informational only.
"""
import json
import statistics as st
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
P_ON_TRACK = 0.70
P_SLOW = 0.30
BOOT = 2000
SEED = 20260907
# PM 09-07: every P(hit gate) MUST carry these three, or it is
# pseudo-precision a professor can rightly reject. Where the unit
# structure is awkward, emit p=null + PACE_UNKNOWN, never a number.
REPRO_ID = "traj-lcg-v1-seed20260907-boot2000"


def evidence(n_units, has_bootstrap, independence):
    """Mandatory metadata block beside any trajectory probability."""
    return {"n_independent_units": n_units,
            "bootstrap": (REPRO_ID if has_bootstrap else None),
            "independence_satisfied": independence}


def j(name):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return None


def jl(name):
    try:
        return [json.loads(x) for x in
                (RES / name).read_text().splitlines() if x.strip()]
    except Exception:
        return []


def theil_sen(ys):
    """Robust slope per unit index (median of pairwise slopes)."""
    n = len(ys)
    if n < 3:
        return 0.0
    sl = [(ys[j] - ys[i]) / (j - i)
          for i in range(n) for j in range(i + 1, n)]
    return st.median(sl)


def lcg(seed):
    s = seed
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s / 0x7FFFFFFF


def a3_trajectory():
    led = jl("a3v21_window_evaluation.jsonl")
    dec = j("a3_decision.json") or {}
    el = [e for e in led if e.get("state") != "SYSTEM_EXCLUDED"]
    deltas = [e.get("delta_pnl") or 0.0 for e in el]
    n = len(deltas)
    gate = dec.get("registered_gate_n", 50)
    if n == 0:
        return {"id": "A3-v2.1", "tier": "T3-entry-timing",
                "state": "BLOCKED", "note": "no eligible windows"}
    cum = sum(deltas)
    mean = cum / n
    remaining = max(0, gate - n)
    # required average over remaining windows merely to reach 0
    req_final = (-cum / remaining) if remaining > 0 else None
    # informational velocity + bootstrap P(mean>=0 at gate)
    slope = theil_sen([sum(deltas[:i + 1]) / (i + 1)
                       for i in range(n)])
    rng = lcg(SEED)
    hits = 0
    for _ in range(BOOT):
        proj = cum
        for _r in range(remaining):
            proj += deltas[int(next(rng) * n)]
        if proj / gate >= 0:
            hits += 1
    p_hit = hits / BOOT
    if n >= gate:
        # gate reached; SPRT-style CI decides — trajectory descriptive
        state = ("IMPROVING_TOO_SLOW" if mean < 0 and slope > 0
                 else "FLAT" if abs(slope) < 1e-4
                 else "REGRESSING" if mean < 0 else "PROMISING")
    else:
        state = ("ON_TRACK" if p_hit > P_ON_TRACK else
                 "IMPROVING_TOO_SLOW" if p_hit < P_SLOW and slope > 0
                 else "PROMISING")
    # headline effect from the authoritative decision artifact
    # (dollars/eligible -> cents); ledger drives trajectory shape
    eff_c = round((dec.get("primary_effect_per_eligible", mean))
                  * 100, 2)
    return {"id": "A3-v2.1", "tier": "T3-entry-timing",
            "current_c_per_eligible": eff_c,
            "cumulative_c": round(cum * 100, 1), "n": n,
            "gate_n": gate, "target": "> 0 at registered gate",
            "gap_c": round(-eff_c, 2),
            "required_avg_over_remaining_c": (round(req_final * 100, 1)
                                              if req_final else None),
            "recent_slope": round(slope, 4),
            "p_reach_zero_by_gate": round(p_hit, 3),
            "trajectory_evidence": evidence(
                n, True,
                "per-window paired deltas; distinct 15-min windows "
                "treated independent — mild regime autocorrelation "
                "possible, so P is indicative not exact"),
            "decision_authority": dec.get("decision"),
            "state": state,
            "note": "frozen bootstrap-CI gate is the authority; "
                    "trajectory is informational"}


def texec_trajectory():
    os_ = j("online_status.json") or {}
    tr = {t["key"]: t for t in (os_.get("treatments") or [])}
    ter = tr.get("t_exec_reg")
    if not ter:
        return None
    rows = jl("treatments.jsonl")
    # per-window paired diff vs champion (informational velocity)
    diffs = []
    for r in rows:
        ev = r.get("ev") or {}
        a, b = ev.get("t_exec_reg"), ev.get("champion")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diffs.append(a - b)
    slope = theil_sen([sum(diffs[:i + 1]) / (i + 1)
                       for i in range(len(diffs))]) if diffs else 0.0
    return {"id": "t_exec_reg", "tier": "T4-execution",
            "current_paired_c_per_$1": round(ter.get("mean_diff", 0)
                                             * 100, 1),
            "n": ter.get("n"),
            "llr": round(ter.get("llr", 0), 2),
            "promote_llr": round(ter.get("upper", 5.66), 2),
            "reject_llr": round(ter.get("lower", -2.30), 2),
            "recent_slope": round(slope, 4),
            "p_reach_gate": None,
            "trajectory_evidence": evidence(
                len(diffs), False,
                "SPRT statistic — deliberately NOT bootstrapped; LLR "
                "is not linearly extrapolable, so no P is emitted"),
            "state": "PROMISING_PACE_UNKNOWN",
            "note": "SPRT is the decision authority; LLR is not "
                    "linearly extrapolable. Traffic was frozen by "
                    "the SEV — track velocity from n=375."}


def simple(id_, tier, current, target, gap, state, note,
           n_units=None, independence="n/a", extra=None):
    d = {"id": id_, "tier": tier, "current": current,
         "target": target, "gap": gap, "state": state,
         "p_reach_gate": None,          # no honest velocity series
         "trajectory_evidence": evidence(n_units, False,
                                         independence),
         "note": note}
    if extra:
        d.update(extra)
    return d


def main():
    qual = (j("model_qualification.json") or {}).get("models") or {}
    f1 = j("f1_capture_qualification.json") or {}
    t1 = j("t1_1_result.json") or {}
    board = []
    board.append(a3_trajectory())
    te = texec_trajectory()
    if te:
        board.append(te)

    # T2 control/challenger — BSS has no stored metric time-series
    def bss(m):
        return ((qual.get(m) or {}).get("walk_forward") or {}).get(
            "median_fold_bss")
    board.append(simple(
        "kb2", "T2-probability", bss("kb2"), "> market (BSS>0 durable)",
        None, "CONTROL",
        "serving control; ~0.998 corr with market — a thin edge",
        n_units=5, independence="5 chronological walk-forward folds"))
    board.append(simple(
        "kb4", "T2-probability", bss("kb4"), "beat kb2 + market durably",
        (round(bss("kb4") - bss("kb2"), 4)
         if bss("kb4") is not None and bss("kb2") is not None
         else None),
        "PROMISING_PACE_UNKNOWN",
        "challenger; only ~+0.002 BSS over kb2, no independent "
        "metric time-series to estimate arrival — needs a registered "
        "forward paired test to earn PROVEN",
        n_units=5, independence="5 folds; no metric time-series for "
        "a velocity estimate — P deliberately null"))

    # F1 — ESS vs 150
    sh = f1.get("proposed_v2_1_stitched") or {}
    ess = sh.get("ess_min_across_horizons")
    board.append(simple(
        "F1", "T0-capture", ess, ">= 150 ESS/horizon",
        (round(150 - ess, 1) if isinstance(ess, (int, float))
         else None),
        "BELOW_BENCHMARK_PLAUSIBLY_ON_TRACK",
        "not failing — independent information accrues slowly in a "
        "trending regime; needs a registered ESS-velocity pace test "
        "before EXTEND vs IMPROVING_TOO_SLOW can be called",
        n_units=(round(ess) if isinstance(ess, (int, float))
                 else None),
        independence="ESS already autocorrelation-adjusts; no stored "
        "ESS time-series yet, so P is deliberately null",
        extra={"verdict": f1.get("verdict")}))

    # T1.2 — blocked, not failing
    board.append(simple(
        "T1.2", "T1-forecast", None, "beat incumbent T1 (pinball)",
        None, "BLOCKED", "awaiting F1 gate; a blocked hypothesis is "
        "not a failure"))

    # t9 temp shadow (T1)
    board.append(simple(
        "t9", "T1-forecast", "pinball ~30.87",
        "beat consensus pinball ~30.40", "~+0.47 pinball (worse)",
        "BELOW_BENCHMARK_TEMP_SHADOW",
        "h30 retrain improved val MSE ~41%; kept ONE confirmation "
        "window — if the val gain does not show in fresh prequential "
        "out-of-sample, RETIRE",
        independence="aggregate pinball, no per-unit series here",
        extra={"coverage80": 0.807}))

    doc = {"generated_ts": int(time.time()),
           "law": "trajectory is informational; frozen gates/SPRT "
                  "remain the decision authority",
           "thresholds": {"on_track_p": P_ON_TRACK,
                          "slow_p": P_SLOW},
           "components": board}
    (RES / "trajectory_board.json").write_text(
        json.dumps(doc, indent=1))
    for c in board:
        print(f"  {c['id']:12s} {c.get('tier',''):18s} "
              f"{c['state']}")
    print(f"trajectory_board: {len(board)} components")


if __name__ == "__main__":
    main()
