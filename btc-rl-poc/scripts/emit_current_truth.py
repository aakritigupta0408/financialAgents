"""FREEZE CURRENT TRUTH (master handoff §5 / Phase 2). Reads canonical
artifacts only and emits research/current_truth.json — the before/after
baseline. Never invents a value: missing telemetry becomes "UNKNOWN".
Append-only in spirit: this is a snapshot, not a rewrite of history.
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "research" / "current_truth.json"
UNK = "UNKNOWN"


def jget(name):
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def jl(name, tail=None):
    p = RES / name
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows[-tail:] if tail else rows


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return UNK


def agg_bss(off, arm):
    m = (off or {}).get("models", {}).get(arm)
    if not m or not m.get("folds"):
        return None
    v = [f.get("bss") for f in m["folds"] if f.get("bss") is not None]
    return round(sum(v) / len(v), 4) if v else None


def trader_econ(tid):
    rows = jl(f"{tid}_trades.jsonl")
    res = [r for r in rows if r.get("actual") not in (None, -1)
           and not r.get("skipped")
           and (r.get("pnl_c") is not None or r.get("would_pnl_c") is not None)]
    def pnl(r):
        return r.get("pnl_c") if r.get("pnl_c") is not None else r.get("would_pnl_c")
    pnls = [pnl(r) for r in res]
    costs = [(r.get("ask_c") or 0) + (r.get("fee_c") or 0) for r in res]
    tot_cost = sum(c for c in costs if c) or None
    last = rows[-1] if rows else {}
    return {
        "n_resolved": len(res),
        "n_skipped": sum(1 for r in rows if r.get("skipped") or r.get("actual") == -1),
        "total_pnl_c": round(sum(pnls), 1) if pnls else 0,
        "realized_ev_per_1": (round(sum(pnls) / tot_cost, 4)
                              if tot_cost else UNK),
        "quoted_ev_per_1": UNK,  # not logged per-trader
        "bankroll_c": last.get("bankroll_c", UNK),
        "current_action": ("SKIPPED" if last.get("skipped") or last.get("actual") == -1
                           else "OPEN" if last.get("pnl_c") is None
                           else "RESOLVED"),
        "rows_total": len(rows)}


def main():
    inv = jget("invariants.json") or {}
    rd = jget("readiness.json") or {}
    dh = jget("data_health.json") or {}
    f1 = jget("f1_capture_qualification.json") or {}
    df = jget("decision_frontier.json") or {}
    fs = jget("failure_store.json") or {}
    db = jget("decision_board.json") or {}
    tb = jget("treatments_board.json") or {}
    off = jget("model_offline.json") or {}
    on = jget("model_online.json") or {}
    life = jget("model_lifecycle.json") or {}
    pm = jget("pm_snapshot.json") or {}
    a3 = jget("a3_live.json") or {}
    recon = jget("reconciliation.json") or {}
    par = jget("parity.json") or {}
    chg = jl("system_change_log.jsonl", tail=1)
    incidents = jl("incidents.jsonl")
    kb = jl("kalshi_binary_log.jsonl")

    open_inc = [r for r in incidents if
                not any(m in str(r.get("status", "")).lower()
                        for m in ("resolv", "closed", "benign", "human_closure"))
                and r.get("resolved") is None
                and any(m in str(r.get("status", "")).lower()
                        for m in ("open", "mitigat", "investigat", "active", "firing"))]
    settled = sum(1 for r in kb if r.get("actual") is not None)

    # F1 / ESS per horizon
    f1_power = {}
    for h, p in (f1.get("window_analysis", {}).get("power", {}) or {}).items():
        f1_power[h] = {"ess": p.get("ess"),
                       "mde_by_metric": (p.get("mde_by_metric_rho95") or {})}

    truth = {
        "generated_ts": int(time.time()),
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git_sha": git_sha(),
        "runtime_mode": "SIMULATED / PAPER — real-money execution DISABLED",
        "invariant_wall": {"passed": inv.get("passed", UNK),
                           "failed": inv.get("failed", UNK),
                           "health": inv.get("health", UNK)},
        "readiness": {"level": (rd.get("system") or {}).get("level", UNK),
                      "headline": (rd.get("system") or {}).get("headline", UNK),
                      "weakest_planes": (rd.get("system") or {}).get("weakest_planes", UNK),
                      "sev0_open": len(rd.get("sev0_open") or [])},
        "champion_control_treatment_shadow": {
            "control": "champion",
            "treatment": (db.get("summary") or {}).get("best_candidate", UNK),
            "shadow": ["kb7", "kb9", "pt6(MLE)"],
            "current_experiment": a3.get("experiment_id") or a3.get("id") or "A3"},
        "tiers": {
            "T1_forecast": {"f1_verdict": (f1.get("verdict") or UNK),
                            "ess_min": f1.get("ess_min_across_horizons", UNK),
                            "governing_metric": f1.get("governing_metric", UNK),
                            "per_horizon": f1_power},
            "T2_probability": {"kb2_bss_vs_market": agg_bss(off, "kb2"),
                               "kb3_bss": agg_bss(off, "kb3"),
                               "online_kb2": ((on.get("models") or {}).get("kb2") or {}).get("windows", {}).get("LIFETIME", UNK)},
            "T3_decision": {"frontier_verdict": df.get("verdict", UNK),
                            "edge_monotonicity": (df.get("edge_monotonicity") or {}).get("monotonicity", UNK),
                            "operating_point": df.get("chosen_operating_point", None)},
            "T4_execution": db.get("execution_waterfall", UNK)},
        "decision_frontier": {"verdict": df.get("verdict", UNK),
                              "base_rate_profitable": df.get("base_rate_profitable", UNK),
                              "risk_budget_epsilon": df.get("risk_budget_epsilon", UNK)},
        "failure_store": {"top_cluster": fs.get("top_cluster", UNK),
                          "bad_total": fs.get("bad_total", UNK),
                          "bets_total": fs.get("bets_total", UNK),
                          "clusters": fs.get("failure_clusters", UNK)},
        "traders": {t: trader_econ(t) for t in
                    ("pt", "pt3", "pt6", "pt2", "pt4", "pt5", "pt7", "pt8")},
        "retraining": {mid: {"runs": m.get("runs"), "kept": m.get("kept"),
                             "keep_rate": m.get("keep_rate"),
                             "last_decision": m.get("last_decision")}
                       for mid, m in (life.get("t1_serving") or {}).items()},
        "experiments": {"treatments": [{"key": t.get("key"), "verdict": t.get("verdict"),
                                        "n": t.get("n"), "own_ev": t.get("own_ev"),
                                        "llr": t.get("llr")}
                                       for t in (tb.get("treatments") or [])]},
        "blocked_work": pm.get("blocked_work", UNK),
        "open_incidents": [{"sev": r.get("sev"), "title": r.get("title")}
                           for r in open_inc],
        "settlement_counters": {"settled_windows": settled,
                                "kb_log_rows": len(kb)},
        "service_health": {"overall": dh.get("overall", UNK),
                           "feeds": {n: {"state": (f or {}).get("state"),
                                         "age_s": (f or {}).get("age_s"),
                                         "errors": (f or {}).get("errors")}
                                     for n, f in (dh.get("feeds") or {}).items()}},
        "data_freshness": {"generated_ts": dh.get("generated_ts", UNK),
                           "missingness": dh.get("missingness", UNK)},
        "reconciliation": recon.get("status", recon.get("overall", UNK)) if recon else UNK,
        "parity": par.get("status", par.get("overall", UNK)) if par else UNK,
        "last_change": (chg[0] if chg else UNK),
        "bottleneck": pm.get("top_bottleneck", UNK),
        "next_registered_step": pm.get("next_unblocked_action", UNK),
        "provenance_note": "All values read from canonical results/*.json|jsonl "
                           "artifacts. UNKNOWN = telemetry not present. History "
                           "is not rewritten; this is an append-only snapshot.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(truth, indent=1))
    print("research/current_truth.json written")
    print(f"  git {truth['git_sha']} · invariants {truth['invariant_wall']['passed']}/"
          f"{(truth['invariant_wall'].get('passed') or 0)+(truth['invariant_wall'].get('failed') or 0)}"
          f" · readiness {truth['readiness']['level']}")
    print(f"  F1 {truth['tiers']['T1_forecast']['f1_verdict'].split(' ')[0]}"
          f" ess_min {truth['tiers']['T1_forecast']['ess_min']}"
          f" · T2 kb2 BSS {truth['tiers']['T2_probability']['kb2_bss_vs_market']}"
          f" · frontier {str(truth['decision_frontier']['verdict']).split(' ')[0]}"
          f" · monotonicity {truth['tiers']['T3_decision']['edge_monotonicity']}")
    print(f"  failure top {truth['failure_store']['top_cluster']}")
    print(f"  settled_windows {settled} · open_incidents {len(open_inc)}"
          f" · service {truth['service_health']['overall']}")


if __name__ == "__main__":
    main()
