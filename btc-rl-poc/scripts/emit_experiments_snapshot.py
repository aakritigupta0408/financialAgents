"""§5-11 — PROFESSIONAL A/B PLATFORM (backend). Canonical experiment evaluator.

Statistical unit = market_window_id (§6). Primary trader metric = paired delta of
REALIZED_PAPER_EV_PER_ELIGIBLE_WINDOW (§7) with a moving-block bootstrap CI (owner:
btc_rl/economics.paired_delta). Reports raw observations vs windows vs paired
windows vs ESS separately (§6). Integrity checks (§10): eligibility/side/
contamination. Mechanism decomposition (§9) where data supports it. Never uses win
rate as the primary trader decision metric.

Each trader treatment is paired against the CONTROL (pt) on the intersection of
eligible windows. Writes results/experiments_snapshot.json.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import economics as E  # noqa: E402

RES = ROOT / "results"
CONTROL = {"id": "pt", "log": "pt_trades.jsonl", "name": "The $1K Desk"}
TREATMENTS = [
    {"id": "pt3", "log": "pt3_trades.jsonl", "name": "The Disciplined",
     "hypothesis": "A tighter conviction gate (0.77 vs 0.62) raises EV/eligible-window",
     "mechanism": "PT3_TAU 0.77"},
    {"id": "pt6", "log": "pt6_trades.jsonl", "name": "The MLE",
     "hypothesis": "An MLE edge model with EV>=10c improves selective EV",
     "mechanism": "pt6-mle edge logit (shadow)"},
]


def _rows(log):
    p = RES / log
    if not p.exists():
        return []
    out = []
    for l in p.open():
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def _per_window_pnl(rows):
    """window close_ts -> realized pnl_c for that window (sum if multiple entries).
    Untraded/skipped windows are absent (0 contributed at pairing time)."""
    w = {}
    for r in rows:
        if r.get("skipped") or r.get("pnl_c") is None or r.get("close_ts") is None:
            continue
        w[r["close_ts"]] = w.get(r["close_ts"], 0.0) + r["pnl_c"]
    return w


def _eligible_windows(rows):
    """All windows the trader SAW (traded or eligible-but-skipped) — the coverage
    denominator and the pairing universe."""
    return {r["close_ts"] for r in rows if r.get("close_ts") is not None}


def _ess(diffs, block=6):
    """Effective sample size with a crude block-correlation inflation: ESS = n / (1 +
    2*sum_k rho_k) approximated by the lag-1 autocorrelation scaled by block size."""
    n = len(diffs)
    if n < 3:
        return n
    mu = sum(diffs) / n
    var = sum((d - mu) ** 2 for d in diffs) / n or 1e-9
    ac1 = sum((diffs[i] - mu) * (diffs[i - 1] - mu) for i in range(1, n)) / (n * var)
    inflation = max(1.0, 1.0 + 2.0 * max(0.0, ac1) * (block - 1) / block)
    return int(n / inflation)


def _evaluate(treat):
    c_rows, t_rows = _rows(CONTROL["log"]), _rows(treat["log"])
    c_pnl, t_pnl = _per_window_pnl(c_rows), _per_window_pnl(t_rows)
    c_elig, t_elig = _eligible_windows(c_rows), _eligible_windows(t_rows)
    shared = sorted(c_elig & t_elig)                      # paired eligible windows
    # per-eligible-window pnl (untraded eligible window contributes 0) — §7 metric
    ctrl = [c_pnl.get(w, 0.0) for w in shared]
    trt = [t_pnl.get(w, 0.0) for w in shared]
    delta = E.paired_delta(ctrl, trt)
    diffs = [trt[i] - ctrl[i] for i in range(len(shared))]
    ess = _ess(diffs)
    # integrity (§10)
    integrity = {
        "eligibility_overlap": round(len(shared) / max(1, len(c_elig | t_elig)), 3),
        "control_only_windows": len(c_elig - t_elig),
        "treatment_only_windows": len(t_elig - c_elig),
        "shadow_treatment": treat["id"] == "pt6",
        "note": "unit=market_window_id; per-eligible-window pnl (untraded=0)"}
    # verdict / boundary (§11): CI vs 0 with a simple promote/reject/continue rule
    if delta is None:
        verdict, state = "INSUFFICIENT_N", "REGISTERED"
    elif delta["ci95"][0] > 0:
        verdict, state = "TREATMENT_WINS", "PROMOTE_CANDIDATE"
    elif delta["ci95"][1] < 0:
        verdict, state = "TREATMENT_LOSES", "REJECT_CANDIDATE"
    else:
        verdict, state = "NO_SIGNIFICANT_DIFFERENCE", "CONTINUE"
    c_summary = E.summarize_trades(c_rows, 100000, len(c_elig))
    t_summary = E.summarize_trades(t_rows, 100000, len(t_elig))
    return {
        "experiment_id": f"trader-{treat['id']}-vs-pt",
        "hypothesis": treat["hypothesis"],
        "control": CONTROL["id"], "treatment": treat["id"],
        "shadow": treat["id"] == "pt6",
        "unit": "market_window_id",
        "estimand": "paired_delta(realized_paper_ev_per_eligible_window_c)",
        "primary_metric": "REALIZED_PAPER_EV_PER_ELIGIBLE_WINDOW (paired)",
        "sample_sizes": {
            "raw_observations_control": len(c_rows),
            "raw_observations_treatment": len(t_rows),
            "windows_control": len(c_elig), "windows_treatment": len(t_elig),
            "paired_windows": len(shared), "effective_n": ess},
        "effect": delta,
        "control_summary": {"ev_per_eligible_c": c_summary["realized_ev_per_eligible_c"],
                            "coverage": c_summary["coverage"]},
        "treatment_summary": {"ev_per_eligible_c": t_summary["realized_ev_per_eligible_c"],
                              "coverage": t_summary["coverage"]},
        "mechanism_decomposition": {
            "registered_mechanism": treat["mechanism"],
            "net_treatment_ev_per_eligible_c": (None if delta is None else delta["paired_delta"]),
            "gross_price_improvement": "UNAVAILABLE",
            "missed_winner_cost": "UNAVAILABLE",
            "fees_delta": "UNAVAILABLE",
            "adverse_selection": "UNAVAILABLE",
            "note": "component decomposition requires per-window counterfactual fills; "
                    "UNAVAILABLE not fabricated (§9)"},
        "integrity": integrity,
        "guardrails": {"max_drawdown_c_treatment": t_summary["max_drawdown_c"],
                       "bad_entry_rate_treatment": t_summary["bad_entry_rate"]},
        "sequential": {"boundary": "CI95 excludes 0", "state": state},
        "verdict": verdict,
        "timeline": ["REGISTERED", "SMOKE", "SHADOW" if treat["id"] == "pt6" else "LIVE",
                     state],
        "slices_note": "any per-segment breakdown is EXPLORATORY unless pre-registered",
    }


def main():
    exps = [_evaluate(t) for t in TREATMENTS]
    from btc_rl import contract_truth as _ct
    settle_prov = ("EXACT_BRTI" if _ct.runtime_enabled() else "PROXY_LEGACY_COINBASE_CANDLE")
    snap = {"schema_version": "experiments-1", "generated_at": __import__("time").time(),
            "metric_owner": "btc_rl/economics.paired_delta",
            "unit_default": "market_window_id",
            "primary_trader_metric": "REALIZED_PAPER_EV_PER_ELIGIBLE_WINDOW (paired)",
            "evidence_class": "ONLINE_PAPER_RETROSPECTIVE",
            "settlement_provenance": settle_prov,
            "caveats": [
                "Paper ledgers settle on " + settle_prov + " until DT-01 is activated in "
                "the live daemon; treat EV as provisional until exact-BRTI settlement.",
                "SHADOW treatments (pt6) stake 0 — their 'wins' are counterfactual "
                "selective abstention (avoiding control losses), not realized cash.",
                "Retrospective online-paper evidence, NOT prospective proof (§27)."],
            "note": "win rate is NOT a primary trader decision metric (§7). Offline and "
                    "online kept separate; this is ONLINE paper evidence.",
            "experiments": exps}
    (RES / "experiments_snapshot.json").write_text(json.dumps(snap, indent=1))
    print(f"experiments_snapshot: {len(exps)} trader A/Bs (control=pt)")
    for e in exps:
        d = e["effect"]
        print(f"  {e['experiment_id']}: paired_n={e['sample_sizes']['paired_windows']} "
              f"ess={e['sample_sizes']['effective_n']} "
              f"delta={d['paired_delta'] if d else None} "
              f"ci={d['ci95'] if d else None} -> {e['verdict']}")


if __name__ == "__main__":
    main()
