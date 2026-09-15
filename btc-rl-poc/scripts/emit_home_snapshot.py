"""DT-07/DT-08 §10 — HOME snapshot. ONE compact backend response the HOME page
renders without any browser-side science. Every number comes from the canonical
owners (btc_rl/metrics, btc_rl/economics); the frontend only formats and plots.

Carries schema_version, generated_at, source_versions (§9) and per-block provenance
(§5). Offline and online evidence are separate objects (§7). Trader roster capped at
1 control + 4 treatments (§11); unused slots render EMPTY, never fabricated.

Writes results/home_snapshot.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import economics as E  # noqa: E402

RES = ROOT / "results"
SCHEMA_VERSION = "home-1"
METRIC_DEFS_VERSION = "econ-1/prob-1"

# roster: 1 control + up to 4 treatments (§11)
ROSTER = [
    {"id": "pt", "name": "The $1K Desk", "role": "CONTROL", "log": "pt_trades.jsonl",
     "strategy": "Follower (leaderboard leader, PT_TAU 0.62)"},
    # Confidence-Gated Follower treatments (2026-09-15) — same policy (follow the
    # leader only when confidence >= 0.20, skip coin-flips, hold to close), $300 each,
    # differing ONLY in stake. Replaced the prior featured treatments (pt3/kb/pb).
    {"id": "cg5", "name": "Gated ·5%", "role": "TREATMENT", "log": "cg5_trades.jsonl",
     "strategy": "Confidence-Gated Follower, 5% stake (conf>=0.20)"},
    {"id": "cg10", "name": "Gated ·10%", "role": "TREATMENT", "log": "cg10_trades.jsonl",
     "strategy": "Confidence-Gated Follower, 10% stake (conf>=0.20)"},
    {"id": "cg33", "name": "Gated ·33%", "role": "TREATMENT", "log": "cg33_trades.jsonl",
     "strategy": "Confidence-Gated Follower, 33% stake (RUIN-RISK experiment)"},
    {"id": "pt6", "name": "The MLE", "role": "TREATMENT", "log": "pt6_trades.jsonl",
     "strategy": "MLE edge logit (shadow, EV>=10c)"},
]


def _jload(name, root=RES):
    p = root / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _rows(name):
    p = RES / name
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
            continue
    return out


def _eligible_universe(logs):
    """Distinct window close_ts the desk saw across active trader logs (§ coverage
    denominator). DERIVED provenance."""
    seen = set()
    for lg in logs:
        for r in _rows(lg):
            if r.get("close_ts") is not None:
                seen.add(r["close_ts"])
    return len(seen)


def _trader_summary(t, n_eligible):
    rows = _rows(t["log"])
    pnl_key = "pnl_c"
    starting = None
    settled = [r for r in rows if r.get(pnl_key) is not None and not r.get("skipped")]
    if settled:
        # starting capital = ending bankroll - realized pnl (robust to schema)
        ending_bankroll = next((r.get("bankroll_c") for r in reversed(settled)
                                if r.get("bankroll_c") is not None), None)
        realized = sum(r[pnl_key] for r in settled)
        starting = (ending_bankroll - realized) if ending_bankroll is not None else 100000
    econ = E.summarize_trades(rows, starting or 100000, n_eligible)
    last_k = [{"ticker": r.get("ticker"), "side": r.get("side"), "pnl_c": r.get("pnl_c"),
               "win": r.get("win"), "close_ts": r.get("close_ts")}
              for r in settled[-10:]]
    return {
        "trader_id": t["id"], "name": t["name"], "role": t["role"],
        "strategy_type": t["strategy"],
        "status": "SHADOW" if t["id"] == "pt6" else "ACTIVE",
        "online_summary": econ,                       # §7 online/prospective paper
        "offline_summary": None,                       # replay not yet wired per-trader
        "latest_action": (settled[-1].get("ticker") if settled else None),
        "last_k_trades": last_k,
        "provenance": "OBSERVED (ledger) -> DERIVED (btc_rl.economics)",
    }


def _oracle_strip():
    """Current live contract + Oracle belief. Populated from the running desk's
    online_status.json (BRTI composite + current KXBTC15M market). p_mech/p_oracle
    are the Oracle's CURRENT belief for this contract (computed from decision-time
    state) — shown for the live window; this does NOT claim the runtime settles on
    exact BRTI (runtime_contract_truth stays honest). UNAVAILABLE only when the desk
    has no fresh contract; never fabricated."""
    import math
    frozen = _jload("oracle_frozen.json", ROOT / "research" / "oracle") or {}
    rh = _jload("brti_runtime_health.json") or {}
    runtime = (rh.get("migration") or {})
    st = _jload("online_status.json") or {}
    U = "UNAVAILABLE"
    # Authoritative current window from EXACT BRTI (independent of daemon lag): the
    # window is [last 15-min boundary, next boundary]; target = 60s BRTI avg at open.
    brti = target = tte_s = p_mech = p_oracle = odelta = contract = None
    _traj = []
    try:
        from btc_rl import contract_truth as CT
        from btc_rl import prospective_capture as PC
        now = int(time.time())
        close_ts = ((now // 900) + 1) * 900
        open_ts = close_ts - 900
        cs = CT.contract_state(open_ts, close_ts, now)
        # exact target from BRTI 60s open-average; current LEVEL from the live values
        # endpoint (rolling 1h of 1s samples, always fresh) — robust between windows;
        # falls back to contract_state / daemon composite only if the live call fails.
        cur = None
        _payload = []
        try:
            import data.adapters.brti as _b
            pr = _b.probe()
            _payload = (pr.get("sample") or {}).get("data", {}).get("payload", [])
            if _payload:
                cur = float(max(_payload, key=lambda x: x["time"])["value"])
        except Exception:
            cur = None
        cur = cur or cs.get("current_brti") or ((st.get("brti") or {}) or {}).get("price")
        # target: exact 60s open-average; at rollover the reconstruction lags a few s,
        # so fall back to the daemon's floor_strike (== the official target from Kalshi).
        _pm = st.get("pm") or {}
        _kb = (st.get("kalshi_binary") or {}).get("last") or {}
        tgt = cs.get("official_target") or _pm.get("strike") or _kb.get("strike")
        if tgt and cur and cs.get("time_remaining_s", 0) > 1:
            brti = round(cur, 2)
            target = round(tgt, 2)
            tte_s = cs["time_remaining_s"]
            p_mech = round(PC.p_mech(cur, tgt, tte_s), 4)
            p_oracle = round(PC.p_oracle(cur, tgt, tte_s), 4)
        # real p_mech(t) trajectory for the elapsed part of THIS window (§6) — no fabrication
        try:
            if tgt and _payload:
                pts = sorted(((float(x["value"]), x["time"] / 1000.0) for x in _payload
                              if open_ts <= x["time"] / 1000.0 <= now),
                             key=lambda p: p[1])          # chronological
                if len(pts) > 4:
                    step = max(1, len(pts) // 40)
                    for v, t in pts[::step]:
                        rem = close_ts - t
                        if rem > 1:
                            _traj.append({"t_left_min": round(rem / 60.0, 2),
                                          "p_mech": round(PC.p_mech(v, tgt, rem), 4)})
        except Exception:
            pass
    except Exception:
        pass
    # market probability for the current window from the desk (if its ticker matches)
    pm = st.get("pm") or {}
    kb = (st.get("kalshi_binary") or {}).get("last") or {}
    p_market = pm.get("model_p_up") if False else kb.get("mkt_p_up")
    if p_mech is None:                                  # BRTI path failed -> daemon fallback
        brti = ((st.get("brti") or {}) or {}).get("price")
        target = pm.get("strike") or kb.get("strike")
        contract = pm.get("ticker") or kb.get("ticker")
    else:
        contract = pm.get("ticker") or kb.get("ticker") or "KXBTC15M (current window)"
    if p_oracle is not None and p_market is not None:
        odelta = round(p_oracle - p_market, 4)
    fresh = bool(brti and target)
    return {
        "contract": contract,
        # absolute epochs so the client can tick the countdown every second
        "close_ts": ((int(time.time()) // 900) + 1) * 900,
        "now_ts": int(time.time()),
        "official_brti": round(brti, 2) if (fresh and brti) else U,
        "official_target": round(target, 2) if (fresh and target) else U,
        "distance": round(brti - target, 2) if (fresh and brti and target) else U,
        "time_remaining_min": round(tte_s / 60.0, 1) if (fresh and tte_s) else U,
        "trajectory": _traj,
        "p_mech": p_mech if p_mech is not None else U,
        "p_oracle": p_oracle if p_oracle is not None else U,
        "oracle_delta": odelta if odelta is not None else U,
        "p_market": p_market if (fresh and p_market is not None) else U,
        # directional conviction of the LIVE p_oracle (it recomputes each tick as
        # price/time move — this is a lean, never a frozen "lock")
        "oracle_state": (U if p_oracle is None else
                         "LEANING UP" if p_oracle > 0.65 else
                         "LEANING DOWN" if p_oracle < 0.35 else "TOSS-UP"),
        "oracle_spec_hash": frozen.get("spec_hash"),
        "benchmark_provenance": ("EXACT CF-BRTI (current window computed live)"
                                 if p_mech is not None else
                                 "BRTI 4-venue composite (daemon fallback)"),
        "runtime_contract_truth": ("ACTIVE_EXACT_BRTI" if runtime.get("runtime_enabled")
                                   else "LEGACY / ACTIVATION_PENDING"),   # §12/§51
        "provenance": "p_mech/p_oracle from exact BRTI current-window state; "
                      "p_market from live desk; UNAVAILABLE not fabricated",
    }


def _trader_family():
    """Track F/G — the current active trader family T0-T4 with model identity and
    BACKTEST vs LIVE separated (never blended). All numbers backend-computed."""
    rep = _jload("replay_result.json", ROOT / "research" / "replay") or {}
    eff = _jload("t1_effect_result.json", ROOT / "research" / "replay") or {}
    t2 = _jload("t2_meta_ml_result.json", ROOT / "research" / "traders") or {}
    t3 = _jload("t3_rl_result.json", ROOT / "research" / "traders") or {}
    hold = rep.get("final_holdout", {}) or {}

    def bt(name):
        h = hold.get(name) or {}
        return {"ev_per_eligible_c": h.get("ev_per_eligible_c"),
                "ev_per_trade_c": h.get("ev_per_trade_c"),
                "coverage": h.get("coverage"), "accuracy": h.get("accuracy"),
                "total_pnl_c": h.get("total_pnl_c"), "max_drawdown_c": h.get("max_drawdown_c"),
                "label": "HISTORICAL UNSEEN HOLDOUT (93 windows)"}
    live = {"n": 0, "state": "COLLECTING (pending DT-01 live activation)"}
    return [
        {"id": "T0", "name": "Baseline Follower", "role": "CONTROL", "type": "Rule-based",
         "tagline": "Simple. Consistent. Reference.", "icon": "crown",
         "blurb": "Takes every eligible Oracle trade at a fixed stake — the reference policy.",
         "reason": "Positive on unseen windows.",
         "backtest": bt("T0"), "live": live, "verdict": (rep.get("offline_verdicts") or {}).get("T0")},
        {"id": "T1", "name": "Selective Edge", "role": "FORMAL_TREATMENT", "type": "Rule-based",
         "tagline": "Quality over quantity.", "icon": "bolt",
         "blurb": "Trades only when the Oracle disagrees strongly with the market (≥15pp).",
         "reason": "Higher EV per trade, lower drawdown than the baseline.",
         "backtest": bt("T1"), "live": live,
         "verdict": (rep.get("offline_verdicts") or {}).get("T1"),
         "holdout_effect": {"paired_delta_c": eff.get("paired_delta_ev_per_eligible_c"),
                            "ci95": eff.get("moving_block_bootstrap_95ci_c"),
                            "status": eff.get("verdict")}},
        {"id": "T2", "name": "Meta-ML", "role": "SHADOW", "type": "Machine-learned",
         "tagline": "Learn to take, not just predict.", "icon": "brain",
         "blurb": "Learns whether a given Oracle trade is worth taking, and skips the rest.",
         "reason": "No learnable signal beyond the Oracle on this data.",
         "backtest": {"best_holdout_auc": t2.get("best_holdout_auc")},
         "live": live, "verdict": t2.get("verdict")},
        {"id": "T3", "name": "RL Policy", "role": "SHADOW", "type": "Reinforcement learning",
         "tagline": "Sequential decisions.", "icon": "chart",
         "blurb": "Learns how much capital to size on the Oracle's trade (the side is fixed).",
         "reason": "Only learned to bet big — no real sizing skill vs baselines.",
         "backtest": (t3.get("risk_adjusted") or {}),
         "live": live, "verdict": t3.get("verdict")},
        {"id": "T4", "name": "Composite", "role": "NOT_QUALIFIED", "type": "Combination",
         "tagline": "Stronger together.", "icon": "layers",
         "blurb": "Combines the best components — unlocked only after each qualifies on its own.",
         "reason": "Waiting on component qualification.",
         "backtest": None, "live": None, "verdict": "NOT_QUALIFIED"},
    ]


def main():
    ct = _jload("current_truth.json", ROOT / "research") or {}
    logs = [t["log"] for t in ROSTER]
    n_eligible = _eligible_universe(logs)
    traders = [_trader_summary(t, n_eligible) for t in ROSTER]
    # pad to 5 slots with EMPTY (§11) — roster already 5, but keep the contract explicit
    while len(traders) < 5:
        traders.append({"trader_id": None, "role": "EMPTY_TREATMENT_SLOT"})

    snap = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFS_VERSION,
        "generated_at": time.time(),
        "source_versions": {
            "git_sha": ct.get("git_sha"),
            "current_truth_utc": ct.get("generated_utc"),
            "economics_owner": "btc_rl/economics.py",
            "probability_owner": "btc_rl/metrics.py"},
        "paper_banner": "PAPER / SIMULATION — REAL MONEY DISABLED — RESEARCH, NOT ADVICE",
        "system_health": ct.get("service_health", "UNKNOWN"),
        "oracle_strip": _oracle_strip(),
        "eligible_windows_universe": {"value": n_eligible, "provenance": "DERIVED"},
        "traders": traders,
        "trader_family": _trader_family(),          # Track F/G: T0-T4 backtest vs live
        "active_experiment": ct.get("experiments"),
        "incident_watch": {"open_incidents": ct.get("open_incidents"),
                           "dt01_activation": "PASS_WITH_WATCH (runtime flag OFF)"},
        "bottleneck": ct.get("bottleneck"),
        "note": "HOME renders this verbatim; no scientific/economic formula runs in the "
                "browser (enforced by tests/test_no_frontend_science.py).",
    }
    (RES / "home_snapshot.json").write_text(json.dumps(snap, indent=1))
    print(f"home_snapshot: {len([t for t in traders if t.get('trader_id')])} traders, "
          f"{n_eligible} eligible windows, health={snap['system_health']}")
    for t in traders:
        if t.get("trader_id"):
            o = t["online_summary"]
            print(f"  {t['trader_id']:4s} {t['role']:9s} pnl_c={o['realized_pnl_c']} "
                  f"ev/trade={o['ev_per_trade_c']} cov={o['coverage']} wr={o['win_rate']}")


if __name__ == "__main__":
    main()
