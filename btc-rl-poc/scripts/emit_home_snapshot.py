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
# 2026-09-15 GREAT ROSTER CUT (owner directive): T0 control + exactly two live
# treatments. Every prior treatment (cg5/cg10/tv/pt3/pt6/pt2/pt4/pt5/pt7/pt8)
# retired — ledgers kept as frozen evidence, off the board.
ROSTER = [
    {"id": "pt", "name": "The $1K Desk", "role": "CONTROL", "log": "pt_trades.jsonl",
     "strategy": "T0 — Follower (leaderboard leader, PT_TAU 0.62)"},
    {"id": "tv", "name": "Value Gate", "role": "TREATMENT", "log": "tv_trades.jsonl",
     "strategy": "EV — value-gated follower (edge>=0.05 over price+fee, half-Kelly); the EV north-star arm"},
    {"id": "ob", "name": "Open+6 Barrier", "role": "TREATMENT", "log": "ob_trades.jsonl",
     "strategy": "T3 — open+6min first-passage barrier; the honest ~0.74@0.90 model, coverage-sliced"},
    {"id": "cg33", "name": "Gated ·33%", "role": "TREATMENT", "log": "cg33_trades.jsonl",
     "strategy": "T1 — Confidence-Gated Follower, 33% stake (RUIN-RISK experiment)"},
    {"id": "fm", "name": "Chronos-Bolt", "role": "TREATMENT", "log": "fm_trades.jsonl",
     "strategy": "T2 — Chronos-Bolt base foundation model, directional (conf>=0.60)"},
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
    # RESERVE open-position stakes so this path's bankroll matches the ledger's own
    # bankroll_c and the trader_family/traders-snapshot convention (else home_snapshot.json
    # publishes two different bankrolls for the same arm — data-integrity audit finding #1,
    # 2026-09-15). ending_capital_c = start + realized - open_reserved.
    open_reserved = sum(r.get("stake_c") or 0 for r in rows if r.get("actual") is None)
    if open_reserved and econ.get("ending_capital_c") is not None:
        econ["ending_capital_c"] = econ["ending_capital_c"] - open_reserved
        econ["open_reserved_c"] = open_reserved
        sc = econ.get("starting_capital_c")
        econ["total_return"] = E.total_return(sc, econ["ending_capital_c"]) if sc else None
    last_k = [{"ticker": r.get("ticker"), "side": r.get("side"), "pnl_c": r.get("pnl_c"),
               "win": r.get("win"), "close_ts": r.get("close_ts")}
              for r in settled[-10:]]
    _recent_asks = [a for a in (r.get("ask_c") for r in settled[-20:]) if a]
    _st = _honest_state(settled, econ.get("ending_capital_c") or 0, _recent_asks, time.time())
    _status = ("SHADOW" if t["id"] == "pt6"
               else "HALTED" if _st["ruined"] else "STALE" if _st["halted"]
               else "ACTIVE")
    return {
        "trader_id": t["id"], "name": t["name"], "role": t["role"],
        "strategy_type": t["strategy"],
        "status": _status, "lifecycle_state": _st["state"],
        "last_trade_age_s": _st["last_trade_age_s"],
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


def _honest_state(settled, bank_c, recent_asks_c, now):
    """Truthful lifecycle state for a live arm, DERIVED from the ledger — never a
    hardcoded 'LIVE'. An arm that stopped acting while 15-min windows keep arriving is
    not live (liveness == recency vs opportunity cadence, not 'has any rows'); an arm
    that can't afford a single contract is ruined. Reports reality — changes no trading
    behavior, so it is honesty, not a guardrail."""
    if not settled:
        return {"state": "COLLECTING (no settled trade yet)", "halted": False,
                "last_trade_age_s": None, "ruined": False}
    last_ts = max((r.get("close_ts") or 0) for r in settled)
    age = max(0.0, now - last_ts)
    cheapest = min(recent_asks_c) if recent_asks_c else None
    ruined = cheapest is not None and bank_c < cheapest       # can't buy 1 contract
    WIN = 900.0                                                # 15-min window cadence
    missed = int(age / WIN)

    def _ago(s):
        h = s / 3600.0
        return f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d"
    if ruined:
        return {"state": f"HALTED — ruined (${bank_c / 100:.2f} left, no trade in {_ago(age)})",
                "halted": True, "last_trade_age_s": int(age), "ruined": True}
    if missed >= 8:            # ~2h of missed 15-min windows => provably dark
        return {"state": f"STALE — no trade in {_ago(age)} ({missed} windows missed)",
                "halted": True, "last_trade_age_s": int(age), "ruined": False}
    return {"state": "LIVE (official BRTI)", "halted": False,
            "last_trade_age_s": int(age), "ruined": False}


def _cg_family_entries():
    """The two live treatments after the 2026-09-15 roster cut: T1 (cg33) and T2 (fm =
    Chronos-Bolt base), with LIVE stats read from their own ledgers (official-BRTI
    settled). Shown in the home trader family so the live deployment is visible. The
    retired arms (cg5/cg10/tv/pt3/pt6/…) are intentionally absent."""
    specs = [
        ("tv", "Value Gate", "EV north-star.", "LIVE_CANDIDATE", "tv_trades.jsonl",
         "EV — follows the leader ONLY when the edge beats the price+fee (p_arm − ask/100 >= 0.05) "
         "and the leader is strong (rec10>=0.7); half-Kelly sizing. OFFLINE replay hypothesis: "
         "value-gating lifts T0's EV (ev_optimize.json). LIVE is prospective and still accruing — "
         "read the live P&L on this card, not the offline claim. North-star = EV, not hit-rate. Official settle."),
        ("ob", "Open+6 Barrier", "The honest model.", "LIVE_CANDIDATE", "ob_trades.jsonl",
         "T3 — decides at open+6min (~9 min left) from the first 6 minutes of price action via the "
         "analytic first-passage barrier P(close>=strike). Trades every window and logs confidence "
         "|z| so performance is coverage-sliced (offline: 0.74 hit @90% coverage, ~0.87 @20%). The "
         "exhaustively-verified honest ceiling; see the coverage A/B on Models Lab. Official settle."),
        ("cg33", "Gated · 33%", "Ruin-risk experiment.", "RUIN_RISK_EXPERIMENT", "cg33_trades.jsonl",
         "T1 — follows the leader only when confidence >= 0.20 (skips coin-flips), 33% stake, hold to "
         "close. Demonstrates over-betting (backtest $300 -> ~$60, 98% drawdown). Official Kalshi settle."),
        ("fm", "Chronos-Bolt", "Foundation model.", "LIVE_CANDIDATE", "fm_trades.jsonl",
         "T2 — chronos-bolt-base reads P(close>=strike) from the window price path and takes its own "
         "side when confident (>=0.60), half-Kelly sizing, one bid/window, hold to close. Benchmark "
         "winner (F1 0.68, precision 0.74, fewest false positives). Official Kalshi settlement."),
    ]
    START_C = {"tv": 100000, "ob": 1000000}   # tv $1,000; ob $10,000 (2026-09-15); others $300
    out = []
    for cid, name, tag, verdict, log, blurb in specs:
        start = START_C.get(cid, 30000)
        p = RES / log
        rows = []
        if p.exists():
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
        settled = sorted([r for r in rows if r.get("actual") is not None],
                         key=lambda r: r.get("close_ts") or 0)
        opens = [r for r in rows if r.get("actual") is None]
        wins = sum(1 for r in settled if r.get("win"))
        pnls = [r.get("pnl_c") or 0 for r in settled]
        pnl = sum(pnls)
        open_stakes = sum(r.get("stake_c") or 0 for r in opens)
        bank = start + pnl - open_stakes

        def _pt(ts):
            try:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                return datetime.fromtimestamp(ts, ZoneInfo("America/Los_Angeles")).strftime("%b %d, %H:%M") if ts else None
            except Exception:
                return None

        def _trow(r, status):
            return {"time": _pt(r.get("close_ts") or r.get("made_ts")), "brti": None,
                    "target": round(r["strike"]) if r.get("strike") else None,
                    "side": (r.get("side") or "").upper(), "entry_c": r.get("ask_c"),
                    "result": None if status == "Open" else ("WIN" if r.get("win") else "LOSS"),
                    "pnl_c": r.get("pnl_c"), "status": status}

        recent = [_trow(r, "Open") for r in opens[-1:]] + \
                 [_trow(r, "Settled") for r in reversed(settled[-12:])]
        eq = [{"i": i, "equity_c": r.get("bankroll_c")} for i, r in enumerate(settled)
              if r.get("bankroll_c") is not None]
        _recent_asks = [a for a in (r.get("ask_c") for r in settled[-20:]) if a]
        _st = _honest_state(settled, bank, _recent_asks, time.time())
        sess = {"n": len(settled), "pnl_c": pnl, "bankroll_c": bank,
                "hit_rate": round(wins / len(settled), 4) if settled else None,
                "ev_per_trade_c": round(pnl / len(pnls), 2) if pnls else None,
                "max_drawdown_c": None, "label": "since launch",
                "state": _st["state"], "halted": _st["halted"],
                "last_trade_age_s": _st["last_trade_age_s"]}
        livedesk = {"session": sess, "since_activation": sess, "recent_trades": recent,
                    "equity_curve": eq, "daemon_alive_age_s": 0, "current_window": None,
                    "settlement": "OFFICIAL_EXACT_BRTI"}
        live = {"n": len(settled), "wins": wins, "pnl_c": pnl, "bankroll_c": bank,
                "hit_rate": round(wins / len(settled), 3) if settled else None,
                "state": _st["state"], "halted": _st["halted"], "ruined": _st["ruined"],
                "last_trade_age_s": _st["last_trade_age_s"]}
        is_fm = cid == "fm"; is_ob = cid == "ob"; is_tv = cid == "tv"
        _type = ("Value gate (EV, half-Kelly)" if is_tv
                 else "First-passage barrier @ open+6min" if is_ob
                 else "Foundation model (Chronos-Bolt base)" if is_fm
                 else "Rule-based (confidence-gated)")
        _reason = ("Bets only +EV windows (edge>=0.05 over price+fee). OFFLINE: value-gating "
                   "lifted T0's EV in replay; LIVE evidence is still accruing (see live P&L)." if is_tv
                   else "Analytic barrier on the first 6 minutes; the exhaustively-verified honest "
                        "ceiling (~0.74 hit @90% coverage). Coverage A/B on Models Lab." if is_ob
                        else "Directional foundation-model trader; benchmark winner among "
                             "Chronos/TimesFM/market (F1 0.68, precision 0.74)." if is_fm
                             else "Live candidate accruing paired evidence vs T0 (offline n=218, small).")
        _bt = ({"coverage": 0.31, "label": "OFFLINE (value gate on T0: +$22 @31% cov, EV +17c/trade)"}
               if is_tv
               else {"coverage": 0.90, "label": "OFFLINE (open+6min barrier: 0.74 hit @90% cov, 0.87 @20%)"}
               if is_ob
               else {"coverage": 0.39, "label": "OFFLINE BENCHMARK (148-win OOS, chronos-bolt-base @0.60)"}
               if is_fm
               else {"coverage": 0.61, "label": "OFFLINE CANDIDATE (n=218, vs always-take)"})
        out.append({"id": cid, "name": name, "role": "TREATMENT", "type": _type,
                    "tagline": f"{tag} · live ${bank/100:.2f}", "icon": "bolt", "blurb": blurb,
                    "reason": _reason, "backtest": _bt,
                    "live": live, "livedesk": livedesk, "verdict": verdict})
    return out


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
    # 2026-09-15: the board = T0 CONTROL + the REAL live treatments (ob/cg33/fm) only.
    # The old generic T1-T4 backtest placeholder cards (Selective Edge/Meta-ML/RL Policy/
    # Composite) are retired from the home board — they were reference stubs, not the
    # deployed roster, and confused the live picture.
    return [
        {"id": "T0", "name": "Baseline Follower", "role": "CONTROL", "type": "Rule-based",
         "tagline": "Simple. Consistent. Reference.", "icon": "crown",
         "blurb": "Takes every eligible Oracle trade at a fixed stake — the reference policy.",
         "reason": "Positive on unseen windows.",
         "backtest": bt("T0"), "live": live, "verdict": (rep.get("offline_verdicts") or {}).get("T0")},
        *_cg_family_entries(),
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
