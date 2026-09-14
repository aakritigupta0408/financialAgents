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
    {"id": "pt3", "name": "The Disciplined", "role": "TREATMENT", "log": "pt3_trades.jsonl",
     "strategy": "High-conviction (PT3_TAU 0.77)"},
    {"id": "pt6", "name": "The MLE", "role": "TREATMENT", "log": "pt6_trades.jsonl",
     "strategy": "MLE edge logit (shadow, EV>=10c)"},
    {"id": "kb", "name": "KB One-Shot", "role": "TREATMENT", "log": "kb_bets.jsonl",
     "strategy": "Per-window edge bet"},
    {"id": "pb", "name": "Conviction Book", "role": "TREATMENT", "log": "pb_bets.jsonl",
     "strategy": "kb5-gated entries"},
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
    """Current contract + p_mech/p_oracle/p_market. UNAVAILABLE when live contract
    state is not present (never fabricated)."""
    ct = _jload("current_truth.json", ROOT / "research") or {}
    frozen = _jload("oracle_frozen.json", ROOT / "research" / "oracle") or {}
    rh = _jload("brti_runtime_health.json") or {}
    runtime = (rh.get("migration") or {})
    return {
        "contract": "KXBTC15M",
        "official_brti": ct.get("current_brti", "UNAVAILABLE"),
        "official_target": ct.get("official_target", "UNAVAILABLE"),
        "p_mech": "UNAVAILABLE", "p_oracle": "UNAVAILABLE",
        "oracle_delta": "UNAVAILABLE", "p_market": "UNAVAILABLE",
        "oracle_state": "SEARCHING",
        "oracle_spec_hash": frozen.get("spec_hash"),
        "runtime_contract_truth": ("ACTIVE_EXACT_BRTI" if runtime.get("runtime_enabled")
                                   else "LEGACY / ACTIVATION_PENDING"),   # §12/§51
        "provenance": "OBSERVED where available; UNAVAILABLE not fabricated",
        "note": "live p_mech/p_oracle populate once the runtime oracle path is active "
                "(EXACT_BRTI_RUNTIME_ENABLED); not shown from worktree evidence alone",
    }


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
