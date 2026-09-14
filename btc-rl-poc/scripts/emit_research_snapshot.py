"""Research OS — live-capture health + research live snapshot (directive §4, §29).

Pure presentation/join adapter for the private Research Narrator. Reads the
canonical artifacts (event ledger, daemon heartbeat, contract inventory, dataset
meta, model ladder) and publishes:
  results/live_capture_health.json    — always-live collector heartbeat (§4)
  results/research_live_snapshot.json  — lanes, jobs, recent events, dataset
                                         progress, best model, blockers (§29-32)
Computes no science; renders what the research engine already recorded.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

R = ROOT / "results"
HEALTH = R / "live_capture_health.json"
SNAP = R / "research_live_snapshot.json"


def _j(p, d=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return d if d is not None else {}


def _tail_jsonl(p, n=1):
    if not p.exists():
        return []
    out = []
    for l in p.open():
        l = l.strip()
        if l:
            out.append(l)
    rows = []
    for l in out[-n:]:
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    return rows


def capture_health():
    st = _j(R / "online_status.json")
    now = time.time()
    age = round(now - st.get("alive_at", 0), 1) if st.get("alive_at") else None
    alive = age is not None and age < 180
    kb = _tail_jsonl(R / "kalshi_binary_log.jsonl", 1)
    cur = kb[0].get("ticker") if kb else None
    inv = _j(R / "research" / "true15m" / "contract_inventory_report.json") \
        if (R / "research").exists() else _j(ROOT / "research" / "true15m" / "contract_inventory_report.json")
    doc = {
        "schema_version": "live-capture-health-1", "generated_at": now,
        "status": "LIVE" if alive else "STALE",
        "heartbeat_age_s": age,
        "current_contract": cur,
        "contracts_total_historical": inv.get("total_windows"),
        "note": "Always-live collector: btc_rl.online freezes each KXBTC15M window at "
                "T0 and records exact BRTI settlement at T1, independent of all research jobs.",
    }
    HEALTH.write_text(json.dumps(doc, indent=1))
    return doc


def _lane(id_, name, status, detail, progress=None, blocked=None):
    return {"id": id_, "name": name, "status": status, "detail": detail,
            "progress": progress, "blocked_reason": blocked}


def _av_lane(total):
    """L2 status derived from the most advanced AV mini-gate result on disk."""
    base = ROOT / "research" / "true15m"
    minis = sorted(base.glob("av_backfill_mini*.json"))
    if not minis:
        return _lane("L2", "Alpha Vantage backfill", "QUEUED",
                     "broad cross-asset/equity/news backbone", f"0/{total}",
                     "not started")
    d = _j(minis[-1])
    cov = d.get("coverage_matrix", {})
    covs = ", ".join(f"{s} {v['n']}/{d.get('cohort_windows')}" for s, v in cov.items())
    return _lane("L2", "Alpha Vantage backfill",
                 "PASS" if d.get("pit_pass") else "FAIL",
                 f"MINI-{d.get('mini')} · {covs}", f"{d.get('cohort_windows')}/{total}",
                 "cohort-only so far; full backfill pending (BTC dynamics come from BRTI, "
                 "not AV crypto)")


def research_snapshot(health):
    inv = _j(ROOT / "research" / "true15m" / "contract_inventory_report.json")
    dmeta = _j(R / "open_oracle_15m_dataset.meta.json")
    ladder = _j(R / "open_oracle_15m_ladder.json")
    brtiC = _j(ROOT / "research" / "true15m" / "brti_state_coverage.json")
    total = inv.get("total_windows") or 0
    raw_cov = (inv.get("raw_60s_observation_coverage") or {}).get("n") or 0
    verdict = ladder.get("verdict", "—")
    cls = ladder.get("classification", "—")

    lanes = [
        _lane("L0", "Live capture", "RUNNING" if health["status"] == "LIVE" else "STALE",
              f"current {health.get('current_contract') or '—'}", None,
              None if health["status"] == "LIVE" else "daemon heartbeat stale"),
        _lane("L1", "Contract / BRTI inventory", "COMPLETE" if total else "QUEUED",
              f"{total} settled windows, labels 100%", f"{total}/{total}"),
        _av_lane(total),
        _lane("L3", "Exchange market data", "PARTIAL",
              "cross-venue PIT layer exists for recent windows", f"{raw_cov}/{total}",
              "historical coverage limited to captured windows"),
        _lane("L4", "Derivatives / options", "PARTIAL",
              "OKX derivatives adapter live; options capture exists", None,
              "historical backfill not built"),
        _lane("L5", "News / macro", "PARTIAL", "AV news capture exists", None,
              "historical vintage backfill not built"),
        _lane("L6", "Feature factory", "PARTIAL",
              "F-CONTRACT price-path family built; broad universe pending", None,
              "awaiting rich-source backfill"),
        _lane("L7", "Data integrity", "PASS" if dmeta.get("post_open_information_violations") == 0 else "FAIL",
              "PIT + one-per-window green on the built dataset", None),
        _lane("L8", "Model research", "COMPLETE" if verdict != "—" else "QUEUED",
              f"baseline ladder M0-M4; verdict {verdict}", None),
        _lane("L9", "Falsification", "PASS" if ladder.get("falsification") else "QUEUED",
              "label-shuffle placebo run", None),
        _lane("L10", "UI / Research Narrator", "RUNNING", "private live feed online", None),
    ]

    events = EV.recent(limit=40)
    disc = next((e for e in reversed(events) if e.get("event_type") == "DISCOVERY"), None)
    blockers = [l for l in lanes if l["blocked_reason"]]

    doc = {
        "schema_version": "research-live-snapshot-1", "generated_at": time.time(),
        "headline": (
            "I am determining what the 6,337 historical contracts genuinely knew at T0. "
            + ("BRTI coverage audited (coarse 2h: {c}/{n}; fine 5m: {f}). ".format(
                c=(brtiC.get("core_cohorts", {}).get("COARSE_BTC_STATE (>=2h @15m res)", {}) or {}).get("n", "?"),
                n=brtiC.get("windows_audited", total),
                f=(brtiC.get("core_cohorts", {}).get("FINE_BTC_STATE (real 5m sub-minute)", {}) or {}).get("n", "?"))
               if brtiC else "BRTI coverage audit running. ")
            + "Alpha Vantage cross-asset history and derivatives backfill in parallel. "
              "No model competes until feature coverage and the full leakage gate are frozen."),
        "why": "The genuine BTC-state core is only as large as the real pre-T0 BRTI lookback. "
               "We measure it before building features, so nothing assumes coverage it lacks.",
        "next": "Build COARSE_BTC_STATE features on the ~6,189 large core + FINE features on "
                "the ~363 sub-minute block; expand AV cross-asset; then coverage matrix, "
                "TRUE15M_DATASET_V1, full integrity gate, frozen split — then models.",
        "capture": health,
        "lanes": lanes,
        "blockers": [{"lane": b["id"], "reason": b["blocked_reason"]} for b in blockers],
        "dataset_progress": {
            "historical_windows": total,
            "official_label_coverage": "100%",
            "raw_brti_reconstruction": f"{round(100*raw_cov/total,1) if total else 0}%",
            "raw_brti_reconstruction_n": raw_cov,
            "verified_t0_dataset_windows": dmeta.get("market_window_n"),
            "class_balance_up": dmeta.get("class_balance_up"),
            "brti_state_coverage": ({h: v.get("any_n") for h, v in
                                     (brtiC.get("per_horizon") or {}).items()}
                                    if brtiC else None),
            "core_cohorts": brtiC.get("core_cohorts") if brtiC else None,
            "note": "Official outcomes/targets from Kalshi settled records cover all "
                    f"{total} windows; exact 60s-BRTI PATHS reconstructed for {raw_cov} "
                    "only. We do NOT claim path reconstruction for all windows.",
        },
        "current_best_model": {
            "f_contract_result": verdict,             # F-CONTRACT price-path family only
            "f_contract_classification": cls,
            "scope": "contract / pre-open price-path information only",
            "global_true15m_conclusion": "NOT_YET_DETERMINED",
            "note": "The richer feature universe (momentum/volume/volatility/RSI-MACD/"
                    "cross-asset/microstructure/derivatives/options/news/macro/regimes) "
                    "is not yet built or tested."},
        "baseline": {
            "OFFICIAL_COURSE_BASELINE": "NOT_FOUND",
            "INTERNAL_RESEARCH_BASELINE": "RESOLVED",
            "name": "INTERNAL_ADOPTED_BASELINE_V1",
            "primary_comparison": "empirical class-frequency baseline",
            "primary_metrics": ["log_loss", "brier"],
            "external_benchmark": "Kalshi-at-open",
            "market_relative_requirement": "BSS_vs_Kalshi_at_open > 0",
            "note": "INTERNAL adopted benchmark, NOT a course baseline — repo has no "
                    "official one. See research/true15m/INTERNAL_ADOPTED_BASELINE_V1.yaml"},
        "latest_discovery": {"title": disc.get("title"), "narrative": disc.get("narrative")} if disc else None,
        "recent_events": events,           # bounded list; UI appends by seq cursor (§30)
        "max_seq": max((e.get("seq", 0) for e in events), default=0),
        "system_health": {"events_total": len(EV.recent(limit=100000)),
                          "capture": health["status"]},
    }
    SNAP.write_text(json.dumps(doc, indent=1))
    return doc


def build():
    h = capture_health()
    s = research_snapshot(h)
    print(f"research_snapshot: capture={h['status']} lanes={len(s['lanes'])} "
          f"events={len(s['recent_events'])} "
          f"f_contract={s['current_best_model']['f_contract_result']} "
          f"global={s['current_best_model']['global_true15m_conclusion']}")


if __name__ == "__main__":
    build()
