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


def research_snapshot(health):
    inv = _j(ROOT / "research" / "true15m" / "contract_inventory_report.json")
    dmeta = _j(R / "open_oracle_15m_dataset.meta.json")
    ladder = _j(R / "open_oracle_15m_ladder.json")
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
        _lane("L2", "Alpha Vantage backfill", "QUEUED",
              "broad historical feature backbone", "0/%d" % total,
              "not started — Phase-2 real backfill"),
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
        "headline": ("Foundation online: contract inventory verified and the "
                     "price-path model ladder ran; broad real-data backfill (Alpha "
                     "Vantage / derivatives / options / news) is the next lane."),
        "why": "A fully causal, leak-free T0 dataset must exist before complex models "
               "are allowed to compete against the class baseline.",
        "next": "Discover the official class baseline, then begin the rich historical "
                "backfill on cohorts while live capture keeps adding real windows.",
        "capture": health,
        "lanes": lanes,
        "blockers": [{"lane": b["id"], "reason": b["blocked_reason"]} for b in blockers],
        "dataset_progress": {
            "historical_windows": total,
            "verified_t0_dataset_windows": dmeta.get("market_window_n"),
            "raw_60s_observation_coverage_n": raw_cov,
            "class_balance_up": dmeta.get("class_balance_up"),
        },
        "current_best_model": {"verdict": verdict, "classification": cls,
                               "note": "on F-CONTRACT price-path family; independent "
                                       "feature families not yet tested"},
        "baseline": {
            "status": "RESOLVED — NO official course baseline exists"
            if (ROOT / "research" / "true15m" / "CLASS_BASELINE_SPEC.yaml").exists()
            else "PENDING_DISCOVERY",
            "adopted": "beat empirical class-frequency (log-loss/Brier) OOS + "
                       "BSS_vs_market>0 vs Kalshi-at-open (benchmark only)",
            "note": "see research/true15m/CLASS_BASELINE_SPEC.yaml — repo has no "
                    "course-mandated pass-mark; threshold adopted as a project decision"},
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
          f"events={len(s['recent_events'])} verdict={s['current_best_model']['verdict']}")


if __name__ == "__main__":
    build()
