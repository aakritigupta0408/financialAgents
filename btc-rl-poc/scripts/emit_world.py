"""Emit results/world.json — the data layer behind the Quant Universe
pages (owner's two-worlds redesign brief, 2026-08-29).

Feeds: the System Clock (every scheduled thing, with REAL last-run
evidence), Agent HQ (who acts, on what trigger, with what authority),
per-region health for the world map, and the cost ledger.

Honesty rules, enforced here:
  * every last-run timestamp comes from an actual artifact (a cron log
    mtime, a results-file mtime, the daemon heartbeat) — never assumed;
  * schedule rows are cross-checked against the LIVE crontab: a job
    this file describes that is absent from crontab is marked missing;
  * costs we do not meter are emitted as amount=null with
    "not metered" — a $0 that is genuinely $0 (public/demo APIs,
    self-hosted compute) is stated as $0 with its basis, because $0 is
    information; an invented number would be a lie with decimals.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402


def _mtime(p: Path):
    try:
        return int(p.stat().st_mtime)
    except OSError:
        return None


def _age_min(ts, now):
    return round((now - ts) / 60, 1) if ts else None


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _retrain_ts(status):
    """Unix ts of the daemon's last gated retrain, from the ISO stamp
    it publishes (per-arm val_mse before/after lives beside it)."""
    at = (status.get("last_retrain") or {}).get("at")
    if not at:
        return None
    from datetime import datetime
    try:
        return int(datetime.fromisoformat(at).timestamp())
    except ValueError:
        return None


def main():
    now = int(time.time())
    cron = subprocess.run(["crontab", "-l"], capture_output=True,
                          text=True).stdout
    status = _json(RES / "online_status.json") or {}
    alive = status.get("alive_at")
    audit = _json(RES / "audit_report.json") or {}
    dboard = _json(RES / "decision_board.json") or {}
    manifest = _json(RES / "site_manifest.json") or {}

    logs = {
        "publish": Path("/tmp/btc_publish.log"),
        "watchdog": Path("/tmp/btc_watchdog_cron.log"),
        "audit": Path("/tmp/btc_audit.log"),
        "introspect": Path("/tmp/btc_introspect.log"),
    }

    def clock_row(every, what, owner, source, last_ts, cron_tag=None,
                  detail=None):
        in_cron = (cron_tag in cron) if cron_tag else None
        age = _age_min(last_ts, now)
        health = "green"
        if cron_tag and not in_cron:
            health = "red"
        elif last_ts is None:
            health = "amber"
        return {"every": every, "what": what, "owner": owner,
                "source": source, "detail": detail,
                "last_run_ts": last_ts, "age_min": age,
                "in_crontab": in_cron, "health": health}

    clock = [
        clock_row("30 s", "market poll + pipeline loop (candles, book, "
                  "trades, Kalshi quotes → features → arm inference)",
                  "daemon", f"btc_rl/online.py POLL_SECONDS="
                  f"{O.POLL_SECONDS}", int(alive) if alive else None,
                  detail="heartbeat = online_status.alive_at"),
        clock_row("1 min", "publish site + data to gh-pages",
                  "cron · publisher", "scripts/publish_dashboard.py",
                  _mtime(logs["publish"]), "publish_dashboard.py"),
        clock_row("5 min", "watchdog — restart daemon if heartbeat "
                  "stale >5 min", "cron · watchdog",
                  "scripts/watchdog.py", _mtime(logs["watchdog"]),
                  "watchdog.py",
                  detail="authority: RESTART the daemon process only. "
                  "Logs only when it acts — a days-old log means "
                  "days of health, not neglect"),
        clock_row("10 min", "audit chain (~17 independent steps: audit "
                  "→ boards → A3 → reconcile → canaries → readiness)",
                  "cron · auditor",
                  "scripts/audit_chain.py", _mtime(logs["audit"]),
                  "audit_chain.py"),
        clock_row("1 h", "gated retraining of neural arms (candidate "
                  "vs holdout MSE; keep or revert)", "daemon",
                  f"btc_rl/online.py RETRAIN_EVERY={O.RETRAIN_EVERY}s",
                  _retrain_ts(status),
                  detail="evidence = online_status.last_retrain "
                  "(per-arm val_mse before/after + kept/reverted)"),
        clock_row("1 h at :20", "model introspection → analyst data "
                  "(weights, trace, representativeness, oracle)",
                  "cron · analyst", "tests/introspect_model_internals"
                  ".py", _mtime(RES / "model_internals.json"),
                  "introspect_model_internals.py"),
        clock_row("on settlement", "score window, update online "
                  "learners + calibrators, append treatment ledger",
                  "daemon", "btc_rl/online.py settle path",
                  _mtime(RES / "treatments.jsonl"),
                  detail="evidence = treatments.jsonl append"),
        clock_row("15 s", "site pages re-fetch published JSON",
                  "browser", "client-side poll", None,
                  detail="stateless; no backend work"),
    ]

    agents = [
        {"name": "The Daemon", "role": "pipeline executor",
         "trigger": f"continuous, {O.POLL_SECONDS}s loop",
         "authority": "READ market + WRITE ledgers + paper-trade the "
         "pre-registered policies",
         "cannot": "change any policy constant, promote a treatment, "
         "touch real money (production key quarantined)",
         "last_ts": int(alive) if alive else None,
         "evidence": "online_status.json heartbeat"},
        {"name": "Watchdog", "role": "reliability",
         "trigger": "cron, every 5 min",
         "authority": "restart the daemon when the heartbeat is stale",
         "cannot": "modify code, data, or policy",
         "last_ts": _mtime(logs["watchdog"]),
         "evidence": "watchdog_log.jsonl + cron log"},
        {"name": "Auditor", "role": "evaluation integrity",
         "trigger": "cron, every 10 min",
         "authority": "recompute all published metrics from raw "
         "ledgers (dual implementation vs the daemon)",
         "cannot": "alter ledgers; it only reads and re-derives",
         "last_ts": _mtime(logs["audit"]),
         "evidence": "audit_report.json"},
        {"name": "Fable — The Analyst", "role": "research analyst",
         "trigger": "hourly, after introspection refresh; plus "
         "session work with the owner",
         "authority": "READ everything + PROPOSE (analyst notes, "
         "treatment proposals, retirement proposals)",
         "cannot": "promote a treatment, modify production policy, "
         "spend money — every live-money decision terminates with "
         "the human (PROGRAM.md §3)",
         "last_ts": _mtime(RES / "model_internals.json"),
         "evidence": "commentary.jsonl (dated, kept even when wrong)"},
        {"name": "Publisher", "role": "communication",
         "trigger": "cron, every 1 min",
         "authority": "copy site + results snapshots to gh-pages",
         "cannot": "modify source data",
         "last_ts": _mtime(logs["publish"]),
         "evidence": "publish log"},
        {"name": "Builder agents", "role": "engineering",
         "trigger": "session-based, launched by the owner's directives",
         "authority": "PROPOSE + BUILD site/scripts changes, verified "
         "by screenshot + parity fixtures before commit",
         "cannot": "run unattended or alter trading policy",
         "last_ts": None,
         "evidence": "git history + DECISIONS.md"},
        {"name": "The Owner (human)", "role": "governance",
         "trigger": "—",
         "authority": "the only authority that can promote, retire, "
         "or change live policy",
         "cannot": "—", "last_ts": None,
         "evidence": "DECISIONS.md 'decided by' column"},
    ]

    tick_age = (now - alive) / 60 if alive else None
    audit_age = _age_min(audit.get("generated_ts"), now)
    integ = (dboard.get("integrity") or {})
    regions = {
        "market_ocean": {
            "label": "Market Ocean — external feeds",
            "status": "green" if tick_age is not None and tick_age < 5
            else "red",
            "why": f"daemon heartbeat {tick_age:.1f} min old"
            if tick_age is not None else "no heartbeat"},
        "forecast_country": {
            "label": "Forecast Country — tier-1 price arms",
            "status": "green" if audit_age is not None
            and audit_age < 30 else "amber",
            "why": f"audit refreshed {audit_age} min ago; last "
            f"retrain {_age_min(_retrain_ts(status), now)} min ago"
            if audit_age is not None else "no recent audit"},
        "probability_city": {
            "label": "Probability City — kb arms",
            "status": "green" if audit_age is not None
            and audit_age < 30 else "amber",
            "why": f"audit refreshed {audit_age} min ago"
            if audit_age is not None else "no recent audit"},
        "decision_hq": {
            "label": "Decision HQ — gates + experiments",
            "status": "green" if integ.get("health") == "green"
            else "red",
            "why": f"treatment ledger {integ.get('age_min')} min old, "
            f"{integ.get('rows')} paired windows"},
        "execution_port": {
            "label": "Execution Port — quotes, fills, fees",
            "status": "amber",
            "why": "structurally healthy; the measured 4.9¢/$1 "
            "execution gap is the desk's biggest open leak"},
        "ledger_valley": {
            "label": "Ledger Valley — append-only records",
            "status": "green",
            "why": "all ledgers append-only; publisher shipping"},
    }

    du = subprocess.run(["du", "-sk", str(RES)], capture_output=True,
                        text=True).stdout.split("\t")[0]
    storage_mb = round(int(du) / 1024, 1) if du.strip().isdigit() \
        else None
    costs = [
        {"item": "Coinbase market data", "amount": 0.0,
         "basis": "public API, no key, no fee"},
        {"item": "Kalshi market data + paper orders", "amount": 0.0,
         "basis": "demo environment; production key quarantined"},
        {"item": "alternative.me Fear & Greed", "amount": 0.0,
         "basis": "public API"},
        {"item": "Compute (daemon + retrains + crons)", "amount": 0.0,
         "basis": "self-hosted on the owner's Mac (caffeinate keeps "
         "it awake); $0 marginal, electricity not metered"},
        {"item": f"Storage ({storage_mb} MB results/)", "amount": 0.0,
         "basis": "local disk + GitHub Pages hosting, free tier"},
        {"item": "Publishing (GitHub Pages)", "amount": 0.0,
         "basis": "free tier"},
        {"item": "LLM — analyst notes + builder agents",
         "amount": None,
         "basis": "not metered — runs inside the owner's Claude "
         "subscription; per-run token metering is a queued "
         "workstream, and until it exists this row stays null "
         "rather than invented"},
    ]

    doc = {"generated_ts": now, "clock": clock, "agents": agents,
           "regions": regions, "costs": costs,
           "facts": {
               "poll_seconds": O.POLL_SECONDS,
               "retrain_every_s": O.RETRAIN_EVERY,
               "daemon_alive_at": alive,
               "windows_scored": integ.get("rows"),
               "config_source": manifest.get("config_source")}}
    (RES / "world.json").write_text(json.dumps(doc, indent=1))
    print(f"world.json: {len(clock)} clock rows, {len(agents)} agents, "
          f"{len(regions)} regions, {len(costs)} cost rows")


if __name__ == "__main__":
    main()
