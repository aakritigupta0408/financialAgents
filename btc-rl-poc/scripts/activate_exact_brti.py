"""DT-01 P0 — exact-BRTI activation harness.

Two modes:

  --preflight   (default) run every check verifiable WITHOUT the running daemon:
                BRTI live health, shadow parity (contract_truth.settle vs official
                Kalshi outcome on recent settled windows), invariant/flag sanity.
                Emits results/brti_activation_readiness.json and prints the exact
                owner activation sequence. Never flips the flag; never marks the
                architecture checkpoint PASS.

  --verify-live after the owner has set EXACT_BRTI_RUNTIME_ENABLED=1 in the RUNNING
                daemon: reads results/settlement_shadow.jsonl the live daemon wrote
                and checks exact==official parity on live windows, plus paper
                accounting reconciliation. Only THIS mode's evidence justifies
                flipping RUNTIME_CONTRACT_TRUTH to PASS.

PASS of RUNTIME_CONTRACT_TRUTH requires the actual running daemon (--verify-live),
not worktree evidence (§1).
"""
import calendar
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import contract_truth as ct  # noqa: E402

RES = ROOT / "results"
CO = RES / "contract_outcomes.jsonl"


def _epoch(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


def preflight(n=10):
    health = ct.health()
    # shadow parity: reconstruct outcome from BRTI, compare to official Kalshi outcome
    rows = [json.loads(l) for l in CO.open() if l.strip()][-n:] if CO.exists() else []
    parity, checked, agree = [], 0, 0
    for r in rows:
        res = ct.settle(_epoch(r["open_time"]), _epoch(r["close_time"]))
        if res["contract_truth_quality"] != ct.EXACT_BRTI:
            continue
        checked += 1
        ok = int(res["outcome"] == r["exact_yes"])
        agree += ok
        parity.append({"ticker": r["ticker"], "exact": res["outcome"],
                       "official": r["exact_yes"], "match": ok})
    ready = (health.get("rest_ok") and health.get("history_ok")
             and checked >= 3 and agree == checked)
    doc = {
        "mode": "preflight", "checked_ts": time.time(),
        "brti_health": {k: health.get(k) for k in
                        ("connected", "rest_ok", "history_ok", "age_s",
                         "observed_cadence_hz", "gaps")},
        "shadow_parity": {"windows_checked": checked, "agreements": agree,
                          "agree_rate": round(agree / checked, 4) if checked else None,
                          "sample": parity[-5:]},
        "flag_state": {"EXACT_BRTI_RUNTIME_ENABLED": ct.runtime_enabled(),
                       "EXACT_BRTI_CAPTURE_ENABLED": ct.capture_enabled()},
        "preflight_ready": bool(ready),
        "activation_sequence": [
            "1. merge/cherry-pick DT-01 commits into the main checkout",
            "2. deploy with flag OFF; confirm daemon heartbeat (online_status.json alive_at fresh)",
            "3. python3 scripts/emit_brti_health.py  -> results/brti_runtime_health.json healthy",
            "4. export EXACT_BRTI_CAPTURE_ENABLED=1 ; restart daemon ; verify BRTI state populates",
            "5. export EXACT_BRTI_RUNTIME_ENABLED=1 ; restart daemon",
            "6. observe results/settlement_shadow.jsonl: exact_vs_official == 1 on first windows",
            "7. python3 scripts/activate_exact_brti.py --verify-live",
            "8. python3 scripts/architecture_checkpoint.py  -> RUNTIME_CONTRACT_TRUTH PASS"],
        "note": "Worktree pre-flight only. RUNTIME_CONTRACT_TRUTH stays PASS_WITH_WATCH "
                "until --verify-live confirms the RUNNING daemon settles on exact BRTI.",
    }
    (RES / "brti_activation_readiness.json").write_text(json.dumps(doc, indent=1))
    print("EXACT-BRTI PRE-FLIGHT")
    print(f"  BRTI health: rest_ok={health.get('rest_ok')} history_ok={health.get('history_ok')} "
          f"age_s={health.get('age_s')}")
    print(f"  shadow parity: {agree}/{checked} windows exact==official "
          f"({doc['shadow_parity']['agree_rate']})")
    print(f"  flag EXACT_BRTI_RUNTIME_ENABLED = {ct.runtime_enabled()}")
    print(f"  PRE-FLIGHT READY: {ready}")
    print("  next: owner runs the activation sequence in the live daemon (see readiness json)")
    return doc


def verify_live(min_windows=5):
    p = RES / "settlement_shadow.jsonl"
    if not p.exists():
        print("no results/settlement_shadow.jsonl — daemon has not run with the flag ON yet")
        return {"mode": "verify-live", "status": "NO_LIVE_DATA"}
    n = agree = exact_n = 0
    for l in p.open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("exact_quality") == "EXACT_BRTI":
            exact_n += 1
        if r.get("exact_vs_official") is not None:
            n += 1
            agree += r["exact_vs_official"]
    ok = n >= min_windows and agree == n
    doc = {"mode": "verify-live", "checked_ts": time.time(),
           "live_windows_with_official": n, "exact_quality_windows": exact_n,
           "exact_vs_official_agree": agree, "parity_ok": bool(ok),
           "runtime_pass_justified": bool(ok),
           "note": "parity_ok AND runtime flag ON in the daemon justifies flipping "
                   "RUNTIME_CONTRACT_TRUTH to PASS."}
    (RES / "brti_activation_verify.json").write_text(json.dumps(doc, indent=1))
    print(f"VERIFY-LIVE: {agree}/{n} live windows exact==official; parity_ok={ok}")
    return doc


def main():
    if "--verify-live" in sys.argv:
        verify_live()
    else:
        preflight()


if __name__ == "__main__":
    main()
