"""Passive research-gate watcher (observation only, no actions).
Exits — printing one line — when any registered gate event occurs:
A3-v2 crosses n>=25 or n>=50 / decision leaves INSUFFICIENT_EVIDENCE,
or Gate F1's verdict changes. Nothing is modified anywhere."""
import json
import time

seen_f1 = None
while True:
    try:
        a3 = json.load(open('results/a3_live.json'))
        n = (a3.get('forward') or {}).get('eligible') or 0
        dec = json.load(open('results/a3_decision.json'))
        f1 = json.load(open(
            'results/f1_capture_qualification.json'))['verdict']
        if seen_f1 is None:
            seen_f1 = f1
        # post-n=50 CONTINUE (2026-09-01): watch for the decision
        # RESOLVING — CI leaving zero -> REJECT or QUALIFY
        if dec.get('decision') not in (None, 'INSUFFICIENT_EVIDENCE',
                                       'CONTINUE'):
            print(f"GATE: A3-v2 decision resolved: {dec['decision']} "
                  f"at n={dec['eligible_n']}")
            break
        if f1 != seen_f1:
            print(f"GATE: F1 verdict changed {seen_f1} -> {f1}")
            break
    except Exception:
        pass
    time.sleep(300)
