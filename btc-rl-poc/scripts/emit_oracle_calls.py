"""M14 — "The Oracle Call" (owner spec 2026-08-29): predict the
window's settlement 2-9 minutes IN, speaking ONLY when the claim can
honestly carry 80-90% certainty; otherwise say NO CALL.

Pre-registered rules (never fitted after the fact):
  * envelope: first decision row with 6 <= mins_left <= 13
    (= 2-9 minutes into the 15-minute window);
  * caller: kb2 (the pre-registered deliverable arm) — its claimed
    0.8-0.9 band verified 91% on the exploratory sample; kb5 logged
    as a shadow second caller for comparison, never merged;
  * call iff claimed confidence >= 0.80; else NO CALL (logged too —
    coverage is half the product);
  * prequential: rows are stamped pre-settle by the daemon; this
    emitter (10-min cron) only READS them, so every call is
    reconstructible and time-safe (leakage canaries guard the tape);
  * promise metric: selective accuracy of calls must be >= 0.80;
    report with Wilson 95% CI and coverage. The promise failing is a
    result, not a formatting problem.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CONF = 0.80
LO_MIN, HI_MIN = 6.0, 13.0
CALLERS = ("kb2", "kb5")

# M14-v2 (owner: "more frequency — 1 in 4"): caller kb5 at 0.72,
# which scored 81% at 47% coverage on the exploratory sample.
# SELECTION-EFFECT DISCIPLINE: that threshold was CHOSEN on the same
# sample, so v2's promise is judged prequentially ONLY on windows
# settling after V2_REGISTERED_TS. v1 keeps running unchanged.
V2_CALLER = "kb5"
V2_CONF = 0.72
V2_REGISTERED_TS = 1788053460     # 2026-08-29 ~18:11 PT

# M14-early (owner: "within the FIRST 2-3 minutes of the window"):
# envelope mins_left in [12, 13.5]; rule = kb2>=0.75, else kb9>=0.75
# (union scored 81% @ 22% coverage exploratory; kb2 alone 88% @ 14%).
# Same selection-effect discipline: judged only on windows settling
# after EARLY_REGISTERED_TS.
EARLY_LO, EARLY_HI = 12.0, 13.5
EARLY_RULES = (("kb2", 0.75), ("kb9", 0.75))
EARLY_REGISTERED_TS = 1788054900   # 2026-08-29 ~18:35 PT


def wilson(k, n):
    if not n:
        return None
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    now = int(time.time())
    rows = [json.loads(l) for l in
            (RES / "kalshi_binary_log.jsonl").open() if l.strip()]
    # first in-envelope decision row per (ticker, variant)
    first = {}
    for r in sorted(rows, key=lambda r: -(r.get("mins_left") or 0)):
        if r.get("mins_left") is None or r.get("p_up") is None:
            continue
        if not (LO_MIN <= r["mins_left"] <= HI_MIN):
            continue
        v = r.get("variant") or "kb"
        if v in CALLERS:
            first.setdefault((r["ticker"], v), r)

    out = {"generated_ts": now, "config": {
        "conf_threshold": CONF, "envelope_mins_left": [LO_MIN, HI_MIN],
        "primary_caller": "kb2", "shadow_caller": "kb5",
        "promise": "selective accuracy of calls >= 0.80"}}
    for v in CALLERS:
        wins = [r for (tk, vv), r in first.items() if vv == v]
        settled = [r for r in wins if r.get("actual") is not None]
        calls = [r for r in settled
                 if max(r["p_up"], 1 - r["p_up"]) >= CONF]
        hits = sum(1 for r in calls
                   if (r["p_up"] >= 0.5) == bool(r["actual"]))
        sel_acc = hits / len(calls) if calls else None
        out[v] = {
            "windows_seen": len(wins), "settled": len(settled),
            "calls": len(calls),
            "coverage": round(len(calls) / len(settled), 3)
            if settled else None,
            "hits": hits,
            "selective_accuracy": round(sel_acc, 3)
            if sel_acc is not None else None,
            "wilson_ci95": wilson(hits, len(calls)),
            "promise_met": (sel_acc >= 0.80) if sel_acc is not None
            else None,
            "no_calls": len(settled) - len(calls),
        }
    # ---- M14-v2: kb5@0.72, prequential from registration ------------
    v2_wins = [r for (tk, vv), r in first.items() if vv == V2_CALLER]
    v2_eval = [r for r in v2_wins if r.get("actual") is not None
               and (r.get("close_ts") or 0) >= V2_REGISTERED_TS]
    v2_calls = [r for r in v2_eval
                if max(r["p_up"], 1 - r["p_up"]) >= V2_CONF]
    v2_hits = sum(1 for r in v2_calls
                  if (r["p_up"] >= 0.5) == bool(r["actual"]))
    out["v2"] = {
        "caller": V2_CALLER, "conf_threshold": V2_CONF,
        "registered_ts": V2_REGISTERED_TS,
        "exploratory_basis": "81% acc @ 47% coverage, n=80 — "
        "threshold chosen on that sample, so it does NOT count "
        "toward the promise",
        "settled_since_registration": len(v2_eval),
        "calls": len(v2_calls),
        "coverage": round(len(v2_calls) / len(v2_eval), 3)
        if v2_eval else None,
        "hits": v2_hits,
        "selective_accuracy": round(v2_hits / len(v2_calls), 3)
        if v2_calls else None,
        "wilson_ci95": wilson(v2_hits, len(v2_calls)),
        "promise_met": (v2_hits / len(v2_calls) >= 0.80)
        if v2_calls else None,
    }
    # ---- M14-early: first-2-3-minutes caller ------------------------
    ef = {}
    for r in sorted(rows, key=lambda r: -(r.get("mins_left") or 0)):
        if r.get("mins_left") is None or r.get("p_up") is None:
            continue
        if not (EARLY_LO <= r["mins_left"] <= EARLY_HI):
            continue
        ef.setdefault((r["ticker"], r.get("variant") or "kb"), r)
    eby = {}
    for (tk, v), r in ef.items():
        eby.setdefault(tk, {})[v] = r

    def early_call(w):
        for arm, th in EARLY_RULES:
            r = w.get(arm)
            if r is None:
                continue
            if max(r["p_up"], 1 - r["p_up"]) >= th:
                return r["p_up"] >= 0.5, arm
        return None, None

    e_settled = {tk: w for tk, w in eby.items()
                 if any(x.get("actual") is not None for x in w.values())
                 and (max(x.get("close_ts") or 0
                          for x in w.values()) >= EARLY_REGISTERED_TS)}
    e_calls, e_hits = 0, 0
    for tk, w in e_settled.items():
        side, arm = early_call(w)
        if side is None:
            continue
        e_calls += 1
        outcome = bool(next(x for x in w.values()
                            if x.get("actual") is not None)["actual"])
        e_hits += 1 if side == outcome else 0
    out["early"] = {
        "envelope_mins_left": [EARLY_LO, EARLY_HI],
        "rules": [list(x) for x in EARLY_RULES],
        "registered_ts": EARLY_REGISTERED_TS,
        "exploratory_basis": "union 81% @ 22% cov (n=37); kb2 alone "
        "88% @ 14% — does not count toward the promise",
        "settled_since_registration": len(e_settled),
        "calls": e_calls, "hits": e_hits,
        "coverage": round(e_calls / len(e_settled), 3)
        if e_settled else None,
        "selective_accuracy": round(e_hits / e_calls, 3)
        if e_calls else None,
        "wilson_ci95": wilson(e_hits, e_calls),
        "promise_met": (e_hits / e_calls >= 0.80) if e_calls else None,
    }
    # ---- THE experiment: control kb2 vs treatment kb9, early ----
    out["early_ct"] = {}
    for role, arm in (("control", "kb2"), ("treatment", "kb9")):
        ws = [w[arm] for w in eby.values() if arm in w
              and w[arm].get("actual") is not None
              and (w[arm].get("close_ts") or 0) >= EARLY_REGISTERED_TS]
        cs = [r for r in ws
              if max(r["p_up"], 1 - r["p_up"]) >= 0.75]
        h = sum(1 for r in cs
                if (r["p_up"] >= 0.5) == bool(r["actual"]))
        out["early_ct"][role] = {
            "arm": arm, "threshold": 0.75, "settled": len(ws),
            "calls": len(cs), "hits": h,
            "selective_accuracy": round(h / len(cs), 3) if cs else None,
            "wilson_ci95": wilson(h, len(cs)),
            "coverage": round(len(cs) / len(ws), 3) if ws else None}
    # live pending call, if any (most recent unsettled window)
    pend = [r for (tk, vv), r in first.items()
            if vv == "kb2" and r.get("actual") is None]
    if pend:
        r = max(pend, key=lambda r: r.get("close_ts") or 0)
        c = max(r["p_up"], 1 - r["p_up"])
        out["live"] = {"ticker": r["ticker"],
                       "call": ("UP" if r["p_up"] >= 0.5 else "DOWN")
                       if c >= CONF else "NO CALL",
                       "claimed_conf": round(c, 3),
                       "mins_left_at_read": r["mins_left"]}
    (RES / "oracle_calls.json").write_text(json.dumps(out, indent=1))
    k2 = out.get("kb2", {})
    print(f"oracle_calls: kb2 {k2.get('calls')} calls / "
          f"{k2.get('settled')} settled · sel_acc "
          f"{k2.get('selective_accuracy')} CI {k2.get('wilson_ci95')} "
          f"· coverage {k2.get('coverage')} · promise_met "
          f"{k2.get('promise_met')}")


if __name__ == "__main__":
    main()
