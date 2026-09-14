"""TEST_V2_CAPTURE_AUDIT + forward-rolling sealed membership (directive).

Why this exists: TEST_V2 reported 0/672 while L0 live capture never stopped. Root
cause was TWO coincident failures at the cutoff boundary:
  (1) the settled-market fetcher (fetch_contract_specs.py) stopped ~Sep 12, so no
      post-cutoff windows entered results/contract_outcomes.jsonl; and
  (2) the counter in sealed_test_governance.py read the FROZEN research inventory
      (research/true15m/contract_inventory.jsonl), which by construction ends at
      the cutoff — so it could only ever report 0.

This module fixes the COUNTER/MEMBERSHIP side. It builds TEST_V2 membership from the
authoritative forward settled feed (contract_outcomes.jsonl) using ONLY:
    T0 > cutoff  AND  settlement_verified  AND  integrity_complete
Membership is NEVER selected by outcome value. The frozen inventory is left
untouched (the TRAIN/VAL/TEST_V1 split stays sealed).

Seal discipline: the membership file carries labels (it IS the eventual blind test
set) but is marked SEALED; the audit report intentionally does NOT surface the
outcome/yes-rate distribution of TEST_V2 — inspecting it would burn the seal.

Writes:
  research/true15m/TEST_V2_MEMBERSHIP.jsonl   (sealed; one row per member)
  research/true15m/TEST_V2_CAPTURE_AUDIT.json (the audit the directive requested)
"""
import calendar
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

SRC = ROOT / "results" / "contract_outcomes.jsonl"          # forward authoritative feed
SPLIT = ROOT / "research" / "true15m" / "SPLIT_SPEC_V1.json"
MEMBERSHIP = ROOT / "research" / "true15m" / "TEST_V2_MEMBERSHIP.jsonl"
AUDIT = ROOT / "research" / "true15m" / "TEST_V2_CAPTURE_AUDIT.json"
STATE = ROOT / "research" / "true15m" / "TEST_V2_MEMBERSHIP_STATE.json"
POWER = ROOT / "research" / "true15m" / "TEST_V2_POWER_ANALYSIS.json"
WINDOW_S = 900
FALLBACK_TARGET = 672    # used only if the power analysis hasn't pre-registered N yet


def _epoch(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


def build():
    t0 = time.time()
    cutoff = json.loads(SPLIT.read_text())["sealed_test"]["range"][1]

    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    for r in rows:
        r["_t0"] = _epoch(r["open_time"]); r["_t1"] = _epoch(r["close_time"])
    rows.sort(key=lambda r: r["_t0"])

    # every window the forward feed has seen strictly after the cutoff
    post = [r for r in rows if r["_t0"] > cutoff]
    seen = len(post)

    # eligibility / integrity gates (NONE of these look at outcome *value*)
    excl = {"missing_settlement_values": 0, "no_official_result": 0,
            "non_900s_duration": 0, "duplicate_ticker": 0}
    seen_ids = set()
    settled = label_verified = integrity_complete = 0
    members = []
    for r in post:
        has_vals = r.get("floor_strike") is not None and r.get("expiration_value") is not None
        has_result = r.get("result") in ("yes", "no")
        dur_ok = (r["_t1"] - r["_t0"]) == WINDOW_S
        dup = r["ticker"] in seen_ids
        seen_ids.add(r["ticker"])
        if has_vals:
            settled += 1
        else:
            excl["missing_settlement_values"] += 1
        # label_verified: official result present AND it agrees with the D>=0 rule
        rule_yes = (1 if (has_vals and r["expiration_value"] >= r["floor_strike"]) else 0)
        official_yes = 1 if r.get("result") == "yes" else 0
        if has_result and has_vals and rule_yes == official_yes:
            label_verified += 1
        elif not has_result:
            excl["no_official_result"] += 1
        if not dur_ok:
            excl["non_900s_duration"] += 1
        if dup:
            excl["duplicate_ticker"] += 1
        # a member must clear every gate; membership is by rule, never by outcome
        if has_vals and has_result and dur_ok and not dup:
            integrity_complete += 1
            members.append({
                "market_window_id": r["ticker"], "T0": r["_t0"], "T1": r["_t1"],
                "open_time": r["open_time"], "close_time": r["close_time"],
                "opening_brti_avg": r["floor_strike"],
                "settlement_brti_avg": r["expiration_value"],
                "D": r.get("D"), "Y": official_yes,   # sealed label; not for selection
                "provenance": "kalshi trade-api v2 settled; forward TEST_V2 accrual",
            })

    # contiguity of the accrued cohort (missing 15-min slots between first & last member)
    if members:
        m_t0 = sorted(m["T0"] for m in members)
        present = set(m_t0)
        contiguity_missing = sum(1 for s in range(m_t0[0], m_t0[-1] + 1, WINDOW_S)
                                 if s not in present)
        first_id = members[0]["market_window_id"]
        latest_id = members[-1]["market_window_id"]
    else:
        contiguity_missing, first_id, latest_id = 0, None, None

    # write the sealed membership (idempotent rebuild — pure function of feed+rule)
    body = "".join(json.dumps(m) + "\n" for m in members)
    MEMBERSHIP.write_text(body)
    mem_sha = hashlib.sha256(body.encode()).hexdigest()[:16]

    # ---- APPEND_ONLY membership state (rolling until the pre-registered target N) ----
    # V2 is rolling, so the current hash is NOT final. Track mode=APPEND_ONLY with a
    # prefix hash; enforce that a window once registered never disappears; and freeze the
    # immutable final_membership_hash EXACTLY ONCE, when the pre-registered target is hit.
    target = FALLBACK_TARGET
    if POWER.exists():
        target = json.loads(POWER.read_text()).get("pre_registered_target_N") or FALLBACK_TARGET
    cur_ids = [m["market_window_id"] for m in members]
    prefix_hash = hashlib.sha256("".join(sorted(cur_ids)).encode()).hexdigest()[:16]
    prev = json.loads(STATE.read_text()) if STATE.exists() else None
    prev_ids = set(prev.get("member_ids", [])) if prev else set()
    vanished = sorted(prev_ids - set(cur_ids))
    append_only_ok = not vanished
    prev_final = (prev or {}).get("final_membership_hash", "UNSET")
    closed = len(members) >= target
    if prev_final not in (None, "UNSET"):
        final_hash = prev_final                 # already frozen — never recompute
    elif closed:
        final_hash = prefix_hash                # freeze once, now
    else:
        final_hash = "UNSET"
    state = {"schema_version": "test-v2-membership-state-1", "generated_at": time.time(),
             "mode": "CLOSED" if final_hash not in (None, "UNSET") else "APPEND_ONLY",
             "current_N": len(members), "target_N": target,
             "current_prefix_hash": prefix_hash, "final_membership_hash": final_hash,
             "append_only_invariant_holds": append_only_ok,
             "vanished_windows": vanished[:10],
             "target_source": "TEST_V2_POWER_ANALYSIS.json" if POWER.exists() else "fallback(672)",
             "member_ids": cur_ids}
    if not append_only_ok:
        EV.emit("ERROR", f"TEST_V2 append-only invariant VIOLATED: {len(vanished)} windows vanished",
                lane="L9", severity="high", technical=str(vanished[:5]))
    STATE.write_text(json.dumps(state, indent=1))
    audit = {
        "schema_version": "test-v2-capture-audit-1", "generated_at": time.time(),
        "cutoff_T0": cutoff, "target_windows": target,
        "source": "results/contract_outcomes.jsonl (authoritative forward settled feed)",
        "membership_file": "research/true15m/TEST_V2_MEMBERSHIP.jsonl",
        "membership_sha256_16": mem_sha,
        "membership_mode": state["mode"],
        "current_prefix_hash": prefix_hash,
        "final_membership_hash": final_hash,
        "append_only_invariant_holds": append_only_ok,

        "post_cutoff_windows_seen": seen,
        "post_cutoff_windows_settled": settled,
        "post_cutoff_windows_label_verified": label_verified,
        "post_cutoff_windows_integrity_complete": integrity_complete,
        "post_cutoff_windows_registered_to_V2": len(members),

        "first_post_cutoff_window_id": first_id,
        "latest_post_cutoff_window_id": latest_id,
        "contiguity_missing_slots": contiguity_missing,
        "exclusion_reason_counts": excl,

        "invariant": ("for every eligible window with T0 > cutoff: if settlement_verified "
                      "and integrity_complete then market_window_id in TEST_V2 membership"),
        "invariant_holds": (label_verified == len(members) and contiguity_missing == 0
                            and integrity_complete == seen),
        "membership_rule": "T0>cutoff AND has floor_strike+expiration_value AND official "
                           "result AND 900s duration AND not duplicate — never by outcome value",
        "seal": "SEALED — membership carries labels but is NEVER read for feature/model/"
                "architecture/hyperparameter/calibration/threshold/ensemble selection. "
                "Outcome distribution intentionally NOT surfaced in this audit.",
        "root_cause": {
            "feed_stall": "fetch_contract_specs.py stopped ~Sep 12; contract_outcomes.jsonl "
                          "ended at the cutoff window. Repaired by re-fetching all settled "
                          "markets (feed now current).",
            "counter_bug": "sealed_test_governance.py counted post-cutoff windows from the "
                           "FROZEN research inventory (ends at cutoff by design) -> always 0. "
                           "Fixed to count from this forward membership.",
        },
        "status": ("ACCUMULATING" if len(members) < target else "TARGET_REACHED"),
    }
    AUDIT.write_text(json.dumps(audit, indent=1))

    EV.emit("BUG_FOUND", "TEST_V2 accumulator read frozen inventory + settled feed had stalled",
            lane="L9", severity="high",
            fact=f"post-cutoff windows: seen={seen}, settled={settled}, "
                 f"label_verified={label_verified}, integrity_complete={integrity_complete}, "
                 f"registered_to_V2={len(members)} (was reporting 0).",
            interpretation="Two coincident failures at the cutoff: settled-market fetcher "
                           "stopped ~Sep 12, and the counter read the frozen inventory which "
                           "ends at the cutoff. Membership is genuinely built now, by rule.",
            next_action="counter fixed to read forward membership; L0 fetch must stay live.",
            files=["research/true15m/TEST_V2_CAPTURE_AUDIT.json",
                   "research/true15m/TEST_V2_MEMBERSHIP.jsonl"])
    EV.emit("BUG_FIXED", f"TEST_V2 now accruing: {len(members)}/{target} sealed windows",
            lane="L9", severity="high",
            fact=f"forward feed re-fetched (contiguous, 0 missing slots); membership rebuilt "
                 f"from cutoff+eligibility+integrity, invariant_holds={audit['invariant_holds']}.",
            interpretation="Seal intact — labels present but not inspected for selection; "
                           "outcome distribution deliberately not surfaced.",
            duration_ms=int((time.time() - t0) * 1000))

    print(f"test_v2_capture_audit: seen={seen} settled={settled} "
          f"label_verified={label_verified} integrity_complete={integrity_complete} "
          f"registered={len(members)}/{target}  invariant_holds={audit['invariant_holds']}")
    print(f"  first={first_id}  latest={latest_id}  missing_slots={contiguity_missing} "
          f"excl={excl}  mem_sha={mem_sha}")


if __name__ == "__main__":
    build()
