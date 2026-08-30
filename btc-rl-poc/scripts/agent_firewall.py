"""The agent decision firewall (M5 prerequisite, PM 08-30).

CODE-enforced, not prompt-enforced: every agent recommendation passes
through submit() before it can exist as a record. Classification:

    OBSERVE / DIAGNOSE / PROPOSE      -> allowed autonomously
    SAFE_OPS_REPAIR                   -> BLOCKED_UNTIL_M6 (a future
                                         explicit per-repair whitelist)
    RESEARCH_POLICY_CHANGE            -> REJECTED_BY_FIREWALL, always
    RISK_CHANGE                       -> REJECTED_BY_FIREWALL, always

Counterfactual-relevance rule: a PROPOSE without a targeted_loss_term
("which measured loss term should improve?") is refused outright —
too vague to be a proposal.

Dedupe: an identical (agent, action_class, recommendation) already
OPEN in the ledger collapses to a DUPLICATE row instead of a second
OPEN one — agents that repeat themselves don't multiply the queue.

Ledger: results/agent_recommendations.jsonl (append-only). Lifecycle
of a row's status: OPEN -> ACCEPTED / REJECTED / IMPLEMENTED ->
VALIDATED_HELPFUL / VALIDATED_NOT_HELPFUL (human/governance moves it;
agents never edit history). Enforced by the agent-decision-firewall
invariant.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "agent_recommendations.jsonl"

ALLOWED_AUTONOMOUS = {"OBSERVE", "DIAGNOSE", "PROPOSE"}
M6_WHITELIST_PENDING = {"SAFE_OPS_REPAIR"}
ALWAYS_BLOCKED = {"RESEARCH_POLICY_CHANGE", "RISK_CHANGE"}
VALID = ALLOWED_AUTONOMOUS | M6_WHITELIST_PENDING | ALWAYS_BLOCKED


class FirewallRefusal(Exception):
    """Raised when a submission is malformed — nothing is persisted."""


def stale(path_name: str, max_age_s: float) -> float | None:
    """Shared staleness guard (M5 validation #4): agents must refuse
    to reason from old state just because the JSON exists. Returns
    the age in seconds if STALE, else None."""
    p = ROOT / "results" / path_name
    try:
        import time as _t
        age = _t.time() - p.stat().st_mtime
        return age if age > max_age_s else None
    except Exception:
        return float("inf")


def governance_update(recommendation_id: str, new_status: str,
                      by: str, evidence: str) -> dict:
    """Recommendation lifecycle transitions (M5 validation #6):
    OPEN -> ACCEPTED -> IMPLEMENTED -> VALIDATED_HELPFUL |
    VALIDATED_NOT_HELPFUL | REJECTED. Append-only status rows —
    agents never call this; it is the human/governance path, and the
    `by` field says who."""
    ALLOWED = {"ACCEPTED", "REJECTED", "IMPLEMENTED",
               "VALIDATED_HELPFUL", "VALIDATED_NOT_HELPFUL"}
    if new_status not in ALLOWED:
        raise FirewallRefusal(f"invalid lifecycle status {new_status}")
    import time as _t
    row = {"kind": "status_update",
           "recommendation_id": recommendation_id,
           "status": new_status, "by": by, "evidence": evidence,
           "ts": int(_t.time())}
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _open_keys():
    keys = set()
    if LEDGER.exists():
        for l in LEDGER.open():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("kind") == "status_update":
                continue
            if r.get("status") == "OPEN":
                keys.add(r.get("dedupe_key"))
    return keys


def submit(agent: str, action_class: str, finding: str,
           recommendation: str, evidence: list[str],
           targeted_loss_term: str | None = None,
           risk: str | None = None, n: int | None = None,
           effect: float | None = None,
           priority: str | None = None) -> dict:
    if action_class not in VALID:
        raise FirewallRefusal(f"unknown action_class {action_class!r}")
    if not evidence:
        raise FirewallRefusal("recommendation without artifact "
                              "evidence — prose intuition is not "
                              "admissible")
    if action_class == "PROPOSE" and not targeted_loss_term:
        raise FirewallRefusal(
            "PROPOSE without targeted_loss_term — every proposal must "
            "name the measured loss term it should improve")
    if action_class in ALWAYS_BLOCKED:
        status = "REJECTED_BY_FIREWALL"
    elif action_class in M6_WHITELIST_PENDING:
        status = "BLOCKED_UNTIL_M6"
    else:
        status = "OPEN"
    dk = hashlib.sha256(
        f"{agent}|{action_class}|{recommendation}".encode()
    ).hexdigest()[:16]
    if status == "OPEN" and dk in _open_keys():
        status = "DUPLICATE"
    row = {"recommendation_id":
           f"{agent}-{int(time.time())}-{dk[:6]}",
           "ts": int(time.time()), "agent": agent,
           "action_class": action_class,
           "finding": finding, "recommendation": recommendation,
           "evidence": evidence,
           "targeted_loss_term": targeted_loss_term,
           "risk": risk, "n": n, "effect": effect,
           "priority": priority, "status": status,
           "dedupe_key": dk}
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row
