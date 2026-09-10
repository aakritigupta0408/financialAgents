"""Emit results/freeze_convergence.json (PM 09-10) — surfaces the
invariant-wall state vs the daemon's cached fail-closed state so
Watchtower can show WHEN they disagree and when convergence is due.
Observation only (Teaching/Reliability plane)."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CACHE_SLA_S = 60          # daemon _FC_CACHE ttl
POLL_S = 30               # daemon loop cadence


def main():
    now = time.time()
    try:
        inv = json.loads((RES / "invariants.json").read_text())
        wall = "GREEN" if inv.get("health") == "green" else "RED"
        wall_age = round(now - (RES / "invariants.json")
                         .stat().st_mtime)
    except Exception:
        wall, wall_age = "UNKNOWN", None
    try:
        st = json.loads((RES / "online_status.json").read_text())
        cached = st.get("runtime_state")
        why = st.get("runtime_state_why")
        hb_age = round(now - (st.get("alive_at") or 0))
    except Exception:
        cached, why, hb_age = "UNKNOWN", None, None
    # they "disagree" when the wall is GREEN but the desk still holds
    # a FREEZE that blames the invariant wall
    disagree = (wall == "GREEN" and cached == "FREEZE_NEW_ENTRIES"
                and "invariant" in str(why or "").lower())
    doc = {"generated_ts": int(now),
           "invariant_wall_state": wall,
           "invariant_wall_age_s": wall_age,
           "daemon_cached_state": cached,
           "daemon_cached_why": why,
           "daemon_heartbeat_age_s": hb_age,
           "state_age_s": wall_age,
           "expected_next_refresh_s": (CACHE_SLA_S + POLL_S),
           "disagree": disagree,
           "note": ("wall GREEN but desk still FREEZE — convergence "
                    "should complete within cache+poll SLA; a stale "
                    "hold past that is a freeze-state-convergence bug"
                    if disagree else "wall and desk agree")}
    (RES / "freeze_convergence.json").write_text(
        json.dumps(doc, indent=1))
    print(f"freeze_convergence: wall={wall} cached={cached} "
          f"disagree={disagree}")


if __name__ == "__main__":
    main()
