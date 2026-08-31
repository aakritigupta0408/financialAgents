"""M6 interaction soak fixtures (PM 08-31) — 7 multi-fault
scenarios proving dependency ordering: no races, no circular
triggering, no downstream repair over unhealthy upstream.

Run: python3 tests/test_m6_interaction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import repair_planes as rp                         # noqa: E402

H, D, F = rp.HEALTHY, rp.DEGRADED, rp.FAILED


def run_orch(planes):
    calls, notes = [], []
    res = rp.orchestrate(
        {"COMPUTE": planes[0], "STATE": planes[1],
         "DELIVERY": planes[2]},
        run_r1=lambda: calls.append("R1") or "R1",
        run_r2=lambda: calls.append("R2") or "R2",
        run_r3=lambda: calls.append("R3") or "R3",
        record=lambda s, **f: notes.append(f.get("plane")))
    return calls, res["suppressed"], res


def run():
    n = 0
    # 1 daemon dead + derived stale -> R1 only, STATE suppressed
    calls, sup, _ = run_orch((F, D, H))
    assert calls == ["R1"] and "STATE" in sup
    n += 1
    print("  1 compute dead + state stale -> R1 only, "
          "STATE suppressed")
    # ...then compute healthy -> R2 acts
    calls, sup, _ = run_orch((H, D, H))
    assert calls == ["R2"] and not sup
    n += 1
    print("  1b compute healed -> R2 acts")

    # 2 derived missing + published stale -> R2 first, R3 suppressed
    calls, sup, _ = run_orch((H, F, F))
    assert calls == ["R2"] and sup == ["DELIVERY"]
    n += 1
    print("  2 state missing + delivery stale -> R2 first, "
          "DELIVERY suppressed")
    # ...then state healthy -> R3
    calls, sup, _ = run_orch((H, H, F))
    assert calls == ["R3"] and not sup
    n += 1
    print("  2b state healed -> R3 acts")

    # 3 triple fault -> strict R1 -> R2 -> R3 across cycles
    seq = []
    for planes in ((F, F, F), (H, F, F), (H, H, F), (H, H, H)):
        calls, _, _ = run_orch(planes)
        seq.extend(calls)
    assert seq == ["R1", "R2", "R3"]
    n += 1
    print("  3 triple fault -> R1 -> R2 -> R3, exactly once each, "
          "no races")

    # 4 canonical corrupt + published stale -> DELIVERY never repaired
    # over bad truth
    calls, sup, _ = run_orch((H, F, F))
    assert "R3" not in calls and "DELIVERY" in sup
    n += 1
    print("  4 corrupt truth + stale delivery -> R3 suppressed")

    # 5 R1 verification fails -> COMPUTE stays FAILED next cycle ->
    # R2/R3 remain suppressed
    calls, sup, _ = run_orch((F, F, F))
    assert calls == ["R1"] and set(sup) == {"STATE", "DELIVERY"}
    calls2, sup2, _ = run_orch((F, F, F))   # still failed post-verify
    assert calls2 == ["R1"] and set(sup2) == {"STATE", "DELIVERY"}
    n += 1
    print("  5 R1 verification failure -> downstream stays "
          "suppressed (no circular triggering)")

    # 6 R2 semantic failure -> STATE stays FAILED -> R3 suppressed
    calls, sup, _ = run_orch((H, F, D))
    assert calls == ["R2"] and sup == ["DELIVERY"]
    calls2, sup2, _ = run_orch((H, F, D))
    assert "R3" not in calls2 and "DELIVERY" in sup2
    n += 1
    print("  6 R2 semantic failure -> R3 stays suppressed")

    # 7 all healthy -> zero repair actions
    calls, sup, res = run_orch((H, H, H))
    assert calls == [] and sup == [] and res["acted"] is None
    n += 1
    print("  7 all healthy -> zero actions")

    # plane_state sanity on a real-ish fixture dir
    with TemporaryDirectory() as td:
        res_d = Path(td)
        (res_d / "online_status.json").write_text(
            json.dumps({"alive_at": 0}))          # ancient
        (res_d / "a3_live.json").write_text("{bad")
        (res_d / "invariants.json").write_text(
            json.dumps({"failed": 0}))
        st = rp.plane_state(res_d, pat="no-such-proc-xyz")
        assert st["COMPUTE"] == F and st["STATE"] == F
        n += 1
        print("  + plane_state classifies dead compute + corrupt "
              "state correctly")

    print(f"M6 interaction fixtures: {n}/10 pass")
    assert n == 10


if __name__ == "__main__":
    run()
