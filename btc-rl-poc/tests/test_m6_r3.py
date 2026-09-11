"""M6.3 chaos-certification matrix for R3 (RETRY_PUBLISHER).

17 contract fixtures in 14 groups; destination simulated with
file:// URLs; publisher simulated with injectable commands. The
flagship: publisher returns rc=0 but the DESTINATION is wrong —
command success is never trusted.

Run: python3 tests/test_m6_r3.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import meta_monitor                                # noqa: E402
import repair_r3 as r3                             # noqa: E402

PY = sys.executable
GOOD = ("import shutil,sys;"
        "shutil.copy(sys.argv[1]+'/a3_live.json',"
        "sys.argv[2]+'/a3_live.json')")
NOOP = "pass"
WRONG_GEN = ("import json,sys;"
             "json.dump({'experiment_id':'A3-v1.1','generated_ts':"
             "9999999999,'forward':{'eligible':56}},"
             "open(sys.argv[2]+'/a3_live.json','w'))")
WRONG_N = ("import json,sys,time;"
           "d=json.load(open(sys.argv[1]+'/a3_live.json'));"
           "d['forward']['eligible']-=1;"
           "json.dump(d,open(sys.argv[2]+'/a3_live.json','w'))")
FAIL_CMD = "import sys;sys.exit(1)"


def mkcfg(td, cmd=GOOD, local_n=12):
    res = Path(td) / "res"
    dest = Path(td) / "dest"
    res.mkdir()
    dest.mkdir()
    cfg = dict(r3.CFG)
    cfg.update({
        "res": res,
        "heal_log": res / "self_heal.jsonl",
        "incidents": res / "incidents.jsonl",
        "maintenance": res / "maintenance.flag",
        "artifact": "a3_live.json",
        "published_url": (dest / "a3_live.json").as_uri(),
        "publish_cmd": [PY, "-c", cmd, str(res), str(dest)],
        "published_lag_trigger_s": 900,
    })
    (res / "a3_live.json").write_text(json.dumps(
        {"experiment_id": "A3-v2", "spec_hash_ok": True,
         "generated_ts": int(time.time()),
         "forward": {"eligible": local_n}}))
    (res / "invariants.json").write_text(json.dumps({"failed": 0}))
    return cfg, res, dest


def publish_good(cfg, dest):
    import shutil
    shutil.copy(cfg["res"] / "a3_live.json", dest / "a3_live.json")


def states(cfg):
    if not cfg["heal_log"].exists():
        return []
    return [json.loads(l) for l in
            cfg["heal_log"].read_text().splitlines()]


def verify(cfg):
    """Independent verify with the fixture's destination wired into
    the module CFG the verifier consults."""
    old = dict(r3.CFG)
    r3.CFG.update({"published_url": cfg["published_url"],
                   "fetch_timeout_s": 10})
    try:
        return meta_monitor.verify_repairs(res_dir=cfg["res"],
                                           min_age_s=0)
    finally:
        r3.CFG.clear()
        r3.CFG.update(old)


def run():
    n = 0
    # 1 published deleted -> republish -> destination-verified RESTORED
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        assert r3.run_r3(cfg) == "REPAIR_ATTEMPTED"
        st = [s["state"] for s in states(cfg)]
        assert st[:3] == ["DETECTED", "CONTAINED", "REPAIR_ATTEMPTED"]
        out = verify(cfg)
        assert out[0]["state"] == "VERIFICATION_PASS", out
        assert states(cfg)[-1]["state"] == "RESTORED"
        assert "M6-R3" in cfg["incidents"].read_text()
        n += 1
        print("  1/16 missing publication -> republished -> "
              "destination-verified RESTORED (+incident)")

    # 2 published stale
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        (dest / "a3_live.json").write_text(json.dumps(
            {"experiment_id": "A3-v2",
             "generated_ts": int(time.time()) - 5000,
             "forward": {"eligible": 3}}))
        assert r3.run_r3(cfg) == "REPAIR_ATTEMPTED"
        det = next(s for s in states(cfg) if s["state"] == "DETECTED")
        assert det["trigger"] == "PUBLISHED_STALE"
        n += 1
        print("  2 published stale -> detected")

    # 3 published corrupted
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        (dest / "a3_live.json").write_text("{broken")
        r3.run_r3(cfg)
        det = next(s for s in states(cfg) if s["state"] == "DETECTED")
        assert det["trigger"] == "PUBLISHED_HASH_MISMATCH"
        n += 1
        print("  3 published corrupt -> detected")

    # 4/5 wrong artifact + wrong generation published
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        (dest / "a3_live.json").write_text(json.dumps(
            {"experiment_id": "A3-v1.1",
             "generated_ts": int(time.time()),
             "forward": {"eligible": 56}}))
        r3.run_r3(cfg)
        det = next(s for s in states(cfg) if s["state"] == "DETECTED")
        assert det["trigger"] == "PUBLISHED_HASH_MISMATCH"
        n += 1
        print("  4/5 wrong artifact / wrong generation -> detected")

    # 6 healthy -> no repair
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        publish_good(cfg, dest)
        assert r3.run_r3(cfg) == "HEALTHY_NO_TRIGGER"
        assert not cfg["heal_log"].exists()
        n += 1
        print("  6 healthy publication -> no trigger")

    # 7/8 canonical stale / corrupt -> refuse (dependency ordering)
    for name, mut in (("stale", lambda p: os.utime(
            p, (time.time() - 9000,) * 2)),
            ("corrupt", lambda p: p.write_text("{x"))):
        with TemporaryDirectory() as td:
            cfg, res, dest = mkcfg(td)
            mut(res / "a3_live.json")
            assert r3.run_r3(cfg) == "UPSTREAM_UNHEALTHY"
            assert states(cfg)[-1]["state"] == "FAILED_CLOSED"
            n += 1
            print(f"  7/8 canonical {name} -> R3 refuses "
                  "(R1/R2 territory)")

    # 9 publisher rc!=0 -> verify fails -> FAILED_CLOSED
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td, cmd=FAIL_CMD)
        assert r3.run_r3(cfg) == "REPAIR_ATTEMPTED"
        out = verify(cfg)
        assert out[0]["state"] == "VERIFICATION_FAIL"
        assert states(cfg)[-1]["state"] == "FAILED_CLOSED"
        n += 1
        print("  9 publisher rc=1 -> VERIFICATION_FAIL")

    # 10 rc=0 but destination absent — THE trust test
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td, cmd=NOOP)
        assert r3.run_r3(cfg) == "REPAIR_ATTEMPTED"
        att = next(s for s in states(cfg)
                   if s["state"] == "REPAIR_ATTEMPTED")
        assert att["publisher_rc"] == 0
        out = verify(cfg)
        assert out[0]["state"] == "VERIFICATION_FAIL"
        assert states(cfg)[-1]["state"] == "FAILED_CLOSED"
        n += 1
        print("  10 publisher SAYS success, destination absent -> "
              "VERIFICATION_FAIL (command rc never trusted)")

    # 11 rc=0 wrong generation content
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td, cmd=WRONG_GEN)
        r3.run_r3(cfg)
        out = verify(cfg)
        assert out[0]["state"] == "VERIFICATION_FAIL"
        assert out[0]["checks"]["experiment_id_matches"] is False
        n += 1
        print("  11 rc=0 wrong-generation content -> "
              "VERIFICATION_FAIL")

    # 12 semantic mismatch (published n < local n at repair)
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td, cmd=WRONG_N)
        r3.run_r3(cfg)
        out = verify(cfg)
        assert out[0]["state"] == "VERIFICATION_FAIL"
        assert out[0]["checks"]["semantic_eligible_n"] is False
        n += 1
        print("  12 semantic n-mismatch -> VERIFICATION_FAIL")

    # 13 retry exhaustion
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        now = time.time()
        with cfg["heal_log"].open("a") as f:
            for dt in (600, 300):
                f.write(json.dumps({"ts": now - dt,
                                    "repair_id": r3.REPAIR_ID,
                                    "state": "REPAIR_ATTEMPTED"})
                        + "\n")
        assert r3.run_r3(cfg) == "FAILED_CLOSED"
        st = [s["state"] for s in states(cfg)]
        assert "AUTO_REPAIR_EXHAUSTED" in st
        assert "open" in cfg["incidents"].read_text()
        n += 1
        print("  13 retry exhaustion -> FAILED_CLOSED, incident open")

    # 14 maintenance suppression
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        cfg["maintenance"].touch()
        assert r3.run_r3(cfg) == "MAINTENANCE_NO_TRIGGER"
        n += 1
        print("  14 maintenance -> no action")

    # 15 idempotence / no double publish
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        r3.run_r3(cfg)
        first = (dest / "a3_live.json").read_text()
        assert r3.run_r3(cfg) == "HEALTHY_NO_TRIGGER"
        assert (dest / "a3_live.json").read_text() == first
        n += 1
        print("  15 idempotent — second pass no-op, identical "
              "published state")

    # 17 scientific state unchanged through repair+verify
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        before = (res / "a3_live.json").read_text()
        r3.run_r3(cfg)
        verify(cfg)
        assert (res / "a3_live.json").read_text() == before
        n += 1
        print("  16/17 canonical source byte-untouched")

    # interaction ordering: canonical stale means BOTH R2-style and
    # R3 refuse — earliest broken dependency (compute) repairs first
    with TemporaryDirectory() as td:
        cfg, res, dest = mkcfg(td)
        os.utime(res / "a3_live.json", (time.time() - 9000,) * 2)
        assert r3.run_r3(cfg) == "UPSTREAM_UNHEALTHY"
        n += 1
        print("  +interaction: multi-fault -> R3 defers to "
              "upstream planes")

    print(f"M6-R3 chaos matrix: {n}/16 groups "
          "(17 contract fixtures + interaction) pass")
    assert n == 16


if __name__ == "__main__":
    run()
